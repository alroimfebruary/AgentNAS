"""FSD50K benchmark: 2D audio multi-label classification, 200 classes."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from autospace.benchmarks.base import BaseBenchmark

_SAMPLE_RATE = 22050
_N_MELS = 96           # Official NAS-Bench-360: 96 mel bands (H-04 fix)
_HOP_LENGTH = 220      # int(22050 * 0.01) = 220
_N_FFT = 1920          # Official: 96 * 20 = 1920 (H-04 fix)
_WIN_LENGTH = 661      # Official: int(22050 * 0.03) = 661 (H-04 fix)
_TARGET_TIME_FRAMES = 101  # 1-second chunks → 101 frames
_TOP_DB = 150          # Matches official NAS-Bench-360


class FSD50KDataset(Dataset):
    """Dataset for FSD50K audio chunks.

    Loads WAV files, converts to mel-spectrogram on-the-fly.
    """

    def __init__(
        self,
        csv_path: str,
        audio_dir: str,
        label_map: dict[str, int],
        num_classes: int = 200,
        noise_dir: str | None = None,
        noise_prob: float = 0.0,
    ):
        self.audio_dir = Path(audio_dir)
        self.label_map = label_map
        self.num_classes = num_classes
        self.noise_dir = Path(noise_dir) if noise_dir else None
        self.noise_prob = noise_prob

        # Parse CSV: columns are "files,labels,ext"
        # Column 2 (ext) is the clip ID — used for clip-level mAP aggregation
        self.entries: list[tuple[str, list[int], int]] = []  # (filepath, label_ids, clip_id)
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Skip header
            for row in reader:
                if len(row) < 2:
                    continue
                filepath = row[0].strip()
                if filepath.startswith("data/"):
                    filepath = filepath[5:]
                label_str = row[1].strip()
                label_ids = []
                for lbl in label_str.split(","):
                    lbl = lbl.strip()
                    if lbl in self.label_map:
                        label_ids.append(self.label_map[lbl])
                # Extract clip ID from column 2 (ext) or derive from filename
                if len(row) > 2 and row[2].strip().isdigit():
                    clip_id = int(row[2].strip())
                else:
                    # Fallback: extract numeric prefix from filename (e.g., "219892_0000.wav" → 219892)
                    import re as _re
                    m = _re.match(r".*?(\d+)_\d+\.wav", filepath)
                    clip_id = int(m.group(1)) if m else hash(filepath) & 0x7FFFFFFF
                if label_ids:
                    self.entries.append((filepath, label_ids, clip_id))

        # Load noise files if provided
        self._noise_files: list[str] = []
        if self.noise_dir and self.noise_dir.exists():
            self._noise_files = [
                str(p) for p in self.noise_dir.glob("*.wav")
            ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        filepath, label_ids, clip_id = self.entries[idx]

        # Load audio
        audio_path = self.audio_dir / filepath
        waveform = self._load_audio(str(audio_path))

        # Background noise mixing during training
        if self.noise_prob > 0 and np.random.random() < self.noise_prob:
            waveform = self._add_noise(waveform)

        # Convert to mel-spectrogram
        mel = self._to_mel(waveform)

        # Multi-hot label encoding
        target = torch.zeros(self.num_classes)
        for lid in label_ids:
            target[lid] = 1.0

        return mel, target, torch.tensor(clip_id, dtype=torch.long)

    def _load_audio(self, path: str) -> np.ndarray:
        """Load WAV file as float32 numpy array."""
        import soundfile as sf

        try:
            audio, sr = sf.read(path, dtype="float32")
        except Exception:
            # Fallback: return silence
            return np.zeros(int(_SAMPLE_RATE * 1.0), dtype=np.float32)

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        return audio

    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Mix in random background noise."""
        if not self._noise_files:
            return audio
        noise_path = np.random.choice(self._noise_files)
        import soundfile as sf

        try:
            noise, _ = sf.read(noise_path, dtype="float32")
            if len(noise.shape) > 1:
                noise = noise.mean(axis=1)
            # Repeat/truncate to match length
            if len(noise) < len(audio):
                reps = int(np.ceil(len(audio) / len(noise)))
                noise = np.tile(noise, reps)
            noise = noise[: len(audio)]
            # Random SNR
            snr = np.random.uniform(0.1, 0.5)
            return audio + snr * noise
        except Exception:
            return audio

    def _to_mel(self, audio: np.ndarray) -> torch.Tensor:
        """Convert waveform to mel-spectrogram tensor."""
        import librosa

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=_SAMPLE_RATE,
            n_fft=_N_FFT,
            hop_length=_HOP_LENGTH,
            win_length=_WIN_LENGTH,
            n_mels=_N_MELS,
        )
        # Log mel — matches official NAS-Bench-360 FSD50K normalization (R1 fix)
        mel = librosa.power_to_db(mel, top_db=_TOP_DB)  # ref=1.0 (default), clips to top_db range
        mel += _TOP_DB / 2  # Offset by top_db/2
        mel -= mel.mean()  # Per-sample zero-centering

        # Pad or crop to target time frames
        if mel.shape[1] < _TARGET_TIME_FRAMES:
            pad_width = _TARGET_TIME_FRAMES - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant")
        else:
            mel = mel[:, :_TARGET_TIME_FRAMES]

        return torch.from_numpy(mel).float().unsqueeze(0)  # (1, n_mels, time)


class PrecomputedFSD50KDataset(Dataset):
    """Fast dataset loading pre-computed mel-spectrograms from .pt files.

    Supports SpecAugment-style augmentation (time/frequency masking) during training.
    """

    def __init__(self, pt_path: str, augment: bool = False):
        # Prefer /dev/shm copy (staged by pod_launcher.sh for FSD tasks) to
        # avoid cross-pod GPFS bandwidth contention. tmpfs-backed mmap reads
        # never hit the network. Fall back to GPFS path if not staged.
        shm_path = os.path.join("/dev/shm/fsd50k", os.path.basename(pt_path))
        actual_path = shm_path if os.path.exists(shm_path) else pt_path
        data = torch.load(actual_path, weights_only=False, map_location="cpu", mmap=True)
        self.spectrograms = data["spectrograms"]  # (N, 1, 96, 101)
        self.labels = data["labels"]  # (N, 200)
        self.clip_ids = data.get("clip_ids", None)  # (N,) int — may be absent in old files

        # Sanity check: reject precomputed data with wrong mel parameters
        n_mels = self.spectrograms.shape[2]
        if n_mels != _N_MELS:
            raise ValueError(
                f"Precomputed data has n_mels={n_mels}, expected {_N_MELS}. "
                f"Re-run scripts/precompute_fsd50k.py to regenerate with correct parameters."
            )
        self.augment = augment

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = self.spectrograms[idx].clone() if self.augment else self.spectrograms[idx]

        if self.augment:
            spec = self._spec_augment(spec)

        clip_id = self.clip_ids[idx] if self.clip_ids is not None else torch.tensor(0, dtype=torch.long)
        return spec, self.labels[idx], clip_id

    @staticmethod
    def _spec_augment(spec: torch.Tensor) -> torch.Tensor:
        """SpecAugment: random time and frequency masking."""
        import random
        _, n_freq, n_time = spec.shape  # (1, 96, 101)

        # Frequency masking (1-2 bands, width 5-15)
        for _ in range(random.randint(1, 2)):
            f_width = random.randint(5, min(15, n_freq // 4))
            f_start = random.randint(0, n_freq - f_width)
            spec[:, f_start : f_start + f_width, :] = 0.0

        # Time masking (1-2 bands, width 5-20)
        for _ in range(random.randint(1, 2)):
            t_width = random.randint(5, min(20, n_time // 4))
            t_start = random.randint(0, n_time - t_width)
            spec[:, :, t_start : t_start + t_width] = 0.0

        return spec


class FSD50KBenchmark(BaseBenchmark):
    """FSD50K: mel-spectrogram (1, 96, 101) → 200 multi-label classes.

    Uses pre-computed mel-spectrograms when available (fast tensor loading).
    Falls back to on-the-fly WAV → mel conversion if pre-computed data not found.
    Metric: 1 - mAP (lower=better).
    """

    def __init__(self, data_root: str):
        super().__init__(data_root, "fsd50k")
        self._data_dir = Path(data_root) / "fsd50k"
        self._precomputed_dir = self._data_dir / "precomputed"
        self._label_map: dict[str, int] | None = None

    def _get_label_map(self) -> dict[str, int]:
        if self._label_map is None:
            lbl_path = self._data_dir / "chunks" / "lbl_map.json"
            with open(lbl_path) as f:
                self._label_map = json.load(f)
        return self._label_map

    @property
    def _has_precomputed(self) -> bool:
        return (self._precomputed_dir / "fsd50k_tr.pt").exists()

    def get_search_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        if self._has_precomputed:
            train_ds = PrecomputedFSD50KDataset(
                str(self._precomputed_dir / "fsd50k_tr.pt"), augment=True
            )
            val_ds = PrecomputedFSD50KDataset(
                str(self._precomputed_dir / "fsd50k_val.pt"), augment=False
            )
        else:
            label_map = self._get_label_map()
            train_ds = FSD50KDataset(
                csv_path=str(self._data_dir / "chunks" / "tr.csv"),
                audio_dir=str(self._data_dir),
                label_map=label_map,
                noise_dir=str(self._data_dir / "noise_22050"),
                noise_prob=0.75,
            )
            val_ds = FSD50KDataset(
                csv_path=str(self._data_dir / "chunks" / "val.csv"),
                audio_dir=str(self._data_dir),
                label_map=label_map,
            )
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True,
                       persistent_workers=(num_workers > 0)),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True,
                       persistent_workers=(num_workers > 0)),
        )

    def get_final_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        from torch.utils.data import ConcatDataset

        if self._has_precomputed:
            train_ds = PrecomputedFSD50KDataset(
                str(self._precomputed_dir / "fsd50k_tr.pt"), augment=True
            )
            val_ds = PrecomputedFSD50KDataset(
                str(self._precomputed_dir / "fsd50k_val.pt"), augment=True
            )
            test_ds = PrecomputedFSD50KDataset(
                str(self._precomputed_dir / "fsd50k_eval.pt"), augment=False
            )
        else:
            label_map = self._get_label_map()
            train_ds = FSD50KDataset(
                csv_path=str(self._data_dir / "chunks" / "tr.csv"),
                audio_dir=str(self._data_dir),
                label_map=label_map,
                noise_dir=str(self._data_dir / "noise_22050"),
                noise_prob=0.75,
            )
            val_ds = FSD50KDataset(
                csv_path=str(self._data_dir / "chunks" / "val.csv"),
                audio_dir=str(self._data_dir),
                label_map=label_map,
                noise_dir=str(self._data_dir / "noise_22050"),
                noise_prob=0.75,
            )
            test_ds = FSD50KDataset(
                csv_path=str(self._data_dir / "chunks" / "eval.csv"),
                audio_dir=str(self._data_dir),
                label_map=label_map,
            )

        trainval_ds = ConcatDataset([train_ds, val_ds])
        return (
            DataLoader(trainval_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True,
                       persistent_workers=(num_workers > 0)),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True,
                       persistent_workers=(num_workers > 0)),
        )

    def loss_fn(self) -> torch.nn.Module:
        return torch.nn.BCEWithLogitsLoss()

    def compute_metric(
        self,
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
        all_extras: list[torch.Tensor] | None = None,
    ) -> float:
        """1 - mAP (lower=better).

        If clip IDs are provided (via all_extras), aggregates chunk predictions
        per clip via mean pooling before computing mAP (official FSD50K protocol).
        Otherwise computes chunk-level mAP (proxy for search phase).
        """
        preds = torch.cat(all_preds, dim=0).sigmoid().cpu().numpy()
        targets = torch.cat(all_targets, dim=0).cpu().numpy()

        # Clip-level aggregation if clip IDs available (B1 fix)
        if all_extras is not None:
            clip_ids = torch.cat(all_extras, dim=0).cpu().numpy()
            unique_clips = np.unique(clip_ids)
            clip_preds = np.zeros((len(unique_clips), preds.shape[1]))
            clip_targets = np.zeros((len(unique_clips), targets.shape[1]))
            for i, cid in enumerate(unique_clips):
                mask = clip_ids == cid
                clip_preds[i] = preds[mask].mean(axis=0)
                clip_targets[i] = targets[mask].max(axis=0)  # Union of chunk labels
            preds = clip_preds
            targets = clip_targets

        from sklearn.metrics import average_precision_score

        aps = []
        for i in range(preds.shape[1]):
            if targets[:, i].sum() > 0:
                ap = average_precision_score(targets[:, i], preds[:, i])
                aps.append(ap)
        mAP = np.mean(aps) if aps else 0.0
        return 1.0 - mAP

    def input_shape(self) -> tuple[int, ...]:
        return (1, _N_MELS, _TARGET_TIME_FRAMES)  # (1, 96, 101)

    def output_spec(self) -> dict[str, Any]:
        return {"num_classes": 200, "task_type": "multi_label"}
