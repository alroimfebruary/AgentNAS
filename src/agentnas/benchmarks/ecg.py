"""ECG benchmark: 1D classification, 4 classes (cardiac rhythms)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from agentnas.benchmarks.base import BaseBenchmark

# Label encoding: N=0, A=1, O=2, ~=3
_LABEL_MAP = {"N": 0, "A": 1, "O": 2, "~": 3}
_WINDOW_SIZE = 1000
_STRIDE = 500
# Official NAS-Bench-360 uses class-dependent stride divisors for oversampling (H-03 fix)
_CLASS_STRIDE_DIVISORS = {0: 1, 1: 7, 2: 2, 3: 20}


def _zscore(signal: np.ndarray) -> np.ndarray:
    """Per-sample z-score normalization."""
    mean = signal.mean()
    std = signal.std()
    if std < 1e-8:
        return signal - mean
    return (signal - mean) / std


def _window_signal(
    signal: np.ndarray, window_size: int, stride: int
) -> list[np.ndarray]:
    """Slice a 1D signal into fixed-size windows."""
    windows = []
    for start in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[start : start + window_size])
    if not windows:
        # Pad short signals
        padded = np.zeros(window_size, dtype=signal.dtype)
        padded[: len(signal)] = signal
        windows.append(padded)
    return windows


class ECGBenchmark(BaseBenchmark):
    """ECG: (1, 1000) → 4 classes.

    Preprocessing: per-sample z-score, 1000-sample windows (stride 500).
    Splits: 80/10/10 via train_test_split(random_state=0).
    Evaluation: window predictions → per-recording majority vote → macro F1.
    Metric: 1 - macro F1 (lower=better).
    """

    loss_name = "cross_entropy"
    metric_name = "f1"
    sota_metric = 0.28
    sota_description = "Human SOTA 1-F1 = 0.28 (F1 = 0.72)"

    explorer_preprocessed_name = "challenge2017.pkl"

    def __init__(self, data_root: str):
        super().__init__(data_root, "ecg")
        self._data_dir = Path(data_root) / "ecg"

    def prepare_explorer_source(self, out_dir: Path, real_root: Path) -> None:
        """Save a copy of challenge2017.pkl with the final-mode test split removed.

        Replicates the benchmark loader's split (train_test_split(indices,
        test_size=0.1, random_state=0)) to identify and drop the ~10% real-test
        recordings, keeping the ~90% train+val portion.
        """
        out_path = out_dir / "challenge2017.pkl"
        src_path = real_root / "ecg" / "challenge2017.pkl"
        if not src_path.exists():
            raise FileNotFoundError(f"ecg source missing: {src_path}")
        with open(src_path, "rb") as f:
            data = pickle.load(f, encoding="bytes")
        signals = data["data"]
        labels = data["label"]
        n = len(signals)
        indices = np.arange(n)
        train_idx, _test_idx = train_test_split(indices, test_size=0.1, random_state=0)
        keep = set(train_idx.tolist())
        filtered = {
            "data": [signals[i] for i in range(n) if i in keep],
            "label": [labels[i] for i in range(n) if i in keep],
        }
        with open(out_path, "wb") as f:
            pickle.dump(filtered, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_raw(self) -> tuple[list[np.ndarray], list[int]]:
        """Load raw ECG recordings and labels."""
        with open(self._data_dir / "challenge2017.pkl", "rb") as f:
            data = pickle.load(f, encoding="bytes")
        signals = data["data"]
        labels = [_LABEL_MAP[l] for l in data["label"]]
        return signals, labels

    def _prepare_windows(
        self, signals: list[np.ndarray], labels: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Z-score normalize + window all signals.

        Uses class-dependent stride for ALL splits (matches official NAS-Bench-360).
        The official code applies the same slide_and_cut to train, val, and test.

        Returns:
            x: (N_windows, 1, 1000) float32
            y: (N_windows,) int64
            recording_ids: (N_windows,) int — which recording each window belongs to
        """
        all_windows = []
        all_labels = []
        all_rec_ids = []
        for rec_id, (sig, label) in enumerate(zip(signals, labels)):
            sig = sig.astype(np.float64)
            sig = _zscore(sig)
            # Class-dependent stride for all splits (matches official)
            stride = _STRIDE // _CLASS_STRIDE_DIVISORS.get(label, 1)
            windows = _window_signal(sig, _WINDOW_SIZE, stride)
            for w in windows:
                all_windows.append(w.astype(np.float32))
                all_labels.append(label)
                all_rec_ids.append(rec_id)

        x = np.stack(all_windows)[:, np.newaxis, :]  # (N, 1, 1000)
        y = np.array(all_labels, dtype=np.int64)
        rec_ids = np.array(all_rec_ids, dtype=np.int64)
        return x, y, rec_ids

    def _split_recordings(self, mode: str = "search"):
        """Split at recording level, then window.

        mode="search": 80/10/10 train/val/test
        mode="final": 90/10 trainval/test (no val)
        """
        signals, labels = self._load_raw()
        indices = np.arange(len(signals))

        # Official NAS-Bench-360 splits (matches read_data_physionet_4_with_val / _4)
        if mode == "search":
            # 80/10/10: test_size=0.2 then 50/50 val/test
            train_idx, temp_idx = train_test_split(
                indices, test_size=0.2, random_state=0
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, random_state=0
            )
            return (
                [signals[i] for i in train_idx],
                [labels[i] for i in train_idx],
                [signals[i] for i in val_idx],
                [labels[i] for i in val_idx],
            )
        else:  # final: 90/10 (different partition, matches official)
            train_idx, test_idx = train_test_split(
                indices, test_size=0.1, random_state=0
            )
            return (
                [signals[i] for i in train_idx],
                [labels[i] for i in train_idx],
                [signals[i] for i in test_idx],
                [labels[i] for i in test_idx],
            )

    def _make_loader(
        self, x: np.ndarray, y: np.ndarray, rec_ids: np.ndarray,
        batch_size: int, num_workers: int, shuffle: bool,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(x), torch.from_numpy(y),
            torch.from_numpy(rec_ids),
        )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )

    def get_search_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        train_sig, train_lab, val_sig, val_lab = self._split_recordings("search")
        train_x, train_y, train_rec = self._prepare_windows(train_sig, train_lab, )
        val_x, val_y, val_rec = self._prepare_windows(val_sig, val_lab, )
        return (
            self._make_loader(train_x, train_y, train_rec, batch_size, num_workers, shuffle=True),
            self._make_loader(val_x, val_y, val_rec, batch_size, num_workers, shuffle=False),
        )

    def get_final_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        trainval_sig, trainval_lab, test_sig, test_lab = self._split_recordings("final")
        trainval_x, trainval_y, trainval_rec = self._prepare_windows(trainval_sig, trainval_lab, )
        test_x, test_y, test_rec = self._prepare_windows(test_sig, test_lab, )
        return (
            self._make_loader(trainval_x, trainval_y, trainval_rec, batch_size, num_workers, shuffle=True),
            self._make_loader(test_x, test_y, test_rec, batch_size, num_workers, shuffle=False),
        )

    def loss_fn(self) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss()

    def compute_metric(
        self,
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
        all_extras: list[torch.Tensor] | None = None,
    ) -> float:
        """1 - macro F1 (lower=better).

        If recording IDs are provided (via all_extras), aggregates window
        predictions per recording via majority vote before computing F1.
        Otherwise falls back to window-level F1.
        """
        preds = torch.cat(all_preds, dim=0)
        pred_classes = preds.argmax(dim=1).cpu().numpy() if preds.dim() == 2 else preds.cpu().numpy()
        target_classes = torch.cat(all_targets, dim=0).cpu().numpy()

        from sklearn.metrics import f1_score
        from collections import Counter

        if all_extras is not None:
            # Recording-level majority voting
            rec_ids = torch.cat(all_extras, dim=0).cpu().numpy()
            rec_preds = {}  # rec_id → list of predicted classes
            rec_labels = {}  # rec_id → true label (same for all windows of a recording)

            for pred, target, rec_id in zip(pred_classes, target_classes, rec_ids):
                rid = int(rec_id)
                rec_preds.setdefault(rid, []).append(int(pred))
                rec_labels[rid] = int(target)

            # Majority vote per recording
            final_preds = []
            final_targets = []
            for rid in sorted(rec_preds.keys()):
                votes = Counter(rec_preds[rid])
                majority = votes.most_common(1)[0][0]
                final_preds.append(majority)
                final_targets.append(rec_labels[rid])

            f1 = f1_score(final_targets, final_preds, average="macro", zero_division=0)
        else:
            # Fallback: window-level F1
            f1 = f1_score(target_classes, pred_classes, average="macro", zero_division=0)

        return 1.0 - f1

    def input_shape(self) -> tuple[int, ...]:
        return (1, 1000)

    def output_spec(self) -> dict[str, Any]:
        return {"num_classes": 4}
