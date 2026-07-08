"""Pre-compute mel-spectrograms for FSD50K to eliminate data loading bottleneck.

Converts WAV files → mel-spectrograms and saves as .pt tensor files.
One file per split: fsd50k_tr.pt, fsd50k_val.pt, fsd50k_eval.pt
Each is a dict: {"spectrograms": tensor(N, 1, 96, 101), "labels": tensor(N, 200)}
"""

import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

DATA_DIR = Path("NAS-Bench-360-Benchmarks/fsd50k")
OUT_DIR = DATA_DIR / "precomputed"
OUT_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 22050
N_MELS = 96            # Official: 96 mel bands (H-04 fix)
HOP_LENGTH = 220       # int(22050 * 0.01)
N_FFT = 1920           # Official: 96 * 20 (H-04 fix)
WIN_LENGTH = 661       # Official: int(22050 * 0.03) (H-04 fix)
TARGET_FRAMES = 101
TOP_DB = 150           # Matches official


def load_label_map():
    with open(DATA_DIR / "chunks" / "lbl_map.json") as f:
        return json.load(f)


def wav_to_mel(path: str) -> np.ndarray:
    """Convert WAV to mel-spectrogram (96, 101) float32."""
    try:
        audio, sr = sf.read(path, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
    except Exception as e:
        print(f"WARNING: Audio load failed: {path}: {e}", flush=True)
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_mels=N_MELS,
    )
    # Official NAS-Bench-360 normalization (R1 fix)
    mel = librosa.power_to_db(mel, top_db=TOP_DB)  # ref=1.0, clips to top_db range
    mel += TOP_DB / 2  # Offset
    mel -= mel.mean()  # Per-sample zero-centering

    # Pad or crop time axis
    if mel.shape[1] < TARGET_FRAMES:
        mel = np.pad(mel, ((0, 0), (0, TARGET_FRAMES - mel.shape[1])))
    else:
        mel = mel[:, :TARGET_FRAMES]

    return mel.astype(np.float32)


def process_split(csv_name: str, out_name: str):
    label_map = load_label_map()
    num_classes = len(label_map)

    # Parse CSV
    entries = []
    csv_path = DATA_DIR / "chunks" / csv_name
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
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
                if lbl in label_map:
                    label_ids.append(label_map[lbl])
            # Extract clip ID from column 2 or filename (C1 fix)
            if len(row) > 2 and row[2].strip().isdigit():
                clip_id = int(row[2].strip())
            else:
                import re
                m = re.match(r".*?(\d+)_\d+\.wav", filepath)
                clip_id = int(m.group(1)) if m else hash(filepath) & 0x7FFFFFFF
            if label_ids:
                entries.append((filepath, label_ids, clip_id))

    print(f"Processing {csv_name}: {len(entries)} entries")

    # Pre-compute in chunks to manage memory
    chunk_size = 50000
    all_mels = []
    all_labels = []
    all_clip_ids = []

    for start in range(0, len(entries), chunk_size):
        chunk = entries[start:start + chunk_size]
        mels = []
        labels = []
        clip_ids = []

        for filepath, label_ids, clip_id in tqdm(chunk, desc=f"{csv_name} [{start}:]"):
            audio_path = str(DATA_DIR / filepath)
            mel = wav_to_mel(audio_path)
            mels.append(mel)

            target = np.zeros(num_classes, dtype=np.float32)
            for lid in label_ids:
                target[lid] = 1.0
            labels.append(target)
            clip_ids.append(clip_id)

        all_mels.append(np.stack(mels)[:, np.newaxis, :, :])  # (N, 1, 96, 101)
        all_labels.append(np.stack(labels))
        all_clip_ids.append(np.array(clip_ids, dtype=np.int64))

    spectrograms = np.concatenate(all_mels, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    clip_ids_arr = np.concatenate(all_clip_ids, axis=0)

    print(f"Saving {out_name}: spectrograms={spectrograms.shape}, labels={labels.shape}, clip_ids={clip_ids_arr.shape}")
    torch.save({
        "spectrograms": torch.from_numpy(spectrograms),
        "labels": torch.from_numpy(labels),
        "clip_ids": torch.from_numpy(clip_ids_arr),
    }, OUT_DIR / out_name)
    print(f"Saved {out_name} ({(OUT_DIR / out_name).stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    splits = [
        ("tr.csv", "fsd50k_tr.pt"),
        ("val.csv", "fsd50k_val.pt"),
        ("eval.csv", "fsd50k_eval.pt"),
    ]

    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    for csv_name, out_name in splits:
        if target == "all" or target in csv_name:
            process_split(csv_name, out_name)

    print("Done!")
