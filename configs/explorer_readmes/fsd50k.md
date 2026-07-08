# FSD50K

**Task Type:** 2D Multi-label Prediction
**Domain:** Sound event classification
**Problem:** Multi-label audio classification (200 sound event categories)
**Metric:** 1 - mAP (mean Average Precision, lower is better)

Audio clips have been preprocessed into 96×101 log-mel spectrograms. Each clip can belong to any number of the 200 labels.

## Data Format

Precomputed spectrograms are stored as PyTorch `.pt` files. Raw audio chunks and per-split metadata CSVs are also present.

### Precomputed features (primary)

In `precomputed/`:

| File | Contents |
|------|----------|
| `fsd50k_tr.pt` | Training set |
| `fsd50k_val.pt` | Validation set |

Each `.pt` file is a torch dict with:
- `spectrograms`: float32 tensor, shape (N, 1, 96, 101)
- `labels`: float32 tensor, shape (N, 200) — binary multi-hot labels
- `clip_ids`: int64 tensor, shape (N,) — source-clip identifier for each chunk (used for clip-level mAP aggregation)

### Metadata and raw audio

- `chunks/tr.csv`, `chunks/val.csv` — per-chunk metadata
- `chunks/lbl_map.json` — label name → index mapping
- `dev_22050_chunks/` — raw 22050 Hz audio chunks for training+validation
- `noise_22050/` — noise samples for augmentation

## Loading

```python
import torch

train = torch.load('precomputed/fsd50k_tr.pt', weights_only=False, map_location='cpu', mmap=True)
val = torch.load('precomputed/fsd50k_val.pt', weights_only=False, map_location='cpu', mmap=True)

train_x = train['spectrograms']   # (N_tr, 1, 96, 101) float32
train_y = train['labels']          # (N_tr, 200) float32
train_clip_ids = train['clip_ids'] # (N_tr,) int64
```

## Details

- Audio sample rate: 22,050 Hz, ~1 second per chunk (chunked from variable-length clips)
- Spectrogram: 96 mel bands × 101 frames
- Each chunk inherits its source clip's labels; a single clip typically has multiple chunks. Evaluation aggregates chunk predictions per clip (mean pool) before computing per-class AP.
- Loss: BCEWithLogitsLoss (multi-label). Output raw logits — no sigmoid.
