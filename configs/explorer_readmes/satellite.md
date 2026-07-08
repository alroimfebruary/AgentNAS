# Satellite

**Task Type:** 1D Point Prediction
**Domain:** Land cover classification from satellite time-series
**Problem:** Multi-class classification (24 land-cover categories)
**Metric:** 0-1 error (1 - top-1 accuracy, lower is better)

Each sample is a 46-step temporal sequence of satellite measurements. Labels indicate the land cover class at that pixel/site.

## Data Format

NumPy `.npy` dictionary files. In this directory you have access to:

- `satellite_train.npy` — pickled dict with keys `data` and `label`:
  - `data`: float64 array, shape (900000, 46), normalized roughly to [-1.0, 1.3]
  - `label`: float64 array, shape (900000,), values in [1, 24] (1-indexed)

(Test samples are held out for evaluation and not accessible from this directory.)

## Loading

```python
import numpy as np

train_dict = np.load('satellite_train.npy', allow_pickle=True).item()
features = train_dict['data']     # (900000, 46)
labels = train_dict['label']       # (900000,) values in [1, 24]

# Remap labels to 0-indexed for CrossEntropyLoss
labels = (labels - 1).astype(np.int64)   # now in [0, 23]
```

## Preprocessing (benchmark convention)

1. Label remap: 1-indexed → 0-indexed
2. Z-score normalization
3. Dimension expansion: (N, 46) → (N, 1, 46) to add a channel dimension

## Splits (in this directory)

- Training: 800,000 samples (first 800K of `satellite_train.npy`)
- Validation: 100,000 samples (last 100K of `satellite_train.npy`)
- Test: held out, not accessible here

Loss: CrossEntropyLoss.
