# DeepSEA

**Task Type:** 1D Multi-label Prediction
**Domain:** DNA sequence analysis for chromatin state prediction
**Problem:** Multi-label binary classification (36 chromatin features)
**Metric:** 1 - mean per-class AUROC (lower is better)

Each sample is a 1000-base-pair DNA sequence one-hot encoded over {A, C, G, T}. Each label is a binary indicator for one of 36 chromatin features.

## Data Format

A NumPy compressed archive. In this directory you have access to:

- `deepsea_filtered.npz` — compressed archive with keys:
  - `x_train`: uint8 array, shape (71753, 1000, 4) — training one-hot sequences
  - `y_train`: uint8 array, shape (71753, 36) — training multi-label targets
  - `x_val`:   uint8 array, shape (2490, 1000, 4) — validation sequences
  - `y_val`:   uint8 array, shape (2490, 36) — validation targets

(Test arrays have been removed from this file and are not accessible from this directory.)

## Loading

```python
import numpy as np

data = np.load('deepsea_filtered.npz')
x_train = data['x_train']  # (71753, 1000, 4) uint8
y_train = data['y_train']  # (71753, 36) uint8

# Transpose to channels-first for model input
x_train = x_train.transpose(0, 2, 1).astype('float32')  # (71753, 4, 1000)
```

## Preprocessing (benchmark convention)

- Transpose: (N, 1000, 4) → (N, 4, 1000) for channels-first convolutions
- Cast inputs to float32 and labels to float32

Loss: BCEWithLogitsLoss (multi-label). Output raw logits — no sigmoid.

## Splits (in this directory)

- Training: 71,753 samples
- Validation: 2,490 samples
- Test: held out, not accessible here
