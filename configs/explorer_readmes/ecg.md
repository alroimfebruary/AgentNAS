# ECG

**Task Type:** 1D Point Prediction
**Domain:** Cardiac rhythm classification from electrocardiogram signals
**Problem:** Multi-class classification (4 rhythm types: Normal, AFib, Other, Noisy)
**Metric:** 1 - macro F1 (lower is better)

Each recording is a 1D ECG signal (single lead) sampled at 300 Hz, with a variable total duration (commonly 30 or 60 seconds). The benchmark slices each recording into fixed-size windows and aggregates predictions per-recording via majority vote.

## Data Format

A single pickle file. In this directory you have access to:

- `challenge2017.pkl` — dict with two parallel lists:
  - `data`: list of variable-length 1D float arrays (each a single ECG recording)
  - `label`: list of single-character strings — one of `N`, `A`, `O`, `~`

(Final-test recordings have been removed from this file. What you see is the train+val portion. The benchmark further subdivides this at runtime.)

## Loading

```python
import pickle
import numpy as np

with open('challenge2017.pkl', 'rb') as f:
    data = pickle.load(f, encoding='bytes')

signals = data['data']   # list[np.ndarray], each (T_i,) float
labels = data['label']    # list[str], length len(signals)

# Example: number of recordings + a sample
print(len(signals), signals[0].shape, labels[0])
```

## Preprocessing (benchmark convention)

1. Per-signal z-score normalization
2. Slide windows of length 1000 with stride 500 (class-dependent stride for oversampling of minority classes)
3. Each window becomes one (1, 1000) training sample labelled by its source recording's class
4. Evaluation: window-level predictions are aggregated per recording via majority vote, then macro F1 is computed

## Label Encoding

- `N` → 0 (Normal sinus rhythm)
- `A` → 1 (Atrial fibrillation)
- `O` → 2 (Other rhythm)
- `~` → 3 (Too noisy to classify)

Loss: CrossEntropyLoss on windowed samples.

## Splits

The benchmark computes splits at runtime using `sklearn.model_selection.train_test_split`. In this directory, final-test recordings have already been removed; you will observe roughly 7,675 recordings.
