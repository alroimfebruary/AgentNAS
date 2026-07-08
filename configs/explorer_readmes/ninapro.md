# NinaPro

**Task Type:** 2D Point Prediction
**Domain:** Electromyography (EMG) signals for hand gesture recognition
**Problem:** Multi-class classification (18 hand gestures)
**Metric:** 0-1 error (1 - top-1 accuracy, lower is better)

Each sample is a short EMG recording from 16 sensors over 52 time steps, normalized to [0, 1]. The model classifies the recorded gesture.

## Data Format

NumPy `.npy` files. In this directory you have access to:

- `ninapro_train.npy`: float64 array, shape (2638, 52, 16)
- `label_train.npy`:   float64 array, shape (2638,) — 18 unique class values
- `ninapro_val.npy`:   float64 array, shape (659, 52, 16)
- `label_val.npy`:     float64 array, shape (659,)

(Test samples are held out for evaluation and not accessible from this directory.)

## Loading

```python
import numpy as np

data_train = np.load('ninapro_train.npy')   # (2638, 52, 16)
label_train = np.load('label_train.npy')    # (2638,)
data_val = np.load('ninapro_val.npy')       # (659, 52, 16)
label_val = np.load('label_val.npy')        # (659,)
```

## Preprocessing (benchmark convention)

Raw data is (N, 52, 16). The benchmark transposes to (N, 16, 52) and adds a channel dimension to get (N, 1, 16, 52) before passing to the model:

```python
data_train = data_train.transpose(0, 2, 1)[:, np.newaxis, :, :]   # (2638, 1, 16, 52)
```

Loss: CrossEntropyLoss.
