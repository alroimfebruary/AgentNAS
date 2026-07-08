# Task Dataset

## Specifications
- Input shape: (12, 8, 8) — channels-first, 12 channels
- Number of classes: 3
- Loss function: CrossEntropyLoss
- Metric: 1 - accuracy (lower is better)

## Data Format

Four NumPy `.npy` files:

| File | Shape | Dtype |
|------|-------|-------|
| `train_x.npy` | (49998, 12, 8, 8) | float32 |
| `train_y.npy` | (49998,) | int64 |
| `valid_x.npy` | (9999, 12, 8, 8) | float32 |
| `valid_y.npy` | (9999,) | int64 |

## Loading

```python
import numpy as np
train_x = np.load('train_x.npy')  # (49998, 12, 8, 8)
train_y = np.load('train_y.npy')  # (49998,)
valid_x = np.load('valid_x.npy')  # (9999, 12, 8, 8)
valid_y = np.load('valid_y.npy')  # (9999,)
```

## Splits in this directory
- Training: 49,998 samples
- Validation: 9,999 samples
- Labels are integers in [0, 2]
- Class distribution is balanced across splits

Test samples are held out for evaluation and are not accessible from this directory.
