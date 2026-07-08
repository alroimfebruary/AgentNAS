# Task Dataset

## Specifications
- Input shape: (3, 64, 64) — channels-first
- Number of classes: 4
- Loss function: CrossEntropyLoss
- Metric: 1 - accuracy (lower is better)

## Data Format

Four NumPy `.npy` files:

| File | Shape | Dtype |
|------|-------|-------|
| `train_x.npy` | (45000, 3, 64, 64) | float32 |
| `train_y.npy` | (45000,) | int64 |
| `valid_x.npy` | (15000, 3, 64, 64) | float32 |
| `valid_y.npy` | (15000,) | int64 |

## Loading

```python
import numpy as np
train_x = np.load('train_x.npy')  # (45000, 3, 64, 64)
train_y = np.load('train_y.npy')  # (45000,)
valid_x = np.load('valid_x.npy')  # (15000, 3, 64, 64)
valid_y = np.load('valid_y.npy')  # (15000,)
```

## Splits in this directory
- Training: 45,000 samples
- Validation: 15,000 samples
- Labels are integers in [0, 3]
- Class distribution is balanced across splits

Test samples are held out for evaluation and are not accessible from this directory.
