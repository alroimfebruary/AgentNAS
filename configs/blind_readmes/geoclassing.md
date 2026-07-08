# Task Dataset

## Specifications
- Input shape: (3, 60, 60) — channels-first
- Number of classes: 10
- Loss function: CrossEntropyLoss
- Metric: 1 - accuracy (lower is better)

## Data Format

Four NumPy `.npy` files:

| File | Shape | Dtype |
|------|-------|-------|
| `train_x.npy` | (43821, 3, 60, 60) | float32 |
| `train_y.npy` | (43821,) | int64 |
| `valid_x.npy` | (8785, 3, 60, 60) | float32 |
| `valid_y.npy` | (8785,) | int64 |

## Loading

```python
import numpy as np
train_x = np.load('train_x.npy')  # (43821, 3, 60, 60)
train_y = np.load('train_y.npy')  # (43821,)
valid_x = np.load('valid_x.npy')  # (8785, 3, 60, 60)
valid_y = np.load('valid_y.npy')  # (8785,)
```

## Splits in this directory
- Training: 43,821 samples
- Validation: 8,785 samples
- Labels are integers in [0, 9]

Test samples are held out for evaluation and are not accessible from this directory.
