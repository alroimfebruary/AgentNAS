# Task Dataset

## Specifications
- Input shape: (3, 32, 32) — channels-first
- Number of classes: 100
- Loss function: CrossEntropyLoss
- Metric: 1 - accuracy (lower is better)

## Data Format

Python pickle files (no extension):

| File | Contents |
|------|----------|
| `train` | Training+validation source: 50,000 samples |
| `meta` | Label index mapping |

The `train` pickle contains a dict (keys are bytes):
- `b'data'`: uint8 array, shape (50000, 3072) — flattened (3, 32, 32) images in row-major order
- `b'fine_labels'`: list of 50,000 ints — labels in [0, 99]
- `b'coarse_labels'`: list of 50,000 ints — coarser 20-category labels (unused here)

## Loading

```python
import pickle
import numpy as np

with open('train', 'rb') as f:
    data = pickle.load(f, encoding='bytes')

images = np.array(data[b'data']).reshape(-1, 3, 32, 32)  # (50000, 3, 32, 32) uint8
labels = np.array(data[b'fine_labels'])                   # (50000,) int
```

## Splits in this directory
- Training: 40,000 samples (first 80% of `train`)
- Validation: 10,000 samples (last 20% of `train`, sequential split)

Test samples are held out for evaluation and are not accessible from this directory.
