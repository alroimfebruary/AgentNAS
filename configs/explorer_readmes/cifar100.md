# CIFAR-100

**Task Type:** 2D Point Prediction
**Domain:** Natural image classification
**Problem:** Multi-class classification (100 fine-grained categories)
**Metric:** 0-1 error (1 - top-1 accuracy, lower is better)

CIFAR-100 is a collection of 32×32 colour natural images covering 100 fine-grained object categories (also grouped into 20 superclasses).

## Data Format

A single pickle file (no extension) holds the training+validation set. In this directory you have access to:

- `train` — Python-pickled dictionary with keys:
  - `b'data'`: uint8 array, shape (50000, 3072), flattened row-major (3, 32, 32) images
  - `b'fine_labels'`: list of 50000 ints, fine-grained labels in [0, 99]
  - `b'coarse_labels'`: list of 50000 ints, superclass labels in [0, 19]
- `meta` — pickled dictionary with label-name mappings

(Test samples are held out for evaluation and not accessible from this directory.)

## Loading

```python
import pickle
import numpy as np

with open('train', 'rb') as f:
    data = pickle.load(f, encoding='bytes')

images = np.array(data[b'data']).reshape(-1, 3, 32, 32)  # (50000, 3, 32, 32) uint8
labels = np.array(data[b'fine_labels'])                   # (50000,) int
```

## Splits (in this directory)

- Training: 40,000 samples (first 80% of `train`)
- Validation: 10,000 samples (last 20% of `train`, sequential split)

Loss: CrossEntropyLoss.
