# Spherical CIFAR-100

**Task Type:** 2D Point Prediction
**Domain:** Spherically-projected images (CIFAR-100 mapped onto a sphere)
**Problem:** Image classification (100 classes)
**Metric:** 0-1 error (1 - top-1 accuracy, lower is better)

Each CIFAR-100 image has been projected onto a spherical surface, yielding a 60×60 RGB input that exhibits distortions due to the projection. The task is standard multi-class classification over the 100 original CIFAR-100 categories.

## Data Format

A single pickle file `s2_cifar100` with a nested dict structure. In this directory you have access to:

- `s2_cifar100` — pickled dict with key `'train'` (the test split has been removed):
  - `data['train']['images']`: uint8 array, shape (50000, 3, 60, 60) — RGB, channels-first
  - `data['train']['labels']`: int array, shape (50000,), values in [0, 99]

(Test samples are held out for evaluation and not accessible from this directory.)

## Loading

```python
import pickle

with open('s2_cifar100', 'rb') as f:
    data = pickle.load(f, encoding='bytes')

train = data['train']
images = train['images']  # (50000, 3, 60, 60) uint8
labels = train['labels']   # (50000,) int
# Model input expects float32 in [0, 1]
images = images.astype('float32') / 255.0
```

## Splits (in this directory)

- Training: 40,000 samples (first 80% of `train`)
- Validation: 10,000 samples (last 20% of `train`, sequential split)

Loss: CrossEntropyLoss.
