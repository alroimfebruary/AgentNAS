# GeoClassing

This dataset is constructed using images from BigEarthNet. The GeoClassing dataset is comprised of BigEarthNet samples with corresponding labels that indicate the location being captured in the image.

The data is in a channels-first format with a shape of (n, 3, 60, 60). In this directory you have access to:
- Training set: 43,821 samples
- Validation set: 8,785 samples

(Test samples are held out for evaluation and not accessible from this directory.)

There are ten classes in the dataset, distributed across the subsets:

- Ireland: 0
- Lithuania: 1
- Austria: 2
- Finland: 3
- Serbia: 4
- Belgium: 5
- Luxembourg: 6
- Kosovo: 7
- Switzerland: 8
- Portugal: 9

## Files

- `train_x.npy`: (43821, 3, 60, 60) float32
- `train_y.npy`: (43821,) int64
- `valid_x.npy`: (8785, 3, 60, 60) float32
- `valid_y.npy`: (8785,) int64

## Loading

```python
import numpy as np
train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')
valid_x = np.load('valid_x.npy')
valid_y = np.load('valid_y.npy')
```
