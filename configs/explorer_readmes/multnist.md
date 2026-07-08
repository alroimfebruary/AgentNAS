# MultNIST

The MultNIST dataset is a constructed dataset from MNIST Images. The intention of this dataset is to require machine learning models to do more than just image classification but also perform a calculation, in this case multiplication followed by a mod. For each image, three MNIST Images were randomly chosen and combined together through the colour channels, resulting in a three colour-channel image so each MNIST image represents one colour channel.

The data is in a channels-first format with a shape of (n, 3, 28, 28). In this directory you have access to:
- Training set: 50,000 samples
- Validation set: 10,000 samples

(Test samples are held out for evaluation and not accessible from this directory.)

There are ten classes in the dataset, distributed evenly between train and val (7,000 examples of each class across splits).

The label of each image is generated using the formula `(r * b * g) % 10` where r, g, and b are the red, green, and blue colour channels respectively. An example of a MultNIST image would be an rgb configuration of 3, 7, and 4, which would result in a label of 4 ((3 * 7 * 4) % 10).

## Files

- `train_x.npy`: (50000, 3, 28, 28) float32
- `train_y.npy`: (50000,) int64
- `valid_x.npy`: (10000, 3, 28, 28) float32
- `valid_y.npy`: (10000,) int64

## Loading

```python
import numpy as np
train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')
valid_x = np.load('valid_x.npy')
valid_y = np.load('valid_y.npy')
```
