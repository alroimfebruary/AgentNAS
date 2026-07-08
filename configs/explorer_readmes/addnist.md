# AddNIST

The AddNIST dataset is a constructed dataset from MNIST Images. The intention of this dataset is to require machine learning models to do more than just image classification but also perform a calculation, in this case addition. For each image, three MNIST Images were randomly chosen and combined together through the colour channels, resulting in a three colour-channel image so each MNIST image represents one colour channel.

The data is in a channels-first format with a shape of (n, 3, 28, 28) where n is the number of samples in the corresponding set. In this directory you have access to:
- Training set: 45,000 samples
- Validation set: 15,000 samples

(Test samples are held out for evaluation and not accessible from this directory.)

There are twenty classes in the dataset, distributed evenly between the training and validation subsets (3,500 examples of each class across splits).

The label of each image is generated using the formula `(r + b + g) - 1` where r, g, and b are the red, green, and blue colour channels respectively. An example of an AddNIST image would be an rgb configuration of 3, 7, and 4, which would result in a label of 13 ((3 + 7 + 4) - 1).

## Files

- `train_x.npy`: (45000, 3, 28, 28) float32
- `train_y.npy`: (45000,) int64
- `valid_x.npy`: (15000, 3, 28, 28) float32
- `valid_y.npy`: (15000,) int64

## Loading

```python
import numpy as np
train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')
valid_x = np.load('valid_x.npy')
valid_y = np.load('valid_y.npy')
```
