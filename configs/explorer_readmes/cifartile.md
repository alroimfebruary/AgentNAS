# CIFARTile

The CIFARTile dataset is constructed using batches of images from the popular CIFAR-10 dataset.

Each CIFARTile image is a joining of four CIFAR-10 images in a 2x2 grid pattern. The task is for a model to identify the number of distinct CIFAR-10 classes that appear in the CIFARTile image. The original CIFAR-10 has 10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck. An example CIFARTile image might include two images of horses, a car, and a bird.

The data is in a channels-first format with a shape of (n, 3, 64, 64). In this directory you have access to:
- Training set: 45,000 samples
- Validation set: 15,000 samples

(Test samples are held out for evaluation and not accessible from this directory.)

There are four classes in the dataset, distributed evenly between train and val (17,500 examples of each class across splits).

Each image's label represents the number of CIFAR-10 classes that appear minus one, for zero-indexing:

- 0: All sub-images belong to the same CIFAR-10 class.
- 1: There are two CIFAR-10 classes among the sub-images.
- 2: There are three CIFAR-10 classes among the sub-images.
- 3: All sub-images belong to different CIFAR-10 classes.

## Files

- `train_x.npy`: (45000, 3, 64, 64) float32
- `train_y.npy`: (45000,) int64
- `valid_x.npy`: (15000, 3, 64, 64) float32
- `valid_y.npy`: (15000,) int64

## Loading

```python
import numpy as np
train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')
valid_x = np.load('valid_x.npy')
valid_y = np.load('valid_y.npy')
```
