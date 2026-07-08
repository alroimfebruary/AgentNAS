# Language

The Language dataset is a constructed dataset using words from aspell dictionaries. The intention of this dataset is to require machine learning models to not only perform image classification but also linguistic analysis to figure out which letter frequency is associated with each language. For each Language image we selected four six-letter words using the standard latin alphabet and removed any words with letters that used diacritics (such as é or ü) or included 'y' or 'z'.

We encode these words on a graph with one axis representing the index of the 24-character long string (the four words joined together) and the other representing the letter (going A-X).

The data is in a channels-first format with a shape of (n, 1, 24, 24). In this directory you have access to:
- Training set: 50,000 samples
- Validation set: 10,000 samples

(Test samples are held out for evaluation and not accessible from this directory.)

There are ten classes in the dataset, distributed evenly between train and val (7,000 examples of each class across splits).

The ten classes and corresponding numerical label are:

- English: 0
- Dutch: 1
- German: 2
- Spanish: 3
- French: 4
- Portuguese: 5
- Swahili: 6
- Zulu: 7
- Finnish: 8
- Swedish: 9

## Files

- `train_x.npy`: (50000, 1, 24, 24) float32
- `train_y.npy`: (50000,) int64
- `valid_x.npy`: (10000, 1, 24, 24) float32
- `valid_y.npy`: (10000,) int64

## Loading

```python
import numpy as np
train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')
valid_x = np.load('valid_x.npy')
valid_y = np.load('valid_y.npy')
```
