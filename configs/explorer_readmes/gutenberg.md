# Gutenberg

The Gutenberg dataset is a constructed dataset containing phrases from famous literary works made available by Project Gutenberg (www.gutenberg.org), which provides free ebooks of literary works that are no longer under US copyright protection.

This dataset was created by accessing several works by six popular authors (see label mapping below). The works downloaded were English translations chosen to represent a variety of cultures and time periods. Basic text preprocessing was applied over each text: removing punctuation, converting letters with diacritics to the base letter, and removing "structure" words (e.g., 'Chapter', 'Scene', 'Prologue').

Consecutive sequences of three words, each between 3 and 6 letters long, were extracted. In each sequence, the three words were padded up to 6 characters with spaces, then concatenated together to produce an 18-character string. These strings are the base for image creation. Training, test, and validation sequences were chosen such that there is no overlap between any sequence across any data split.

The strings are encoded into images: the x-axis represents each character's index in the 18-character string, and the y-axis represents the corresponding letter (the axis is arranged alphabetically A-Z with a space represented underneath the Z).

The data is in a channels-first format with a shape of (n, 1, 27, 18). In this directory you have access to:
- Training set: 45,000 samples
- Validation set: 15,000 samples

(Test samples are held out for evaluation and not accessible from this directory.)

There are six classes in the dataset, distributed evenly between train and val (11,000 examples of each class across splits):

- aquinas: 0
- confucius: 1
- hawthorne: 2
- plato: 3
- shakespeare: 4
- tolstoy: 5

## Files

- `train_x.npy`: (45000, 1, 27, 18) float32
- `train_y.npy`: (45000,) int64
- `valid_x.npy`: (15000, 1, 27, 18) float32
- `valid_y.npy`: (15000,) int64

## Loading

```python
import numpy as np
train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')
valid_x = np.load('valid_x.npy')
valid_y = np.load('valid_y.npy')
```
