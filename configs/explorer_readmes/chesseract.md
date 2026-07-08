# Chesseract

The Chesseract dataset is constructed from publicly accessible Chess games played by Grandmasters: Bobby Fischer, Garry Kasparov, Magnus Carlsen, Viswanathan Anand, Hikaru Nakamura, Anatoly Karpov, Fabiano Caruana, and Mikhail Tal.

The final 15% of board states were extracted, where the Nth board state refers to the position after N moves.

Chesseract uses 12 channels to encode the information rather than the traditional 2 or 3 for grayscale and RGB images respectively. Each channel represents a singular piece type and colour combination. In essence, this allows representation of the pieces and their position on the board in a one-hot encoded format.

The data is in a channels-first format with a shape of (n, 12, 8, 8). In this directory you have access to:
- Training set: 49,998 samples
- Validation set: 9,999 samples

(Test samples are held out for evaluation and not accessible from this directory.)

There are three classes in the dataset, distributed as evenly as possible between train and val (23,333 examples of each class across splits):

- White wins: 0
- Draw: 1
- Black wins: 2

## Files

- `train_x.npy`: (49998, 12, 8, 8) float32
- `train_y.npy`: (49998,) int64
- `valid_x.npy`: (9999, 12, 8, 8) float32
- `valid_y.npy`: (9999,) int64

## Loading

```python
import numpy as np
train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')
valid_x = np.load('valid_x.npy')
valid_y = np.load('valid_y.npy')
```
