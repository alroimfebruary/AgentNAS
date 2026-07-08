# Cosmic

**Task Type:** 2D Dense Prediction
**Domain:** Cosmic ray detection in telescope images
**Problem:** Binary dense segmentation (pixel-wise cosmic ray identification)
**Metric:** 1 - AUROC (lower is better)

Each telescope image contains sparse cosmic-ray artifacts that must be identified at the pixel level. An additional "ignore" mask marks pixels that should be excluded from both loss and metric computation.

## Data Format

Serialized PyTorch dataset files, plus the loader module they depend on. In this directory you have access to:

- `cosmic_train.pt` — 4,347 training samples (serialized PyTorch Dataset)
- `cosmic_valid.pt` — 483 validation samples
- `cosmic_dataset.py` — required module for deserializing the `.pt` files
- `npy_train/` — raw training `.npy` image files (nested subdirectories, ~25K files). The `.pt` files reference these via relative paths.

(Test `.pt` and raw test directories are held out and not accessible here.)

Each raw `.npy` file is a (3, H, W) float64 array:
- `data[0]` — grayscale image
- `data[1]` — binary cosmic-ray mask (0 or 1)
- `data[2]` — ignore mask (1 where pixel is excluded)

The `.pt` datasets apply a fixed crop of offset (64, 64), yielding 128×128 samples.

## Loading

```python
# The .pt files require cosmic_dataset.py in sys.path / cwd for deserialization
from cosmic_dataset import load_cosmic

train_dataset = load_cosmic('cosmic_train.pt')
valid_dataset = load_cosmic('cosmic_valid.pt')

# Each item yields (image, mask, ignore) as numpy arrays of shape (128, 128)
image, mask, ignore = train_dataset[0]
```

## Loss and Metric Details

Loss is applied as `BCEWithLogitsLoss(logits * (1 - ignore), mask * (1 - ignore))`: the ignore mask excludes certain pixels from loss computation. Metric is computed similarly by thresholding logits at 0.5 and aggregating TP/TN/FP/FN over non-ignored pixels.

## Splits (in this directory)

- Training: 4,347 samples
- Validation: 483 samples
- Test: held out, not accessible here

Binary segmentation — cosmic rays are sparse (most pixels are background).
