# PSICOV

**Task Type:** 2D Dense Prediction
**Domain:** Protein structure prediction — inter-residue distance map prediction
**Problem:** Dense regression over protein pair distances
**Metric:** MAE at long-range separation with d ≤ 8Å (lower is better)

For each protein, the input is a 2D feature map constructed from multiple residue-pair features (coevolution signals, potentials, secondary structure, etc.). The output is a full L×L distance matrix where L is the sequence length. The benchmark pads to a fixed size and evaluates MAE on specific subsets of residue pairs.

## Data Format

Data is organized in split-specific subdirectories. In this directory you have access to:

- `deepcov/`:
  - `distance/` — ground-truth distance matrices (`.npy`), 3,456 files
  - `features/` — input features (`.pkl`), 3,456 files
- `deepcov.lst` — per-protein IDs and sequence lengths
- `train_ecod.lst` — training split definition (subset of `deepcov` IDs)

(Test-set protein directories and their split-list files are held out and not accessible here. Validation proteins are defined as those in `deepcov` that are not in `train_ecod.lst`.)

### Distance files (.npy)

Each `.npy` file is a shape (3,) object array containing:
- `arr[0]`: sequence length (int)
- `arr[1]`: sequence string
- `arr[2]`: distance matrix, shape (L, L), float16

### Feature files (.pkl)

Each `.pkl` file is a dict with:
- `seq`: sequence string (length L)
- `ccmpred`: (L, L) float
- `freecon`: (L, L) float
- `potential`: (L, L) float
- `entropy`: (L, 1) float
- `ss`: (L, 3) one-hot secondary structure
- `sa`: (L, 1) float (solvent accessibility)
- `pssm`: (L, 20) float

## Loading

```python
import numpy as np
import pickle

dist_arr = np.load('deepcov/distance/<id>.npy', allow_pickle=True)
L = dist_arr[0]
sequence = dist_arr[1]
distance_matrix = dist_arr[2]  # (L, L) float16

with open('deepcov/features/<id>.pkl', 'rb') as f:
    features = pickle.load(f)
ccmpred = features['ccmpred']   # (L, L)
pssm = features['pssm']         # (L, 20)
```

## Preprocessing (benchmark convention)

The benchmark assembles a 57-channel 2D representation of shape (57, L, L) from the feature fields, then pads to LMAX=512 (or uses 128 in the search stage). Input and output are both 2D maps of shape (C, L, L).

## Splits (in this directory)

- Training: proteins listed in `train_ecod.lst` (~3,356 samples)
- Validation: proteins in `deepcov/` but not in `train_ecod.lst` (~100 samples)
- Test: held out, not accessible here

Variable-length sequences padded to fixed size. Loss: MSELoss.
