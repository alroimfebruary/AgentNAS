# Darcy Flow

**Task Type:** 2D Dense Prediction
**Domain:** Partial Differential Equation (PDE) solving — fluid flow through porous media
**Problem:** Dense regression (predicting the full 2D solution field from coefficient fields)
**Metric:** Relative L2 error (lower is better)

Given multi-field input coefficients describing spatially varying material properties, the task is to predict the full 2D pressure/flow solution field at matching spatial resolution.

## Data Format

A MATLAB `.mat` file. In this directory you have access to:

- `piececonst_r421_N1024_smooth1.mat` — 1024 samples with smoothness parameter 1

Each `.mat` file contains the following fields, all with shape (1024, 421, 421):

| Field | Dtype | Description |
|-------|-------|-------------|
| `Kcoeff` | float64 | Permeability coefficient field |
| `Kcoeff_x` | float64 | X-direction gradient |
| `Kcoeff_y` | float64 | Y-direction gradient |
| `coeff` | uint8 | Discretized coefficient representation |
| `sol` | float64 | Pressure/flow solution field (prediction target) |

(A second `.mat` file — with a different smoothness parameter — contains the held-out test samples and is not accessible from this directory.)

## Loading

```python
from scipy.io import loadmat

data = loadmat('piececonst_r421_N1024_smooth1.mat')
Kcoeff = data['Kcoeff']      # (1024, 421, 421)
Kcoeff_x = data['Kcoeff_x']  # (1024, 421, 421)
Kcoeff_y = data['Kcoeff_y']  # (1024, 421, 421)
coeff = data['coeff']         # (1024, 421, 421) uint8
sol = data['sol']             # (1024, 421, 421) — target
```

## Preprocessing (benchmark convention)

- 421×421 grid is subsampled by factor 5 to 85×85
- Input is constructed from the coefficient fields (3-channel)
- Input and output are both z-score normalized (zero-mean, unit variance per dimension)
- A positional grid encoding is appended to the input channels

## Splits (in this directory)

- Training: 900 samples (indices [0:900])
- Validation: 100 samples (indices [900:1000])
- Samples at index 1000+ are unused. Test set comes from a separate file not accessible here.

Loss: LpLoss (relative L2 norm).
