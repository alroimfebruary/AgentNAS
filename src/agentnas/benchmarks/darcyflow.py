"""Darcy Flow benchmark: 2D dense regression (PDE solving)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader, TensorDataset

from agentnas.benchmarks.base import BaseBenchmark

_SUBSAMPLE_RATE = 5  # 421 → 85
_GRID_SIZE = 85


class UnitGaussianNormalizer:
    """Normalize data to zero mean, unit variance. Stores stats for inverse."""

    def __init__(self, x: torch.Tensor, eps: float = 1e-5):
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)
        self.eps = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean


class RelativeL2Loss(torch.nn.Module):
    """Relative L2 loss: mean(||pred - true||_2 / ||true||_2)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dims
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        diff_norm = torch.norm(pred_flat - target_flat, p=2, dim=1)
        target_norm = torch.norm(target_flat, p=2, dim=1)
        return (diff_norm / (target_norm + 1e-8)).mean()


class DarcyFlowBenchmark(BaseBenchmark):
    """Darcy Flow: (3, 85, 85) → (1, 85, 85) dense regression.

    Input: 1 coefficient channel + 2 positional grid channels.
    Subsampled from 421×421 by factor 5.
    Splits: 900 train / 100 val from smooth1.mat, 100 test from smooth2.mat.
    Metric: Relative L2 error (lower=better).
    """

    def __init__(self, data_root: str):
        super().__init__(data_root, "darcyflow")
        self._data_dir = Path(data_root) / "darcyflow"
        self._x_normalizer = None
        self._y_normalizer = None
        self._cached_data = None  # Cache to avoid reloading + renormalizing

    def _load_mat(self, filename: str) -> tuple[np.ndarray, np.ndarray]:
        """Load coefficient and solution fields from .mat file."""
        path = self._data_dir / filename
        data = scipy.io.loadmat(path)
        coeff = data["Kcoeff"]  # (1024, 421, 421) float64 — NOT "coeff" which is uint8 (2.1 fix)
        sol = data["sol"]  # (1024, 421, 421) float64
        return coeff.astype(np.float32), sol.astype(np.float32)

    def _subsample(self, field: np.ndarray) -> np.ndarray:
        """Subsample from 421 to 85 by taking every 5th point."""
        return field[:, ::_SUBSAMPLE_RATE, ::_SUBSAMPLE_RATE]

    def _make_grid(self) -> np.ndarray:
        """Create 2D positional grid (2, 85, 85)."""
        coords = np.linspace(0, 1, _GRID_SIZE, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(coords, coords)
        return np.stack([grid_x, grid_y], axis=0)  # (2, 85, 85)

    def _prepare_data(self):
        """Load, subsample, normalize, and split all data. Cached after first call."""
        if self._cached_data is not None:
            return self._cached_data
        coeff1, sol1 = self._load_mat("piececonst_r421_N1024_smooth1.mat")
        coeff2, sol2 = self._load_mat("piececonst_r421_N1024_smooth2.mat")

        # Subsample
        coeff1 = self._subsample(coeff1)  # (1024, 85, 85)
        sol1 = self._subsample(sol1)
        coeff2 = self._subsample(coeff2)
        sol2 = self._subsample(sol2)

        # Split: 900 train, 100 val from smooth1; 100 test from smooth2
        train_x_raw = coeff1[:900][:, np.newaxis, :, :]  # (900, 1, 85, 85)
        train_y = sol1[:900][:, np.newaxis, :, :]
        val_x_raw = coeff1[900:1000][:, np.newaxis, :, :]
        val_y = sol1[900:1000][:, np.newaxis, :, :]
        test_x_raw = coeff2[:100][:, np.newaxis, :, :]
        test_y = sol2[:100][:, np.newaxis, :, :]

        # Convert to tensors for normalization
        train_x_t = torch.from_numpy(train_x_raw).squeeze(1)  # (900, 85, 85)
        train_y_t = torch.from_numpy(train_y).squeeze(1)
        val_x_t = torch.from_numpy(val_x_raw).squeeze(1)
        val_y_t = torch.from_numpy(val_y).squeeze(1)
        test_x_t = torch.from_numpy(test_x_raw).squeeze(1)
        test_y_t = torch.from_numpy(test_y).squeeze(1)

        # Normalize using train stats
        self._x_normalizer = UnitGaussianNormalizer(train_x_t)
        self._y_normalizer = UnitGaussianNormalizer(train_y_t)

        train_x_norm = self._x_normalizer.encode(train_x_t).unsqueeze(1)
        val_x_norm = self._x_normalizer.encode(val_x_t).unsqueeze(1)
        test_x_norm = self._x_normalizer.encode(test_x_t).unsqueeze(1)

        train_y_norm = self._y_normalizer.encode(train_y_t).unsqueeze(1)
        val_y_norm = self._y_normalizer.encode(val_y_t).unsqueeze(1)
        test_y_norm = self._y_normalizer.encode(test_y_t).unsqueeze(1)

        # Append positional grid to input
        grid = torch.from_numpy(self._make_grid())  # (2, 85, 85)
        train_x_final = torch.cat([train_x_norm, grid.unsqueeze(0).expand(900, -1, -1, -1)], dim=1)
        val_x_final = torch.cat([val_x_norm, grid.unsqueeze(0).expand(100, -1, -1, -1)], dim=1)
        test_x_final = torch.cat([test_x_norm, grid.unsqueeze(0).expand(100, -1, -1, -1)], dim=1)

        self._cached_data = {
            "train_x": train_x_final, "train_y": train_y_norm,
            "val_x": val_x_final, "val_y": val_y_norm,
            "test_x": test_x_final, "test_y": test_y_norm,
        }
        return self._cached_data

    def _make_loader(
        self, x: torch.Tensor, y: torch.Tensor, batch_size: int,
        num_workers: int, shuffle: bool,
    ) -> DataLoader:
        dataset = TensorDataset(x, y)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )

    def get_search_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        data = self._prepare_data()
        return (
            self._make_loader(data["train_x"], data["train_y"], batch_size, num_workers, shuffle=True),
            self._make_loader(data["val_x"], data["val_y"], batch_size, num_workers, shuffle=False),
        )

    def get_final_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        data = self._prepare_data()
        trainval_x = torch.cat([data["train_x"], data["val_x"]], dim=0)
        trainval_y = torch.cat([data["train_y"], data["val_y"]], dim=0)
        return (
            self._make_loader(trainval_x, trainval_y, batch_size, num_workers, shuffle=True),
            self._make_loader(data["test_x"], data["test_y"], batch_size, num_workers, shuffle=False),
        )

    def loss_fn(self) -> torch.nn.Module:
        return RelativeL2Loss()

    def compute_metric(
        self,
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
        all_extras: list[torch.Tensor] | None = None,
    ) -> float:
        """Relative L2 error (lower=better)."""
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        # Denormalize if normalizer is available
        if self._y_normalizer is not None:
            preds_flat = preds.squeeze(1)
            targets_flat = targets.squeeze(1)
            preds_denorm = self._y_normalizer.decode(preds_flat)
            targets_denorm = self._y_normalizer.decode(targets_flat)
        else:
            preds_denorm = preds.reshape(preds.shape[0], -1)
            targets_denorm = targets.reshape(targets.shape[0], -1)
        preds_flat = preds_denorm.reshape(preds_denorm.shape[0], -1)
        targets_flat = targets_denorm.reshape(targets_denorm.shape[0], -1)
        diff_norm = torch.norm(preds_flat - targets_flat, p=2, dim=1)
        target_norm = torch.norm(targets_flat, p=2, dim=1)
        return (diff_norm / (target_norm + 1e-8)).mean().item()

    def input_shape(self) -> tuple[int, ...]:
        return (3, 85, 85)

    def output_spec(self) -> dict[str, Any]:
        return {"output_shape": (1, 85, 85)}
