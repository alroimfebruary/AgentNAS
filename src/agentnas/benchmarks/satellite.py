"""Satellite benchmark: 1D spectral classification, 24 land-cover classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from agentnas.benchmarks.base import BaseBenchmark


def _zscore_per_sample(x: np.ndarray) -> np.ndarray:
    """Per-sample z-score normalization along the feature axis."""
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std


class SatelliteBenchmark(BaseBenchmark):
    """Satellite: (1, 46) → 24 classes.

    Input: spectral features (N, 46) float64, per-sample z-score, expand to (N, 1, 46).
    Labels: 1-24 remapped to 0-23.
    Splits: first 800K train / last 100K val (from train file) / test from separate file.
    Metric: 1 - top-1 accuracy (lower=better).
    """

    def __init__(self, data_root: str):
        super().__init__(data_root, "satellite")
        self._data_dir = Path(data_root) / "satellite"

    def _load_npy_dict(self, filename: str) -> tuple[np.ndarray, np.ndarray]:
        """Load pickle-encoded npy dict with 'data' and 'label' keys."""
        data = np.load(self._data_dir / filename, allow_pickle=True).item()
        x = data["data"].astype(np.float64)   # (N, 46)
        y = data["label"].astype(np.int64)     # (N,) values 1-24
        # Remap labels from 1-24 to 0-23
        y = y - 1
        # Per-sample z-score
        x = _zscore_per_sample(x).astype(np.float32)
        # Expand to (N, 1, 46) for 1D conv
        x = x[:, np.newaxis, :]
        return x, y

    def _load_splits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load train file and split into train/val."""
        x_all, y_all = self._load_npy_dict("satellite_train.npy")
        split_idx = len(x_all) - 100000  # Val = last 100K, train = rest (M6 fix)
        train_x, val_x = x_all[:split_idx], x_all[split_idx:]
        train_y, val_y = y_all[:split_idx], y_all[split_idx:]
        return train_x, train_y, val_x, val_y

    def _load_test(self) -> tuple[np.ndarray, np.ndarray]:
        return self._load_npy_dict("satellite_test.npy")

    def _make_loader(
        self, x: np.ndarray, y: np.ndarray, batch_size: int,
        num_workers: int, shuffle: bool,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(x), torch.from_numpy(y)
        )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )

    def get_search_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        train_x, train_y, val_x, val_y = self._load_splits()
        return (
            self._make_loader(train_x, train_y, batch_size, num_workers, shuffle=True),
            self._make_loader(val_x, val_y, batch_size, num_workers, shuffle=False),
        )

    def get_final_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        train_x, train_y, val_x, val_y = self._load_splits()
        test_x, test_y = self._load_test()
        trainval_x = np.concatenate([train_x, val_x], axis=0)
        trainval_y = np.concatenate([train_y, val_y], axis=0)
        return (
            self._make_loader(trainval_x, trainval_y, batch_size, num_workers, shuffle=True),
            self._make_loader(test_x, test_y, batch_size, num_workers, shuffle=False),
        )

    def loss_fn(self) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss()

    def compute_metric(
        self,
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
        all_extras: list[torch.Tensor] | None = None,
    ) -> float:
        """1 - top-1 accuracy (lower=better)."""
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        predicted = preds.argmax(dim=1)
        accuracy = (predicted == targets).float().mean().item()
        return 1.0 - accuracy

    def input_shape(self) -> tuple[int, ...]:
        return (1, 46)

    def output_spec(self) -> dict[str, Any]:
        return {"num_classes": 24}
