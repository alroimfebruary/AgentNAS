"""NAS-Unseen benchmarks: unified loader for all 7 unseen tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from autospace.benchmarks.base import BaseBenchmark

# Task name → (directory name, input shape, num_classes)
_TASK_SPECS: dict[str, tuple[str, tuple[int, ...], int]] = {
    "addnist":     ("AddNIST",     (3, 28, 28),  20),
    "language":    ("Language",     (1, 24, 24),  10),
    "multnist":    ("MultNIST",    (3, 28, 28),  10),
    "cifartile":   ("CIFARTile",   (3, 64, 64),  4),
    "gutenberg":   ("Gutenberg",   (1, 27, 18),  6),
    "geoclassing": ("GeoClassing", (3, 60, 60),  10),
    "chesseract":  ("Chesseract",  (12, 8, 8),   3),
}


class UnseenNASBenchmark(BaseBenchmark):
    """NAS-Unseen task loader.

    Uniform format: {train,valid,test}_{x,y}.npy.
    Loss: CrossEntropyLoss.
    Metric: 1 - accuracy (lower=better).
    """

    def __init__(self, data_root: str, task_name: str | None = None):
        if task_name is None:
            raise ValueError("task_name is required for UnseenNASBenchmark")
        if task_name not in _TASK_SPECS:
            raise ValueError(
                f"Unknown unseen task '{task_name}'. "
                f"Available: {sorted(_TASK_SPECS.keys())}"
            )
        super().__init__(data_root, task_name)
        dir_name, self._input_shape, self._num_classes = _TASK_SPECS[task_name]
        self._data_dir = Path(data_root) / dir_name

    def _load_split(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Load x and y arrays for a split (train/valid/test)."""
        x = np.load(self._data_dir / f"{split}_x.npy").astype(np.float32)
        y = np.load(self._data_dir / f"{split}_y.npy").astype(np.int64)
        return x, y

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
        train_x, train_y = self._load_split("train")
        val_x, val_y = self._load_split("valid")
        return (
            self._make_loader(train_x, train_y, batch_size, num_workers, shuffle=True),
            self._make_loader(val_x, val_y, batch_size, num_workers, shuffle=False),
        )

    def get_final_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        train_x, train_y = self._load_split("train")
        val_x, val_y = self._load_split("valid")
        test_x, test_y = self._load_split("test")
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
        return self._input_shape

    def output_spec(self) -> dict[str, Any]:
        return {"num_classes": self._num_classes}
