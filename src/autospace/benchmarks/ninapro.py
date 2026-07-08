"""NinaPro benchmark: 2D EMG classification, 18 gesture classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from autospace.benchmarks.base import BaseBenchmark


class NinaProBenchmark(BaseBenchmark):
    """NinaPro: (1, 16, 52) → 18 classes.

    Input: EMG signals (N, 52, 16) float64, transposed to (N, 1, 16, 52) for 2D conv.
    Pre-split into train/val/test npy files.
    Metric: 1 - top-1 accuracy (lower=better).
    """

    def __init__(self, data_root: str):
        super().__init__(data_root, "ninapro")
        self._data_dir = Path(data_root) / "ninapro"

    def _load_split(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Load a single split. Returns (x, y) as numpy arrays."""
        x = np.load(self._data_dir / f"ninapro_{split}.npy")  # (N, 52, 16)
        y = np.load(self._data_dir / f"label_{split}.npy")    # (N,)
        # Transpose (N,52,16) → (N,1,16,52) for 2D conv input
        x = x.transpose(0, 2, 1)[:, np.newaxis, :, :].astype(np.float32)
        y = y.astype(np.int64)
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
        val_x, val_y = self._load_split("val")
        return (
            self._make_loader(train_x, train_y, batch_size, num_workers, shuffle=True),
            self._make_loader(val_x, val_y, batch_size, num_workers, shuffle=False),
        )

    def get_final_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        train_x, train_y = self._load_split("train")
        val_x, val_y = self._load_split("val")
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
        return (1, 16, 52)

    def output_spec(self) -> dict[str, Any]:
        return {"num_classes": 18}
