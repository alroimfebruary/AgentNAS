"""CIFAR-100 benchmark: 2D image classification, 100 classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from autospace.benchmarks.base import BaseBenchmark


class CIFAR100Benchmark(BaseBenchmark):
    """CIFAR-100: (3, 32, 32) → 100 classes.

    Splits: 40K train / 10K val (last 20% of official train, sequential) / 10K test.
    Metric: 1 - top-1 accuracy (lower=better).
    """

    def __init__(self, data_root: str):
        super().__init__(data_root, "cifar100")
        self._data_dir = Path(data_root) / "cifar100"

    def _load_pickle(self, filename: str) -> dict:
        path = self._data_dir / filename
        with open(path, "rb") as f:
            return pickle.load(f, encoding="bytes")

    def _load_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and split data. Returns (train_x, train_y, val_x, val_y)."""
        # Official train: 50K samples
        train_data = self._load_pickle("train")
        images = train_data[b"data"]  # (50000, 3072)
        labels = np.array(train_data[b"fine_labels"])  # (50000,)

        # Reshape to (N, 3, 32, 32) and normalize to float32 [0, 1]
        images = images.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0

        # Val = last 20% of official train (sequential slice, no shuffle)
        split_idx = 40000
        train_x, val_x = images[:split_idx], images[split_idx:]
        train_y, val_y = labels[:split_idx], labels[split_idx:]

        return train_x, train_y, val_x, val_y

    def _load_test(self) -> tuple[np.ndarray, np.ndarray]:
        test_data = self._load_pickle("test")
        images = test_data[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        labels = np.array(test_data[b"fine_labels"])
        return images, labels

    def _make_loader(
        self, x: np.ndarray, y: np.ndarray, batch_size: int,
        num_workers: int, shuffle: bool,
    ) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(x), torch.from_numpy(y).long()
        )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )

    def get_search_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        train_x, train_y, val_x, val_y = self._load_all()
        train_loader = self._make_loader(train_x, train_y, batch_size, num_workers, shuffle=True)
        val_loader = self._make_loader(val_x, val_y, batch_size, num_workers, shuffle=False)
        return train_loader, val_loader

    def get_final_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        train_x, train_y, val_x, val_y = self._load_all()
        test_x, test_y = self._load_test()
        # Combine train + val
        trainval_x = np.concatenate([train_x, val_x], axis=0)
        trainval_y = np.concatenate([train_y, val_y], axis=0)
        trainval_loader = self._make_loader(trainval_x, trainval_y, batch_size, num_workers, shuffle=True)
        test_loader = self._make_loader(test_x, test_y, batch_size, num_workers, shuffle=False)
        return trainval_loader, test_loader

    def loss_fn(self) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss()

    def compute_metric(
        self,
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
        all_extras: list[torch.Tensor] | None = None,
    ) -> float:
        """1 - top-1 accuracy (lower=better)."""
        preds = torch.cat(all_preds, dim=0)  # (N, 100)
        targets = torch.cat(all_targets, dim=0)  # (N,)
        predicted = preds.argmax(dim=1)
        accuracy = (predicted == targets).float().mean().item()
        return 1.0 - accuracy

    def input_shape(self) -> tuple[int, ...]:
        return (3, 32, 32)

    def output_spec(self) -> dict[str, Any]:
        return {"num_classes": 100}
