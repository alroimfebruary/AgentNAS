"""Spherical CIFAR-100 benchmark: 2D image classification on spherical projections."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from agentnas.benchmarks.base import BaseBenchmark


class SphericalBenchmark(BaseBenchmark):
    """Spherical CIFAR-100: (3, 60, 60) → 100 classes.

    Splits: 40K train / 10K val (last 20% of train, sequential) / 10K test.
    Metric: 1 - top-1 accuracy (lower=better).
    """

    loss_name = "cross_entropy"
    metric_name = "accuracy"
    sota_metric = 0.4823
    sota_description = "SOTA 1-accuracy = 0.4823 (accuracy 51.77%)"

    explorer_blind_renames = {"s2_cifar100": "data.pkl"}
    explorer_preprocessed_name = "s2_cifar100"

    def __init__(self, data_root: str):
        super().__init__(data_root, "spherical")
        self._data_dir = Path(data_root) / "spherical"

    def prepare_explorer_source(self, out_dir: Path, real_root: Path) -> None:
        """Save a copy of s2_cifar100 with only the 'train' entry (no 'test')."""
        import pickle as _pickle

        out_path = out_dir / "s2_cifar100"
        src_path = real_root / "spherical" / "s2_cifar100"
        if not src_path.exists():
            raise FileNotFoundError(f"spherical source missing: {src_path}")
        with open(src_path, "rb") as f:
            data = _pickle.load(f, encoding="bytes")
        filtered = {"train": data["train"]}
        with open(out_path, "wb") as f:
            _pickle.dump(filtered, f, protocol=_pickle.HIGHEST_PROTOCOL)

    def _load_pickle(self) -> dict:
        path = self._data_dir / "s2_cifar100"
        with open(path, "rb") as f:
            return pickle.load(f, encoding="bytes")

    def _load_splits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and split train data into train/val."""
        data = self._load_pickle()
        train_all = data["train"]
        images = train_all["images"].astype(np.float32) / 255.0  # (50K, 3, 60, 60)
        labels = train_all["labels"]  # (50K,)

        split_idx = 40000
        train_x, val_x = images[:split_idx], images[split_idx:]
        train_y, val_y = labels[:split_idx], labels[split_idx:]
        return train_x, train_y, val_x, val_y

    def _load_test(self) -> tuple[np.ndarray, np.ndarray]:
        data = self._load_pickle()
        test_all = data["test"]
        images = test_all["images"].astype(np.float32) / 255.0
        labels = test_all["labels"]
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
        return (3, 60, 60)

    def output_spec(self) -> dict[str, Any]:
        return {"num_classes": 100}
