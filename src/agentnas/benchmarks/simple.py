"""Convenience benchmark for the common custom-task case: classification
from pre-split .npy files.

Subclass, set three class attributes, register — no methods needed::

    from agentnas.benchmarks.registry import register_benchmark
    from agentnas.benchmarks.simple import NpyClassificationBenchmark

    @register_benchmark("my_task")
    class MyTask(NpyClassificationBenchmark):
        task = "my_task"
        data_shape = (3, 32, 32)  # per-sample input shape, e.g. (C, H, W)
        n_classes = 10

Expected data layout under ``<data_root>/<task>/`` (override ``data_subdir``
or ``split_files`` to change)::

    train_x.npy  train_y.npy
    valid_x.npy  valid_y.npy
    test_x.npy   test_y.npy

where ``*_x.npy`` has shape ``(N, *data_shape)`` and ``*_y.npy`` holds integer
labels of shape ``(N,)``. The Explorer allowlist is derived automatically from
the train/val split files, so test-set isolation holds by construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from agentnas.benchmarks.base import BaseBenchmark


class NpyClassificationBenchmark(BaseBenchmark):
    """Classification benchmark backed by {train,valid,test}_{x,y}.npy files.

    Loss: CrossEntropyLoss. Metric: 1 - top-1 accuracy (lower=better).
    """

    # --- Required in subclasses ---
    task: str = ""  # registered task name
    data_shape: tuple[int, ...] = ()  # per-sample input shape
    n_classes: int = 0

    # --- Overridable ---
    split_files: dict[str, tuple[str, str]] = {
        "train": ("train_x.npy", "train_y.npy"),
        "val": ("valid_x.npy", "valid_y.npy"),
        "test": ("test_x.npy", "test_y.npy"),
    }

    loss_name = "cross_entropy"
    metric_name = "accuracy"

    def __init__(self, data_root: str):
        if not (self.task and self.data_shape and self.n_classes):
            raise TypeError(
                f"{type(self).__name__} must set the class attributes "
                f"'task', 'data_shape', and 'n_classes'."
            )
        super().__init__(data_root, self.task)
        self._data_dir = Path(data_root) / self.data_subdir
        if self.explorer_allowed_files is None:
            # Only train/val files are visible to the Explorer.
            self.explorer_allowed_files = tuple(
                name
                for split in ("train", "val")
                for name in self.split_files[split]
            )

    def _load_split(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        x_name, y_name = self.split_files[split]
        x = np.load(self._data_dir / x_name).astype(np.float32)
        y = np.load(self._data_dir / y_name).astype(np.int64)
        return x, y

    def _make_loader(
        self, x: np.ndarray, y: np.ndarray, batch_size: int,
        num_workers: int, shuffle: bool,
    ) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
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
        return tuple(self.data_shape)

    def output_spec(self) -> dict[str, Any]:
        return {"num_classes": self.n_classes}
