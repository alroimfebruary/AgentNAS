"""Full-control custom task: the same toy data, written against BaseBenchmark.

Use this shape when your data doesn't fit NpyClassificationBenchmark —
custom file formats, dense outputs, special metrics or losses. Everything
the pipeline needs lives in this one class:

- loaders (search: train/val; final: train+val vs test)
- loss_fn / compute_metric (metric must be lower=better)
- input_shape / output_spec
- LLM-facing metadata (loss_name, metric_name, optional SOTA)
- the Explorer visibility spec — for custom tasks, ALWAYS the allowlist:
  files not listed in `explorer_allowed_files` can never reach the LLM.

Validate before running a search:
    agentnas validate-task examples/custom_task/full_custom_task.py \\
        --data-root ./toy_data
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from agentnas.benchmarks.base import BaseBenchmark
from agentnas.benchmarks.registry import register_benchmark


@register_benchmark("toy_task_full")
class ToyTaskFullBenchmark(BaseBenchmark):
    """(3, 16, 16) → 4 classes. Metric: 1 - accuracy (lower=better)."""

    # LLM-facing metadata
    loss_name = "cross_entropy"
    metric_name = "accuracy"
    # sota_metric = 0.05        # set when a published baseline exists;
    # sota_description = "..."  # without it, prompts degrade gracefully

    data_subdir = "toy_task"  # this task's directory under --data-root

    # REQUIRED for custom tasks: the Explorer sees ONLY these paths (plus a
    # README). test_{x,y}.npy are absent from the list, so they cannot leak.
    explorer_allowed_files = (
        "train_x.npy", "train_y.npy", "valid_x.npy", "valid_y.npy",
    )

    def __init__(self, data_root: str):
        # Must be I/O-free: no data reads here (validate-task enforces this).
        super().__init__(data_root, "toy_task_full")
        self._dir = Path(data_root) / self.data_subdir

    def explorer_readme(self) -> str:
        # Shown to the Data Explorer LLM. Omit this method to get an
        # auto-generated summary of shapes, loss, metric, and files.
        return (
            "# toy_task_full\n\n"
            "Synthetic 4-class image classification, (3, 16, 16) float32 "
            "inputs.\n\n"
            "- `train_x.npy` / `valid_x.npy`: input arrays (N, 3, 16, 16)\n"
            "- `train_y.npy` / `valid_y.npy`: integer labels (N,)\n"
        )

    def _split(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        x = np.load(self._dir / f"{name}_x.npy").astype(np.float32)
        y = np.load(self._dir / f"{name}_y.npy").astype(np.int64)
        return x, y

    def _loader(
        self, x: np.ndarray, y: np.ndarray, batch_size: int,
        num_workers: int, shuffle: bool,
    ) -> DataLoader:
        dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
        )

    def get_search_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        train_x, train_y = self._split("train")
        val_x, val_y = self._split("valid")
        return (
            self._loader(train_x, train_y, batch_size, num_workers, shuffle=True),
            self._loader(val_x, val_y, batch_size, num_workers, shuffle=False),
        )

    def get_final_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        train_x, train_y = self._split("train")
        val_x, val_y = self._split("valid")
        test_x, test_y = self._split("test")
        trainval_x = np.concatenate([train_x, val_x], axis=0)
        trainval_y = np.concatenate([train_y, val_y], axis=0)
        return (
            self._loader(trainval_x, trainval_y, batch_size, num_workers, shuffle=True),
            self._loader(test_x, test_y, batch_size, num_workers, shuffle=False),
        )

    def loss_fn(self) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss()

    def compute_metric(
        self,
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
        all_extras: list[torch.Tensor] | None = None,
    ) -> float:
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        accuracy = (preds.argmax(dim=1) == targets).float().mean().item()
        return 1.0 - accuracy

    def input_shape(self) -> tuple[int, ...]:
        return (3, 16, 16)

    def output_spec(self) -> dict[str, Any]:
        return {"num_classes": 4}
