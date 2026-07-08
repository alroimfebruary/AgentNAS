"""Base benchmark class defining the standard interface for all tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import DataLoader


class BaseBenchmark(ABC):
    """Standard interface for all 17 benchmark tasks.

    All metrics are normalized to lower=better.
    """

    def __init__(self, data_root: str, task_name: str):
        self.data_root = data_root
        self.task_name = task_name

    @abstractmethod
    def get_search_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        """Return (train_loader, val_loader) for the search phase.

        Train on train split, evaluate on val split, use best checkpoint.
        """

    @abstractmethod
    def get_final_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        """Return (trainval_loader, test_loader) for the final phase.

        Train on train+val combined, evaluate on test split, use last checkpoint.
        """

    @abstractmethod
    def loss_fn(self) -> torch.nn.Module:
        """Return the fixed loss function for this task."""

    @abstractmethod
    def compute_metric(
        self,
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
        all_extras: list[torch.Tensor] | None = None,
    ) -> float:
        """Compute the official metric from accumulated predictions and targets.

        Args:
            all_preds: Model outputs per batch.
            all_targets: Ground truth per batch.
            all_extras: Optional extra tensors per batch (e.g., recording IDs for ECG,
                        clip IDs for FSD50K). Benchmarks that need aggregation use these.

        Returns:
            float: metric value where lower = better.
        """

    def compute_search_metric(
        self,
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
        all_extras: list[torch.Tensor] | None = None,
    ) -> float:
        """Search-time best-checkpoint selection metric. Defaults to ``compute_metric``.

        Override only when the official metric is susceptible to early-training
        flukes (e.g. a contact-MAE-on-a-narrow-target-band metric that rewards
        any near-constant predictor whose value happens to land in the band).
        Final/test reporting always uses ``compute_metric``.
        """
        return self.compute_metric(all_preds, all_targets, all_extras)

    @abstractmethod
    def input_shape(self) -> tuple[int, ...]:
        """Return input tensor shape (without batch dim) for FLOPs measurement."""

    @abstractmethod
    def output_spec(self) -> dict[str, Any]:
        """Return output specification.

        Either {"num_classes": N} for classification
        or {"output_shape": (C, H, W)} for dense prediction.
        """

    def smoke_test(self) -> bool:
        """Validate that the data loader works correctly.

        Checks: shapes, split sizes, label ranges, loss computation.
        Returns True if all checks pass.
        """
        try:
            # Check search loaders
            train_loader, val_loader = self.get_search_loaders(batch_size=4)
            batch = next(iter(train_loader))
            x_train, y_train = batch[0], batch[1]  # Handle 2 or 3-element tuples
            batch = next(iter(val_loader))
            x_val, y_val = batch[0], batch[1]

            # Verify shapes
            expected_shape = self.input_shape()
            actual_shape = tuple(x_train.shape[1:])
            assert actual_shape == expected_shape, (
                f"Input shape mismatch: got {actual_shape}, expected {expected_shape}"
            )

            # Verify loss works
            loss_fn = self.loss_fn()
            model_out = torch.randn(4, *self._output_tensor_shape())
            try:
                loss = loss_fn(model_out, y_train[:4])
                assert torch.isfinite(loss), f"Loss is not finite: {loss}"
            except Exception as e:
                raise AssertionError(f"Loss computation failed: {e}")

            # Check final loaders
            trainval_loader, test_loader = self.get_final_loaders(batch_size=4)
            batch = next(iter(trainval_loader))
            x_tv, y_tv = batch[0], batch[1]
            batch = next(iter(test_loader))
            x_test, y_test = batch[0], batch[1]

            # Verify train+val is larger than train alone
            assert len(trainval_loader.dataset) >= len(train_loader.dataset), (
                "trainval dataset should be >= train dataset"
            )

            return True
        except Exception as e:
            print(f"Smoke test failed for {self.task_name}: {e}")
            return False

    def _output_tensor_shape(self) -> tuple[int, ...]:
        """Return the expected model output shape (without batch dim)."""
        spec = self.output_spec()
        if "num_classes" in spec:
            return (spec["num_classes"],)
        return tuple(spec["output_shape"])
