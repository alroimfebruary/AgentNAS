"""Cosmic ray benchmark: 2D binary segmentation with ignore mask."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from autospace.benchmarks.base import BaseBenchmark


class CosmicDataset(Dataset):
    """Wraps the PairedDatasetImagePath to produce (input, target, ignore) tensors.

    Input: channel 0 (image) → (1, 128, 128).
    Target: channel 1 (mask) → (1, 128, 128).
    Ignore: channel 2 → (1, 128, 128).
    """

    def __init__(self, pt_path: str):
        # The .pt files contain PairedDatasetImagePath objects that require
        # the cosmic_dataset module to be importable for deserialization.
        cosmic_dir = Path(pt_path).parent
        if str(cosmic_dir) not in sys.path:
            sys.path.insert(0, str(cosmic_dir))

        from cosmic_dataset import load_cosmic
        self._inner = load_cosmic(pt_path)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask, ignore = self._inner[idx]
        # image, mask, ignore are (128, 128) numpy arrays
        x = torch.from_numpy(np.asarray(image, dtype=np.float32)).unsqueeze(0)  # (1,128,128)
        # Pack mask and ignore into a 2-channel target for loss computation
        mask_t = torch.from_numpy(np.asarray(mask, dtype=np.float32)).unsqueeze(0)
        ignore_t = torch.from_numpy(np.asarray(ignore, dtype=np.float32)).unsqueeze(0)
        target = torch.cat([mask_t, ignore_t], dim=0)  # (2, 128, 128)
        return x, target


class MaskedBCEWithLogitsLoss(torch.nn.Module):
    """BCEWithLogitsLoss that ignores pixels by zeroing inputs before BCE.

    Matches official NAS-Bench-360: criterion(outputs*(1-ignore), mask*(1-ignore))
    Expects target tensor with 2 channels: [mask, ignore].
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (B, 1, H, W), target: (B, 2, H, W)
        mask = target[:, 0:1, :, :]    # ground truth
        ignore = target[:, 1:2, :, :]  # ignore mask (1 = ignore)

        # Official: zero out BOTH pred and target where ignored, BEFORE BCE (H-02 fix)
        valid = 1.0 - ignore
        pred_masked = pred * valid
        mask_masked = mask * valid
        return torch.nn.functional.binary_cross_entropy_with_logits(pred_masked, mask_masked)


class CosmicBenchmark(BaseBenchmark):
    """Cosmic: (1, 128, 128) → (1, 128, 128) binary segmentation.

    Uses ignore mask for loss computation.
    Loss: BCEWithLogitsLoss with ignore mask.
    Metric: 1 - AUROC (lower=better).
    """

    def __init__(self, data_root: str):
        super().__init__(data_root, "cosmic")
        self._data_dir = Path(data_root) / "cosmic"

    def get_search_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        train_ds = CosmicDataset(str(self._data_dir / "cosmic_train.pt"))
        val_ds = CosmicDataset(str(self._data_dir / "cosmic_valid.pt"))
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True),
        )

    def get_final_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        from torch.utils.data import ConcatDataset

        train_ds = CosmicDataset(str(self._data_dir / "cosmic_train.pt"))
        val_ds = CosmicDataset(str(self._data_dir / "cosmic_valid.pt"))
        test_ds = CosmicDataset(str(self._data_dir / "cosmic_test.pt"))
        trainval_ds = ConcatDataset([train_ds, val_ds])
        return (
            DataLoader(trainval_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True),
        )

    def loss_fn(self) -> torch.nn.Module:
        return MaskedBCEWithLogitsLoss()

    def compute_metric(
        self,
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
        all_extras: list[torch.Tensor] | None = None,
    ) -> float:
        """1 - AUROC (lower=better), computed on non-ignored pixels."""
        preds = torch.cat(all_preds, dim=0)    # (N, 1, H, W)
        targets = torch.cat(all_targets, dim=0)  # (N, 2, H, W)

        mask = targets[:, 0:1, :, :]    # ground truth
        ignore = targets[:, 1:2, :, :]  # ignore mask

        # Flatten and filter out ignored pixels
        valid = (1.0 - ignore).bool().flatten()
        pred_flat = preds.sigmoid().flatten()[valid].cpu().numpy()
        mask_flat = mask.flatten()[valid].cpu().numpy()

        from sklearn.metrics import roc_auc_score

        if mask_flat.sum() == 0 or mask_flat.sum() == len(mask_flat):
            return 1.0  # Degenerate case
        auroc = roc_auc_score(mask_flat, pred_flat)
        return 1.0 - auroc

    def input_shape(self) -> tuple[int, ...]:
        return (1, 128, 128)

    def output_spec(self) -> dict[str, Any]:
        return {"output_shape": (1, 128, 128), "task_type": "segmentation"}
