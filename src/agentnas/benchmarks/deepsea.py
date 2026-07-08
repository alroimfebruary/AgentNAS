"""DeepSEA benchmark: 1D multi-label classification, 36 chromatin features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from agentnas.benchmarks.base import BaseBenchmark


class DeepSEABenchmark(BaseBenchmark):
    """DeepSEA: (4, 1000) → 36 multi-label classes.

    Input: DNA sequences (N, 1000, 4) uint8, transposed to (N, 4, 1000).
    Pre-split npz with x_train/y_train/x_val/y_val/x_test/y_test.
    Loss: BCEWithLogitsLoss (multi-label).
    Metric: 1 - mean per-class AUROC (lower=better).
    """

    loss_name = "bce_with_logits"
    metric_name = "auroc"
    sota_metric = 0.30
    sota_description = "SOTA 1-AUROC = 0.30 (mean per-class AUROC = 0.70)"

    explorer_blind_renames = {"deepsea_filtered.npz": "data.npz"}
    explorer_preprocessed_name = "deepsea_filtered.npz"

    def __init__(self, data_root: str):
        super().__init__(data_root, "deepsea")
        self._data_dir = Path(data_root) / "deepsea"

    def prepare_explorer_source(self, out_dir: Path, real_root: Path) -> None:
        """Save a copy of deepsea_filtered.npz with only train/val arrays."""
        out_path = out_dir / "deepsea_filtered.npz"
        src_path = real_root / "deepsea" / "deepsea_filtered.npz"
        if not src_path.exists():
            raise FileNotFoundError(f"deepsea source missing: {src_path}")
        data = np.load(src_path)
        np.savez(
            out_path,
            x_train=data["x_train"],
            y_train=data["y_train"],
            x_val=data["x_val"],
            y_val=data["y_val"],
        )

    def _load_all(self) -> dict[str, np.ndarray]:
        """Load the npz file and transpose inputs."""
        data = np.load(self._data_dir / "deepsea_filtered.npz")
        result = {}
        for split in ("train", "val", "test"):
            x = data[f"x_{split}"].astype(np.float32)  # (N, 1000, 4)
            x = x.transpose(0, 2, 1)  # (N, 4, 1000)
            y = data[f"y_{split}"].astype(np.float32)   # (N, 36)
            result[f"x_{split}"] = x
            result[f"y_{split}"] = y
        return result

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
        data = self._load_all()
        return (
            self._make_loader(data["x_train"], data["y_train"], batch_size, num_workers, shuffle=True),
            self._make_loader(data["x_val"], data["y_val"], batch_size, num_workers, shuffle=False),
        )

    def get_final_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        data = self._load_all()
        trainval_x = np.concatenate([data["x_train"], data["x_val"]], axis=0)
        trainval_y = np.concatenate([data["y_train"], data["y_val"]], axis=0)
        return (
            self._make_loader(trainval_x, trainval_y, batch_size, num_workers, shuffle=True),
            self._make_loader(data["x_test"], data["y_test"], batch_size, num_workers, shuffle=False),
        )

    def loss_fn(self) -> torch.nn.Module:
        return torch.nn.BCEWithLogitsLoss()

    def compute_metric(
        self,
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
        all_extras: list[torch.Tensor] | None = None,
    ) -> float:
        """1 - mean per-class AUROC (lower=better)."""
        preds = torch.cat(all_preds, dim=0).sigmoid().cpu().numpy()
        targets = torch.cat(all_targets, dim=0).cpu().numpy()

        from sklearn.metrics import roc_auc_score

        aucs = []
        for i in range(preds.shape[1]):
            # Skip classes with no positive or all-positive labels
            if targets[:, i].sum() > 0 and targets[:, i].sum() < len(targets):
                auc = roc_auc_score(targets[:, i], preds[:, i])
                aucs.append(auc)
        mean_auroc = np.mean(aucs) if aucs else 0.0
        return 1.0 - mean_auroc

    def input_shape(self) -> tuple[int, ...]:
        return (4, 1000)

    def output_spec(self) -> dict[str, Any]:
        return {"num_classes": 36, "task_type": "multi_label"}
