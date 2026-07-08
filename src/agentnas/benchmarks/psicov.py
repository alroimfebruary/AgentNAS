"""PSICOV benchmark: 2D dense regression for protein distance map prediction.

Training: Random 128×128 crops from padded proteins.
Inference: Eval loader tiles proteins into 128×128 patches with protein IDs.
           compute_metric stitches predictions back and evaluates MAE.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from agentnas.benchmarks.base import BaseBenchmark

_LMAX = 512
_PAD_SIZE = 10
_WINDOW = 128  # Training and inference window size
_STRIDE = 128  # Sliding window stride (non-overlapping by default)
_PAD_VALUE_TARGET: float = 100.0
_TRAIN_LABEL_CAP: float = 16.0


def _load_protein_ids(lst_path: Path) -> list[str]:
    ids = []
    with open(lst_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                ids.append(parts[0])
    return ids


def _load_features(feat_path: Path) -> dict:
    with open(feat_path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _load_distance(dist_path: Path) -> tuple[int, np.ndarray]:
    data = np.load(dist_path, allow_pickle=True)
    seq_len = int(data[0])
    dist_matrix = data[2].astype(np.float32)
    return seq_len, dist_matrix


def _assemble_features(features: dict, seq_len: int) -> np.ndarray:
    """Assemble 57-channel 2D representation from feature dict."""
    L = seq_len
    feat = np.zeros((57, L, L), dtype=np.float32)

    feat[0] = features["ccmpred"].astype(np.float32)[:L, :L]
    feat[1] = features["freecon"].astype(np.float32)[:L, :L]
    feat[2] = features["potential"].astype(np.float32)[:L, :L]

    pssm = features["pssm"].astype(np.float32)[:L, :]
    for c in range(22):
        feat[3 + c] = np.tile(pssm[:, c:c+1], (1, L))
        feat[25 + c] = np.tile(pssm[:, c:c+1].T, (L, 1))

    ss = features["ss"].astype(np.float32)[:, :L]
    for c in range(3):
        feat[47 + c] = np.tile(ss[c:c+1, :].T, (1, L))
        feat[50 + c] = np.tile(ss[c:c+1, :], (L, 1))

    sa = features["sa"].astype(np.float32)[:L]
    feat[53] = np.tile(sa[:, np.newaxis], (1, L))
    feat[54] = np.tile(sa[np.newaxis, :], (L, 1))

    entropy = features["entropy"].astype(np.float32)[:L]
    feat[55] = np.tile(entropy[:, np.newaxis], (1, L))
    feat[56] = np.tile(entropy[np.newaxis, :], (L, 1))

    return feat


def _pad_protein(arr: np.ndarray, pad_size: int, pad_value: float = 0.0) -> np.ndarray:
    """Pad a (C, L, L) array with ``pad_value``, offset by pad_size//2.

    Inputs use ``pad_value=0.0`` (zeros for missing features). Targets use
    ``pad_value=100.0`` so the metric mask `(tgt > 0) & (tgt <= 8)` excludes
    padding and the loss penalises any constant predictor that sits in the
    contact range.
    """
    c, L, _ = arr.shape
    target = L + pad_size
    padded = np.full((c, target, target), pad_value, dtype=arr.dtype)
    offset = pad_size // 2
    padded[:, offset:offset + L, offset:offset + L] = arr
    return padded


class PsicovTrainDataset(Dataset):
    """Training dataset: random 128×128 crops from padded proteins.

    Pre-assembles and pads all proteins in float16 during __init__ so that
    __getitem__ is a fast crop + cast. This allows num_workers=0 without
    starving the GPU — the bottleneck shifts from I/O to a trivial memcpy.
    """

    def __init__(self, protein_ids: list[str], data_dir: Path, subset_name: str,
                 window: int = _WINDOW, pad_size: int = _PAD_SIZE,
                 label_cap: float | None = _TRAIN_LABEL_CAP):
        self.window = window
        self.label_cap = label_cap

        feat_dir = data_dir / subset_name / "features"
        dist_dir = data_dir / subset_name / "distance"
        self._cached: list[tuple[np.ndarray, np.ndarray]] = []
        for pid in protein_ids:
            features = _load_features(feat_dir / f"{pid}.pkl")
            seq_len, dist_matrix = _load_distance(dist_dir / f"{pid}-cb.npy")
            feat = _assemble_features(features, seq_len)       # (57, L, L) float32
            target = dist_matrix[np.newaxis, :seq_len, :seq_len]  # (1, L, L) float32
            
            target = np.nan_to_num(target, nan=0.0)
            feat_padded = _pad_protein(feat, pad_size, pad_value=0.0).astype(np.float16)
            target_padded = _pad_protein(
                target, pad_size, pad_value=_PAD_VALUE_TARGET,
            ).astype(np.float16)
            self._cached.append((feat_padded, target_padded))

    def __len__(self) -> int:
        return len(self._cached)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        feat_padded, target_padded = self._cached[idx]

        H = feat_padded.shape[1]
        W = self.window

        if H <= W:
            crop_feat = np.zeros((57, W, W), dtype=np.float32)
            # Outer-tile fill is the same padding sentinel so loss-padding stays consistent.
            crop_tgt = np.full((1, W, W), _PAD_VALUE_TARGET, dtype=np.float32)
            crop_feat[:, :H, :H] = feat_padded  # float16 → float32 on copy
            crop_tgt[:, :H, :H] = target_padded
        else:
            rx = random.randint(0, H - W)
            ry = random.randint(0, H - W)
            crop_feat = feat_padded[:, rx:rx+W, ry:ry+W].astype(np.float32)
            crop_tgt = target_padded[:, rx:rx+W, ry:ry+W].astype(np.float32)

        if self.label_cap is not None:
            np.minimum(crop_tgt, self.label_cap, out=crop_tgt)

        return torch.from_numpy(crop_feat), torch.from_numpy(crop_tgt)


class PsicovEvalDataset(Dataset):
    """Eval dataset: tiles each protein into 128×128 patches with protein IDs.

    Each sample is one tile. compute_metric stitches them back per protein.
    """

    def __init__(self, protein_ids: list[str], data_dir: Path, subset_name: str,
                 window: int = _WINDOW, stride: int = _STRIDE, pad_size: int = _PAD_SIZE):
        self.window = window
        self.stride = stride
        self.pad_size = pad_size

        # Pre-compute all tiles
        self.tiles: list[tuple[np.ndarray, np.ndarray, int, int, int, int]] = []
        # (feat_crop, tgt_crop, protein_idx, row_offset, col_offset, padded_size)

        for pidx, pid in enumerate(protein_ids):
            feat_dir = data_dir / subset_name / "features"
            dist_dir = data_dir / subset_name / "distance"

            features = _load_features(feat_dir / f"{pid}.pkl")
            seq_len, dist_matrix = _load_distance(dist_dir / f"{pid}-cb.npy")

            feat = _assemble_features(features, seq_len)
            target = dist_matrix[np.newaxis, :seq_len, :seq_len]
            target = np.nan_to_num(target, nan=0.0)

            feat_padded = _pad_protein(feat, pad_size, pad_value=0.0)
            target_padded = _pad_protein(target, pad_size, pad_value=_PAD_VALUE_TARGET)

            H = feat_padded.shape[1]

            if H <= window:
                # Small protein: single tile. Outer-tile fill uses the padding
                # sentinel so the metric mask filters it out the same way as
                # the inner border.
                crop_feat = np.zeros((57, window, window), dtype=np.float32)
                crop_tgt = np.full((1, window, window), _PAD_VALUE_TARGET, dtype=np.float32)
                crop_feat[:, :H, :H] = feat_padded
                crop_tgt[:, :H, :H] = target_padded
                self.tiles.append((crop_feat, crop_tgt, pidx, 0, 0, H))
            else:
                # Generate non-overlapping tiles covering the full protein
                positions = list(range(0, H - window + 1, stride))
                # Ensure last tile covers the end
                if positions[-1] + window < H:
                    positions.append(H - window)

                for ri in positions:
                    for ci in positions:
                        crop_feat = feat_padded[:, ri:ri+window, ci:ci+window].copy()
                        crop_tgt = target_padded[:, ri:ri+window, ci:ci+window].copy()
                        self.tiles.append((crop_feat, crop_tgt, pidx, ri, ci, H))

        self.n_proteins = len(protein_ids)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat, tgt, pidx, row, col, padded_size = self.tiles[idx]
        # Pack metadata into a single tensor: [protein_idx, row_offset, col_offset, padded_size]
        meta = torch.tensor([pidx, row, col, padded_size], dtype=torch.long)
        return torch.from_numpy(feat), torch.from_numpy(tgt), meta


class PsicovBenchmark(BaseBenchmark):
    """PSICOV: (57, 128, 128) → (1, 128, 128) dense distance regression.

    Training: Random 128×128 crops from padded proteins.
    Eval: 128×128 sliding window tiles, stitched in compute_metric.
    Metric: MAE at distance ≤ 8Å, sequence separation ≥ 24 (lower=better).
    """

    def __init__(self, data_root: str):
        super().__init__(data_root, "psicov")
        self._data_dir = Path(data_root) / "psicov"

    def _get_splits(self) -> tuple[list[str], list[str], list[str]]:
        deepcov_ids = _load_protein_ids(self._data_dir / "deepcov.lst")
        psicov_ids = _load_protein_ids(self._data_dir / "psicov.lst")
        return deepcov_ids[100:], deepcov_ids[:100], psicov_ids

    def get_search_loaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        train_ids, val_ids, _ = self._get_splits()
        train_ds = PsicovTrainDataset(train_ids, self._data_dir, "deepcov")
        val_ds = PsicovEvalDataset(val_ids, self._data_dir, "deepcov")
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

        train_ids, val_ids, test_ids = self._get_splits()
        train_ds = PsicovTrainDataset(train_ids, self._data_dir, "deepcov")
        val_train_ds = PsicovTrainDataset(val_ids, self._data_dir, "deepcov")
        trainval_ds = ConcatDataset([train_ds, val_train_ds])
        test_ds = PsicovEvalDataset(test_ids, self._data_dir, "psicov")
        return (
            DataLoader(trainval_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True),
        )

    def loss_fn(self) -> torch.nn.Module:
        return torch.nn.MSELoss()

    def compute_search_metric(
        self,
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
        all_extras: list[torch.Tensor] | None = None,
    ) -> float:
        preds = torch.cat(all_preds, dim=0).float()
        targets = torch.cat(all_targets, dim=0).float()
        capped = torch.minimum(targets, torch.full_like(targets, _TRAIN_LABEL_CAP))
        return (preds - capped).abs().mean().item()

    def compute_metric(
        self,
        all_preds: list[torch.Tensor],
        all_targets: list[torch.Tensor],
        all_extras: list[torch.Tensor] | None = None,
    ) -> float:
        """MAE at distance ≤ 8Å, separation ≥ 24 (lower=better).

        Stitches 128×128 tile predictions back into full proteins using
        tile metadata from all_extras, then evaluates per protein.
        """
        preds = torch.cat(all_preds, dim=0).cpu()    # (N_tiles, 1, 128, 128)
        targets = torch.cat(all_targets, dim=0).cpu()

        # If no extras (e.g., training eval without tiling), evaluate directly on patches
        if all_extras is None:
            return self._eval_patches(preds, targets)

        metas = torch.cat(all_extras, dim=0).cpu()  # (N_tiles, 4) = [pidx, row, col, padded_size]

        # Group tiles by protein and stitch
        protein_preds = {}  # pidx → (accumulated_pred, counts)
        protein_targets = {}

        for i in range(preds.shape[0]):
            pidx = metas[i, 0].item()
            row = metas[i, 1].item()
            col = metas[i, 2].item()
            H = metas[i, 3].item()

            if pidx not in protein_preds:
                # Use at least _WINDOW size for small proteins whose tiles
                # were zero-padded to _WINDOW×_WINDOW in PsicovEvalDataset
                buf_size = max(H, _WINDOW)
                protein_preds[pidx] = (torch.zeros(1, buf_size, buf_size), torch.zeros(1, buf_size, buf_size))
                protein_targets[pidx] = torch.zeros(1, buf_size, buf_size)

            pred_acc, count_acc = protein_preds[pidx]
            W = _WINDOW
            pred_acc[0, row:row+W, col:col+W] += preds[i, 0]
            count_acc[0, row:row+W, col:col+W] += 1.0
            # Use first tile's target for each region (they should be identical)
            protein_targets[pidx][0, row:row+W, col:col+W] = targets[i, 0]

        # Evaluate per protein
        total_mae = 0.0
        count = 0
        offset = _PAD_SIZE // 2

        for pidx in sorted(protein_preds.keys()):
            pred_acc, count_acc = protein_preds[pidx]
            pred_avg = pred_acc / count_acc.clamp(min=1)
            tgt = protein_targets[pidx]

            # Find actual protein region. Real residue-pair distances are in
            # (0, ~50] Å. Padding cells carry the sentinel _PAD_VALUE_TARGET
            # (100.0) and cells NaN'd in the source distance map are zeroed
            # via nan_to_num. Both endpoints are excluded.
            tgt_2d = tgt[0]
            real = (tgt_2d > 0) & (tgt_2d < _PAD_VALUE_TARGET)
            if not real.any():
                continue

            rows = real.any(dim=1).nonzero(as_tuple=True)[0]
            cols = real.any(dim=0).nonzero(as_tuple=True)[0]
            if len(rows) == 0:
                continue

            r0, r1 = rows[0].item(), rows[-1].item() + 1
            c0, c1 = cols[0].item(), cols[-1].item() + 1
            L = r1 - r0

            pred_crop = pred_avg[0, r0:r1, c0:c1]
            tgt_crop = tgt_2d[r0:r1, c0:c1]

            # Symmetrize
            pred_sym = (pred_crop + pred_crop.T) / 2.0

            # Separation mask: |i - j| >= 24
            idx = torch.arange(L)
            sep = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
            sep_mask = sep >= 24

            # Distance threshold: ≤ 8Å
            dist_mask = tgt_crop <= 8.0
            valid = sep_mask & dist_mask & (tgt_crop > 0)

            if valid.sum() == 0:
                continue

            mae = (pred_sym[valid] - tgt_crop[valid]).abs().mean().item()
            total_mae += mae
            count += 1

        return total_mae / max(count, 1)

    def _eval_patches(self, preds, targets):
        """Fallback: evaluate directly on patches (no stitching)."""
        total_mae = 0.0
        count = 0
        for k in range(preds.shape[0]):
            p, t = preds[k, 0], targets[k, 0]
            real = (t > 0) & (t < _PAD_VALUE_TARGET)
            if not real.any():
                continue
            L = min(p.shape[0], t.shape[0])
            idx = torch.arange(L)
            sep = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
            valid = (sep >= 24) & (t[:L, :L] <= 8.0) & (t[:L, :L] > 0)
            if valid.sum() == 0:
                continue
            p_sym = (p[:L, :L] + p[:L, :L].T) / 2.0
            mae = (p_sym[valid] - t[:L, :L][valid]).abs().mean().item()
            total_mae += mae
            count += 1
        return total_mae / max(count, 1)

    def input_shape(self) -> tuple[int, ...]:
        return (57, _WINDOW, _WINDOW)  # 128×128 patches

    def output_spec(self) -> dict[str, Any]:
        return {"output_shape": (1, _WINDOW, _WINDOW)}
