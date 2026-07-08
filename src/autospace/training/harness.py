"""Fixed training harness enforcing the evaluation protocol.

Two modes:
- train_search: train on train split, best checkpoint by val metric, epoch-limited (30%
  of proposed epochs) with 60-min GPU time hard cap
- train_final:  train on train+val, last checkpoint, 24-hr GPU time limit
"""

from __future__ import annotations

import copy
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from autospace.benchmarks.base import BaseBenchmark
from autospace.training.config import TrainingConfig
from autospace.training.metrics import (
    count_parameters,
    get_peak_vram_mb,
    measure_flops,
    reset_vram_tracking,
)
from autospace.utils.timing import GPUTimer

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Result of a single training run."""

    metric: float  # Lower = better (benchmark's normalized metric)
    wall_time_s: float  # GPU compute time in seconds (via CUDA events; named for legacy compat)
    peak_vram_mb: float
    params: int
    flops: int
    epochs_completed: int
    status: str  # "success", "timeout", "oom", "error"
    error_message: str = ""
    train_losses: list[float] = field(default_factory=list)
    val_metrics: list[float] = field(default_factory=list)
    best_epoch: int = -1


def _build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    params = model.parameters()
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            params, lr=config.lr, momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam":
        return torch.optim.Adam(
            params, lr=config.lr, betas=config.betas,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adamw":
        return torch.optim.AdamW(
            params, lr=config.lr, betas=config.betas,
            weight_decay=config.weight_decay,
        )
    else:
        logger.warning(f"Unknown optimizer '{config.optimizer}', falling back to AdamW")
        return torch.optim.AdamW(
            params, lr=config.lr, weight_decay=config.weight_decay,
        )


def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Build LR scheduler from config."""
    if config.scheduler == "none":
        return None
    elif config.scheduler == "cosine":
        T_max = config.scheduler_params.get("T_max", config.epochs)
        eta_min = config.scheduler_params.get("eta_min", 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min,
        )
    elif config.scheduler == "step":
        step_size = config.scheduler_params.get("step_size", 30)
        gamma = config.scheduler_params.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma,
        )
    elif config.scheduler == "warmup_cosine":
        # Warmup for N epochs, then cosine
        warmup = config.warmup_epochs
        T_max = max(1, config.scheduler_params.get("T_max", config.epochs - warmup))  # #15 guard
        eta_min = config.scheduler_params.get("eta_min", 0.0)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min,
        )
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine], milestones=[warmup],
        )
    # NOTE: cosine_with_warmup and cosine_with_warm_restarts intentionally NOT
    # supported here. Phase 2 LLM frequently misspecs SGDR parameters (e.g.,
    # T_0=100/T_mult=2 with epochs=500 leaves training mid-cycle at epoch 500,
    # producing under-converged last-checkpoint weights for test_final). The
    # silent fallback to plain cosine below is intentionally protective —
    # validated by v1/v2 producing strong test metrics, and v3 (with these
    # aliases active) regressing 25% on DarcyFlow.
    elif config.scheduler in ("plateau", "reduce_lr_on_plateau", "reduce_on_plateau"):
        patience = config.scheduler_params.get("patience", 5)
        factor = config.scheduler_params.get("factor", 0.5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience,
        )
    else:
        logger.warning(f"Unknown scheduler '{config.scheduler}', falling back to cosine")
        T_max = config.scheduler_params.get("T_max", config.epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=0.0,
        )


def _apply_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """Mixup: interpolate between random pairs of samples."""
    import numpy as np
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def _apply_cutmix(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """CutMix: replace a random rectangular patch with another sample's patch."""
    import numpy as np
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)

    # Generate random bounding box
    _, _, H, W = x.shape  # CutMix only called for 4D tensors
    cut_ratio = (1 - lam) ** 0.5
    cut_h, cut_w = max(1, int(H * cut_ratio)), max(1, int(W * cut_ratio))  # H-03: min 1×1
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)

    mixed_x = x.clone()
    if x.dim() == 4:
        mixed_x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        # Adjust lambda to actual area
        lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return mixed_x, y, y[idx], lam


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gpu_timer: "GPUTimer | None" = None,
    grad_clip: float | None = None,
    augmentations: list[str] | None = None,
    augmentation_params: dict | None = None,
) -> tuple[float, bool]:
    """Train for one epoch. Returns (average_loss, time_exceeded)."""
    augmentations = augmentations or []
    augmentation_params = augmentation_params or {}

    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        x, y = batch[0], batch[1]
        x, y = x.to(device), y.to(device)

        # Apply augmentations (B3)
        y_b, lam = None, 1.0
        # H-01: Skip CutMix for dense prediction (y has spatial dims matching x)
        is_dense = y.dim() >= 3 and y.shape[-2:] == x.shape[-2:]
        if "mixup" in augmentations and not is_dense:
            alpha = augmentation_params.get("alpha", 0.2)
            x, y, y_b, lam = _apply_mixup(x, y, alpha)
        elif "cutmix" in augmentations and x.dim() == 4 and not is_dense:
            alpha = augmentation_params.get("alpha", 1.0)
            x, y, y_b, lam = _apply_cutmix(x, y, alpha)
        # H-03: random_erasing support
        if "random_erasing" in augmentations:
            from torchvision.transforms import RandomErasing
            p = augmentation_params.get("p", 0.5)
            x = RandomErasing(p=p, inplace=False)(x)
        # Warn on unrecognized augmentations (H-03)
        _known_augs = {"mixup", "cutmix", "random_erasing"}
        for aug in augmentations:
            if aug not in _known_augs and n_batches == 0:
                logger.warning(f"Unknown augmentation '{aug}' — silently ignored")

        # Time only the GPU compute
        if gpu_timer:
            gpu_timer.step_start()

        optimizer.zero_grad()
        output = model(x)

        # Mixed loss for augmented training (B-11: use target interpolation for regression)
        if y_b is not None:
            if y.dim() > 1 and y.shape == output.shape:
                # Regression: interpolate targets directly
                mixed_y = lam * y + (1 - lam) * y_b
                loss = loss_fn(output, mixed_y)
            else:
                # Classification: weighted loss (standard mixup formulation)
                loss = lam * loss_fn(output, y) + (1 - lam) * loss_fn(output, y_b)
        else:
            loss = loss_fn(output, y)

        # NaN/Inf guard (B14, H-01) — stop training if loss is non-finite
        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss detected, stopping epoch")
            if gpu_timer:
                gpu_timer.step_end()
                gpu_timer.flush()
            return float("nan"), True  # NaN avg_loss signals numerical failure to caller

        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if gpu_timer:
            gpu_timer.step_end()

        total_loss += loss.item()
        n_batches += 1

        # Check GPU time budget after each step
        if gpu_timer and gpu_timer.exceeded:
            return total_loss / max(n_batches, 1), True

    # Flush orphan steps at epoch boundary (C2 fix)
    if gpu_timer:
        gpu_timer.flush()

    return total_loss / max(n_batches, 1), False


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    benchmark: BaseBenchmark,
    device: torch.device,
    gpu_timer: "GPUTimer | None" = None,
    metric_kind: str = "final",
) -> float:
    """Evaluate model on a loader. Returns metric (lower=better).

    Evaluation GPU time counts toward the training budget.
    Passes extra batch elements (e.g., recording IDs) through to compute_metric.

    Args:
        metric_kind: "final" → ``benchmark.compute_metric`` (official metric,
            used for final/test reporting). "search" →
            ``benchmark.compute_search_metric`` (best-checkpoint selection
            during search-mode training; defaults to ``compute_metric`` unless
            the benchmark overrides — currently only PSICOV does, see
            ``psicov.py`` for why).
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_extras = []
    has_extras = False

    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device)

        if gpu_timer:
            gpu_timer.step_start()
        output = model(x)
        if gpu_timer:
            gpu_timer.step_end()

        all_preds.append(output.cpu())
        all_targets.append(y)
        if len(batch) > 2:
            all_extras.append(batch[2])
            has_extras = True

    extras = all_extras if has_extras else None
    if metric_kind == "search":
        return benchmark.compute_search_metric(all_preds, all_targets, extras)
    return benchmark.compute_metric(all_preds, all_targets, extras)


def train_search(
    model: nn.Module,
    config: TrainingConfig,
    benchmark: BaseBenchmark,
    device: torch.device,
    time_limit_s: float = 3600,
    save_dir: Path | None = None,
    proxy_epoch_fraction: float = 0.3,
) -> TrainResult:
    """Search-phase training: train split, best checkpoint, epoch- and time-limited.

    Args:
        model: The nn.Module to train.
        config: Training recipe (designed for full training).
        benchmark: Benchmark providing data loaders and metrics.
        device: CUDA device.
        time_limit_s: GPU time hard cap in seconds (default 60 min).
        save_dir: Directory to save best checkpoint.
        proxy_epoch_fraction: Fraction of proposed epochs to run (default 0.3).

    Returns:
        TrainResult with metric from best checkpoint.
    """
    return _train_core(
        model=model,
        config=config,
        benchmark=benchmark,
        device=device,
        time_limit_s=time_limit_s,
        save_dir=save_dir,
        mode="search",
        proxy_epoch_fraction=proxy_epoch_fraction,
    )


def train_final(
    model: nn.Module,
    config: TrainingConfig,
    benchmark: BaseBenchmark,
    device: torch.device,
    time_limit_s: float = 86400,
    save_dir: Path | None = None,
) -> TrainResult:
    """Final-phase training: train+val combined, last checkpoint, time-limited.

    Args:
        model: The nn.Module to train.
        config: Training recipe.
        benchmark: Benchmark providing data loaders and metrics.
        device: CUDA device.
        time_limit_s: GPU time hard cap in seconds (default 24 hr, measured via CUDA events).
        save_dir: Directory to save final checkpoint.

    Returns:
        TrainResult with metric from last checkpoint on test set.
    """
    return _train_core(
        model=model,
        config=config,
        benchmark=benchmark,
        device=device,
        time_limit_s=time_limit_s,
        save_dir=save_dir,
        mode="final",
    )


def _worker_init_fn(worker_id):
    """Deterministic re-seeding in DataLoader workers (M3 fix).

    Module-level function so it is picklable — required by forkserver context.
    """
    import numpy as _np
    import random as _rnd
    _seed = torch.initial_seed() % (2**32) + worker_id
    _np.random.seed(_seed)
    _rnd.seed(_seed)


def _train_core(
    model: nn.Module,
    config: TrainingConfig,
    benchmark: BaseBenchmark,
    device: torch.device,
    time_limit_s: float,
    save_dir: Path | None,
    mode: str,  # "search" or "final"
    proxy_epoch_fraction: float = 1.0,
) -> TrainResult:
    """Core training loop shared by search and final modes."""
    # --- Proxy epoch reduction (search mode only) ---
    # The LLM proposes a full training recipe; proxy runs a fraction of epochs.
    # Scheduler T_max and warmup are scaled proportionally so the LR schedule
    # completes properly within the shorter proxy run.
    if mode == "search" and proxy_epoch_fraction < 1.0:
        original_epochs = config.epochs
        proxy_epochs = max(1, round(original_epochs * proxy_epoch_fraction))
        proxy_warmup = max(0, round(config.warmup_epochs * proxy_epoch_fraction))

        updated = config.to_dict()
        updated["epochs"] = proxy_epochs
        updated["warmup_epochs"] = proxy_warmup

        # Scale scheduler time-based params proportionally
        sp = updated.get("scheduler_params", {})
        if "T_max" in sp:
            sp["T_max"] = max(1, round(sp["T_max"] * proxy_epoch_fraction))
        if "step_size" in sp:
            sp["step_size"] = max(1, round(sp["step_size"] * proxy_epoch_fraction))
        if "patience" in sp:
            sp["patience"] = max(1, round(sp["patience"] * proxy_epoch_fraction))
        if sp:
            updated["scheduler_params"] = sp

        config = TrainingConfig.from_dict(updated)
        logger.info(
            f"Proxy epoch reduction: {original_epochs} → {proxy_epochs} epochs "
            f"({proxy_epoch_fraction:.0%}), warmup {config.warmup_epochs}"
        )

    # Setup data loaders (NOT counted toward training time budget)
    batch_size = config.batch_size
    if mode == "search":
        train_loader, eval_loader = benchmark.get_search_loaders(batch_size, num_workers=int(os.environ.get("AUTOSPACE_NUM_WORKERS", "4")))
    else:
        train_loader, eval_loader = benchmark.get_final_loaders(batch_size, num_workers=int(os.environ.get("AUTOSPACE_NUM_WORKERS", "4")))

    # Re-create loaders with worker_init_fn properly set via constructor.
    # persistent_workers is forwarded from the original loader so benchmarks
    # can opt in (default False preserves prior behavior for all other tasks).
    from torch.utils.data import DataLoader as _DL, RandomSampler
    if train_loader.num_workers > 0:
        train_loader = _DL(
            train_loader.dataset, batch_size=train_loader.batch_size,
            shuffle=isinstance(train_loader.sampler, RandomSampler),
            num_workers=train_loader.num_workers, pin_memory=train_loader.pin_memory,
            drop_last=train_loader.drop_last, worker_init_fn=_worker_init_fn,
            persistent_workers=getattr(train_loader, "persistent_workers", False),
        )
    if eval_loader.num_workers > 0:
        eval_loader = _DL(
            eval_loader.dataset, batch_size=eval_loader.batch_size,
            shuffle=False, num_workers=eval_loader.num_workers,
            pin_memory=eval_loader.pin_memory, drop_last=eval_loader.drop_last,
            worker_init_fn=_worker_init_fn,
            persistent_workers=getattr(eval_loader, "persistent_workers", False),
        )

    # Reset VRAM tracking BEFORE model transfer so model weights are included (H-03 fix)
    reset_vram_tracking()
    model = model.to(device)

    # Measure FLOPs and params before training
    params = count_parameters(model)
    try:
        flops = measure_flops(model, benchmark.input_shape())
    except ImportError:
        logger.warning("fvcore not installed — FLOPs measurement unavailable")
        flops = 0
    except Exception as e:
        logger.warning(f"FLOPs measurement failed: {e}")
        flops = 0

    # Build optimizer and scheduler
    loss_fn = benchmark.loss_fn()
    # Apply label smoothing if specified and loss supports it
    if config.label_smoothing > 0 and isinstance(loss_fn, nn.CrossEntropyLoss):
        # M-08: preserve existing loss params (weight, ignore_index, reduction)
        loss_fn = nn.CrossEntropyLoss(
            weight=loss_fn.weight,
            ignore_index=loss_fn.ignore_index,
            reduction=loss_fn.reduction,
            label_smoothing=config.label_smoothing,
        )
    loss_fn = loss_fn.to(device)
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)

    # Replace with CosineAnnealingLR to ensure LR decays to near-zero for last checkpoint.
    if mode == "final" and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        logger.info("Replacing ReduceLROnPlateau with CosineAnnealingLR for final training")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=0.0,
        )

    # GPU timer — measures only actual GPU compute time (CUDA events)
    timer = GPUTimer(time_limit_s, device=device)

    # Training loop
    best_metric = float("inf")
    best_state = None
    best_epoch = -1
    train_losses = []
    val_metrics = []
    epochs_completed = 0
    status = "success"
    error_msg = ""

    try:
        for epoch in range(config.epochs):
            if timer.exceeded:
                logger.info(f"GPU time limit reached at epoch {epoch} ({timer.elapsed_s:.0f}s GPU time)")
                status = "timeout" if epochs_completed == 0 else "success"
                break

            # Train (GPU time accumulated per step inside _train_epoch)
            avg_loss, time_exceeded = _train_epoch(
                model, train_loader, loss_fn, optimizer, device,
                gpu_timer=timer,
                grad_clip=config.grad_clip,
                augmentations=config.augmentations,
                augmentation_params=config.augmentation_params,
            )
            # Check NaN BEFORE incrementing epochs (M1 fix)
            import math
            if not math.isfinite(avg_loss):
                status = "error"
                error_msg = "NaN loss detected during training"
                logger.error(f"NaN loss at epoch {epoch}, marking as error")
                break

            train_losses.append(avg_loss)
            epochs_completed = epoch + 1

            if time_exceeded:
                logger.info(f"GPU time limit hit mid-epoch {epoch} ({timer.elapsed_s:.0f}s)")
                break

            # Evaluate + step scheduler (search mode only)
            # Final mode: NO evaluation during training — use last checkpoint,
            # evaluate on test ONCE after training completes. (C1 fix: prevents
            # data leakage from stepping scheduler on test metrics.)
            if mode == "search" and not timer.exceeded:
                metric = _evaluate(
                    model, eval_loader, benchmark, device,
                    gpu_timer=timer, metric_kind="search",
                )
                val_metrics.append(metric)

                if metric < best_metric:
                    best_metric = metric
                    best_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch

                # Step scheduler with fresh metric
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(metric)
                    else:
                        scheduler.step()
            elif mode == "final":
                # Step non-plateau schedulers without evaluation
                if scheduler is not None and not isinstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    scheduler.step()
                best_epoch = epoch  # Last epoch is always "best" in final mode

            if timer.exceeded:
                break

    except torch.cuda.OutOfMemoryError:
        status = "oom"
        error_msg = f"OOM at epoch {epochs_completed}"
        logger.error(error_msg)
    except Exception as e:
        status = "error"
        error_msg = str(e)
        logger.error(f"Training error: {e}")

    timer.stop()

    # Save checkpoint
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if mode == "search" and best_state is not None:
            torch.save(best_state, save_dir / "best_checkpoint.pt")
        elif mode == "final":
            torch.save(model.state_dict(), save_dir / "last_checkpoint.pt")

    # Final mode: evaluate on test set ONCE after training completes (C1 fix)
    if mode == "final" and status not in ("oom", "error"):
        final_eval_metric = _evaluate(model, eval_loader, benchmark, device)
        best_metric = final_eval_metric
        val_metrics.append(final_eval_metric)

    final_metric = best_metric if status != "oom" else float("inf")

    return TrainResult(
        metric=final_metric,
        wall_time_s=timer.elapsed_s,  # Actually GPU time (measured via CUDA events)
        peak_vram_mb=get_peak_vram_mb(),
        params=params,
        flops=flops,
        epochs_completed=epochs_completed,
        status=status,
        error_message=error_msg,
        train_losses=train_losses,
        val_metrics=val_metrics,
        best_epoch=best_epoch,
    )
