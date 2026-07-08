"""GDAS bilevel search loop for agentnas one-shot NAS.

Trains a `MixedSlot`-based supernet via two interleaved optimizers:
    - weight optimizer (SGD + cosine) over all non-alpha parameters,
    - architecture optimizer (Adam) over the per-slot alpha parameters,
alternating per-batch on a held-out (train_loader, val_loader) split.

The Gumbel-softmax temperature `tau` is annealed linearly from `tau_init` to
`tau_final` across all search epochs. Architecture updates are gated by a
warmup of weight-only training so the supernet's branches are minimally
trained before alpha gradients start steering them.

Public API: `run_gdas_search(supernet, mixed, train_loader, val_loader,
loss_fn, device, gdas_cfg, training_cfg, save_dir) -> dict[str, int]`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from agentnas.config import GDASConfig
from agentnas.grammar.supernet import MixedSlot
from agentnas.utils.state import atomic_write_json

logger = logging.getLogger(__name__)


def _alpha_param_ids(mixed: dict[str, MixedSlot]) -> set[int]:
    return {id(ms.alpha) for ms in mixed.values()}


def _split_params(
    supernet: nn.Module, alpha_ids: set[int]
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return (weight_params, alpha_params) — disjoint, by id()."""
    w_params: list[nn.Parameter] = []
    a_params: list[nn.Parameter] = []
    for p in supernet.parameters():
        if id(p) in alpha_ids:
            a_params.append(p)
        else:
            w_params.append(p)
    return w_params, a_params


def _make_optimizers(
    weight_params: list[nn.Parameter],
    alpha_params: list[nn.Parameter],
    training_cfg: dict[str, Any],
    gdas_cfg: GDASConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer, Any]:
    """Construct (weight_optim, arch_optim, weight_scheduler).

    The supernet weight optimizer mirrors `training_cfg["optimizer"]` so that
    the supernet's training dynamics match what agentnas uses for the
    realized model's final-training. Mismatches (e.g. SGD with an
    AdamW-calibrated LR of 1e-3) leave the supernet undertrained and the
    discretized argmax architecture poorly evaluated.
    """
    optim_name = str(training_cfg.get("optimizer", "adamw")).lower()
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    betas = tuple(training_cfg.get("betas", (0.9, 0.999)))
    momentum = float(training_cfg.get("momentum", 0.9))

    if optim_name in ("adamw",):
        w_optim = torch.optim.AdamW(
            weight_params, lr=lr, betas=betas, weight_decay=weight_decay,
        )
    elif optim_name in ("adam",):
        w_optim = torch.optim.Adam(
            weight_params, lr=lr, betas=betas, weight_decay=weight_decay,
        )
    elif optim_name in ("sgd",):
        w_optim = torch.optim.SGD(
            weight_params, lr=lr, momentum=momentum, weight_decay=weight_decay,
        )
    else:
        # Unknown; default to AdamW (matches agentnas's most common config).
        logger.warning(f"Unknown optimizer '{optim_name}' in training_cfg; using AdamW")
        w_optim = torch.optim.AdamW(
            weight_params, lr=lr, betas=betas, weight_decay=weight_decay,
        )

    a_optim = torch.optim.Adam(
        alpha_params, lr=gdas_cfg.arch_lr, betas=(0.5, 0.999), weight_decay=gdas_cfg.arch_wd,
    )
    eta_min = float(training_cfg.get("scheduler_params", {}).get("eta_min", 1e-6))
    w_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, T_max=max(1, gdas_cfg.epochs), eta_min=eta_min,
    )
    return w_optim, a_optim, w_sched


def _tau_for_epoch(epoch: int, total: int, tau_init: float, tau_final: float) -> float:
    if total <= 1:
        return tau_final
    frac = epoch / (total - 1)
    return float(tau_init + (tau_final - tau_init) * frac)


def _set_tau(mixed: dict[str, MixedSlot], tau: float) -> None:
    for ms in mixed.values():
        ms.set_tau(tau)


def _alpha_softmax_snapshot(mixed: dict[str, MixedSlot]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for name, ms in mixed.items():
        out[name] = F.softmax(ms.alpha.detach(), dim=-1).cpu().tolist()
    return out


def _argmax_snapshot(mixed: dict[str, MixedSlot]) -> dict[str, int]:
    return {name: int(ms.alpha.argmax().item()) for name, ms in mixed.items()}


def _next_batch(
    iterator_holder: list[Iterable], loader, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x, y) on `device`. Cycles `loader` when exhausted.

    `iterator_holder` is a 1-element list used as a mutable cell so we can
    rebind the iterator on StopIteration without making the caller hold it.
    """
    while True:
        try:
            batch = next(iterator_holder[0])
            break
        except StopIteration:
            iterator_holder[0] = iter(loader)
    x = batch[0].to(device, non_blocking=True)
    y = batch[1].to(device, non_blocking=True)
    return x, y


def _save_checkpoint(
    save_dir: Path, supernet: nn.Module, w_optim, a_optim, w_sched,
    epoch: int, mixed: dict[str, MixedSlot],
) -> None:
    ckpt_path = save_dir / "search_ckpt.pt"
    tmp_path = ckpt_path.with_suffix(".pt.tmp")
    torch.save({
        "supernet": supernet.state_dict(),
        "w_optim": w_optim.state_dict(),
        "a_optim": a_optim.state_dict(),
        "w_sched": w_sched.state_dict(),
        "epoch": epoch,
        "alpha_argmax": _argmax_snapshot(mixed),
    }, tmp_path)
    tmp_path.replace(ckpt_path)


def _load_checkpoint(
    save_dir: Path, supernet: nn.Module, w_optim, a_optim, w_sched,
) -> int:
    """Return the next epoch to run (0 if no checkpoint)."""
    ckpt_path = save_dir / "search_ckpt.pt"
    if not ckpt_path.exists():
        return 0
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    supernet.load_state_dict(ckpt["supernet"])
    w_optim.load_state_dict(ckpt["w_optim"])
    a_optim.load_state_dict(ckpt["a_optim"])
    w_sched.load_state_dict(ckpt["w_sched"])
    return int(ckpt["epoch"]) + 1


def _append_search_log(save_dir: Path, meta: dict[str, Any], row: dict[str, Any]) -> None:
    """Append one row to gdas_search.json.epochs and atomically rewrite."""
    path = save_dir / "gdas_search.json"
    if path.exists():
        import json as _json
        try:
            data = _json.loads(path.read_text())
        except Exception:
            data = {"meta": meta, "epochs": []}
    else:
        data = {"meta": meta, "epochs": []}
    data["meta"] = meta
    data["epochs"].append(row)
    atomic_write_json(path, data)


def run_gdas_search(
    supernet: nn.Module,
    mixed: dict[str, MixedSlot],
    train_loader,
    val_loader,
    loss_fn,
    device: torch.device,
    gdas_cfg: GDASConfig,
    training_cfg: dict[str, Any],
    save_dir: Path,
    grad_clip: float = 5.0,
) -> dict[str, int]:
    """Run the GDAS bilevel search loop.

    Returns the discretized choice dict (per MixedSlot argmax). Slots not in
    `mixed` are not represented — caller must merge with default_idx.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    supernet = supernet.to(device)

    alpha_ids = _alpha_param_ids(mixed)
    weight_params, alpha_params = _split_params(supernet, alpha_ids)
    w_optim, a_optim, w_sched = _make_optimizers(
        weight_params, alpha_params, training_cfg, gdas_cfg,
    )

    total_epochs = max(1, gdas_cfg.epochs)
    warmup_epochs = gdas_cfg.warmup_epochs or max(5, total_epochs // 10)

    start_epoch = _load_checkpoint(save_dir, supernet, w_optim, a_optim, w_sched)
    if start_epoch > 0:
        logger.info(f"GDAS search resuming from epoch {start_epoch}/{total_epochs}")

    meta = {
        "epochs": total_epochs,
        "warmup_epochs": warmup_epochs,
        "tau_init": gdas_cfg.tau_init,
        "tau_final": gdas_cfg.tau_final,
        "arch_lr": gdas_cfg.arch_lr,
        "arch_wd": gdas_cfg.arch_wd,
        "weight_lr": training_cfg.get("lr"),
        "weight_momentum": training_cfg.get("momentum"),
        "weight_wd": training_cfg.get("weight_decay"),
        "n_mixed_slots": len(mixed),
        "n_weight_params": sum(p.numel() for p in weight_params),
        "n_alpha_params": sum(p.numel() for p in alpha_params),
    }
    logger.info(f"GDAS search meta: {meta}")

    val_iter_holder: list[Iterable] = [iter(val_loader)]

    for epoch in range(start_epoch, total_epochs):
        tau = _tau_for_epoch(epoch, total_epochs, gdas_cfg.tau_init, gdas_cfg.tau_final)
        do_arch_step = epoch >= warmup_epochs

        supernet.train()
        epoch_w_loss_sum, epoch_a_loss_sum, n_w, n_a = 0.0, 0.0, 0, 0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            x_t = batch[0].to(device, non_blocking=True)
            y_t = batch[1].to(device, non_blocking=True)

            # Step 1: weight update on training batch.
            _set_tau(mixed, tau)
            w_optim.zero_grad(set_to_none=True)
            logits = supernet(x_t)
            loss_w = loss_fn(logits, y_t)
            loss_w.backward()
            torch.nn.utils.clip_grad_norm_(weight_params, grad_clip)
            w_optim.step()
            epoch_w_loss_sum += float(loss_w.detach().item()); n_w += 1

            # Step 2: architecture update on val batch (first-order).
            if do_arch_step:
                x_v, y_v = _next_batch(val_iter_holder, val_loader, device)
                _set_tau(mixed, tau)
                a_optim.zero_grad(set_to_none=True)
                logits_v = supernet(x_v)
                loss_a = loss_fn(logits_v, y_v)
                loss_a.backward()
                # No grad clip on alphas — they're tiny and the dynamic range matters.
                a_optim.step()
                epoch_a_loss_sum += float(loss_a.detach().item()); n_a += 1

        w_sched.step()
        wall_s = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "tau": tau,
            "weight_loss": epoch_w_loss_sum / max(1, n_w),
            "arch_loss": (epoch_a_loss_sum / n_a) if n_a > 0 else None,
            "alpha_softmax": _alpha_softmax_snapshot(mixed),
            "argmax": _argmax_snapshot(mixed),
            "wall_s": wall_s,
            "weight_lr": w_sched.get_last_lr()[0],
        }
        _append_search_log(save_dir, meta, row)
        a_loss_str = "-" if row["arch_loss"] is None else f"{row['arch_loss']:.4f}"
        logger.info(
            f"[GDAS epoch {epoch+1}/{total_epochs}] tau={tau:.3f} "
            f"w_loss={row['weight_loss']:.4f} a_loss={a_loss_str} "
            f"wall={wall_s:.1f}s"
        )

        # Periodic checkpoint (every 5 epochs and at end).
        if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
            _save_checkpoint(save_dir, supernet, w_optim, a_optim, w_sched, epoch, mixed)

    return _argmax_snapshot(mixed)
