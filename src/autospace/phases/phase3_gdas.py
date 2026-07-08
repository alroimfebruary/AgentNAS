"""GDAS one-shot search variant of Phase 3.

Consumes Phase 2's slotted_arch.json, builds a Gumbel-STE supernet, runs
the bilevel search loop, argmaxes alpha to a discrete `choices` dict, and
runs a single test_final training pass on the realized architecture using
the same `_run_training_subprocess` machinery as the evolutionary path.

Dispatched from `phase3.run_phase3` when `config.evolution.search_mode ==
"gdas"`. Output dir + state.json + evaluations.jsonl land in the standard
locations so existing report tooling sees them transparently.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch

from autospace.config import PipelineConfig
from autospace.grammar.slots import SlottedArchitecture
from autospace.grammar.supernet import build_supernet
from autospace.utils.logging import ExperimentLogger
from autospace.utils.state import StateManager, atomic_write_json

logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _probe_oom(
    supernet: torch.nn.Module, input_shape: tuple[int, ...],
    device: torch.device, batch_size: int,
) -> str | None:
    """Return None on success, "oom" on OOM, "error: <msg>" otherwise.

    Probes at the actual training batch size with a real forward+backward —
    matches the memory footprint the search loop will see. Catches OOMs
    early so the orchestrator can write `failed_oom` state cleanly instead
    of crashing mid-training. "device busy/unavailable" is treated as OOM
    because the CUDA driver sometimes reports OOM through that error string.
    """
    try:
        supernet.to(device)
        x = torch.randn(batch_size, *input_shape, device=device)
        supernet.train()
        y = supernet(x)
        loss = y.sum()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        # Reset grads so the search loop starts clean.
        for p in supernet.parameters():
            if p.grad is not None: p.grad = None
        del x, y, loss
        if device.type == "cuda": torch.cuda.empty_cache()
        return None
    except torch.cuda.OutOfMemoryError:
        return "oom"
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "device(s) is/are busy" in msg or "unavailable" in msg:
            return "oom"
        return f"error: {e}"
    except Exception as e:
        return f"error: {e}"


def run_phase3_gdas(config: PipelineConfig, resume: bool = True) -> dict[str, Any]:
    """Run GDAS one-shot search for Phase 3."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _seed_everything(config.seed)

    exp_dir = Path(config.experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    state_mgr = StateManager(exp_dir / "state.json")
    if resume and state_mgr.exists:
        state_mgr.load()
        if state_mgr.get("phase") == "completed":
            logger.info("GDAS phase already completed; returning saved results")
            return {
                "status": "success",
                "search": "gdas",
                "test_final_results": state_mgr.get("test_final_results", []),
            }

    # Locate the Phase 2 slotted_arch.json. First try phase3's standard discovery
    # (covers the flat-layout case). Fall back to the nested round-based layout
    # used by run_production.py: experiments/{task}_..._seed{seed}/round_NN/phase2/.
    from autospace.phases.phase3 import _find_slotted_arch
    arch_path = _find_slotted_arch(config)
    if arch_path is None:
        arch_path = _find_slotted_arch_round_layout(config)
    if arch_path is None:
        raise FileNotFoundError(
            f"slotted_arch.json not found for task={config.task} seed={config.seed}. "
            f"Searched flat and round_NN/phase2 layouts under experiments/."
        )
    logger.info(f"GDAS: loading slotted_arch from {arch_path}")
    slotted = SlottedArchitecture.load(str(arch_path))

    # Build benchmark + loaders.
    from autospace.benchmarks.registry import get_benchmark
    benchmark = get_benchmark(config.task, data_root=config.data_root)
    bs = config.gdas.batch_size or int(slotted.training_config.get("batch_size", 32))
    train_loader, val_loader = benchmark.get_search_loaders(
        batch_size=bs, num_workers=4,
    )
    loss_fn = benchmark.loss_fn()
    input_shape = benchmark.input_shape()

    # Build supernet.
    logger.info("GDAS: constructing supernet")
    try:
        supernet, mixed = build_supernet(slotted)
    except torch.cuda.OutOfMemoryError:
        return _record_oom(exp_dir, state_mgr, "oom_build_supernet")
    except Exception as e:
        logger.error(f"GDAS supernet build failed: {e}", exc_info=True)
        state_mgr.update({"phase": "failed", "search": "gdas", "error": str(e)})
        state_mgr.save()
        raise

    n_params = sum(p.numel() for p in supernet.parameters())
    n_alpha = sum(ms.alpha.numel() for ms in mixed.values())
    logger.info(f"GDAS: supernet {n_params/1e6:.2f}M params, {len(mixed)} mixed slots, {n_alpha} alphas")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("GDAS: no GPU available — search will run on CPU (very slow)")

    probe = _probe_oom(supernet, input_shape, device, batch_size=bs)
    if probe == "oom":
        logger.error(
            f"GDAS: OOM on supernet probe (batch_size={bs}, input_shape={input_shape}) "
            f"— skipping task"
        )
        return _record_oom(exp_dir, state_mgr, "oom_probe")
    if probe is not None:
        logger.error(f"GDAS: probe forward failed: {probe}")
        state_mgr.update({"phase": "failed", "search": "gdas", "error": probe})
        state_mgr.save()
        raise RuntimeError(probe)
    logger.info(f"GDAS: probe forward OK on {device} (batch_size={bs})")

    # Run search. Wrap in OOM-tolerant try so mid-training OOM lands on
    # `failed_oom` state instead of crashing the orchestrator. The probe
    # cannot fully detect mid-training OOM because AdamW state + cudnn
    # algo selection allocate additional memory after the first step.
    from autospace.training.gdas_search import run_gdas_search
    search_start = time.time()
    try:
        final_argmax = run_gdas_search(
            supernet=supernet,
            mixed=mixed,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            gdas_cfg=config.gdas,
            training_cfg=slotted.training_config,
            save_dir=exp_dir,
        )
    except torch.cuda.OutOfMemoryError:
        logger.error("GDAS: OOM during search — recording failed_oom and exiting cleanly")
        return _record_oom(exp_dir, state_mgr, "oom_during_search")
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "device(s) is/are busy" in msg or "unavailable" in msg:
            logger.error(f"GDAS: OOM-like RuntimeError during search ({e}) — recording failed_oom")
            return _record_oom(exp_dir, state_mgr, "oom_during_search")
        raise
    search_wall_s = time.time() - search_start
    logger.info(f"GDAS: search complete in {search_wall_s:.0f}s")

    # Free the supernet before launching the test_final subprocess on the same GPU.
    del supernet, mixed
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Discretize: argmax for mixed slots, default_idx for everything else.
    choices = supernet_choices_from_argmax(slotted, final_argmax)
    atomic_write_json(exp_dir / "final_choices.json", choices)
    logger.info(f"GDAS: argmax choices written to {exp_dir / 'final_choices.json'}")

    # Realize and run a single test_final.
    realized_code = slotted.realize(choices)
    config_dict = dict(slotted.training_config)
    # lr_mult typically returns 1.0 here (we leave lr_mult slot at default).
    config_dict["lr"] = float(config_dict.get("lr", 1e-3)) * slotted.lr_mult_for(choices)

    from autospace.phases.phase1 import _find_python_bin, _run_training_subprocess
    python_bin = _find_python_bin()
    save_dir = exp_dir / "test_final"

    logger.info("GDAS: launching test_final on realized architecture")
    test_result = _run_training_subprocess(
        model_code=realized_code,
        config_dict=config_dict,
        task_name=config.task,
        data_root=config.data_root or "",
        gpu_id=0,  # single-GPU search box; if multi-GPU, harness handles it
        mode="final",
        time_limit_s=config.gpu.final_time_limit_s,
        seed=config.seed,
        save_dir=save_dir,
        python_bin=python_bin,
        max_vram_mb=config.gpu.max_vram_mb,
    )

    # Log the single evaluation row.
    exp_logger = ExperimentLogger(exp_dir)
    eval_record = {
        "arch_id": "gdas_argmax",
        "phase": 3,
        "mode": "test_final",
        "metric": test_result["metric"],
        "status": test_result["status"],
        "params": test_result.get("params", 0),
        "flops": test_result.get("flops", 0),
        "peak_vram_mb": test_result.get("peak_vram_mb", 0),
        "wall_time_s": test_result.get("wall_time_s", 0),
        "search_mode": "gdas",
        "choices": choices,
    }
    exp_logger.log_evaluation(eval_record)

    state_mgr.update({
        "phase": "completed",
        "search": "gdas",
        "gdas_summary": {
            "search_epochs": config.gdas.epochs,
            "search_wall_s": search_wall_s,
            "warmup_epochs": config.gdas.warmup_epochs or max(5, config.gdas.epochs // 10),
            "tau_init": config.gdas.tau_init,
            "tau_final": config.gdas.tau_final,
            "arch_lr": config.gdas.arch_lr,
            "n_mixed_slots": len(final_argmax),
            "final_argmax": final_argmax,
            "search_space_size": slotted.search_space_size,
        },
        "test_final_results": [{
            "arch_id": "gdas_argmax",
            "test_metric": test_result["metric"],
            "status": test_result["status"],
            "choices": choices,
        }],
    })
    state_mgr.save()

    return {
        "status": "success",
        "search": "gdas",
        "search_wall_s": search_wall_s,
        "test_metric": test_result["metric"],
        "choices": choices,
    }


def supernet_choices_from_argmax(slotted: SlottedArchitecture, argmax: dict[str, int]) -> dict[str, int]:
    """Merge per-slot argmax with default_idx for slots not in argmax."""
    choices: dict[str, int] = dict(argmax)
    for name, slot in slotted.slots.items():
        choices.setdefault(name, slot.default_idx)
    return choices


def _find_slotted_arch_round_layout(config: PipelineConfig) -> Path | None:
    """Find slotted_arch.json under the nested round-based experiment layout.

    Looks at: experiments/{task}_*_seed{seed}*/round_*/phase2/slotted_arch.json,
    preferring directories whose name embeds `config.llm.model` (so a Sonnet
    GDAS run consumes the Sonnet phase2 output, not a GPT one). Skips
    directories with "_gdas_" or "_phase{N}" suffix (those are CLI outputs,
    not phase-2 sources). Filters by blind_mode and the chosen variant tags.
    """
    exp_root = Path("experiments")
    if not exp_root.exists():
        return None
    seed_tag = f"_seed{config.seed}"
    model_tag = config.llm.model.replace("/", "_")
    matches: list[tuple[int, float, Path]] = []  # (model_match_score, mtime, path)
    for d in exp_root.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith(config.task + "_"):
            continue
        if seed_tag not in d.name:
            continue
        if "_gdas_" in d.name or "_phase" in d.name:
            continue
        if config.blind_mode and "_blind" not in d.name:
            continue
        if (not config.blind_mode) and "_blind" in d.name:
            continue
        # Skip enriched/fixed_seed/wrn variants unless they were explicitly requested.
        if "_enriched" in d.name:
            continue
        if "seed_" in d.name or "wrn16_4seed" in d.name or "resnet18seed" in d.name:
            continue
        for rd in sorted(d.glob("round_*"), reverse=True):
            p = rd / "phase2" / "slotted_arch.json"
            if p.exists():
                model_match = 1 if model_tag in d.name else 0
                matches.append((model_match, p.stat().st_mtime, p))
                break  # one round per source dir is enough
    if not matches:
        return None
    matches.sort(reverse=True)  # model_match desc, then mtime desc
    return matches[0][2]


def _record_oom(exp_dir: Path, state_mgr: StateManager, reason: str) -> dict[str, Any]:
    """Write the OOM artifact set and return a clean result dict."""
    exp_logger = ExperimentLogger(exp_dir)
    exp_logger.log_evaluation({
        "arch_id": "gdas_argmax",
        "phase": 3,
        "mode": "search",
        "metric": float("inf"),
        "status": reason,
        "search_mode": "gdas",
    })
    state_mgr.update({
        "phase": "failed_oom",
        "search": "gdas",
        "reason": reason,
        "test_final_results": [],
    })
    state_mgr.save()
    return {"status": "failed_oom", "reason": reason}
