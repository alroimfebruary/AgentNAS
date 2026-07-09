#!/usr/bin/env python
"""Production launcher: run the full pipeline (Phase 1 → 1.5 modularize → 2 → 3) for a single task.

Usage:
    python scripts/run_production.py --task fsd50k
    python scripts/run_production.py --task fsd50k --llm-model claude-sonnet-4-6
    python scripts/run_production.py --task fsd50k --phase 1     # Run only Phase 1

Environment:
    - Uses configs/production.yaml for hardware settings
    - API keys at ./anthropic_api_key.txt and ./openai_api_key.txt
    - Data at ./NAS-Bench-360-Benchmarks/ and ./NAS-Unseen/
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("production")


def main():
    parser = argparse.ArgumentParser(description="AgentNAS Production Launcher")
    parser.add_argument("--task", required=True, help="Benchmark task name (e.g., fsd50k)")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run specific phase (1/2/3). Default 0 = full pipeline (1→2→3)")
    parser.add_argument("--llm-model", default="claude-sonnet-4-6", help="LLM model name")
    parser.add_argument("--data-root", default=None, help="Data root (absolute path)")
    parser.add_argument("--benchmark-file", default=None,
                        help="Python file registering custom tasks via @register_benchmark")
    parser.add_argument("--config", default=None,
                        help="Pipeline config YAML (default: configs/production.yaml)")
    parser.add_argument("--experiment-dir", default=None, help="Override experiment directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true", help="Start fresh (don't resume)")
    parser.add_argument("--blind-mode", action="store_true",
                        help="Blind mode: suppress domain knowledge (Unseen-NAS ablation)")
    parser.add_argument("--no-modularize", action="store_true",
                        help="Skip Phase 1.5 (stage modularization). Default modularizes "
                             "the Phase 1 winner before Phase 2 — paper results use it.")
    parser.add_argument("--fixed-seed-arch",
                        choices=["auto", "resnet18", "wideresnet_16_4"],
                        default=None,
                        help="Fixed-seed-arch ablation: skip Phase 1 LLM loop and seed "
                             "with ResNet-18 (Unseen-NAS) or WideResNet-16-4 (NAS-Bench-360). "
                             "Use 'auto' to derive family from the task.")
    parser.add_argument("--fixed-seed-epochs", type=int, default=None,
                        help="Cap epochs for the fixed-seed final training (smoke-test knob). "
                             "Only effective with --fixed-seed-arch.")
    args = parser.parse_args()

    task = args.task
    resume = not args.no_resume

    # Resolve paths
    project_root = Path(__file__).parent.parent.resolve()
    if args.config:
        config_path = Path(args.config).resolve()
    else:
        config_path = project_root / "configs" / "production.yaml"
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    data_root = args.data_root
    if not data_root:
        nb360 = project_root / "NAS-Bench-360-Benchmarks"
        unseen = project_root / "NAS-Unseen"
        if nb360.exists():
            data_root = str(nb360)
        elif unseen.exists():
            data_root = str(unseen)
        else:
            logger.error("No data directory found. Use --data-root.")
            sys.exit(1)

    # Base experiment dir: deterministic {task}_{model}_{seed} for safe resume
    from agentnas.config import make_experiment_subdir
    if args.fixed_seed_arch:
        from agentnas.grammar.fixed_seeds import resolve_family
        resolved_family = resolve_family(task, args.fixed_seed_arch)
    else:
        resolved_family = ""
    if args.experiment_dir:
        base_dir = make_experiment_subdir(
            parent=args.experiment_dir, task=task, model=args.llm_model,
            seed=args.seed, phase=None, blind=args.blind_mode,
            fixed_seed=resolved_family,
        )
    else:
        base_dir = make_experiment_subdir(
            parent=str(project_root / "experiments"), task=task,
            model=args.llm_model, seed=args.seed, phase=None,
            blind=args.blind_mode, fixed_seed=resolved_family,
        )

    from agentnas.config import (
        load_pipeline_config,
    )

    # Load production config as base, override with CLI args
    overrides = {"task": task, "seed": args.seed, "data_root": data_root}
    if args.benchmark_file:
        overrides["benchmark_file"] = str(Path(args.benchmark_file).resolve())
    if args.llm_model != "claude-sonnet-4-6":
        overrides["llm.model"] = args.llm_model
    if args.blind_mode:
        overrides["blind_mode"] = True
    if args.fixed_seed_arch:
        overrides["fixed_seed_arch"] = args.fixed_seed_arch
    if args.fixed_seed_epochs is not None:
        overrides["fixed_seed_epochs"] = args.fixed_seed_epochs

    config = load_pipeline_config(config_path, overrides)

    logger.info(f"{'='*60}")
    logger.info(f"AgentNAS Production Run")
    logger.info(f"{'='*60}")
    logger.info(f"Task:       {task}")
    logger.info(f"Model:      {config.llm.model}")
    logger.info(f"GPUs:       {config.gpu.n_gpu}")
    logger.info(f"VRAM limit: {config.gpu.max_vram_mb} MB")
    logger.info(f"Budget:     {config.budget.max_evaluations} evals")
    logger.info(f"Proxy time: {config.gpu.proxy_time_limit_s // 60} min")
    logger.info(f"Final time: {config.gpu.final_time_limit_s // 3600} hr")
    logger.info(f"Data root:  {data_root}")
    logger.info(f"Experiment: {base_dir}")
    logger.info(f"Resume:     {resume}")
    logger.info(f"{'='*60}\n")

    # Dispatch based on --phase
    if args.phase > 0:
        # Explicit single phase
        _run_single_phase(args.phase, base_dir, config, task, resume)
    else:
        # Default: full pipeline Phase 1 → 1.5 modularize → 2 → 3
        # `--no-modularize` drops the 1.5 step entirely.
        if _run_single_phase(1, base_dir, config, task, resume):
            if args.no_modularize:
                logger.info("Phase 1.5 skipped (--no-modularize)")
                modularize_ok = True
            else:
                modularize_ok = _run_modularize_step(base_dir, config, task, resume)
            if modularize_ok:
                for phase in [2, 3]:
                    if not _run_single_phase(phase, base_dir, config, task, resume):
                        break

    logger.info(f"\n{'='*60}")
    logger.info(f"All phases complete for {task}")
    logger.info(f"Results in: {base_dir}")
    logger.info(f"{'='*60}")


def _run_modularize_step(base_dir: str, config, task: str, resume: bool) -> bool:
    """Phase 1.5: modularize the Phase 1 winning arch in place.

    Runs the ModularizerAgent on Phase 1's winning architecture so downstream
    Phase 2 sees a stage-modularized seed. Idempotent — a `modularized.done`
    marker in the audit dir short-circuits repeat runs.

    Returns True on success (or already-done), False on failure.
    """
    phase1_dir = Path(base_dir) / "phase1"
    audit_dir = Path(base_dir) / "phase1_5_modularize"
    audit_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(f"{audit_dir}/run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 1.5 — modularize seed — {task}")
    logger.info(f"{'='*60}")
    t0 = time.time()

    try:
        from agentnas.phases.phase1_5_modularize import run_modularize_in_place
        result = run_modularize_in_place(config, phase1_dir, phase2_nudge=True)
        elapsed = time.time() - t0
        logger.info(
            f"Phase 1.5 completed in {elapsed/60:.1f} min: status={result.get('status')}"
        )
        if result.get("status") in ("success", "already_done"):
            with open(f"{audit_dir}/result_summary.json", "w") as f:
                json.dump(result, f, indent=2, default=str)
            return True
        logger.error(f"Phase 1.5 failed: {result}")
        return False
    except Exception as e:
        logger.error(f"Phase 1.5 crashed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        logging.getLogger().removeHandler(fh)


def _run_single_phase(phase: int, base_dir: str, config, task: str, resume: bool) -> bool:
    """Run a single phase. Returns True if successful, False if failed."""
    phase_dir = f"{base_dir}/phase{phase}"
    config.experiment_dir = phase_dir
    config.phase = phase
    Path(phase_dir).mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(f"{phase_dir}/run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info(f"\n{'='*60}")
    logger.info(f"Phase {phase} — {task}")
    logger.info(f"{'='*60}")
    t0 = time.time()

    try:
        if phase == 1:
            from agentnas.phases.phase1 import run_phase1
            result = run_phase1(config, resume=resume)
        elif phase == 2:
            p1_state = Path(f"{base_dir}/phase1/state.json")
            if not p1_state.exists():
                logger.error("Phase 1 results not found. Run Phase 1 first.")
                return False
            from agentnas.phases.phase2 import run_phase2
            result = run_phase2(config, resume=resume)
        elif phase == 3:
            p2_arch = Path(f"{base_dir}/phase2/slotted_arch.json")
            p3_arch = Path(f"{phase_dir}/slotted_arch.json")
            if p2_arch.exists():
                # Mirror phase2 → phase3 when phase2's slotted_arch.json is newer.
                if not p3_arch.exists() or p2_arch.stat().st_mtime > p3_arch.stat().st_mtime:
                    shutil.copy(p2_arch, p3_arch)
            elif not p3_arch.exists():
                logger.error("Phase 2 slotted_arch.json not found. Run Phase 2 first.")
                return False
            from agentnas.phases.phase3 import run_phase3
            result = run_phase3(config, resume=resume)
        else:
            logger.error(f"Invalid phase: {phase}")
            return False

        elapsed = time.time() - t0
        logger.info(f"Phase {phase} completed in {elapsed/3600:.1f} hours")
        logger.info(f"Result: {json.dumps(result, indent=2, default=str)[:500]}")

        with open(f"{phase_dir}/result_summary.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        if result.get("status") == "failed":
            logger.error(f"Phase {phase} failed: {result.get('reason', 'unknown')}")
            return False

        return True

    except Exception as e:
        logger.error(f"Phase {phase} crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        logging.getLogger().removeHandler(fh)


if __name__ == "__main__":
    main()
