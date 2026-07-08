"""Phase 1: LLM Iterative Refinement orchestrator.

Flow:
1. Forced data exploration (first iteration)
2. Loop: Planner action → dispatch → collect results → feed back
3. Final submission: top-N architectures full-trained
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from agentnas.agents.executor import CodeExecutor
from agentnas.agents.explorer import DataExplorer
from agentnas.agents.planner import (
    ArchProposal,
    EvalRecord,
    Planner,
    PlannerState,
)
from agentnas.benchmarks.registry import get_benchmark, loaded_benchmark_files
from agentnas.config import PipelineConfig
from agentnas.llm.client import LLMClient
from agentnas.llm.logger import LLMLogger
from agentnas.utils.gpu import GPUPool, get_gpu_env
from agentnas.utils.logging import ExperimentLogger
from agentnas.utils.paths import project_root
from agentnas.utils.state import StateManager

logger = logging.getLogger(__name__)


def _parallel_training_limit(n_gpu: int) -> int:
    """Return the max parallel training workers for a wave.

    Normally equals ``n_gpu``. Can be capped via the
    ``AGENTNAS_MAX_PARALLEL_TRAINING`` environment variable to reduce peak
    host-memory pressure for datasets with large in-RAM caches (e.g. PSICOV's
    pre-assembled float16 protein cache).

    The cap is applied to phase-1 proxy search, Wave 1 val-final, and Wave 2
    test-final. A value of 0 or a non-integer is treated as unset.
    """
    raw = (os.environ.get("AGENTNAS_MAX_PARALLEL_TRAINING", "")
           or os.environ.get("AUTOSPACE_MAX_PARALLEL_TRAINING", ""))  # legacy name
    if not raw:
        return n_gpu
    try:
        cap = int(raw)
    except ValueError:
        return n_gpu
    if cap <= 0:
        return n_gpu
    limit = min(n_gpu, cap)
    if limit < n_gpu:
        logger.info(
            f"AGENTNAS_MAX_PARALLEL_TRAINING={cap}: limiting concurrent training "
            f"from {n_gpu} GPUs to {limit} to reduce host memory pressure."
        )
    return limit


def _find_python_bin() -> str:
    """Find the Python binary to use for subprocesses."""
    import sys
    return sys.executable


def _run_training_subprocess(
    model_code: str,
    config_dict: dict[str, Any],
    task_name: str,
    data_root: str,
    gpu_id: int,
    mode: str,
    time_limit_s: float,
    seed: int,
    save_dir: Path,
    python_bin: str,
    max_vram_mb: int = 0,
    proxy_epoch_fraction: float = 0.3, 
) -> dict[str, Any]:
    """Run training in a subprocess on a specific GPU.

    Returns the results dict from the worker.
    """
    work_dir = save_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write model code
    (work_dir / "model.py").write_text(model_code)

    # Write training config
    with open(work_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Write args
    args = {
        "task": task_name,
        "mode": mode,
        "time_limit_s": time_limit_s,
        "data_root": data_root,
        "seed": seed,
        "max_vram_mb": max_vram_mb,
        "proxy_epoch_fraction": proxy_epoch_fraction,
        # Custom-benchmark files to replay in the worker process
        "benchmark_files": loaded_benchmark_files(),
    }
    with open(work_dir / "args.json", "w") as f:
        json.dump(args, f, indent=2)

    # Run worker
    worker_module = "agentnas.training.worker"
    env = get_gpu_env(gpu_id)
    # Ensure the agentnas package is importable even without an installed dist
    env["PYTHONPATH"] = str(project_root() / "src") + ":" + env.get("PYTHONPATH", "")

    timeout = int(time_limit_s + max(3600, time_limit_s * 0.10))  # 1-hr wall-clock buffer minimum, 10% for long runs
    stdout_path = work_dir / "worker_stdout.log"
    stderr_path = work_dir / "worker_stderr.log"

    f_out = open(stdout_path, "w")
    f_err = open(stderr_path, "w")
    proc = None
    timed_out = False
    try:
        proc = subprocess.Popen(
            [python_bin, "-m", worker_module, str(work_dir)],
            stdout=f_out, stderr=f_err, env=env,
            start_new_session=True,  # makes proc the leader of a new process group
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
    finally:
        f_out.close()
        f_err.close()
        
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    if timed_out:
        logger.error(f"Worker timed out (>{timeout}s) — process group killed")
        return {
            "metric": float("inf"), "wall_time_s": time_limit_s,
            "peak_vram_mb": 0, "params": 0, "flops": 0,
            "epochs_completed": 0, "status": "timeout",
            "error_message": "Worker process timed out (group killed)",
            "train_losses": [], "val_metrics": [], "best_epoch": -1,
        }
    if proc.returncode != 0:
        stderr_tail = stderr_path.read_text()[-500:] if stderr_path.exists() else ""
        logger.warning(f"Worker stderr: {stderr_tail}")

    results_path = work_dir / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    else:
        return {
            "metric": float("inf"), "wall_time_s": 0,
            "peak_vram_mb": 0, "params": 0, "flops": 0,
            "epochs_completed": 0, "status": "error",
            "error_message": "No results.json produced by worker",
            "train_losses": [], "val_metrics": [], "best_epoch": -1,
        }


def run_phase1(config: PipelineConfig, resume: bool = True) -> dict[str, Any]:
    """Run Phase 1: LLM Iterative Refinement.

    Returns a summary dict with best architecture info.

    When ``config.fixed_seed_arch`` is set, the LLM loop is short-circuited and
    a hardcoded ResNet-18 / WideResNet-16-4 seed is used instead — see
    ``_run_phase1_fixed_seed``. Standard Phase 1 behavior is unchanged.
    """
    cleanup_state: dict[str, Any] = {"explorer_temp_dir": None}
    try:
        return _run_phase1_main(config, resume, cleanup_state)
    finally:
        etd = cleanup_state["explorer_temp_dir"]
        if etd is not None and etd.exists():
            shutil.rmtree(etd, ignore_errors=True)


def _run_phase1_main(
    config: PipelineConfig,
    resume: bool,
    cleanup_state: dict[str, Any],
) -> dict[str, Any]:
    """Phase 1 main body. See ``run_phase1`` for the public docstring.

    ``cleanup_state`` is a dict the public wrapper uses to track resources
    that must be released on any exit path. Currently tracks the explorer
    temp directory so it is removed even on uncaught exceptions.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import random as _random
    import numpy as _np
    _random.seed(config.seed)
    _np.random.seed(config.seed)

    exp_dir = Path(config.experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    python_bin = _find_python_bin()

    if config.fixed_seed_arch:
        return _run_phase1_fixed_seed(config, exp_dir, python_bin, resume)

    llm_logger = LLMLogger(exp_dir / "llm_logs")
    planner_llm = LLMClient.from_config(config.llm, logger=llm_logger.set_agent("planner"))
    executor_llm = LLMClient.from_config(config.llm, logger=llm_logger.set_agent("executor"))
    explorer_llm = LLMClient.from_config(config.llm, logger=llm_logger.set_agent("explorer"))
    gpu_pool = GPUPool(n_gpu=config.gpu.n_gpu)
    exp_logger = ExperimentLogger(exp_dir)
    state_mgr = StateManager(exp_dir / "state.json")

    # Resume or initialize state
    planner_state = None
    if resume and state_mgr.exists:
        state_mgr.load()
        if state_mgr.get("phase") == "completed":
            logger.info("Phase 1 already completed, returning saved results")
            final_summary = state_mgr.get("final_summary")
            if final_summary is None:
                raise RuntimeError(
                    "state.json reports phase=completed but final_summary is missing — "
                    "likely corrupted state; delete state.json and rerun."
                )
            return final_summary
        logger.info("Resuming from checkpoint")
        if "planner_state" in state_mgr:
            planner_state = PlannerState.from_dict(state_mgr["planner_state"])

    # Load benchmark
    benchmark = get_benchmark(config.task, data_root=config.data_root)

    # Load task README
    task_data_dir = config.task_config.data_dir if config.task_config else config.task
    if config.blind_mode:
        # Blind mode: minimal spec only, no domain knowledge
        tc = config.task_config
        task_readme = (
            f"Task: TASK_X\n"
            f"Input shape: {benchmark.input_shape()}\n"
            f"Output: {benchmark.output_spec()}\n"
            f"Loss: {tc.loss if tc else 'unknown'}\n"
            f"Metric: {tc.metric if tc else 'unknown'} "
            f"({'lower is better' if (tc and tc.lower_is_better) else 'higher is better'})"
        )
    else:
        task_readme = f"Task: {config.task}\nInput shape: {benchmark.input_shape()}\nOutput: {benchmark.output_spec()}"
        readme_path = Path(config.data_root) / task_data_dir / "README.md"

        if not readme_path.exists():
            unseen_root = project_root() / "NAS-Unseen"
            alt_path = unseen_root / task_data_dir / "README.md"
            if alt_path.exists():
                readme_path = alt_path
        if not readme_path.exists():
            # Reproducibility fallback: the NAS-Bench-360 data distribution ships
            # without per-task README.md files. Fall back to the curated README
            # under configs/explorer_readmes/ so the Planner sees the same task
            # description the paper used, even when data was set up as a raw
            # download without our README overlay.
            curated = project_root() / "configs" / "explorer_readmes" / f"{config.task}.md"
            if curated.exists():
                readme_path = curated
                logger.info(f"Using curated task README: {curated}")
        if readme_path.exists():
            task_readme = readme_path.read_text()

    # Build SOTA info string from task config
    sota_info = "No baseline available."
    train_samples = "unknown"
    if config.task_config:
        tc = config.task_config
        if tc.sota_metric is not None:
            if config.blind_mode:
                sota_info = f"**SOTA metric: {tc.sota_metric}** (lower=better)."
            else:
                sota_info = (
                    f"**Human SOTA metric: {tc.sota_metric}** (lower=better). "
                    f"{tc.sota_description}"
                )
    # Estimate training samples from benchmark
    try:
        tl, _ = benchmark.get_search_loaders(batch_size=1, num_workers=0)
        train_samples = f"{len(tl.dataset):,}"
    except Exception as e:
        logger.warning("Could not estimate train_samples: %s", e)

    # Initialize Planner
    planner = Planner(
        llm_client=planner_llm,
        task_readme=task_readme,
        n_gpu=config.gpu.n_gpu,
        vram_mb=config.gpu.max_vram_mb,
        time_limit_min=config.gpu.proxy_time_limit_s // 60,
        budget_total=config.budget.max_evaluations,
        patience=config.budget.patience,
        sota_info=sota_info,
        sota_available=bool(
            config.task_config and config.task_config.sota_metric is not None
        ),
        train_samples=train_samples,
        state=planner_state,
        proxy_epoch_pct=int(config.gpu.proxy_epoch_fraction * 100),
    )

    # Code Executor setup
    task_label = "TASK_X" if config.blind_mode else config.task
    executor = CodeExecutor(
        llm_client=executor_llm,
        task_name=task_label,
        input_shape=benchmark.input_shape(),
        output_spec=benchmark.output_spec(),
        vram_mb=config.gpu.max_vram_mb,
        time_limit_min=config.gpu.proxy_time_limit_s // 60,
        data_root=config.data_root or "",
        max_retries=config.budget.max_executor_retries,
        python_bin=python_bin,
        proxy_epoch_pct=int(config.gpu.proxy_epoch_fraction * 100),
    )

    # Data Explorer setup — resolve correct data root for unseen tasks (D-01 fix)
    from agentnas.benchmarks.registry import _UNSEEN_TASKS, _DATA_ROOTS
    explorer_data_root = config.data_root or ""
    if config.task in _UNSEEN_TASKS:
        from pathlib import Path as _P
        unseen_root = _P.cwd() / _DATA_ROOTS["unseen"]
        if unseen_root.exists():
            explorer_data_root = str(unseen_root)

    # Build a filtered temp directory for the explorer: test files always excluded,
    # blind mode additionally blinds domain-leaking files and renames symlinks.
    # The real data dir (config.data_root) is untouched; only the explorer sees
    # this filtered view. The benchmark loader (final training/evaluation)
    # continues to use the real data dir.
    from agentnas.utils.explorer_data import build_explorer_data_dir
    explorer_temp_dir, explorer_subdir = build_explorer_data_dir(
        real_data_root=explorer_data_root,
        benchmark=benchmark,
        blind_mode=config.blind_mode,
        project_root=project_root(),
    )
    cleanup_state["explorer_temp_dir"] = explorer_temp_dir
    mode_label = "blind" if config.blind_mode else "standard"
    logger.info(f"Explorer data dir ({mode_label} mode): {explorer_temp_dir}")

    explorer = DataExplorer(
        llm_client=explorer_llm,
        data_root=str(explorer_temp_dir),
        data_subdir=explorer_subdir,
        max_steps=config.budget.max_explorer_steps,
        python_bin=python_bin,
    )

    # ---- Phase 1 Main Loop ----

    # Step 1: Forced data exploration (if not already done)
    if not planner.state.explorer_report:
        logger.info("=== Data Exploration (forced first iteration) ===")
        purpose = (
            "Understand the dataset: check statistics, class distribution, "
            "input shape, any domain-specific characteristics, and "
            "considerations for architecture design."
        )
        report, tokens = explorer.explore(purpose)
        planner.set_explorer_report(report)
        # Mirror to a standalone file so the report survives state.json corruption
        # and is trivially extractable (no JSON parse) for reuse across runs.
        (exp_dir / "explorer_report.md").write_text(report)
        logger.info(f"Explorer report ({tokens} tokens): {report[:200]}...")

        # Checkpoint
        state_mgr.update({"planner_state": planner.state.to_dict()})
        state_mgr.save()


    # Step 2: Iterative experiment loop
    max_consecutive_explores = 2  # Prevent explore loops
    consecutive_explores = 0
    empty_iteration_count = 0  # Guard against infinite loop with no proposals
    max_empty_iterations = 10

    while not planner.should_force_stop:
        logger.info(f"=== Iteration {planner.state.iteration + 1} "
                     f"(budget: {planner.budget_remaining}/{planner.budget_total}) ===")

        # Get Planner action
        force_experiment = consecutive_explores >= max_consecutive_explores
        if force_experiment:
            logger.info("Forcing EXPERIMENT (max consecutive explores reached)")

        action_type, action_data = planner.get_action()
        logger.info(f"Planner action: {action_type}")

        if action_type == "stop":
            logger.info("Planner chose to STOP")
            break

        if action_type == "explore" and not force_experiment:
            consecutive_explores += 1
            report, tokens = explorer.explore(action_data)
            planner.set_explorer_report(report)
            (exp_dir / "explorer_report.md").write_text(report)
            state_mgr.update({"planner_state": planner.state.to_dict()})
            state_mgr.save()
            continue

        # Reset explore counter when we experiment
        consecutive_explores = 0

        # If planner said explore but we're forcing experiment, explicitly ask for proposals
        if action_type == "explore" or (action_type == "experiment" and not action_data):
            logger.info("Explicitly requesting architecture proposals from Planner")
            resp = planner.llm.chat(
                planner._build_context_messages() + [{
                    "role": "user",
                    "content": (
                        "You MUST now propose architectures using <EXPERIMENT>. "
                        "Propose up to {n} architectures with <ARCH>, <DESCRIPTION>, "
                        "<REASONING>, and <TRAINING_RECIPE> tags."
                    ).format(n=planner.n_gpu),
                }],
                system=planner._build_system_prompt(),
            )
            planner.total_tokens += resp.total_tokens
            _, action_data = planner._parse_action(resp.content)
            if not isinstance(action_data, list):
                action_data = []

        proposals: list[ArchProposal] = action_data if isinstance(action_data, list) else []

        if not proposals:
            empty_iteration_count += 1
            logger.warning(f"No proposals (empty iteration {empty_iteration_count}/{max_empty_iterations})")
            # Don't increment planner.state.iteration here — only feed_results() should (#12 fix)
            if empty_iteration_count >= max_empty_iterations:
                logger.error("Too many empty iterations without progress, stopping")
                break
            continue

        # Reset empty counter on successful proposals
        empty_iteration_count = 0

        # Limit proposals to available budget
        proposals = proposals[: planner.budget_remaining]

        # --- Step 1: Generate code for all proposals (sequential, needs GPU for validation) ---
        ready_jobs = []  # (arch_id, description, model_code, train_config, tokens)
        results_text_parts = []

        # Build a code lookup keyed by arch_id. Only included in the Executor prompt
        # when the proposal's description actually references that arch (e.g.
        # "arch_006 with wider channels") — avoids bloating every call with codes
        # the Executor doesn't need.
        arch_code_index: dict[str, dict] = {}
        for e in planner.state.evaluations:
            aid = e.get("arch_id")
            if aid and e.get("code"):
                arch_code_index[aid] = {
                    "arch_id": aid, "code": e["code"],
                    "metric": e.get("metric"), "status": e.get("status", ""),
                }

        for proposal in proposals:
            # Use proposal's arch_id (set by parser via next_arch_id) — don't double-increment (H2 fix)
            arch_id = proposal.arch_id
            logger.info(f"Generating code for {arch_id}: {proposal.description[:80]}...")

            # Pull only the archs this proposal explicitly references.
            combined_text = f"{proposal.description}\n{proposal.training_recipe or ''}\n{proposal.reasoning or ''}"
            referenced = []
            seen = set()
            for match in re.findall(r"arch_\d+", combined_text):
                if match in arch_code_index and match not in seen and match != arch_id:
                    referenced.append(arch_code_index[match])
                    seen.add(match)
            past_evals_for_executor = referenced or None

            model_code, train_config, gen_status, tokens = executor.generate(
                arch_description=proposal.description,
                training_recipe=proposal.training_recipe or proposal.reasoning,
                past_evals=past_evals_for_executor,
            )

            if gen_status == "rejected" or model_code is None:
                record = EvalRecord(
                    arch_id=arch_id,
                    description=proposal.description,
                    metric=float("inf"),
                    params=0, peak_vram_mb=0, flops=0, wall_time_s=0,
                    status="rejected", code="",
                    iteration=planner.state.iteration,
                    error_message="Code generation failed after max retries",
                )
                planner.update_after_eval(record)
                exp_logger.log_evaluation({
                    "arch_id": arch_id, "phase": 1,
                    "iteration": planner.state.iteration,
                    "metric": float("inf"), "status": "rejected",
                    "params": 0, "flops": 0, "peak_vram_mb": 0,
                    "wall_time_s": 0, "tokens_used": tokens,
                })
                results_text_parts.append(f"**{arch_id}**: REJECTED (code generation failed)\n")
            else:
                ready_jobs.append((arch_id, proposal.description, model_code, train_config, tokens))

        # --- Step 2: Train all ready architectures in parallel across GPUs ---
        if ready_jobs:
            def _train_one(job_tuple):
                a_id, desc, code, cfg, tok = job_tuple
                gpu_id = gpu_pool.allocate_one()
                try:
                    logger.info(f"Training {a_id} on GPU {gpu_id}")
                    save_dir = exp_dir / "models" / a_id
                    result = _run_training_subprocess(
                        model_code=code,
                        config_dict=cfg.to_dict() if cfg else {},
                        task_name=config.task,
                        data_root=config.data_root or "",
                        gpu_id=gpu_id,
                        mode="search",
                        time_limit_s=config.gpu.proxy_time_limit_s,
                        seed=config.seed,
                        save_dir=save_dir,
                        python_bin=python_bin,
                        max_vram_mb=config.gpu.max_vram_mb,
                        proxy_epoch_fraction=config.gpu.proxy_epoch_fraction,
                    )
                    return (a_id, desc, code, result, tok)
                finally:
                    gpu_pool.release(gpu_id)

            with ThreadPoolExecutor(max_workers=_parallel_training_limit(config.gpu.n_gpu)) as pool:
                futures = {}
                for job in ready_jobs:
                    futures[pool.submit(_train_one, job)] = job
                for future in as_completed(futures):
                    job_tuple = futures[future]
                    a_id, desc, code, cfg, tokens = job_tuple
                    cfg_dict = cfg.to_dict() if cfg else {}
                    try:
                        a_id, desc, code, train_results, tokens = future.result()
                    except Exception as e:
                        logger.error(f"Training {a_id} raised: {e}")
                        train_results = {
                            "metric": float("inf"), "params": 0, "flops": 0,
                            "peak_vram_mb": 0, "wall_time_s": 0,
                            "status": "error", "error_message": str(e),
                            "epochs_completed": 0,
                        }

                    record = EvalRecord(
                        arch_id=a_id, description=desc,
                        metric=train_results["metric"],
                        params=train_results["params"],
                        peak_vram_mb=train_results["peak_vram_mb"],
                        flops=train_results["flops"],
                        wall_time_s=train_results["wall_time_s"],
                        status=train_results["status"],
                        code=code,
                        iteration=planner.state.iteration,
                        error_message=train_results.get("error_message", ""),
                        training_config=cfg_dict,
                    )
                    planner.update_after_eval(record)

                    exp_logger.log_evaluation({
                        "arch_id": a_id, "phase": 1,
                        "iteration": planner.state.iteration,
                        "code": code,
                        "metric": train_results["metric"],
                        "status": train_results["status"],
                        "params": train_results["params"],
                        "flops": train_results["flops"],
                        "peak_vram_mb": train_results["peak_vram_mb"],
                        "wall_time_s": train_results["wall_time_s"],
                        "epochs_completed": train_results["epochs_completed"],
                        "tokens_used": tokens,
                        "training_config": cfg_dict,
                    })

                    metric_str = f"{train_results['metric']:.4f}" if train_results["status"] == "success" else "N/A"
                    results_text_parts.append(
                        f"**{a_id}** ({train_results['status']}): "
                        f"metric={metric_str}, params={train_results['params']:,}, "
                        f"VRAM={train_results['peak_vram_mb']:.0f}MB, "
                        f"time={train_results['wall_time_s']:.0f}s, "
                        f"epochs={train_results['epochs_completed']}\n"
                    )

                    # Checkpoint after each completed evaluation
                    state_mgr.update({"planner_state": planner.state.to_dict()})
                    state_mgr.save()

        # Feed results to Planner
        results_text = "\n".join(results_text_parts)
        if results_text:
            summary = planner.feed_results(results_text)
            logger.info(f"Planner summary: {summary[:200]}...")

        state_mgr.update({"planner_state": planner.state.to_dict()})
        state_mgr.save()

    # ---- Final Submission (two waves) ----
    # Wave 1: full training on train/val split → val metric for all pipeline decisions
    # Wave 2: full training on train+val combined → test metric for analysis only
    logger.info("=== Final Submission ===")

    # Get top-N architectures
    successful = [
        e for e in planner.state.evaluations if e["status"] == "success"
    ]
    successful.sort(key=lambda e: e["metric"])
    top_archs = successful[: config.budget.top_k_final]  # M-01: use top_k_final, not n_gpu

    if not top_archs:
        logger.error("No successful architectures to submit")
        return {"status": "failed", "reason": "No successful architectures"}

    # --- Wave 1: Val-final (full training, train/val split, best checkpoint) ---
    logger.info(f"=== Wave 1: Val-final training {len(top_archs)} architectures ===")

    def _val_final_train_one(arch):
        """Wave 1: Full training on train/val split, best checkpoint by val metric."""
        if not arch.get("code"):
            return None
        gpu_id = gpu_pool.allocate_one()
        try:
            save_dir = exp_dir / "val_final" / arch["arch_id"]
            cfg_dict = arch.get("training_config", {})
            train_results = _run_training_subprocess(
                model_code=arch["code"],
                config_dict=cfg_dict,
                task_name=config.task,
                data_root=config.data_root or "",
                gpu_id=gpu_id,
                mode="search",
                time_limit_s=config.gpu.final_time_limit_s,
                seed=config.seed,
                save_dir=save_dir,
                python_bin=python_bin,
                max_vram_mb=config.gpu.max_vram_mb,
                proxy_epoch_fraction=1.0,
            )
            logger.info(
                f"Val-final {arch['arch_id']}: "
                f"proxy={arch['metric']:.4f}, val_final={train_results['metric']:.4f}"
            )
            return {
                "arch_id": arch["arch_id"],
                "proxy_metric": arch["metric"],
                "val_final_metric": train_results["metric"],
                "status": train_results["status"],
                "code": arch["code"],
                "training_config": cfg_dict,
                "params": train_results.get("params", 0),
            }
        finally:
            gpu_pool.release(gpu_id)

    # Resume support for Wave 1: skip archs already trained successfully (saved
    # in state.json). Patched 2026-05-04: previously a pod restart mid-Wave-1
    # would retrain all 8 archs from scratch.
    val_final_results = []
    if resume:
        prev = state_mgr.get("val_final_results", [])
        if prev:
            val_final_results = list(prev)
            logger.info(f"Phase 1 Wave 1 resume: loaded {len(val_final_results)} prior val_final results")
    done_w1 = {r["arch_id"] for r in val_final_results if r.get("status") == "success"}
    pending_w1 = [a for a in top_archs if a["arch_id"] not in done_w1]
    if len(pending_w1) < len(top_archs):
        logger.info(f"Phase 1 Wave 1: skipping {len(top_archs) - len(pending_w1)} already-trained archs ({sorted(done_w1)})")
    with ThreadPoolExecutor(max_workers=_parallel_training_limit(config.gpu.n_gpu)) as pool:
        futures = {pool.submit(_val_final_train_one, arch): arch["arch_id"] for arch in pending_w1}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                arch_id = futures[future]
                logger.error(f"Val-final training {arch_id} failed: {e}")
                continue
            if result:
                val_final_results.append(result)
                exp_logger.log_evaluation({
                    "arch_id": result["arch_id"], "phase": 1, "mode": "val_final",
                    "metric": result["val_final_metric"], "status": result["status"],
                    "proxy_metric": result.get("proxy_metric"),
                })
                # Persist after each completed arch — supports mid-Wave-1 resume.
                state_mgr.update({
                    "planner_state": planner.state.to_dict(),
                    "val_final_results": val_final_results,
                })
                state_mgr.save()

    # Intermediate checkpoint after Wave 1 (resume safety)
    state_mgr.update({
        "planner_state": planner.state.to_dict(),
        "val_final_results": val_final_results,
    })
    state_mgr.save()

    best_val_final = min(
        (r for r in val_final_results if r["status"] == "success"),
        key=lambda r: r["val_final_metric"],
        default=None,
    )

    # --- Wave 2: Test-final (train+val combined, last checkpoint, analysis only) ---
    logger.info(f"=== Wave 2: Test-final training {len(top_archs)} architectures (analysis only) ===")

    def _test_final_train_one(arch):
        """Wave 2: Full training on train+val, last checkpoint, test eval. Analysis only."""
        if not arch.get("code"):
            return None
        gpu_id = gpu_pool.allocate_one()
        try:
            save_dir = exp_dir / "test_final" / arch["arch_id"]
            cfg_dict = arch.get("training_config", {})
            train_results = _run_training_subprocess(
                model_code=arch["code"],
                config_dict=cfg_dict,
                task_name=config.task,
                data_root=config.data_root or "",
                gpu_id=gpu_id,
                mode="final",
                time_limit_s=config.gpu.final_time_limit_s,
                seed=config.seed,
                save_dir=save_dir,
                python_bin=python_bin,
                max_vram_mb=config.gpu.max_vram_mb,
            )
            logger.info(
                f"Test-final {arch['arch_id']}: "
                f"proxy={arch['metric']:.4f}, test={train_results['metric']:.4f}"
            )
            return {
                "arch_id": arch["arch_id"],
                "test_metric": train_results["metric"],
                "status": train_results["status"],
            }
        finally:
            gpu_pool.release(gpu_id)

    # Resume support for Wave 2 — same pattern as Wave 1.
    test_final_results = []
    if resume:
        prev = state_mgr.get("test_final_results", [])
        if prev:
            test_final_results = list(prev)
            logger.info(f"Phase 1 Wave 2 resume: loaded {len(test_final_results)} prior test_final results")
    done_w2 = {r["arch_id"] for r in test_final_results if r.get("status") == "success"}
    pending_w2 = [a for a in top_archs if a["arch_id"] not in done_w2]
    if len(pending_w2) < len(top_archs):
        logger.info(f"Phase 1 Wave 2: skipping {len(top_archs) - len(pending_w2)} already-trained archs ({sorted(done_w2)})")
    with ThreadPoolExecutor(max_workers=_parallel_training_limit(config.gpu.n_gpu)) as pool:
        futures = {pool.submit(_test_final_train_one, arch): arch["arch_id"] for arch in pending_w2}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                arch_id = futures[future]
                logger.error(f"Test-final training {arch_id} failed: {e}")
                continue
            if result:
                test_final_results.append(result)
                exp_logger.log_evaluation({
                    "arch_id": result["arch_id"], "phase": 1, "mode": "test_final",
                    "metric": result["test_metric"], "status": result["status"],
                })
                # Persist after each completed arch — supports mid-Wave-2 resume.
                state_mgr.update({"test_final_results": test_final_results})
                state_mgr.save()

    # Save final state
    state_mgr.update({
        "planner_state": planner.state.to_dict(),
        "val_final_results": val_final_results,
        "test_final_results": test_final_results,
        "phase": "completed",
    })
    state_mgr.save()

    summary = {
        "phase": 1,
        "task": config.task,
        "total_evaluations": planner.state.budget_used,
        "total_tokens": sum(c.total_input_tokens + c.total_output_tokens
                           for c in [planner_llm, executor_llm, explorer_llm]),
        "tokens_by_agent": {
            "planner": planner_llm.total_input_tokens + planner_llm.total_output_tokens,
            "executor": executor_llm.total_input_tokens + executor_llm.total_output_tokens,
            "explorer": explorer_llm.total_input_tokens + explorer_llm.total_output_tokens,
        },
        "best_proxy": {
            "arch_id": planner.state.best_arch_id,
            "metric": planner.state.best_metric,
        },
        "best_final": best_val_final,
        "val_final_results": val_final_results,
        "test_final_results": test_final_results,
    }

    # Save summary for resume early exit
    state_mgr["final_summary"] = summary
    state_mgr.save()

    logger.info(f"Phase 1 complete: {json.dumps(summary, indent=2, default=str)}")
    return summary


def _run_phase1_fixed_seed(
    config: PipelineConfig,
    exp_dir: Path,
    python_bin: str,
    resume: bool,
) -> dict[str, Any]:
    """Phase 1 ablation: skip LLM loop, use a hardcoded seed architecture.

    Reuses the seed42 explorer report, builds a ResNet-18 / WideResNet-16-4
    seed adapted to the task's input/output spec, runs Wave 1 (val_final) and
    Wave 2 (test_final) with the existing training subprocess, and writes a
    state.json that Phase 2 reads as if produced by the normal LLM loop.
    """
    from agentnas.grammar.fixed_seeds import (
        build_fixed_seed_code,
        resolve_family,
        source_explorer_path,
    )

    state_mgr = StateManager(exp_dir / "state.json")
    if resume and state_mgr.exists:
        state_mgr.load()
        if state_mgr.get("phase") == "completed":
            logger.info("Phase 1 (fixed-seed) already completed, returning saved results")
            return state_mgr.get("final_summary", {"status": "completed"})

    benchmark = get_benchmark(config.task, data_root=config.data_root)
    family = resolve_family(config.task, config.fixed_seed_arch)
    src = source_explorer_path(config.task, project_root())
    if not src.exists():
        raise FileNotFoundError(
            f"Fixed-seed ablation requires source explorer report at {src}. "
            f"Run the corresponding seed42 experiment first."
        )
    explorer_report = src.read_text()
    (exp_dir / "explorer_report.md").write_text(explorer_report)

    code, train_cfg = build_fixed_seed_code(
        family, benchmark.input_shape(), benchmark.output_spec(), config.task,
    )
    if config.fixed_seed_epochs is not None:
        train_cfg["epochs"] = int(config.fixed_seed_epochs)

    arch_id = "arch_001"
    description = f"Fixed seed: {family} ({config.task})"

    gpu_pool = GPUPool(n_gpu=config.gpu.n_gpu)
    exp_logger = ExperimentLogger(exp_dir)

    logger.info(
        f"=== Fixed-seed ablation: family={family}, task={config.task}, "
        f"epochs={train_cfg['epochs']}, batch_size={train_cfg['batch_size']} ==="
    )

    # --- Wave 1: val_final (full train on train/val, best checkpoint by val) ---
    logger.info("=== Wave 1: Val-final training (fixed seed) ===")
    gpu_id = gpu_pool.allocate_one()
    try:
        save_dir = exp_dir / "val_final" / arch_id
        val_final = _run_training_subprocess(
            model_code=code,
            config_dict=train_cfg,
            task_name=config.task,
            data_root=config.data_root or "",
            gpu_id=gpu_id,
            mode="search",
            time_limit_s=config.gpu.final_time_limit_s,
            seed=config.seed,
            save_dir=save_dir,
            python_bin=python_bin,
            max_vram_mb=config.gpu.max_vram_mb,
            proxy_epoch_fraction=1.0,
        )
    finally:
        gpu_pool.release(gpu_id)

    val_final_metric = val_final.get("metric", float("inf"))
    logger.info(f"Val-final {arch_id}: metric={val_final_metric:.4f}")

    val_final_record = {
        "arch_id": arch_id,
        "proxy_metric": val_final_metric,
        "val_final_metric": val_final_metric,
        "status": val_final.get("status", "error"),
        "code": code,
        "training_config": train_cfg,
        "params": val_final.get("params", 0),
    }
    exp_logger.log_evaluation({
        "arch_id": arch_id, "phase": 1, "mode": "val_final",
        "metric": val_final_metric, "status": val_final.get("status", "error"),
        "proxy_metric": val_final_metric,
    })

    # --- Wave 2: test_final (train+val combined, last checkpoint, analysis only) ---
    logger.info("=== Wave 2: Test-final training (fixed seed) ===")
    gpu_id = gpu_pool.allocate_one()
    try:
        save_dir = exp_dir / "test_final" / arch_id
        test_final = _run_training_subprocess(
            model_code=code,
            config_dict=train_cfg,
            task_name=config.task,
            data_root=config.data_root or "",
            gpu_id=gpu_id,
            mode="final",
            time_limit_s=config.gpu.final_time_limit_s,
            seed=config.seed,
            save_dir=save_dir,
            python_bin=python_bin,
            max_vram_mb=config.gpu.max_vram_mb,
        )
    finally:
        gpu_pool.release(gpu_id)

    test_final_metric = test_final.get("metric", float("inf"))
    logger.info(f"Test-final {arch_id}: metric={test_final_metric:.4f}")

    test_final_record = {
        "arch_id": arch_id,
        "test_metric": test_final_metric,
        "status": test_final.get("status", "error"),
    }
    exp_logger.log_evaluation({
        "arch_id": arch_id, "phase": 1, "mode": "test_final",
        "metric": test_final_metric, "status": test_final.get("status", "error"),
    })

    # --- Build the synthetic Phase 1 state Phase 2 will consume ---
    # Phase 2 prefers val_final_results; the proxy slot mirrors val_final_metric
    # so the planner_state.evaluations entry (the fallback path) is also coherent.
    proxy_metric = val_final_metric if val_final.get("status") == "success" else float("inf")
    planner_state = PlannerState(
        iteration=1,
        evaluations=[{
            "arch_id": arch_id,
            "description": description,
            "metric": proxy_metric,
            "params": val_final.get("params", 0),
            "peak_vram_mb": val_final.get("peak_vram_mb", 0.0),
            "flops": val_final.get("flops", 0),
            "wall_time_s": val_final.get("wall_time_s", 0.0),
            "status": val_final.get("status", "error"),
            "code": code,
            "iteration": 0,
            "error_message": val_final.get("error_message", ""),
            "training_config": train_cfg,
        }],
        summaries=[],
        explorer_report=explorer_report,
        budget_used=1,
        patience_counter=0,
        best_metric=proxy_metric,
        best_arch_id=arch_id,
        arch_counter=1,
    )

    summary = {
        "phase": 1,
        "task": config.task,
        "fixed_seed_arch": family,
        "total_evaluations": 1,
        "total_tokens": 0,
        "tokens_by_agent": {"planner": 0, "executor": 0, "explorer": 0},
        "best_proxy": {"arch_id": arch_id, "metric": proxy_metric},
        "best_final": val_final_record,
        "val_final_results": [val_final_record],
        "test_final_results": [test_final_record],
    }

    state_mgr.update({
        "planner_state": planner_state.to_dict(),
        "val_final_results": [val_final_record],
        "test_final_results": [test_final_record],
        "phase": "completed",
        "final_summary": summary,
    })
    state_mgr.save()

    logger.info(
        f"Phase 1 (fixed-seed) complete: {json.dumps(summary, indent=2, default=str)}"
    )
    return summary
