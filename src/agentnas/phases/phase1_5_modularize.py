"""Phase 1.5: Seed-modularization clone hook.

Reads an existing experiment's Phase 1 winning architecture, asks the
``ModularizerAgent`` to refactor it so each repeated stage is wrapped in its
own ``nn.Module`` subclass (with internal norm/act/skip baked in), validates
forward-pass equivalence, then writes a NEW experiment directory whose
``round_01/phase1/`` is a copy of the source's with the seed ``code`` field
replaced by the modularized code.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from agentnas.agents.modularizer import ModularizerAgent
from agentnas.benchmarks.registry import get_benchmark
from agentnas.config import PipelineConfig
from agentnas.llm.client import LLMClient
from agentnas.llm.logger import LLMLogger
from agentnas.utils.state import atomic_write_json

logger = logging.getLogger(__name__)


def _find_source_phase1_state(source_dir: Path) -> Path:
    """Find the Phase 1 state.json under source_dir.

    Tries round_01/phase1/state.json first (Phase 4 multi-round layout), then
    phase1/state.json (single-round layout).
    """
    candidates = [
        source_dir / "round_01" / "phase1" / "state.json",
        source_dir / "phase1" / "state.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No Phase 1 state.json found under {source_dir}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def _pick_top_val_final(state_data: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Return the (entry, index) of the best val_final_results record.

    Best = lowest val_final_metric among status=='success'. Falls back to
    final_summary.best_final if val_final_results is empty.
    """
    val_final = state_data.get("val_final_results", [])
    successful = [
        (i, e) for i, e in enumerate(val_final)
        if e.get("status") == "success"
    ]
    if successful:
        successful.sort(key=lambda ie: ie[1].get("val_final_metric", float("inf")))
        idx, entry = successful[0]
        return entry, idx

    bf = state_data.get("final_summary", {}).get("best_final")
    if bf and bf.get("code"):
        return bf, -1

    raise RuntimeError(
        "Source Phase 1 state has no successful val_final_results and no "
        "final_summary.best_final.code — nothing to modularize."
    )


def _replace_code_in_state(
    state_data: dict[str, Any],
    arch_id: str,
    new_code: str,
) -> int:
    """Replace the ``code`` field in every state.json entry referring to *arch_id*.

    Touches: ``final_summary.best_final``, every matching ``val_final_results``
    entry, every matching ``test_final_results`` entry, and any planner_state
    evaluations record matching the same arch_id.

    Returns the number of code fields rewritten.
    """
    n = 0
    bf = state_data.get("final_summary", {}).get("best_final", {})
    if bf and bf.get("arch_id") == arch_id and bf.get("code") is not None:
        bf["code"] = new_code
        n += 1

    for entry in state_data.get("val_final_results", []):
        if entry.get("arch_id") == arch_id and entry.get("code") is not None:
            entry["code"] = new_code
            n += 1

    for entry in state_data.get("test_final_results", []):
        if entry.get("arch_id") == arch_id and entry.get("code") is not None:
            entry["code"] = new_code
            n += 1

    planner_state = state_data.get("planner_state", {})
    for entry in planner_state.get("evaluations", []):
        if entry.get("arch_id") == arch_id and entry.get("code") is not None:
            entry["code"] = new_code
            n += 1

    return n


def _copy_phase1_dir(source_phase1: Path, target_phase1: Path) -> None:
    """Copy source/round_01/phase1/ -> target/round_01/phase1/ recursively.

    Preserves all artifacts (state.json, evaluations.jsonl, leaderboard.csv,
    llm_logs/, models/, val_final/, test_final/). The target's state.json is
    later modified in-place to swap the seed code.
    """
    target_phase1.parent.mkdir(parents=True, exist_ok=True)
    if target_phase1.exists():
        raise FileExistsError(
            f"Target Phase 1 dir already exists: {target_phase1}. "
            "Refusing to overwrite — delete manually if intentional."
        )
    shutil.copytree(source_phase1, target_phase1)


def _write_target_root_state(target_dir: Path, source_state: dict[str, Any]) -> None:
    """Write target/state.json so resume-detection treats this as an existing experiment dir.

    `make_experiment_subdir` checks for `<dir>/state.json` and returns the
    parent as-is when present (avoiding the auto-name suffix). Writing this
    file marks the modularize-clone target as a valid experiment dir.
    """
    root_state = {
        "current_phase": "phase1_complete",
        "best_proxy_metric": source_state.get("best_proxy_metric", float("inf")),
        "modularization_clone": True,
    }
    atomic_write_json(target_dir / "state.json", root_state)


def _save_audit_artifacts(
    audit_dir: Path,
    original_code: str,
    modularized_code: str,
    validation_report: dict[str, Any],
    messages: list[dict[str, str]],
    tokens_used: int,
    retries_used: int,
    phase2_nudge: bool,
    source_arch_id: str,
    source_seed_metric: float | None,
) -> None:
    """Write the phase1_5_modularize/ subdir contents."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "original_code.py").write_text(original_code)
    (audit_dir / "modularized_code.py").write_text(modularized_code)
    (audit_dir / "validation_report.json").write_text(
        json.dumps(validation_report, indent=2)
    )
    (audit_dir / "llm_log.json").write_text(json.dumps({
        "messages": messages,
        "tokens_used": tokens_used,
        "retries_used": retries_used,
        "source_arch_id": source_arch_id,
        "source_seed_metric": source_seed_metric,
    }, indent=2, default=str))
    (audit_dir / "config.json").write_text(json.dumps({
        "phase2_nudge": phase2_nudge,
    }, indent=2))


def run_modularize_in_place(
    config: PipelineConfig,
    phase1_dir: str | Path,
    phase2_nudge: bool = True,
    max_retries: int = 5,
) -> dict[str, Any]:
    """Modularize the Phase 1 winning arch in place.

    Reads *phase1_dir*/state.json, picks the best val_final arch, runs the
    ``ModularizerAgent``, and rewrites every ``code`` field in state.json
    (best_final, matching val_final_results/test_final_results/planner
    evaluations) so downstream Phase 2 reads the modularized seed.

    Audit artifacts (original code, modularized code, validation report, LLM
    trace) are written to ``<phase1_dir>.parent / phase1_5_modularize/``.

    Idempotent: if that dir already contains ``modularized.done``, this returns
    ``{"status": "already_done", ...}`` without re-running the LLM.

    Raises RuntimeError on hard failures (missing state.json, missing seed code,
    modularizer validation failed after ``max_retries``).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    phase1_dir = Path(phase1_dir)
    state_path = phase1_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"Phase 1 state.json not found at {state_path}. Run Phase 1 first."
        )

    audit_dir = phase1_dir.parent / "phase1_5_modularize"
    done_marker = audit_dir / "modularized.done"
    if done_marker.exists():
        prev = json.loads(done_marker.read_text())
        logger.info(
            f"Modularization already complete for {phase1_dir} — skipping. "
            f"({prev.get('source_arch_id', '?')}, retries={prev.get('retries_used', '?')}, "
            f"tokens={prev.get('tokens_used', '?')})"
        )
        return {**prev, "status": "already_done"}

    source_state = json.loads(state_path.read_text())
    if source_state.get("phase") != "completed":
        raise RuntimeError(
            f"Phase 1 not marked complete (phase={source_state.get('phase')!r}). "
            "Refusing to modularize an in-progress Phase 1."
        )

    seed_entry, _ = _pick_top_val_final(source_state)
    arch_id = seed_entry.get("arch_id", "<unknown>")
    seed_metric = seed_entry.get("val_final_metric", seed_entry.get("metric"))
    original_code = seed_entry.get("code")
    if not original_code:
        raise RuntimeError(
            f"Winning arch (arch_id={arch_id}) has no 'code' field — cannot modularize."
        )

    logger.info(
        f"Phase 1.5: modularizing arch_id={arch_id}, val_final={seed_metric}, "
        f"params={seed_entry.get('params', '?')}"
    )

    audit_dir.mkdir(parents=True, exist_ok=True)
    llm_logger = LLMLogger(audit_dir / "llm_logs")
    llm = LLMClient.from_config(config.llm, logger=llm_logger.set_agent("modularizer"))
    benchmark = get_benchmark(config.task, data_root=config.data_root or None)

    modularizer = ModularizerAgent(
        llm_client=llm,
        task_name=config.task,
        input_shape=benchmark.input_shape(),
        max_retries=max_retries,
        python_bin=sys.executable,
        seed=0,
    )

    t0 = time.time()
    result = modularizer.run(original_code)
    elapsed = time.time() - t0
    logger.info(
        f"Modularizer finished in {elapsed:.1f}s: status={result.status}, "
        f"retries={result.retries_used}, tokens={result.tokens_used}"
    )

    _save_audit_artifacts(
        audit_dir=audit_dir,
        original_code=original_code,
        modularized_code=result.modularized_code,
        validation_report=result.validation_report,
        messages=result.messages,
        tokens_used=result.tokens_used,
        retries_used=result.retries_used,
        phase2_nudge=phase2_nudge,
        source_arch_id=arch_id,
        source_seed_metric=seed_metric,
    )

    if result.status != "success":
        raise RuntimeError(
            f"Modularization failed after {max_retries} retries. "
            f"See {audit_dir / 'validation_report.json'} for details."
        )

    n_replaced = _replace_code_in_state(source_state, arch_id, result.modularized_code)
    if n_replaced == 0:
        raise RuntimeError(
            f"No code field found for arch_id={arch_id} in {state_path}. "
            "Phase 1 state structure may have changed unexpectedly."
        )
    atomic_write_json(state_path, source_state)
    logger.info(f"Rewrote {n_replaced} code fields for arch_id={arch_id} in {state_path}")

    summary = {
        "status": "success",
        "source_arch_id": arch_id,
        "source_seed_metric": seed_metric,
        "phase1_dir": str(phase1_dir),
        "max_abs_diff": result.validation_report.get("max_abs_diff"),
        "retries_used": result.retries_used,
        "tokens_used": result.tokens_used,
        "phase2_nudge": phase2_nudge,
        "n_code_fields_replaced": n_replaced,
        "transplant_strategy": result.validation_report.get("transplant_strategy"),
        "elapsed_s": elapsed,
    }
    done_marker.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def run_clone_and_modularize(
    config: PipelineConfig,
    source_dir: str | Path,
    target_dir: str | Path,
    phase2_nudge: bool = False,
    max_retries: int = 5,
) -> dict[str, Any]:
    """Clone *source_dir* into *target_dir* with a modularized seed.

    Returns a dict summary on success::
        {"status": "success", "source_arch_id": "...", "target_dir": "...",
         "max_abs_diff": ..., "retries_used": ..., "tokens_used": ...,
         "phase2_nudge": bool}

    Raises on hard failures (source missing, validation failed, target exists).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dir not found: {source_dir}")
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(
            f"Target dir already exists and is non-empty: {target_dir}. "
            "Pick a fresh path or delete the existing one."
        )

    source_phase1 = _find_source_phase1_state(source_dir).parent
    source_state_path = source_phase1 / "state.json"
    source_state = json.loads(source_state_path.read_text())

    if source_state.get("phase") != "completed":
        raise RuntimeError(
            f"Source Phase 1 not marked complete (phase={source_state.get('phase')!r}). "
            "Refusing to clone an in-progress run."
        )

    seed_entry, seed_idx = _pick_top_val_final(source_state)
    arch_id = seed_entry.get("arch_id", "<unknown>")
    seed_metric = seed_entry.get(
        "val_final_metric", seed_entry.get("metric", None)
    )
    original_code = seed_entry.get("code")
    if not original_code:
        raise RuntimeError(
            f"Source seed (arch_id={arch_id}) has no 'code' field — cannot modularize."
        )

    logger.info(
        f"Source: {source_dir}\n"
        f"  Phase 1 best: arch_id={arch_id}, val_final={seed_metric}, "
        f"params={seed_entry.get('params', '?')}"
    )

    # Build the modularizer (subprocess validator runs on CPU; LLM is configurable)
    target_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = target_dir / "round_01" / "phase1_5_modularize"
    audit_dir.mkdir(parents=True, exist_ok=True)

    llm_logger = LLMLogger(audit_dir / "llm_logs")
    llm = LLMClient.from_config(
        config.llm, logger=llm_logger.set_agent("modularizer"),
    )
    benchmark = get_benchmark(config.task, data_root=config.data_root or None)
    input_shape = benchmark.input_shape()

    modularizer = ModularizerAgent(
        llm_client=llm,
        task_name=config.task,
        input_shape=input_shape,
        max_retries=max_retries,
        python_bin=sys.executable,
        seed=0,
    )

    t0 = time.time()
    result = modularizer.run(original_code)
    elapsed = time.time() - t0
    logger.info(
        f"Modularizer finished in {elapsed:.1f}s: status={result.status}, "
        f"retries={result.retries_used}, tokens={result.tokens_used}"
    )

    # Persist audit artifacts regardless of success/failure
    _save_audit_artifacts(
        audit_dir=audit_dir,
        original_code=original_code,
        modularized_code=result.modularized_code,
        validation_report=result.validation_report,
        messages=result.messages,
        tokens_used=result.tokens_used,
        retries_used=result.retries_used,
        phase2_nudge=phase2_nudge,
        source_arch_id=arch_id,
        source_seed_metric=seed_metric,
    )

    if result.status != "success":
        raise RuntimeError(
            f"Modularization failed after {max_retries} retries. "
            f"See {audit_dir / 'validation_report.json'} for details."
        )

    # Copy Phase 1 dir
    target_phase1 = target_dir / "round_01" / "phase1"
    _copy_phase1_dir(source_phase1, target_phase1)

    # Rewrite the seed code in the cloned phase1/state.json
    cloned_state_path = target_phase1 / "state.json"
    cloned_state = json.loads(cloned_state_path.read_text())
    n_replaced = _replace_code_in_state(
        cloned_state, arch_id, result.modularized_code,
    )
    if n_replaced == 0:
        raise RuntimeError(
            f"No code field found for arch_id={arch_id} in cloned phase1/state.json. "
            "Source state structure may have changed."
        )
    atomic_write_json(cloned_state_path, cloned_state)
    logger.info(
        f"Rewrote {n_replaced} 'code' field(s) for arch_id={arch_id} in "
        f"{cloned_state_path}"
    )

    # Write target/state.json so Phase 4 picks up at round 1 with Phase 1 complete
    _write_target_root_state(target_dir, source_state)

    return {
        "status": "success",
        "source_arch_id": arch_id,
        "source_seed_metric": seed_metric,
        "target_dir": str(target_dir),
        "max_abs_diff": result.validation_report.get("max_abs_diff"),
        "retries_used": result.retries_used,
        "tokens_used": result.tokens_used,
        "phase2_nudge": phase2_nudge,
        "n_code_fields_replaced": n_replaced,
        "transplant_strategy": result.validation_report.get("transplant_strategy"),
    }
