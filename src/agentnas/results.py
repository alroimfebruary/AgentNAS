"""Experiment results: collect, render, and export the winning architecture.

Understands both layouts:

- a single-phase experiment dir (``agentnas run``): contains ``state.json``
- a full-pipeline base dir (``scripts/run_production.py``): contains
  ``phase1/``, optionally ``phase1_5_modularize/``, ``phase2/``, ``phase3/``

The winner follows the pipeline's own rule — ever-best by ``val_final``
across phases; the reported test metric is the winner's, never the best
test metric (that would be test-set selection).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _best_by_val(val_results: list[dict]) -> dict | None:
    """Lowest val_final_metric among successful entries."""
    ok = [
        r for r in val_results
        if r.get("status") == "success" and r.get("val_final_metric") is not None
    ]
    if not ok:
        return None
    return min(ok, key=lambda r: r["val_final_metric"])


def _test_metric_for(test_results: list[dict], arch_id: str) -> float | None:
    for r in test_results:
        if r.get("arch_id") == arch_id and r.get("status") == "success":
            return r.get("test_metric")
    return None


def _collect_phase(phase_dir: Path) -> dict | None:
    """Summarize one phase dir (phase 1 or 3) from its state.json."""
    state = _load_json(phase_dir / "state.json")
    if state is None:
        return None

    fs = state.get("final_summary") or {}
    val_results = fs.get("val_final_results") or state.get("val_final_results") or []
    test_results = (
        fs.get("test_final_results") or state.get("test_final_results") or []
    )
    best = _best_by_val(val_results)

    # run_production enrichment (has total_evaluations for phase 3)
    summary = _load_json(phase_dir / "result_summary.json") or {}

    phase = state.get("phase")
    if isinstance(phase, str):  # phases store markers like "completed" here
        phase = fs.get("phase") or summary.get("phase")
    if phase is None and "evolution" in state:
        phase = 3

    info: dict[str, Any] = {
        "dir": str(phase_dir),
        "phase": phase,
        "task": fs.get("task"),
        "state_phase_marker": state.get("phase"),
        "total_evaluations": fs.get("total_evaluations")
                             or summary.get("total_evaluations"),
        "total_tokens": fs.get("total_tokens"),
        "tokens_by_agent": fs.get("tokens_by_agent"),
        "val_final_results": val_results,
        "best": None,
    }
    if best is not None:
        info["best"] = {
            "arch_id": best["arch_id"],
            "proxy_metric": best.get("proxy_metric"),
            "val_final_metric": best.get("val_final_metric"),
            "test_metric": _test_metric_for(test_results, best["arch_id"]),
            "params": best.get("params"),
        }
    return info


def _find_best_code(phase_dir: Path, arch_id: str) -> tuple[str, str] | None:
    """Return (code, source_description) for an architecture.

    Prefers the model.py that actually produced the reported test metric.
    """
    for split in ("test_final", "val_final"):
        model_py = phase_dir / split / arch_id / "model.py"
        if model_py.exists():
            return model_py.read_text(), str(model_py)
    for src_name in ("state.json", "result_summary.json"):
        data = _load_json(phase_dir / src_name) or {}
        fs = data.get("final_summary") or data
        best_final = fs.get("best_final") or {}
        if best_final.get("arch_id") == arch_id and best_final.get("code"):
            return best_final["code"], f"{phase_dir / src_name} (best_final.code)"
    return None


def collect(exp_dir: str | Path) -> dict:
    """Gather results across all phases found under exp_dir."""
    exp_dir = Path(exp_dir)
    phases: list[dict] = []

    if (exp_dir / "state.json").exists():
        info = _collect_phase(exp_dir)
        if info:
            phases.append(info)
    else:
        for name in ("phase1", "phase3"):
            sub = exp_dir / name
            if (sub / "state.json").exists():
                info = _collect_phase(sub)
                if info:
                    info["phase"] = info["phase"] or int(name[-1])
                    phases.append(info)

    if not phases:
        raise FileNotFoundError(
            f"No experiment state found under {exp_dir} — expected "
            f"state.json (single phase) or phase1/phase3 subdirectories."
        )

    phase2_state = _load_json(exp_dir / "phase2" / "state.json") or {}
    modularized = (exp_dir / "phase1_5_modularize").exists()

    candidates = [p for p in phases if p["best"] is not None]
    winner_phase = (
        min(candidates, key=lambda p: p["best"]["val_final_metric"])
        if candidates else None
    )

    task = next((p["task"] for p in phases if p.get("task")), None)
    return {
        "experiment_dir": str(exp_dir),
        "task": task,
        "phases": phases,
        "modularized": modularized,
        "search_space": {
            "n_slots": phase2_state.get("n_slots"),
            "size": phase2_state.get("search_space_size"),
        } if phase2_state else None,
        "winner": {
            "phase": winner_phase["phase"],
            "phase_dir": winner_phase["dir"],
            **winner_phase["best"],
        } if winner_phase else None,
    }


def _fmt(x: float | None, digits: int = 4) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def render(results: dict) -> str:
    """Human-readable report."""
    lines: list[str] = []
    lines.append(f"Experiment: {results['experiment_dir']}")
    if results.get("task"):
        lines.append(f"Task: {results['task']}")
    phase_ids = ", ".join(str(p["phase"]) for p in results["phases"])
    extra = " (+ modularize)" if results.get("modularized") else ""
    lines.append(f"Phases with results: {phase_ids}{extra}")

    ss = results.get("search_space")
    if ss and ss.get("n_slots"):
        size = ss.get("size")
        size_s = f", {size:,} configurations" if isinstance(size, (int, float)) else ""
        lines.append(f"Search space: {ss['n_slots']} slots{size_s}")

    w = results.get("winner")
    lines.append("")
    if w is None:
        lines.append("Winner: none — no successful final-trained architecture found.")
    else:
        lines.append(
            f"Winner (ever-best by val_final): phase {w['phase']} / {w['arch_id']}"
        )
        params = f"{w['params']:,}" if w.get("params") else "—"
        lines.append(
            f"  proxy {_fmt(w.get('proxy_metric'))} | "
            f"val_final {_fmt(w.get('val_final_metric'))} | "
            f"test {_fmt(w.get('test_metric'))} | params {params}"
        )
        if w.get("test_metric") is None:
            lines.append("  (no test-final result recorded for the winner)")

    lines.append("")
    lines.append("Per-phase best (by val_final):")
    for p in results["phases"]:
        b = p.get("best")
        if b is None:
            lines.append(f"  phase {p['phase']}: no successful final training")
            continue
        lines.append(
            f"  phase {p['phase']}: {b['arch_id']}  "
            f"val {_fmt(b['val_final_metric'])}  test {_fmt(b['test_metric'])}"
            + (f"  ({p['total_evaluations']} evaluations)"
               if p.get("total_evaluations") else "")
        )

    tok_lines = []
    for p in results["phases"]:
        if p.get("total_tokens"):
            by = p.get("tokens_by_agent") or {}
            detail = ", ".join(f"{k} {v:,}" for k, v in by.items())
            tok_lines.append(
                f"  phase {p['phase']}: {p['total_tokens']:,}"
                + (f" ({detail})" if detail else "")
            )
    if tok_lines:
        lines.append("")
        lines.append("LLM tokens (where recorded):")
        lines.extend(tok_lines)

    return "\n".join(lines)


def export_winner(results: dict, out_path: str | Path) -> Path:
    """Write the winner's standalone model.py with a provenance header."""
    w = results.get("winner")
    if w is None:
        raise ValueError("No winner to export — no successful final training found.")
    found = _find_best_code(Path(w["phase_dir"]), w["arch_id"])
    if found is None:
        raise FileNotFoundError(
            f"No code found for {w['arch_id']} under {w['phase_dir']} "
            f"(looked in test_final/, val_final/, state.json, result_summary.json)."
        )
    code, source = found
    header = (
        f'"""AgentNAS winning architecture — exported by `agentnas results`.\n'
        f"\n"
        f"Experiment: {results['experiment_dir']}\n"
        f"Task: {results.get('task') or 'unknown'}\n"
        f"Phase {w['phase']} / {w['arch_id']} | "
        f"val_final {_fmt(w.get('val_final_metric'))} | "
        f"test {_fmt(w.get('test_metric'))} (lower = better)\n"
        f"Source: {source}\n"
        f'"""\n\n'
    )
    out = Path(out_path)
    out.write_text(header + code.rstrip() + "\n")
    return out
