"""Unit tests for `agentnas results` (synthetic experiment fixtures)."""

from __future__ import annotations

import json

import pytest

from agentnas.results import collect, export_winner, render

CODE = "import torch.nn as nn\n\n\nclass Architecture(nn.Module):\n    pass\n"


def _phase1_state(val: float, test: float) -> dict:
    return {
        "phase": "completed",
        "final_summary": {
            "phase": 1,
            "task": "t_res",
            "total_evaluations": 3,
            "total_tokens": 1000,
            "tokens_by_agent": {"planner": 400, "explorer": 600},
            "best_proxy": {"arch_id": "arch_001", "metric": val},
            "best_final": {
                "arch_id": "arch_001", "code": CODE, "params": 123,
                "proxy_metric": val, "val_final_metric": val, "status": "success",
            },
            "val_final_results": [
                {"arch_id": "arch_001", "proxy_metric": val,
                 "val_final_metric": val, "status": "success", "params": 123},
                {"arch_id": "arch_002", "proxy_metric": 0.9,
                 "val_final_metric": 0.9, "status": "success", "params": 5},
            ],
            "test_final_results": [
                {"arch_id": "arch_001", "test_metric": test, "status": "success"},
            ],
        },
    }


def _phase3_state(val: float, test: float, ok: bool = True) -> dict:
    return {
        "phase": "completed",
        "evolution": {"generation": 5},
        "val_final_results": [
            {"arch_id": "evo_007", "proxy_metric": val,
             "val_final_metric": val,
             "status": "success" if ok else "error", "params": 456},
        ],
        "test_final_results": [
            {"arch_id": "evo_007", "test_metric": test, "status": "success"},
        ],
    }


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_single_phase_dir(tmp_path):
    _write(tmp_path / "state.json", _phase1_state(0.20, 0.22))
    res = collect(tmp_path)
    assert res["task"] == "t_res"
    w = res["winner"]
    assert w["phase"] == 1 and w["arch_id"] == "arch_001"
    assert w["val_final_metric"] == 0.20 and w["test_metric"] == 0.22

    report = render(res)
    assert "ever-best by val_final" in report and "arch_001" in report
    assert "planner 400" in report


def test_full_pipeline_ever_best_picks_phase3(tmp_path):
    _write(tmp_path / "phase1" / "state.json", _phase1_state(0.20, 0.22))
    _write(tmp_path / "phase3" / "state.json", _phase3_state(0.15, 0.18))
    _write(tmp_path / "phase2" / "state.json",
           {"phase": 2, "n_slots": 19, "search_space_size": 1000000})
    (tmp_path / "phase1_5_modularize").mkdir()

    res = collect(tmp_path)
    w = res["winner"]
    assert w["phase"] == 3 and w["arch_id"] == "evo_007"
    assert w["test_metric"] == 0.18
    report = render(res)
    assert "phase 3 / evo_007" in report
    assert "19 slots" in report and "1,000,000" in report
    assert "(+ modularize)" in report


def test_winner_falls_back_when_phase3_failed(tmp_path):
    _write(tmp_path / "phase1" / "state.json", _phase1_state(0.20, 0.22))
    _write(tmp_path / "phase3" / "state.json", _phase3_state(0.10, 0.11, ok=False))
    res = collect(tmp_path)
    assert res["winner"]["phase"] == 1  # phase3's better val was not "success"


def test_export_prefers_test_final_model_py(tmp_path):
    _write(tmp_path / "state.json", _phase1_state(0.20, 0.22))
    exact = "# exact trained file\n" + CODE
    model_py = tmp_path / "test_final" / "arch_001" / "model.py"
    model_py.parent.mkdir(parents=True)
    model_py.write_text(exact)

    out = export_winner(collect(tmp_path), tmp_path / "best.py")
    text = out.read_text()
    assert "# exact trained file" in text
    assert "test 0.2200" in text and str(model_py) in text


def test_export_falls_back_to_state_code(tmp_path):
    _write(tmp_path / "state.json", _phase1_state(0.20, 0.22))
    out = export_winner(collect(tmp_path), tmp_path / "best.py")
    text = out.read_text()
    assert "class Architecture" in text and "best_final.code" in text


def test_no_state_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No experiment state"):
        collect(tmp_path)
