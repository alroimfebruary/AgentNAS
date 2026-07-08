"""Unit tests for the Task Author agent (mock LLM + stub validator)."""

from __future__ import annotations

import sys

from agentnas.agents.task_author import TaskAuthor, _extract_file

GOOD_FILE = """\
from agentnas.benchmarks.registry import register_benchmark
from agentnas.benchmarks.simple import NpyClassificationBenchmark


@register_benchmark("t_authored")
class TAuthored(NpyClassificationBenchmark):
    task = "t_authored"
    data_shape = (3, 8, 8)
    n_classes = 4
"""


class _Resp:
    def __init__(self, content: str):
        self.content = content
        self.total_tokens = 10


class _ScriptedLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages, system=""):
        self.calls.append([dict(m) for m in messages])
        return _Resp(self.responses.pop(0))


class _StubValidator:
    """Scripted (ok, report) results; records calls."""

    def __init__(self, results: list[tuple[bool, str]]):
        self.results = list(results)
        self.calls: list[tuple] = []

    def __call__(self, file_path, task, data_root):
        self.calls.append((file_path, task, data_root))
        return self.results.pop(0)


def _author(llm, validator, tmp_path, **kw):
    (tmp_path / "data" / "t_authored").mkdir(parents=True, exist_ok=True)
    return TaskAuthor(
        llm_client=llm,
        task="t_authored",
        data_root=str(tmp_path / "data"),
        data_subdir="t_authored",
        out_path=tmp_path / "t_authored.py",
        python_bin=sys.executable,
        validate_fn=validator,
        **kw,
    )


def _wrap(content: str) -> str:
    return f"here it is:\n<BENCHMARK_FILE>\n{content}</BENCHMARK_FILE>\n"


def test_extract_file_plain_and_fenced():
    assert _extract_file(_wrap(GOOD_FILE)) == GOOD_FILE.strip()
    fenced = f"<BENCHMARK_FILE>\n```python\n{GOOD_FILE}```\n</BENCHMARK_FILE>"
    assert _extract_file(fenced) == GOOD_FILE.strip()
    assert _extract_file("no file here") is None
    assert _extract_file("<BENCHMARK_FILE>\n\n</BENCHMARK_FILE>") is None


def test_inspect_then_emit_success(tmp_path):
    llm = _ScriptedLLM([
        "```python\nprint('looking at files')\n```",
        _wrap(GOOD_FILE),
    ])
    validator = _StubValidator([(True, "All checks passed")])
    result = _author(llm, validator, tmp_path).author()

    assert result.success and result.attempts == 1 and result.turns == 2
    assert result.file_path.read_text().strip() == GOOD_FILE.strip()
    # Inspection output was fed back before the emission turn
    assert "looking at files" in llm.calls[1][-1]["content"]
    # Validator got the right coordinates
    fp, task, root = validator.calls[0]
    assert fp == result.file_path and task == "t_authored"


def test_validation_failure_feeds_report_and_retries(tmp_path):
    llm = _ScriptedLLM([
        _wrap(GOOD_FILE.replace("n_classes = 4", "n_classes = 0")),
        _wrap(GOOD_FILE),
    ])
    validator = _StubValidator([
        (False, "[FAIL] must set the class attributes"),
        (True, "All checks passed"),
    ])
    result = _author(llm, validator, tmp_path).author()

    assert result.success and result.attempts == 2
    feedback = llm.calls[1][-1]["content"]
    assert "Validation FAILED" in feedback
    assert "[FAIL] must set the class attributes" in feedback
    # Final file on disk is the corrected one
    assert "n_classes = 4" in result.file_path.read_text()


def test_attempt_budget_exhausted(tmp_path):
    llm = _ScriptedLLM([_wrap(GOOD_FILE), _wrap(GOOD_FILE)])
    validator = _StubValidator([
        (False, "[FAIL] broken 1"),
        (False, "[FAIL] broken 2"),
    ])
    result = _author(llm, validator, tmp_path, max_attempts=2).author()

    assert not result.success and result.attempts == 2
    assert "[FAIL] broken 2" in result.last_report
    assert len(validator.calls) == 2


def test_no_progress_aborts(tmp_path):
    llm = _ScriptedLLM(["hmm let me think"] * 6)
    validator = _StubValidator([])
    result = _author(llm, validator, tmp_path).author()

    assert not result.success and result.attempts == 0
    assert result.turns == 4  # aborted at the streak limit, not max_turns
    assert "neither executable code nor" in result.last_report


def test_file_emission_takes_precedence_over_code(tmp_path):
    # A turn containing both a file and code counts as an emission.
    llm = _ScriptedLLM([
        "```python\nprint('x')\n```\n" + _wrap(GOOD_FILE),
    ])
    validator = _StubValidator([(True, "ok")])
    result = _author(llm, validator, tmp_path).author()
    assert result.success and result.attempts == 1 and len(validator.calls) == 1
