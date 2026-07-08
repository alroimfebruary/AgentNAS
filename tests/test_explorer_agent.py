"""Unit tests for DataExplorer robustness (mock LLM, no API calls).

Pins down the failure mode found in the first custom-task smoke run: a model
that emits tool-call XML instead of ```python fences must either be parsed
(fallback) or fail loudly — never produce a fabricated report.
"""

from __future__ import annotations

import sys

import pytest

from agentnas.agents.explorer import DataExplorer, ExplorerError, _extract_codes

XML_CODE_RESPONSE = """I'll inspect the dataset directory.
<function_calls>
<invoke name="execute_code">
<parameter name="code">
print("hello from xml")
</parameter>
</invoke>
</function_calls>"""


class _Resp:
    def __init__(self, content: str):
        self.content = content
        self.total_tokens = 10


class _ScriptedLLM:
    """Returns canned responses; records every call."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages, system=""):
        self.calls.append([dict(m) for m in messages])
        return _Resp(self.responses.pop(0))


def _explorer(llm, tmp_path, max_steps=6) -> DataExplorer:
    (tmp_path / "dataset").mkdir(exist_ok=True)
    return DataExplorer(
        llm_client=llm,
        data_root=str(tmp_path),
        data_subdir="dataset",
        max_steps=max_steps,
        python_bin=sys.executable,
    )


def test_extract_codes_python_fence():
    text = "look:\n```python\nprint(1)\n```\nand\n```python\nprint(2)\n```"
    assert _extract_codes(text) == ["print(1)", "print(2)"]


def test_extract_codes_xml_fallback():
    assert _extract_codes(XML_CODE_RESPONSE) == ['print("hello from xml")']


def test_extract_codes_fence_takes_precedence_over_xml():
    text = "```python\nprint('fence')\n```\n" + XML_CODE_RESPONSE
    assert _extract_codes(text) == ["print('fence')"]


def test_xml_code_executes_and_report_accepted(tmp_path):
    llm = _ScriptedLLM([XML_CODE_RESPONSE, "<REPORT>grounded findings"])
    report, _ = _explorer(llm, tmp_path).explore("look around")
    assert report == "grounded findings"
    # The second call must contain REAL execution output from the XML code.
    feedback = llm.calls[1][-1]["content"]
    assert "Execution output" in feedback and "hello from xml" in feedback


def test_report_after_fenced_execution(tmp_path):
    llm = _ScriptedLLM(["```python\nprint(41 + 1)\n```", "<REPORT>found 42"])
    report, _ = _explorer(llm, tmp_path).explore("math")
    assert report == "found 42"
    assert "42" in llm.calls[1][-1]["content"]


def test_no_code_feedback_names_the_format(tmp_path):
    llm = _ScriptedLLM(["let me think...", "```python\nprint('ok')\n```",
                        "<REPORT>done"])
    report, _ = _explorer(llm, tmp_path).explore("x")
    assert report == "done"
    nudge = llm.calls[1][-1]["content"]
    assert "No executable code" in nudge and "```python" in nudge


def test_persistent_no_code_aborts_early(tmp_path):
    llm = _ScriptedLLM(["thinking about it..."] * 10)
    with pytest.raises(ExplorerError, match="no executable code"):
        _explorer(llm, tmp_path, max_steps=10).explore("x")
    # Aborted after the streak limit, not the full step budget.
    assert len(llm.calls) == 5


def test_final_step_report_without_execution_raises(tmp_path):
    llm = _ScriptedLLM(["no code here", "<REPORT>fabricated stats"])
    with pytest.raises(ExplorerError, match="without a single successful"):
        _explorer(llm, tmp_path, max_steps=2).explore("x")


def test_final_step_report_with_prior_execution_ok(tmp_path):
    llm = _ScriptedLLM(["```python\nprint('data')\n```", "<REPORT>real stats"])
    report, _ = _explorer(llm, tmp_path, max_steps=2).explore("x")
    assert report == "real stats"
