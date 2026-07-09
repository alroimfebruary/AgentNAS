"""Data Explorer agent: analyzes datasets via sandboxed Python code execution.

The explorer receives a "purpose" string from the Planner and executes Python code
in a subprocess to investigate the dataset. It runs for up to 30 autoregressive steps,
producing a <REPORT> at the end.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

from agentnas.llm.client import LLMClient

logger = logging.getLogger(__name__)


class ExplorerError(RuntimeError):
    """The explorer could not ground a report in real code execution.

    Raised instead of returning a fabricated report: a search seeded with
    invented dataset statistics is worse than a failed run.
    """


# Abort after this many consecutive turns without any executable code —
# a model stuck on the wrong output format will not fix itself by step 30,
# and every futile turn grows the context (and the bill).
_MAX_NO_CODE_STREAK = 5

# Execution output shown to the model is truncated to this many characters
# (single-sourced into the system prompt and the executor).
_OUTPUT_TRUNCATE_CHARS = 5000

_EXPLORER_SYSTEM_PROMPT = """\
You are a Data Explorer agent for a neural architecture search pipeline. Your job is \
to analyze a dataset by writing and executing Python code.

## Environment
- You have access to: numpy, pandas, scipy, matplotlib, seaborn, librosa, h5py, \
sklearn, torch, and standard library.
- Data is located at: {data_root}/{data_subdir}/
- You are working in a temporary directory. Save any plots to ./plot.png.

## Workflow
At each step, write Python code in one or more ```python blocks. ALL blocks in \
a single turn are concatenated and executed together in one fresh subprocess \
(notebook-style: imports and variables from an earlier block are visible to \
later blocks in the same turn, but NOT across turns — each turn starts a new \
process). You'll receive stdout+stderr (truncated to {truncate_chars} chars) as the next \
user message.

CRITICAL rules:
- You have NO tools in this conversation: tool-call / function-call XML \
(<function_calls>, <invoke>, bash) does nothing here. The ONLY way to run \
code is a ```python fenced block written directly in your reply.
- NEVER fabricate results. Do NOT write "**Execution output:**" or any output \
block in your own message — outputs come back to you in the NEXT user turn.
- Do NOT produce <REPORT> until you have actually executed at least one \
```python block and observed real output. Reports before any execution will \
be rejected.

When your investigation is complete, write <REPORT> followed by your findings \
as a concise, structured report covering:
- Dataset statistics (samples, features, classes)
- Key characteristics relevant to architecture design
- Any domain-specific observations
- Recommendations for the Planner

## Constraints
- Maximum {max_steps} steps. On step {max_steps}, you MUST produce a <REPORT> \
(the report will be accepted on the final step even without further execution).
- Keep code concise and focused on the investigation purpose.
- Every step costs real tokens: keep prose minimal (the code and its output \
carry the analysis) and produce <REPORT> as soon as the purpose is answered. \
Simple datasets rarely need more than a handful of steps.

## Purpose
{purpose}
"""


def _execute_code(
    code: str, work_dir: Path, python_bin: str, timeout: int = 120
) -> str:
    """Execute Python code in a subprocess. Returns stdout+stderr."""
    work_dir = Path(work_dir).resolve()  # Absolute path to avoid doubling with cwd
    script_path = work_dir / "step.py"
    script_path.write_text(code)

    try:
        result = subprocess.run(
            [python_bin, str(script_path)],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(work_dir),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr
        if len(output) > _OUTPUT_TRUNCATE_CHARS:
            output = output[:_OUTPUT_TRUNCATE_CHARS] + "\n... (truncated)"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: Code execution timed out (>{timeout}s)"
    except Exception as e:
        return f"ERROR: {e}"


def _extract_codes(text: str) -> list[str]:
    """Extract all Python code blocks from LLM response, stripped and non-empty.

    Primary format is ```python fences. As a fallback, code wrapped in
    pseudo-tool-call XML (<invoke name="execute_code"><parameter name="code">
    ...) is accepted — some models emit their tool-use syntax as plain text
    despite the prompt. Whitespace-only blocks are filtered out so they
    cannot satisfy the ≥1-execution gate in the explorer loop.
    """
    matches = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    codes = [m.strip() for m in matches if m.strip()]
    if codes:
        return codes
    # Fallback: pseudo-tool-call markup emitted as plain text.
    matches = re.findall(
        r"<invoke name=\"execute_code\">\s*<parameter name=\"code\">(.*?)</parameter>",
        text,
        re.DOTALL,
    )
    return [m.strip() for m in matches if m.strip()]


class DataExplorer:
    """Analyzes datasets by executing Python code in a sandbox.

    Autoregressive loop: LLM writes code → execute → observe → repeat.
    Produces a <REPORT> when done or after max steps.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        data_root: str,
        data_subdir: str,
        max_steps: int = 30,
        python_bin: str = "",
        exec_timeout_s: int = 120,
    ):
        self.llm = llm_client
        self.data_root = data_root
        self.data_subdir = data_subdir
        self.max_steps = max_steps
        self.python_bin = python_bin or "python"
        self.exec_timeout_s = exec_timeout_s
        self.total_tokens = 0

    def explore(self, purpose: str) -> tuple[str, int]:
        """Run the exploration loop.

        Args:
            purpose: What to investigate (from Planner).

        Returns:
            (report, tokens_used)
        """
        system = _EXPLORER_SYSTEM_PROMPT.format(
            data_root=self.data_root,
            data_subdir=self.data_subdir,
            max_steps=self.max_steps,
            purpose=purpose,
            truncate_chars=_OUTPUT_TRUNCATE_CHARS,
        )

        messages = [
            {
                "role": "user",
                "content": f"Begin your exploration. Purpose: {purpose}",
            }
        ]

        tokens_used = 0
        exec_count = 0
        no_code_streak = 0

        with tempfile.TemporaryDirectory(prefix="agentnas_explore_") as tmpdir:
            work_dir = Path(tmpdir)

            for step in range(1, self.max_steps + 1):
                # Add final step hint by appending to the LAST user message (M-01 fix)
                if step == self.max_steps and messages and messages[-1]["role"] == "user":
                    messages[-1]["content"] += (
                        f"\n\n⚠️ This is step {step}/{self.max_steps} (FINAL STEP). "
                        "You MUST produce a <REPORT> now with your findings."
                    )

                resp = self.llm.chat(messages, system=system)
                tokens_used += resp.total_tokens

                codes = _extract_codes(resp.content)
                # Scan ONLY assistant content for <REPORT>; never the exec output
                # (a model-crafted stdout must not be able to short-circuit the loop).
                has_report = "<REPORT>" in resp.content
                is_final_step = step == self.max_steps

                # Final step: <REPORT> is mandatory — accept it even if code also
                # present. Otherwise a final-step response with both would execute
                # code, the loop would exit, and we'd hit the post-loop fallback
                # "(Explorer did not produce a report...)". A report with zero
                # executions is never accepted — that is a fabrication.
                if has_report and is_final_step and exec_count >= 1:
                    report = resp.content.split("<REPORT>", 1)[1].replace("</REPORT>", "").strip()
                    logger.info(
                        f"Explorer produced report at final step {step} "
                        f"(exec_count={exec_count})"
                    )
                    self.total_tokens += tokens_used
                    return report, tokens_used

                # Non-final step: code takes precedence over <REPORT> — prevents
                # hallucinated single-turn responses from bypassing execution.
                if codes:
                    combined = "\n\n# --- next block ---\n\n".join(codes)
                    logger.info(
                        f"Explorer step {step}: executing {len(codes)} block(s) "
                        f"({len(combined)} chars)"
                    )
                    output = _execute_code(
                        combined, work_dir, self.python_bin,
                        timeout=self.exec_timeout_s,
                    )
                    exec_count += 1
                    no_code_streak = 0
                    messages.append({"role": "assistant", "content": resp.content})
                    messages.append({
                        "role": "user",
                        "content": f"**Execution output (step {step}/{self.max_steps}):**\n```\n{output}\n```",
                    })
                    continue

                if has_report and exec_count >= 1:
                    report = resp.content.split("<REPORT>", 1)[1].replace("</REPORT>", "").strip()
                    logger.info(
                        f"Explorer produced report at step {step} "
                        f"(exec_count={exec_count})"
                    )
                    self.total_tokens += tokens_used
                    return report, tokens_used

                no_code_streak += 1
                if no_code_streak >= _MAX_NO_CODE_STREAK:
                    self.total_tokens += tokens_used
                    raise ExplorerError(
                        f"Explorer produced no executable code for "
                        f"{no_code_streak} consecutive turns (step {step}/"
                        f"{self.max_steps}, {exec_count} successful executions "
                        f"so far). The model is not emitting ```python blocks "
                        f"— check the llm_logs/explorer_*.json transcripts, "
                        f"and consider a stronger model."
                    )

                if has_report:
                    # Report before any execution — reject and nudge.
                    logger.warning(
                        f"Explorer step {step}: rejecting <REPORT> (no prior execution)"
                    )
                    messages.append({"role": "assistant", "content": resp.content})
                    messages.append({
                        "role": "user",
                        "content": (
                            "You produced <REPORT> without executing any code. Reports must "
                            "be grounded in real execution output — fabricated values are "
                            "not acceptable. Write a ```python block that actually inspects "
                            f"{self.data_root}/{self.data_subdir}/ (e.g. list files, load one "
                            "sample, print shape/dtype). Real output will be returned to you."
                        ),
                    })
                    continue

                # No code, no report — say exactly what was wrong and show the
                # only format that executes (models stuck on tool-call syntax
                # will not recover from a vague nudge).
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Step {step}/{self.max_steps}. No executable code was "
                        "found in your reply. Code runs ONLY when written inside "
                        "a fenced block exactly like:\n"
                        "```python\nprint('hello')\n```\n"
                        "Tool-call syntax (<function_calls>, <invoke>, "
                        "execute_code) is NOT available here. Rewrite your code "
                        "in a single ```python block, or produce a <REPORT> if "
                        "your investigation is complete."
                    ),
                })

        # If we get here, max steps exhausted without report
        if exec_count == 0:
            self.total_tokens += tokens_used
            raise ExplorerError(
                f"Explorer exhausted {self.max_steps} steps without a single "
                f"successful code execution — refusing to fabricate a report."
            )
        logger.warning("Explorer exhausted max steps without producing a report")
        self.total_tokens += tokens_used
        return "(Explorer did not produce a report within the step limit)", tokens_used
