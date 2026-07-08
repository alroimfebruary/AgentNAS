"""Task Author agent: writes a custom --benchmark-file from a data directory.

``agentnas init-task`` spawns this agent. It inspects the user's data by
executing Python code (the same sandboxed protocol as the Data Explorer),
emits a complete benchmark file, and iterates against ``agentnas
validate-task`` — run in a fresh subprocess — until the file passes or the
attempt budget is exhausted.

The generated file is an ordinary benchmark file: after generation the
pipeline has no memory of who authored it. The flow never starts a search —
it hard-stops for human review of the produced code and validation report.

Unlike the search-time Data Explorer, this agent MAY see test files (it has
to wire them into the final loaders). That is safe because authoring is
one-shot and pre-search; its transcript never carries into search-time
context, and the search-time Explorer allowlist applies regardless of who
wrote the loader.
"""

from __future__ import annotations

import inspect
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from agentnas.agents.explorer import _execute_code, _extract_codes
from agentnas.llm.client import LLMClient

logger = logging.getLogger(__name__)

_FILE_RE = re.compile(r"<BENCHMARK_FILE>\s*\n?(.*?)</BENCHMARK_FILE>", re.DOTALL)

_EXAMPLE = '''\
from agentnas.benchmarks.registry import register_benchmark
from agentnas.benchmarks.simple import NpyClassificationBenchmark


@register_benchmark("my_task")
class MyTaskBenchmark(NpyClassificationBenchmark):
    task = "my_task"
    data_shape = (3, 32, 32)
    n_classes = 10
'''

_AUTHOR_SYSTEM_PROMPT = """\
You are a Task Author agent for AgentNAS, an LLM-driven neural architecture
search pipeline. Your job: write a complete, correct benchmark definition
file for the user's dataset.

## The dataset
- Files are located at: {data_dir}/
- Registered task name: {task}
- data_subdir (directory name under the pipeline's data root): {data_subdir}
- User's description: {description}

## Phase 1 — inspect
Write Python code in ```python blocks; all blocks in one turn run together
in one fresh subprocess (state does NOT persist across turns) and
stdout+stderr come back in the next user message. Available: numpy, pandas,
scipy, h5py, sklearn, torch, standard library. Inspect the file listing,
formats, shapes, dtypes, label ranges, and split sizes; determine whether
splits are pre-separated files or bundled. NEVER guess or fabricate —
every decision must be grounded in executed output.

## Phase 2 — emit the file
When you understand the data, emit the COMPLETE file wrapped EXACTLY as:
<BENCHMARK_FILE>
... python source ...
</BENCHMARK_FILE>
Emit the file alone in its turn — no ```python blocks in the same message.

Requirements for the file:
- Registers exactly one benchmark: @register_benchmark("{task}").
- PREFER the convenience base when the data fits it (pre-split .npy integer
  classification): subclass NpyClassificationBenchmark and set task /
  data_shape / n_classes (plus split_files / data_subdir when names differ).
- Otherwise subclass BaseBenchmark and implement everything yourself.
- Set data_subdir = "{data_subdir}" when it differs from the task name.
- __init__ must do NO filesystem access (metadata introspection must work
  without the data present).
- The metric must be lower = better (e.g. report 1 - accuracy).
- get_search_loaders returns (train, val); get_final_loaders returns
  (train+val combined, test). If the data ships no test split, carve a
  DETERMINISTIC held-out test split (fixed seed, documented in a comment)
  that never appears in train/val.
- Explorer test-set isolation: set explorer_allowed_files to the train/val
  files only. If a single bundled file holds all splits, instead set
  explorer_preprocessed_name and implement prepare_explorer_source() to
  write a train+val-only copy.
- If the user's description names a published baseline, set sota_metric and
  sota_description (in the lower=better scale); otherwise leave them unset.

## Ground-truth API (the actual installed source you are coding against)

### agentnas/benchmarks/base.py
```python
{base_source}
```

### agentnas/benchmarks/simple.py
```python
{simple_source}
```

### Minimal worked example (convenience route)
```python
{example}
```

## Validation loop
Each emitted file is checked by `agentnas validate-task` (contract, loaders,
split disjointness, explorer isolation, metric sanity). On failure you get
the full report back — fix the issues and emit the complete corrected file
again (never a diff).

## Constraints
- At most {max_turns} turns total and {max_attempts} file emissions.
"""


def _extract_file(text: str) -> str | None:
    """Return the last <BENCHMARK_FILE> payload, unwrapping stray fences."""
    matches = _FILE_RE.findall(text)
    if not matches:
        return None
    content = matches[-1].strip()
    if content.startswith("```"):
        lines = content.splitlines()
        lines = lines[1:]  # drop the opening fence (```python or ```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return content or None


def _default_validate(file_path: Path, task: str, data_root: str) -> tuple[bool, str]:
    """Run `agentnas validate-task` in a fresh subprocess.

    A subprocess keeps the registry of THIS process clean across retry
    attempts and validates exactly what the user will later run.
    """
    proc = subprocess.run(
        [
            sys.executable, "-m", "agentnas.cli", "validate-task",
            str(file_path), "--task", task, "--data-root", data_root,
        ],
        capture_output=True, text=True, timeout=900,
    )
    output = proc.stdout
    if proc.stderr:
        output += "\n[STDERR]\n" + proc.stderr
    return proc.returncode == 0, output.strip()


@dataclass
class AuthorResult:
    success: bool
    file_path: Path
    attempts: int
    turns: int
    tokens: int
    last_report: str


class TaskAuthor:
    """Inspect → author → validate loop producing a benchmark file."""

    def __init__(
        self,
        llm_client: LLMClient,
        task: str,
        data_root: str,
        data_subdir: str,
        out_path: Path,
        description: str = "",
        max_turns: int = 25,
        max_attempts: int = 4,
        python_bin: str = "",
        exec_timeout_s: int = 120,
        validate_fn=None,
    ):
        self.llm = llm_client
        self.task = task
        self.data_root = str(Path(data_root).resolve())
        self.data_subdir = data_subdir
        self.out_path = Path(out_path)
        self.description = description.strip() or "(none provided)"
        self.max_turns = max_turns
        self.max_attempts = max_attempts
        self.python_bin = python_bin or sys.executable
        self.exec_timeout_s = exec_timeout_s
        self.validate_fn = validate_fn or _default_validate

    def _system_prompt(self) -> str:
        from agentnas.benchmarks import base as base_mod
        from agentnas.benchmarks import simple as simple_mod

        return _AUTHOR_SYSTEM_PROMPT.format(
            data_dir=str(Path(self.data_root) / self.data_subdir),
            task=self.task,
            data_subdir=self.data_subdir,
            description=self.description,
            base_source=inspect.getsource(base_mod),
            simple_source=inspect.getsource(simple_mod),
            example=_EXAMPLE,
            max_turns=self.max_turns,
            max_attempts=self.max_attempts,
        )

    def author(self) -> AuthorResult:
        system = self._system_prompt()
        messages = [{
            "role": "user",
            "content": (
                "Begin. Inspect the data first; when you are confident, emit "
                "the complete file in <BENCHMARK_FILE> tags."
            ),
        }]

        attempts = 0
        tokens = 0
        no_progress = 0
        last_report = ""

        with tempfile.TemporaryDirectory(prefix="agentnas_author_") as tmpdir:
            work_dir = Path(tmpdir)

            for turn in range(1, self.max_turns + 1):
                if turn == self.max_turns and messages[-1]["role"] == "user":
                    messages[-1]["content"] += (
                        f"\n\n⚠️ FINAL TURN ({turn}/{self.max_turns}): emit the "
                        "complete <BENCHMARK_FILE> now."
                    )

                resp = self.llm.chat(messages, system=system)
                tokens += resp.total_tokens

                file_content = _extract_file(resp.content)
                if file_content is not None:
                    attempts += 1
                    no_progress = 0
                    self.out_path.write_text(file_content.rstrip() + "\n")
                    ok, report = self.validate_fn(
                        self.out_path, self.task, self.data_root
                    )
                    last_report = report
                    logger.info(
                        f"init-task attempt {attempts}/{self.max_attempts}: "
                        f"{'PASS' if ok else 'FAIL'}"
                    )
                    if ok:
                        return AuthorResult(
                            True, self.out_path, attempts, turn, tokens, report
                        )
                    if attempts >= self.max_attempts:
                        return AuthorResult(
                            False, self.out_path, attempts, turn, tokens, report
                        )
                    messages.append({"role": "assistant", "content": resp.content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Validation FAILED (attempt {attempts}/"
                            f"{self.max_attempts}). Report:\n\n"
                            f"{report[:6000]}\n\n"
                            "Fix every [FAIL] finding and emit the complete "
                            "corrected file in <BENCHMARK_FILE> tags (you may "
                            "run more ```python inspection first if needed)."
                        ),
                    })
                    continue

                codes = _extract_codes(resp.content)
                if codes:
                    no_progress = 0
                    combined = "\n\n# --- next block ---\n\n".join(codes)
                    output = _execute_code(
                        combined, work_dir, self.python_bin,
                        timeout=self.exec_timeout_s,
                    )
                    messages.append({"role": "assistant", "content": resp.content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"**Execution output (turn {turn}/{self.max_turns}):"
                            f"**\n```\n{output}\n```"
                        ),
                    })
                    continue

                no_progress += 1
                if no_progress >= 4:
                    return AuthorResult(
                        False, self.out_path, attempts, turn, tokens,
                        last_report
                        or "model produced neither executable code nor a "
                           "<BENCHMARK_FILE> for 4 consecutive turns",
                    )
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Turn {turn}/{self.max_turns}. Nothing actionable "
                        "found in your reply. Either write inspection code in "
                        "a ```python fenced block, or emit the complete file "
                        "in <BENCHMARK_FILE> ... </BENCHMARK_FILE> tags."
                    ),
                })

        return AuthorResult(
            False, self.out_path, attempts, self.max_turns, tokens,
            last_report or "turn budget exhausted before a file was emitted",
        )
