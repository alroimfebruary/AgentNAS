"""Modularizer agent: refactors a Phase 1 seed Architecture for stage modularization.

Given an Architecture class whose stages are unrolled inside ``forward()`` (or
held as ``nn.ModuleList`` of repeated blocks), this agent rewrites the source so
that each stage is wrapped in its own ``nn.Module`` subclass with all of its
internal norm/act/attention/residual baked in. The top-level ``forward`` then
calls ``stem -> stage1(x) -> stage2(x) -> ... -> pool -> head`` only.

The contract: forward-pass behavior, parameter count, and tensor shapes are
preserved. Validation transplants the original module's state_dict onto the
modularized module and checks ``torch.allclose(rtol=1e-5, atol=1e-6)`` on a
fixed-seed input in eval mode. Up to ``max_retries`` (default 5) retries with
the validation failure fed back as feedback.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentnas.llm.client import LLMClient

logger = logging.getLogger(__name__)


_MODULARIZER_SYSTEM_PROMPT = """\
You are a Modularizer agent. You refactor a PyTorch Architecture's source code \
so that each repeated stage is wrapped in its own self-contained nn.Module \
subclass. The forward-pass behavior must be IDENTICAL to the original \
(bit-equivalent under torch.allclose(rtol=1e-5, atol=1e-6) in eval mode after \
the original's state_dict is transplanted into the refactored module).

## What you must produce

A complete Python source file that defines:

1. All helper classes (RegNetBlock, FNOBlock, etc.) the original used. Copy \
   them VERBATIM — do not rename, do not change ordering, do not modify their \
   layer construction or forward.

2. **NEW per-stage Stage classes** (e.g., `Stage1`, `Stage2`, `Stage3`, …) \
   that each wrap one repeated stage of the network. Each Stage class must:
   - Have a docstring that explicitly states the contract, e.g.,
     `'''INTERNAL: BatchNorm + ReLU + SE + residual baked in. Outer norm/act/skip slots SHOULD be Identity.'''`
   - In `__init__`, instantiate the same blocks/parameters in the SAME order \
     and with the SAME hyperparameters as the original stage construction. \
     (This preserves init-order and weight-shape ordering.)
   - In `forward(self, x)`, perform exactly the same operations the original \
     did for this stage — INCLUDING any per-stage outer norm, activation, or \
     residual that was previously written in the top-level `Architecture.forward`.
   - Have a single tensor in / single tensor out signature.

3. The **rewritten `Architecture` class** with:
   - The SAME helper attributes (`stem`, `pool`, `head`, etc.) as the original, \
     constructed identically.
   - Each repeated stage replaced by a single named `self.stageN = StageN()` \
     attribute (instead of `self.stages = nn.ModuleList([...])`).
   - `Architecture.forward(self, x)` simplified to:
     ```
     # ... data augmentation block (UNCHANGED, copy verbatim) ...
     x = self.stem(x)
     x = self.stage1(x)
     x = self.stage2(x)
     x = self.stage3(x)
     # ... etc ...
     x = self.pool(x).flatten(1)
     return self.head(x)
     ```
     No outer per-stage norm / act / skip operations remain in the top-level forward.

## Hard constraints (any violation rejects the output)

- **Do not add or remove parameters.** Total `sum(p.numel() for p in model.parameters())` \
  must be identical to the original.
- **Do not change op types.** No swapping ReLU ↔ GELU, BN ↔ LN, Conv2d ↔ DWConv, etc.
- **Do not change op order.** Inside each stage, the sequence of operations must \
  match what the original produced when its top-level forward and helper-block \
  forward were inlined.
- **Do not touch the data-augmentation block.** Any `if self.training: ...` block at \
  the top of `Architecture.forward` must be copied verbatim.
- **Do not change the `_init_weights` logic.** Copy it verbatim. The order in which \
  submodules are constructed in `__init__` MUST be the same as the original — keep \
  helper blocks instantiated in the same iteration order — so that init RNG draws \
  align identically.
- **Do not introduce new imports** beyond what the original used.

## Output format

Return ONLY a single ```python ... ``` block containing the complete refactored \
source. No prose outside the code block.

## Original source

The original Phase 1 winning Architecture class is below. Refactor it.

```python
{original_code}
```

## Task description

{task_description}
"""


_VALIDATOR_SCRIPT = r'''
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

ORIG = sys.argv[1]
MOD = sys.argv[2]
INPUT_SHAPE = tuple(int(x) for x in sys.argv[3].split(","))
SEED = int(sys.argv[4])
REPORT_PATH = sys.argv[5]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flat_params(module):
    return [p for p in module.parameters()]


def _shapes(params):
    return sorted([tuple(p.shape) for p in params])


def _max_abs_diff(a, b):
    return float((a - b).abs().max().item())


def _strict_transplant(orig_module, mod_module):
    """Try strict state_dict load. Returns (ok, error_msg)."""
    try:
        mod_module.load_state_dict(orig_module.state_dict(), strict=True)
        return True, ""
    except (RuntimeError, KeyError) as e:
        return False, str(e)


def _ordered_transplant(orig_module, mod_module):
    """Fallback: copy params in flat order. Both modules must have the same
    sequence of parameter shapes (already checked). Iterates over the same
    sorted-by-name order and copies tensor-by-tensor when shapes match.

    This is a SHAPE-AND-ORDER transplant: the i-th parameter (by sorted name)
    of mod gets the i-th parameter of orig, provided shapes match. Combined
    with the modularizer prompt's requirement to preserve construction order,
    this should produce identical weight assignments without depending on
    state_dict key strings.
    """
    orig_named = sorted(orig_module.named_parameters(), key=lambda kv: kv[0])
    mod_named = sorted(mod_module.named_parameters(), key=lambda kv: kv[0])
    if len(orig_named) != len(mod_named):
        return False, f"param count mismatch: orig={len(orig_named)} vs mod={len(mod_named)}"
    with torch.no_grad():
        for (on, op), (mn, mp) in zip(orig_named, mod_named):
            if tuple(op.shape) != tuple(mp.shape):
                return False, (
                    f"shape mismatch at orig={on}{tuple(op.shape)} -> "
                    f"mod={mn}{tuple(mp.shape)}"
                )
            mp.data.copy_(op.data)
    # Also copy buffers (BN running stats etc.) by name+shape
    orig_buf = sorted(orig_module.named_buffers(), key=lambda kv: kv[0])
    mod_buf = sorted(mod_module.named_buffers(), key=lambda kv: kv[0])
    if len(orig_buf) == len(mod_buf):
        with torch.no_grad():
            for (on, ob), (mn, mb) in zip(orig_buf, mod_buf):
                if tuple(ob.shape) == tuple(mb.shape):
                    mb.data.copy_(ob.data)
    return True, ""


report = {
    "param_count_match": False,
    "shape_match": False,
    "transplant_strategy": "",
    "transplant_error": "",
    "max_abs_diff": None,
    "forward_equivalent": False,
    "n_params_orig": 0,
    "n_params_mod": 0,
    "error": "",
}


try:
    torch.manual_seed(SEED)
    orig_mod = _load("orig", ORIG).Architecture()
    torch.manual_seed(SEED)
    mod_mod = _load("mod", MOD).Architecture()

    n_orig = sum(p.numel() for p in orig_mod.parameters())
    n_mod = sum(p.numel() for p in mod_mod.parameters())
    report["n_params_orig"] = n_orig
    report["n_params_mod"] = n_mod
    report["param_count_match"] = (n_orig == n_mod)
    if not report["param_count_match"]:
        report["error"] = f"param count differs: orig={n_orig} vs mod={n_mod}"
        Path(REPORT_PATH).write_text(json.dumps(report, indent=2))
        sys.exit(0)

    s_orig = _shapes(_flat_params(orig_mod))
    s_mod = _shapes(_flat_params(mod_mod))
    report["shape_match"] = (s_orig == s_mod)
    if not report["shape_match"]:
        report["error"] = "sorted parameter shapes differ"
        Path(REPORT_PATH).write_text(json.dumps(report, indent=2))
        sys.exit(0)

    ok, err = _strict_transplant(orig_mod, mod_mod)
    if ok:
        report["transplant_strategy"] = "strict_state_dict"
    else:
        ok2, err2 = _ordered_transplant(orig_mod, mod_mod)
        if ok2:
            report["transplant_strategy"] = "ordered_named_params"
            report["transplant_error"] = f"strict failed: {err}"
        else:
            report["transplant_error"] = (
                f"strict failed: {err}; ordered failed: {err2}"
            )
            report["error"] = "weight transplant failed"
            Path(REPORT_PATH).write_text(json.dumps(report, indent=2))
            sys.exit(0)

    orig_mod.eval()
    mod_mod.eval()
    torch.manual_seed(SEED)
    x = torch.randn(2, *INPUT_SHAPE)
    with torch.no_grad():
        y_orig = orig_mod(x)
        y_mod = mod_mod(x)
    diff = _max_abs_diff(y_orig, y_mod)
    report["max_abs_diff"] = diff
    report["forward_equivalent"] = bool(torch.allclose(y_orig, y_mod, rtol=1e-5, atol=1e-6))
    if not report["forward_equivalent"]:
        report["error"] = f"forward outputs differ: max_abs_diff={diff:.3e}"

except Exception as e:
    report["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"

Path(REPORT_PATH).write_text(json.dumps(report, indent=2))
print(f"VALIDATION_DONE: {json.dumps(report)}")
'''


@dataclass
class ModularizationResult:
    """Result of one modularization attempt."""

    status: str  # "success" | "rejected"
    modularized_code: str = ""
    validation_report: dict[str, Any] = field(default_factory=dict)
    retries_used: int = 0
    tokens_used: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)


def _extract_code_block(text: str, language: str = "python") -> str | None:
    """Extract the first code block of the given language from LLM output."""
    pattern = rf"```{language}\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _validate_modularization(
    original_code: str,
    modularized_code: str,
    input_shape: tuple[int, ...],
    python_bin: str,
    seed: int = 0,
    timeout_s: int = 120,
) -> dict[str, Any]:
    """Validate forward-pass equivalence in a subprocess. Returns the report dict.

    Spawns a Python subprocess with CUDA_VISIBLE_DEVICES="" so validation runs on
    CPU and never collides with running training. Times out at 120s.
    """
    with tempfile.TemporaryDirectory(prefix="agentnas_modularize_") as tmpdir:
        tmpdir = Path(tmpdir)
        orig_path = tmpdir / "orig.py"
        mod_path = tmpdir / "mod.py"
        script_path = tmpdir / "validate.py"
        report_path = tmpdir / "report.json"

        orig_path.write_text(original_code)
        mod_path.write_text(modularized_code)
        script_path.write_text(_VALIDATOR_SCRIPT)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

        shape_str = ",".join(str(s) for s in input_shape)
        try:
            result = subprocess.run(
                [
                    python_bin,
                    str(script_path),
                    str(orig_path),
                    str(mod_path),
                    shape_str,
                    str(seed),
                    str(report_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "error": f"validation timed out (>{timeout_s}s)",
                "forward_equivalent": False,
            }

        if report_path.exists():
            try:
                return json.loads(report_path.read_text())
            except json.JSONDecodeError:
                pass

        # Fallback if subprocess died before writing report
        err = result.stderr or result.stdout or "unknown subprocess failure"
        return {
            "error": f"subprocess failed: {err[-1500:]}",
            "forward_equivalent": False,
        }


def _summarize_failure(report: dict[str, Any]) -> str:
    """Produce a concise human-readable failure message for LLM feedback."""
    if report.get("error"):
        return report["error"]
    if not report.get("param_count_match"):
        return (
            f"Parameter count mismatch: original={report.get('n_params_orig')} "
            f"vs modularized={report.get('n_params_mod')}. "
            "You added or removed parameters — the contract requires identical counts."
        )
    if not report.get("shape_match"):
        return (
            "Parameter shapes differ as a sorted multiset. The set of tensor shapes "
            "in the modularized module does not match the original — you've changed "
            "a layer's dimensions or replaced a layer with a different one."
        )
    if not report.get("forward_equivalent"):
        return (
            f"Forward output mismatch: max_abs_diff={report.get('max_abs_diff')}. "
            "After transplanting the original weights into your refactored module, "
            "the forward pass produced different outputs in eval mode. Check that "
            "your stage classes preserve op order, that you copied any per-stage "
            "outer norm/activation/skip from the top-level forward into the stage's "
            "forward, and that you did not omit any operations."
        )
    return "Unknown validation failure"


class ModularizerAgent:
    """LLM-driven agent that modularizes a Phase 1 seed Architecture.

    Single-shot interface: call ``run(original_code)`` once per seed. Internal
    retry loop drives the LLM up to ``max_retries`` times, feeding back each
    validation failure as a user message.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        task_name: str,
        input_shape: tuple[int, ...],
        max_retries: int = 5,
        python_bin: str = "",
        seed: int = 0,
        exec_timeout_s: int = 120,
    ):
        self.llm = llm_client
        self.task_name = task_name
        self.input_shape = input_shape
        self.max_retries = max_retries
        self.python_bin = python_bin or "python"
        self.seed = seed
        self.exec_timeout_s = exec_timeout_s

    def run(self, original_code: str) -> ModularizationResult:
        """Run the modularization loop. Returns ModularizationResult."""

        task_description = (
            f"Task: {self.task_name}, Input shape: (batch, "
            f"{', '.join(str(s) for s in self.input_shape)})"
        )
        system = _MODULARIZER_SYSTEM_PROMPT.format(
            original_code=original_code,
            task_description=task_description,
        )

        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": (
                    "Refactor the Architecture above per the contract. Wrap each "
                    "repeated stage in its own Stage class with internal norm/act/"
                    "skip baked in, simplify the top-level forward, and ensure "
                    "forward-pass equivalence. Return ONE ```python``` block."
                ),
            }
        ]

        tokens_used = 0
        last_report: dict[str, Any] = {}
        last_code = ""

        for attempt in range(1, self.max_retries + 1):
            logger.info(
                f"Modularizer attempt {attempt}/{self.max_retries} "
                f"(task={self.task_name})"
            )
            resp = self.llm.chat(
                messages,
                system=system,
                max_tokens=max(self.llm.max_tokens, 16384),
            )
            tokens_used += resp.total_tokens

            code = _extract_code_block(resp.content, "python")
            if not code:
                err = "No ```python block found in response."
                logger.warning(f"Modularizer attempt {attempt}: {err}")
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({
                    "role": "user",
                    "content": f"Error: {err} Return the full refactored Architecture in a ```python block.",
                })
                continue

            last_code = code
            report = _validate_modularization(
                original_code=original_code,
                modularized_code=code,
                input_shape=self.input_shape,
                python_bin=self.python_bin,
                seed=self.seed,
                timeout_s=self.exec_timeout_s,
            )
            last_report = report

            if report.get("forward_equivalent") and report.get("param_count_match"):
                logger.info(
                    f"Modularizer succeeded on attempt {attempt} "
                    f"(max_abs_diff={report.get('max_abs_diff')}, "
                    f"transplant={report.get('transplant_strategy')})"
                )
                return ModularizationResult(
                    status="success",
                    modularized_code=code,
                    validation_report=report,
                    retries_used=attempt,
                    tokens_used=tokens_used,
                    messages=messages + [{"role": "assistant", "content": resp.content}],
                )

            failure_msg = _summarize_failure(report)
            logger.warning(f"Modularizer attempt {attempt} validation failed: {failure_msg[:300]}")
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({
                "role": "user",
                "content": (
                    f"Validation FAILED:\n\n{failure_msg}\n\n"
                    "Re-emit the refactored Architecture, fixing the issue. The contract "
                    "(no param adds/removes, no op-type or op-order changes, identical "
                    "construction order, augmentation block untouched) must be respected."
                ),
            })

        logger.error(
            f"Modularizer failed after {self.max_retries} attempts "
            f"(task={self.task_name})"
        )
        return ModularizationResult(
            status="rejected",
            modularized_code=last_code,
            validation_report=last_report,
            retries_used=self.max_retries,
            tokens_used=tokens_used,
            messages=messages,
        )
