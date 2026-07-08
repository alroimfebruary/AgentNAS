"""Code Executor agent: translates architecture descriptions into runnable code.

The executor generates:
  (a) An nn.Module class (model.py)
  (b) A TrainingConfig dict (config.json)

It validates the model with a dummy forward pass and retries on failure (max 10 attempts).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agentnas.llm.client import LLMClient
from agentnas.training.config import TrainingConfig

logger = logging.getLogger(__name__)

_EXECUTOR_SYSTEM_PROMPT = """\
You are a Code Executor for a neural architecture search pipeline. Your job is to \
translate architecture descriptions into working PyTorch code.

## What you produce

1. **Model code**: A Python class named `Architecture` that extends `nn.Module`.
   - Must define `__init__(self)` and `forward(self, x)`.
   - `x` is a tensor of shape `(batch, {input_shape})`.
   - Output shape must be `(batch, {output_spec})`.
   - Import only `torch`, `torch.nn`, `torch.nn.functional`, and standard library.
   - **NO pre-trained weights.** Do NOT use `torchvision.models`, `timm`, or any \
pre-trained checkpoints. All weights must be randomly initialized. This is a \
neural architecture search benchmark — we evaluate architecture design, not transfer learning.

2. **Training config**: A JSON dict with training hyperparameters. See the \
"Training config fields honored by the harness" section below for the full schema \
of supported fields — use ONLY those fields, no others.

## Output format

Wrap your model code in ```python ... ``` and your config in ```json ... ```.

## Constraints
- Maximum {vram_mb} MB VRAM (GPU memory).
- **Proxy training**: Only {proxy_epoch_pct}% of proposed epochs are used. \
{time_limit}-minute GPU time hard cap.
- **Final training**: Full proposed epochs. 24-hour GPU time cap.
- Design the training recipe for full training — the pipeline handles proxy scaling automatically.
- You may ONLY fix bugs (shape mismatches, runtime errors). Do NOT change the \
architecture's functional design.
- **Loss function: {loss_info}**. Output raw logits — do NOT apply softmax, sigmoid, \
or any activation to the final output. The loss function handles this internally.

## Training recipe constraints — the harness has limited, fixed support

The training harness honors ONLY these values. Anything else is silently dropped:

- **optimizer**: one of `sgd`, `adam`, `adamw`. Unknown names fall back to AdamW.
- **scheduler**: one of `cosine`, `step`, `warmup_cosine`, `plateau`, `none`. Unknown \
names fall back to cosine. Use `scheduler_params` for `T_max`, `eta_min`, `step_size`, \
`gamma`, `patience`, `factor`. For `warmup_cosine`, set `warmup_epochs` at the top level.
- **augmentations** (config JSON list): only `"mixup"`, `"cutmix"`, `"random_erasing"` \
are applied by the harness. Any other aug name is logged-then-ignored. Use \
`augmentation_params` for their hyperparameters (e.g., alpha for mixup/cutmix, p for \
random_erasing).
- **Loss is task-fixed** (see above). Do NOT write a custom loss inside the model — \
the harness ignores any model-side loss and always uses the benchmark's loss. \
`label_smoothing` is the only loss-side knob the Planner can set.
- **Not supported** — mixed precision (amp/autocast/fp16/bf16), torch.compile, gradient \
accumulation. Do not include them in the model or config.

## Handling training-recipe requests the harness can't honor

If the Planner's recipe asks for an augmentation outside `{{mixup, cutmix, random_erasing}}` \
(e.g., time_shift, gaussian_noise, horizontal_flip, random_crop, rotation), the correct \
way to honor it is to implement it INLINE inside `Architecture.forward()` **gated on \
`self.training`**, using only `torch` / `torch.nn.functional` ops on the batched tensor. \
Example skeleton:

```python
def forward(self, x):
    if self.training:
        # time shift along last axis, in-batch
        shift = int(torch.randint(-4, 5, (1,)).item())
        x = torch.roll(x, shifts=shift, dims=-1)
        # gaussian noise
        x = x + torch.randn_like(x) * 0.01
    # ... rest of architecture
    return logits
```

If the Planner requests a custom optimizer, scheduler, loss, or "amp/compile/grad-accum", \
IGNORE those items — pick the closest supported value and set the training config JSON \
accordingly. Do not try to instantiate unsupported training-framework components; the \
harness will silently override them.

## Training config fields honored by the harness

```json
{{
  "optimizer": "adamw",              // one of sgd/adam/adamw
  "lr": 0.001,
  "weight_decay": 0.01,
  "momentum": 0.9,                   // SGD only
  "betas": [0.9, 0.999],             // Adam/AdamW only
  "scheduler": "cosine",             // one of cosine/step/warmup_cosine/plateau/none
  "scheduler_params": {{}},
  "warmup_epochs": 0,                // warmup_cosine only
  "batch_size": 32,
  "epochs": 50,
  "augmentations": [],               // subset of mixup/cutmix/random_erasing
  "augmentation_params": {{}},
  "grad_clip": 1.0,
  "label_smoothing": 0.0
}}
```

## Task
{task_description}
"""


def _extract_code_block(text: str, language: str = "python") -> str | None:
    """Extract the first code block of the given language from LLM output."""
    pattern = rf"```{language}\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_json_block(text: str) -> dict | None:
    """Extract JSON config from LLM output."""
    code = _extract_code_block(text, "json")
    if code:
        try:
            return json.loads(code)
        except json.JSONDecodeError:
            pass
    # Fallback: find balanced {...} containing "optimizer" (handles nested objects)
    for i, c in enumerate(text):
        if c == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                if depth == 0:
                    candidate = text[i : j + 1]
                    if '"optimizer"' in candidate:
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            pass
                    break
    return None


def _validate_model_code(
    model_code: str,
    input_shape: tuple[int, ...],
    task_name: str,
    data_root: str,
    python_bin: str,
) -> tuple[bool, str]:
    """Validate model by running a dummy forward pass in a subprocess on CPU.

    Uses CPU to avoid conflicting with GPU training on any device.
    Returns (success, error_message).
    """
    with tempfile.TemporaryDirectory(prefix="agentnas_validate_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Write model code
        (tmpdir / "model.py").write_text(model_code)

        # Write validation script — always CPU, no GPU needed
        val_script = f"""
import sys
sys.path.insert(0, "{tmpdir}")
import torch
import importlib.util

spec = importlib.util.spec_from_file_location("model", "{tmpdir}/model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

model = module.Architecture()
device = torch.device("cpu")
model = model.to(device)
dummy = torch.randn(1, {", ".join(str(s) for s in input_shape)}, device=device)
with torch.no_grad():
    out = model(dummy)
print(f"OUTPUT_SHAPE={{out.shape}}")
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"PARAMS={{n_params}}")
print("VALIDATION_OK")
"""
        (tmpdir / "validate.py").write_text(val_script)

        # No GPU env — run on CPU only
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
        try:
            result = subprocess.run(
                [python_bin, str(tmpdir / "validate.py")],
                capture_output=True, text=True, timeout=60, env=env,
            )
            if "VALIDATION_OK" in result.stdout:
                return True, result.stdout
            else:
                error = result.stderr or result.stdout or "Unknown validation error"
                return False, error[-2000:]  # Truncate long errors
        except subprocess.TimeoutExpired:
            return False, "Validation timed out (>60s)"
        except Exception as e:
            return False, str(e)


class CodeExecutor:
    """Translates architecture descriptions into runnable PyTorch code.

    For each architecture proposed by the Planner:
    1. LLM generates model code + training config
    2. Model is validated (forward pass)
    3. On failure: error fed back to LLM, retry (max 10)
    4. On success: returns model code + config
    """

    def __init__(
        self,
        llm_client: LLMClient,
        task_name: str,
        input_shape: tuple[int, ...],
        output_spec: dict[str, Any],
        vram_mb: int,
        time_limit_min: int,
        data_root: str,
        max_retries: int = 10,
        python_bin: str = "",
        proxy_epoch_pct: int = 30,
    ):
        self.llm = llm_client
        self.task_name = task_name
        self.input_shape = input_shape
        self.output_spec = output_spec
        self.vram_mb = vram_mb
        self.time_limit_min = time_limit_min
        self.data_root = data_root
        self.max_retries = max_retries
        self.python_bin = python_bin or "python"

        # Build output spec string
        if "num_classes" in output_spec:
            self._output_str = str(output_spec["num_classes"])
        else:
            # H-06: flat format "1, 85, 85" not nested "(1, 85, 85)"
            self._output_str = ", ".join(str(s) for s in output_spec["output_shape"])

        # B-07: Determine loss info to tell LLM about multi-label tasks
        task_type = output_spec.get("task_type", "")
        if task_type == "multi_label" or ("output_shape" in output_spec and output_spec.get("num_classes", 0) == 0):
            loss_info = "BCEWithLogitsLoss (multi-label). Output raw logits, NO softmax/sigmoid."
        elif "output_shape" in output_spec:
            loss_info = "Regression loss (MSE or relative L2). Output raw predictions."
        else:
            loss_info = "CrossEntropyLoss (classification). Output raw logits, NO softmax."

        self._system = _EXECUTOR_SYSTEM_PROMPT.format(
            input_shape=", ".join(str(s) for s in input_shape),
            output_spec=self._output_str,
            vram_mb=vram_mb,
            time_limit=time_limit_min,
            proxy_epoch_pct=proxy_epoch_pct,
            task_description=f"Task: {task_name}",
            loss_info=loss_info,
        )

        self.total_tokens = 0

    def generate(
        self,
        arch_description: str,
        training_recipe: str,
        past_evals: list[dict] | None = None,
    ) -> tuple[str | None, TrainingConfig | None, str, int]:
        """Generate and validate model code for an architecture.

        Args:
            arch_description: Architecture description from the Planner.
            training_recipe: Training recipe instructions from the Planner.
            past_evals: Optional list of prior evaluations (dicts with keys
                ``arch_id``, ``code``, ``metric``, ``status``). When provided,
                their code is included as context so the Executor can modify
                specific prior architectures the description refers to instead
                of reinventing from the shape signature alone.

        Returns:
            (model_code, training_config, status, tokens_used)
            model_code is None if all retries failed.
            status is "success" or "rejected".
        """
        past_block = ""
        if past_evals:
            code_parts = []
            for e in past_evals:
                if not e.get("code"):
                    continue
                metric = e.get("metric")
                metric_str = f"{metric:.4f}" if isinstance(metric, (int, float)) and metric != float("inf") else "N/A"
                code_parts.append(
                    f"### {e['arch_id']} (status={e.get('status', '?')}, metric={metric_str})\n"
                    f"```python\n{e['code']}\n```"
                )
            if code_parts:
                past_block = (
                    "## Recent Architectures (code for reference)\n\n"
                    "The Planner's description may reference these by `arch_NNN` — use them as the baseline to modify.\n\n"
                    + "\n\n".join(code_parts) + "\n\n"
                )

        messages = [
            {
                "role": "user",
                "content": (
                    f"{past_block}"
                    f"## Architecture to implement\n\n{arch_description}\n\n"
                    f"## Training recipe\n\n{training_recipe}\n\n"
                    "Produce the `Architecture` class and training config JSON."
                ),
            }
        ]

        tokens_used = 0

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"Code Executor attempt {attempt}/{self.max_retries}")

            resp = self.llm.chat(messages, system=self._system)
            tokens_used += resp.total_tokens

            # Extract code and config
            model_code = _extract_code_block(resp.content, "python")
            config_dict = _extract_json_block(resp.content)

            if not model_code:
                error_msg = "No Python code block found in response."
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({
                    "role": "user",
                    "content": f"Error: {error_msg}\nPlease provide the Architecture class in a ```python block.",
                })
                continue

            # Parse training config (use defaults if missing)
            try:
                config = TrainingConfig.from_dict(config_dict) if config_dict else TrainingConfig()
            except Exception:
                config = TrainingConfig()

            # Validate model
            ok, validation_output = _validate_model_code(
                model_code, self.input_shape, self.task_name,
                self.data_root, self.python_bin,
            )

            if ok:
                logger.info(f"Validation passed on attempt {attempt}")
                self.total_tokens += tokens_used
                return model_code, config, "success", tokens_used

            # Feed error back
            logger.warning(f"Validation failed (attempt {attempt}): {validation_output[:200]}")
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({
                "role": "user",
                "content": (
                    f"The model failed validation with this error:\n\n"
                    f"```\n{validation_output}\n```\n\n"
                    "Please fix the code. Only fix bugs — do NOT change the architecture design."
                ),
            })

        # All retries exhausted
        logger.error(f"Code Executor failed after {self.max_retries} attempts")
        self.total_tokens += tokens_used
        return None, None, "rejected", tokens_used

    def generate_slot_module(
        self,
        class_name: str,
        slot_description: str,
        alternative_description: str,
        input_shape_at_slot: tuple[int, ...],
        output_shape_at_slot: tuple[int, ...],
        interface: str = "single",
        residual_shape: tuple[int, ...] | None = None,
    ) -> tuple[str, str, int]:
        """Generate a self-contained nn.Module class for a slot alternative.

        The LLM produces a complete class that:
        - Is named `class_name` (e.g., SLOT_block0)
        - interface="single": forward(self, x) — single tensor in/out
        - interface="dual": forward(self, x, residual) — two inputs, one output

        Validation: instantiate module, forward pass with dummy tensor(s).

        Returns:
            (module_code, status, tokens_used)
            module_code is empty string if rejected.
        """
        in_shape_str = ", ".join(str(s) for s in input_shape_at_slot)
        out_shape_str = ", ".join(str(s) for s in output_shape_at_slot)
        is_dual = interface == "dual"
        res_shape_str = ", ".join(str(s) for s in residual_shape) if residual_shape else in_shape_str

        # Build forward signature based on interface type
        if is_dual:
            fwd_sig = "def forward(self, x, residual)"
            fwd_comment = (
                f"        # x shape: (batch, {in_shape_str})\n"
                f"        # residual shape: (batch, {res_shape_str})\n"
                f"        # return shape: (batch, {out_shape_str})"
            )
            interface_note = (
                f"This is a **dual-input** slot. `forward(self, x, residual)` receives:\n"
                f"- `x`: the transformed tensor from the main path, shape (batch, {in_shape_str})\n"
                f"- `residual`: the original input before transformation, shape (batch, {res_shape_str})\n"
                f"Your module combines them and returns shape (batch, {out_shape_str})."
            )
        else:
            fwd_sig = "def forward(self, x)"
            fwd_comment = (
                f"        # x shape: (batch, {in_shape_str})\n"
                f"        # return shape: (batch, {out_shape_str})"
            )
            interface_note = (
                f"This is a **single-input** slot. `forward(self, x)` receives a tensor "
                f"of shape (batch, {in_shape_str}) and must return shape (batch, {out_shape_str})."
            )

        system = (
            "You are implementing a PyTorch nn.Module for a neural architecture search slot.\n\n"
            f"## What to produce\n"
            f"A complete Python class named `{class_name}` that extends `nn.Module`.\n\n"
            f"```python\n"
            f"class {class_name}(nn.Module):\n"
            f"    def __init__(self):\n"
            f"        super().__init__()\n"
            f"        # ... your layers here ...\n"
            f"\n"
            f"    {fwd_sig}:\n"
            f"{fwd_comment}\n"
            f"        return x\n"
            f"```\n\n"
            f"## Interface\n{interface_note}\n\n"
            f"## Slot: {slot_description}\n"
            f"## Alternative: {alternative_description}\n\n"
            f"## Constraints\n"
            f"- Import only `torch`, `torch.nn`, `torch.nn.functional`.\n"
            f"- **NO pre-trained weights.** Random initialization only.\n"
            f"- The class must be fully self-contained.\n"
            f"- Wrap your code in a ```python block.\n"
        )

        messages = [{"role": "user", "content": f"Implement `{class_name}` for: {alternative_description}"}]
        tokens_used = 0

        for attempt in range(1, self.max_retries + 1):
            resp = self.llm.chat(messages, system=system)
            tokens_used += resp.total_tokens

            module_code = _extract_code_block(resp.content, "python")
            if not module_code:
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": "No ```python block found. Provide the class."})
                continue

            # Ensure class name matches (H-03 fix: use regex to replace, not prepend)
            if f"class {class_name}" not in module_code:
                module_code = re.sub(
                    r"^class \w+", f"class {class_name}",
                    module_code, count=1, flags=re.MULTILINE,
                )

            # Build validation wrapper based on interface type
            if is_dual:
                test_code = (
                    "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n"
                    f"{module_code}\n\n"
                    f"class Architecture(nn.Module):\n"
                    f"    def __init__(self):\n"
                    f"        super().__init__()\n"
                    f"        self.slot = {class_name}()\n"
                    f"    def forward(self, x):\n"
                    f"        residual = torch.randn_like(x[:, :{residual_shape[0] if residual_shape else input_shape_at_slot[0]}])\n"
                    f"        return self.slot(x, residual)\n"
                )
            else:
                test_code = (
                    "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n"
                    f"{module_code}\n\n"
                    f"class Architecture(nn.Module):\n"
                    f"    def __init__(self):\n"
                    f"        super().__init__()\n"
                    f"        self.slot = {class_name}()\n"
                    f"    def forward(self, x):\n"
                    f"        return self.slot(x)\n"
                )

            ok, validation_output = _validate_model_code(
                test_code, input_shape_at_slot, self.task_name,
                self.data_root, self.python_bin,
            )

            if ok:
                logger.info(f"Module '{alternative_description}' validated (attempt {attempt})")
                self.total_tokens += tokens_used
                return module_code, "success", tokens_used

            logger.warning(f"Module validation failed (attempt {attempt}): {validation_output[:200]}")
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({
                "role": "user",
                "content": (
                    f"The module failed validation:\n```\n{validation_output}\n```\n"
                    f"Fix it. Class: `{class_name}`, forward signature: `{fwd_sig}`."
                ),
            })

        logger.error(f"Module generation failed for '{alternative_description}'")
        self.total_tokens += tokens_used
        return "", "rejected", tokens_used
