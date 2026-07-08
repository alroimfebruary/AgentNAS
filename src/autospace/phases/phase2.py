"""Phase 2: Slot-Based Grammar Generation (Module Approach).

1. Grammar Planner (LLM) reads seed → identifies slots with shapes and alternatives
2. Code Executor generates a self-contained nn.Module class per alternative
3. Output: SlottedArchitecture where realize() = concatenate modules + scaffold
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from autospace.agents.executor import CodeExecutor
from autospace.benchmarks.registry import get_benchmark
from autospace.config import PipelineConfig
from autospace.grammar.slots import Slot, SlotAlternative, SlottedArchitecture
from autospace.llm.client import LLMClient
from autospace.llm.logger import LLMLogger
from autospace.utils.state import StateManager

logger = logging.getLogger(__name__)


def _find_dead_slots(scaffold_code: str, slot_class_names: list[str]) -> list[str]:
    """Return SLOT class names whose bound attribute is never used by the class
    that holds it. Such slots are dead code — evolution mutates them but the
    forward pass ignores their output.

    Walks every class in the scaffold (not just `Architecture`), because scaffolds
    often define helper classes (e.g. `PerChannelCNN`, `EncoderBlock`) that hold
    the slots while `Architecture` just composes those helpers. A slot is "live"
    if there exists some class C where:

      1. `self.<attr> = SLOT_X(...)` appears in C's `__init__`
      2. `self.<attr>` is referenced in any non-`__init__` method of C

    `slot_class_names` should be the SLOT_X class names (e.g. "SLOT_stage1_op"),
    NOT the JSON slot ids — the JSON slot id and the scaffold attribute name are
    not always the same string (the scaffold may bind via `self.s1_op = SLOT_stage1_op()`).
    """
    try:
        tree = ast.parse(scaffold_code)
    except SyntaxError:
        return []

    live: set[str] = set()

    for cls_node in ast.walk(tree):
        if not isinstance(cls_node, ast.ClassDef):
            continue

        # Collect SLOT bindings from this class's __init__: {attr_name: {slot_class, ...}}
        attr_to_slots: dict[str, set[str]] = {}
        used_attrs: set[str] = set()
        for item in cls_node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            if item.name == "__init__":
                for stmt in ast.walk(item):
                    if not isinstance(stmt, ast.Assign):
                        continue
                    rhs = stmt.value
                    if not (isinstance(rhs, ast.Call)
                            and isinstance(rhs.func, ast.Name)
                            and rhs.func.id.startswith("SLOT_")):
                        continue
                    slot_cls = rhs.func.id
                    for target in stmt.targets:
                        if (isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"):
                            attr_to_slots.setdefault(target.attr, set()).add(slot_cls)
            else:
                # Any non-__init__ method's `self.<attr>` references count as "used".
                for node in ast.walk(item):
                    if (isinstance(node, ast.Attribute)
                            and isinstance(node.value, ast.Name)
                            and node.value.id == "self"):
                        used_attrs.add(node.attr)

        for attr, slot_classes in attr_to_slots.items():
            if attr in used_attrs:
                live |= slot_classes

    return [cls for cls in slot_class_names if cls not in live]

# Phase 1.5 modularization companion intervention (variant C of
# MODULARIZATION_STRESS_TEST.md §3.5). Appended to the Grammar Planner system
# prompt when <exp_dir>/../phase1_5_modularize/config.json has phase2_nudge=true.
_MODULARIZATION_NUDGE = """\

## Recognizing self-contained stage modules

If the seed architecture defines a stage as a self-contained nn.Module subclass \
(e.g., class Stage1(nn.Module)), and the docstring or class structure indicates \
that the block already contains its own normalization, activation, and residual \
connection, then:
  - Create ONE op slot per stage (the stage as a whole)
  - Create OUTER norm / act / skip slots BUT mark their alt0 explicitly as Identity \
with the description "Identity (already inside the stage block)"
  - Provide non-Identity alternatives for the outer slots so NAS can ADD an outer \
norm/act if useful, but the seed pick is Identity.
DO NOT unwrap the stage into multiple sub-slots that duplicate functionality \
already inside the block.
"""


_GRAMMAR_PLANNER_SYSTEM = """\
You are a Grammar Planner for neural architecture search. Your job is to design \
a rich, domain-aware search space around a seed architecture.

## What you produce

A JSON object with `scaffold_code` and `slots`.

### scaffold_code
The main Architecture class with searchable components replaced by slot modules:
- `self.xxx = SLOT_xxx()` in __init__ — **NO constructor arguments.** Always `SLOT_xxx()`.
- `x = self.xxx(x)` in forward — **single tensor in, single tensor out.**
- Everything else (stem, final pooling, classification head) stays fixed.

**CRITICAL rules for slot modules:**
- NO constructor arguments. Always `SLOT_xxx()` with no args.
- Dimensions are hardcoded inside each module based on the slot's input/output shape.
- Two interface types exist:
  - **"single"**: `forward(self, x)` — one tensor in, one out. Used for ops, norms, activations, attention.
  - **"dual"**: `forward(self, x, residual)` — two tensors in, one out. Used for skip/compose slots.
- The scaffold must call single slots as `x = self.slot(x)` and dual slots as `x = self.slot(x, residual)`.
- **DO NOT include `class SLOT_xxx` definitions in `scaffold_code`.** The scaffold should
  ONLY contain the `Architecture` class (and any non-slot helper classes it needs). The
  actual `SLOT_xxx` class bodies are inserted automatically before the scaffold during
  assembly. Adding stub `class SLOT_xxx(nn.Module): def forward(self, x): return x`
  forward declarations will silently override the real implementations and cause every
  evolved architecture to collapse to identity-only forward passes.
- **Every slot you instantiate in `__init__` must be called from `forward()` (or a
  helper method that `forward()` invokes).** Unused slots are dead code and will fail
  validation.

### slots
Each slot is a **fine-grained** decision point. You should define **10-20 slots** \
across multiple categories:

**Per-stage operation slots** (for each stage in the network):
- `stageN_op`: The main operation (conv variants, attention, novel domain ops)
- `stageN_norm`: Normalization type
- `stageN_act`: Activation function
- `stageN_attn`: Optional attention mechanism (or identity)
- `stageN_pool`: Pooling / downsampling method (if applicable)

**Cross-cutting parameter slots** (applied globally or per-stage):
- `width_stageN`: Channel width choices (e.g., 32 vs 64 vs 128). Each alternative \
is a full module with that specific width.
- `kernel_stageN`: Kernel size choices (e.g., 3x3 vs 5x5 vs 7x1+1x7)

Skip connections and composition can be implemented either:
- **Inside op modules** (e.g., a "ResBlock with additive skip" handles residual internally, `forward(self, x)`)
- **As separate dual-input slots** (e.g., `SLOT_stage1_skip` with `forward(self, x, residual)`, `interface: "dual"`)
Both approaches are valid. Use whichever gives the LLM more expressive search space.

## Designing good alternatives

For each slot, provide **4-8 alternatives** including:

1. **Seed choice** (always first) — what the seed architecture uses
2. **Domain-specific novel operations** — design NEW operations informed by the \
task's characteristics. Think about what domain structure the data has and what \
operations could exploit it. Examples:
   - For audio: frequency-band processing, multi-scale temporal convolution, \
axis-specific attention
   - For images: multi-scale feature extraction, deformable convolution
   - For 1D signals: dilated causal convolution, wavelet-inspired blocks
3. **Standard alternatives** — well-known ops: conv3x3, conv1x1, depthwise \
separable, squeeze-excite, self-attention
4. **Simpler alternatives** — identity (skip entirely) and zero (disable). \
These let NAS discover that some components are unnecessary ("subtractive insights").

## Slot JSON format

Single-input slot (ops, norms, activations, attention):
```json
{{
  "name": "stage1_op",
  "class_name": "SLOT_stage1_op",
  "description": "Main operation in stage 1.",
  "input_shape": [32, 16, 16],
  "output_shape": [32, 16, 16],
  "interface": "single",
  "seed_description": "Two 3x3 convs with BatchNorm and ReLU",
  "alternatives": [
    "Depthwise separable 3x3 conv with squeeze-excite",
    "Multi-scale block: parallel 1x1 + 3x3 + 5x5, project back",
    "Identity (pass-through)",
    "Zero (outputs zeros)"
  ]
}}
```

Dual-input slot (skip connections, composition — receives both x and residual):
```json
{{
  "name": "stage1_skip",
  "class_name": "SLOT_stage1_skip",
  "description": "How to combine transformed features with the residual.",
  "input_shape": [32, 16, 16],
  "output_shape": [32, 16, 16],
  "interface": "dual",
  "residual_shape": [32, 16, 16],
  "seed_description": "Additive residual: x + residual",
  "alternatives": [
    "Concatenation: cat(x, residual) -> 1x1 conv project",
    "Gated: g=sigmoid(conv(residual)); out = g*x + (1-g)*residual",
    "No skip: ignore residual, return x only"
  ]
}}
```

## Key principles
- **More slots = richer search.** 3 coarse layer slots is too few. Break each layer \
into operation + norm + activation + optional attention. 10-20 slots is the target.
- **Domain creativity matters.** Don't just list standard ConvNet blocks. Think about \
what makes this task unique and design operations that exploit that structure.
- **Include identity and zero** for EVERY slot. NAS should be able to discover that \
any component is unnecessary.
- **Input/output shapes must be EXACT.** They define the module interface contract.
- **The scaffold must be valid Python** (assuming all SLOT_xxx classes exist).
- **NO pre-trained weights.**

## Seed Architecture
```python
{seed_code}
```

## Seed Performance
{seed_metrics}

## Data Explorer Report
{explorer_report}

## Planner's Accumulated Insights
{planner_summary}

## Task
{task_description}
"""


def _get_ranked_seeds(
    phase1_state: StateManager, planner_state: dict, max_seeds: int,
) -> list[tuple[dict, float, str]]:
    """Return up to *max_seeds* ranked seed architectures from Phase 1.

    Returns list of ``(seed_dict, metric_value, source_label)`` sorted best-first.
    """
    val_final = phase1_state.get("val_final_results", [])
    successful_val_final = [e for e in val_final if e.get("status") == "success"]

    if successful_val_final:
        successful_val_final.sort(key=lambda e: e["val_final_metric"])
        return [
            (s, s["val_final_metric"], "val_final")
            for s in successful_val_final[:max_seeds]
        ]

    evaluations = planner_state.get("evaluations", [])
    successful = [e for e in evaluations if e.get("status") == "success"]
    if not successful:
        raise RuntimeError("No successful architectures in Phase 1")
    successful.sort(key=lambda e: e["metric"])
    return [
        (s, s["metric"], "proxy")
        for s in successful[:max_seeds]
    ]


_DOCSTRING_RE = re.compile(r'(""".*?"""|\'\'\'.*?\'\'\')', re.DOTALL)
_LINE_COMMENT_RE = re.compile(r'#.*$', re.MULTILINE)
_CLASS_HEADER_RE = re.compile(r'class\s+SLOT_\w+\s*\([^)]*\)\s*:', re.MULTILINE)


def _normalize_module_code(code: str) -> str:
    """Strip docstrings, line comments, the SLOT class header, and whitespace.

    Used by the dedup pass to detect alternatives whose ``module_code`` is
    byte-equivalent after ignoring formatting and paraphrased descriptions.
    """
    code = _DOCSTRING_RE.sub("", code)
    code = _LINE_COMMENT_RE.sub("", code)
    code = _CLASS_HEADER_RE.sub("", code)
    return re.sub(r"\s+", "", code)


def _dedupe_slot_alternatives(
    slot_name: str,
    class_name: str,
    slot_description: str,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    interface: str,
    residual_shape: tuple[int, ...] | None,
    alternatives: list[SlotAlternative],
    seed_alt_idx: int,
    executor: Any,
    max_retries: int = 3,
) -> tuple[list[SlotAlternative], int]:
    """Detect byte-equivalent ``module_code`` within a slot and re-prompt replacements.

    Strategy:
      1. SHA-256 hash each normalized ``module_code``.
      2. For each hash collision (size ≥ 2): keep the seed if it's in the
         group (else the first), re-prompt the executor for behaviorally
         distinct replacements for the rest.
      3. Up to ``max_retries`` re-prompt rounds. If a duplicate persists after
         all retries, drop it (the slot may shrink below ``max_alts``).
      4. The seed alternative is never re-prompted or dropped.

    Returns the deduplicated alternatives list and the (possibly updated)
    seed_alt_idx.
    """
    if len(alternatives) <= 1:
        return alternatives, seed_alt_idx

    drop_indices: set[int] = set()

    def _other_descs(skip_idx: int) -> str:
        return "; ".join(
            f"({i}) {alternatives[i].description[:160]}"
            for i in range(len(alternatives))
            if i != skip_idx and i not in drop_indices
        )

    for retry in range(max_retries + 1):
        groups: dict[str, list[int]] = {}
        for idx, alt in enumerate(alternatives):
            if idx in drop_indices:
                continue
            h = hashlib.sha256(_normalize_module_code(alt.module_code).encode()).hexdigest()
            groups.setdefault(h, []).append(idx)
        duplicate_groups = [g for g in groups.values() if len(g) >= 2]
        if not duplicate_groups:
            return alternatives, seed_alt_idx
        if retry == max_retries:
            for g in duplicate_groups:
                keeper = seed_alt_idx if seed_alt_idx in g else g[0]
                drop = [i for i in g if i != keeper]
                drop_indices.update(drop)
                logger.warning(
                    f"Slot '{slot_name}': dropping {len(drop)} duplicate alt(s) "
                    f"after {max_retries} retries (kept idx {keeper}, dropped {drop})"
                )
            break
        for g in duplicate_groups:
            keeper = seed_alt_idx if seed_alt_idx in g else g[0]
            for dup_idx in g:
                if dup_idx == keeper:
                    continue
                logger.info(
                    f"Slot '{slot_name}': alt {dup_idx} duplicates alt {keeper} "
                    f"(retry {retry + 1}/{max_retries}) — re-prompting"
                )
                distinct_desc = (
                    f"{alternatives[dup_idx].description} — but BEHAVIORALLY "
                    f"DISTINCT from existing alternatives. Must differ in "
                    f"operator type, capacity (>10% params), or norm/act choice. "
                    f"Already present in this slot: {_other_descs(dup_idx)}."
                )
                module_code, status, _ = executor.generate_slot_module(
                    class_name=class_name,
                    slot_description=slot_description,
                    alternative_description=distinct_desc,
                    input_shape_at_slot=in_shape,
                    output_shape_at_slot=out_shape,
                    interface=interface,
                    residual_shape=residual_shape,
                )
                if status == "success" and module_code:
                    # Keep the original alt's description — `distinct_desc` is a
                    # prompt-only instruction and should not be stored as metadata.
                    alternatives[dup_idx] = SlotAlternative(
                        name=alternatives[dup_idx].name,
                        description=alternatives[dup_idx].description,
                        module_code=module_code,
                    )

    if not drop_indices:
        return alternatives, seed_alt_idx

    new_alts: list[SlotAlternative] = []
    new_seed_idx = seed_alt_idx
    for i, alt in enumerate(alternatives):
        if i in drop_indices:
            if seed_alt_idx >= 0 and i < seed_alt_idx:
                new_seed_idx -= 1
            continue
        new_alts.append(alt)
    return new_alts, new_seed_idx


def _run_phase2_core(
    seed: dict,
    seed_metric: float,
    config: PipelineConfig,
    benchmark: Any,
    llm_client: LLMClient,
    planner_state: dict,
    python_bin: str,
) -> dict[str, Any]:
    """Execute a single Phase 2 attempt for one seed.

    Returns ``{"status": "success", "slotted_arch": ..., "validated_slots": ...}``
    on success, or ``{"status": "failed", "reason": "..."}`` on failure.
    State saving is the caller's responsibility.
    """
    # --- Step 1: Grammar Planner identifies slots ---
    logger.info("Step 1: Grammar Planner identifying slots...")

    explorer_report = planner_state.get("explorer_report", "(No data exploration report available)")
    planner_summaries = planner_state.get("summaries", [])
    _full_summary = "\n\n".join(planner_summaries) if planner_summaries else ""
    planner_summary = _full_summary or "(No summaries available)"

    task_label = "TASK_X" if config.blind_mode else config.task
    system = _GRAMMAR_PLANNER_SYSTEM.format(
        seed_code=seed.get("code", "(unavailable)"),
        seed_metrics=f"Metric: {seed.get('val_final_metric', seed.get('metric', float('inf'))):.4f}, Params: {seed.get('params', 'N/A'):,}",
        task_description=f"Task: {task_label}, Input: {benchmark.input_shape()}, Output: {benchmark.output_spec()}",
        explorer_report=explorer_report,
        planner_summary=planner_summary,
    )

    nudge_marker = Path(config.experiment_dir).parent / "phase1_5_modularize" / "config.json"
    if nudge_marker.exists():
        try:
            nudge_cfg = json.loads(nudge_marker.read_text())
            if nudge_cfg.get("phase2_nudge"):
                system = system + _MODULARIZATION_NUDGE
                logger.info(
                    "Phase 2: Modularization nudge ENABLED "
                    "(self-contained-stage-modules section appended to Grammar Planner system prompt)"
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Phase 2: Could not parse modularization marker {nudge_marker}: {e}")

    max_slots = config.budget.max_slots
    max_alts = config.budget.max_alternatives_per_slot
    resp = llm_client.chat(
        [{"role": "user", "content": (
            f"Analyze the seed architecture and the task domain. "
            f"Produce the JSON with scaffold_code and EXACTLY {max_slots} slots. "
            f"Each slot should have {max_alts - 1} alternatives (plus the seed choice). "
            f"The scaffold must ONLY reference the {max_slots} slots you define — "
            f"do not reference any SLOT_xxx that isn't in your slots list. "
            f"Include domain-specific novel operations, standard alternatives, "
            f"and identity/zero for each slot."
        )}],
        system=system,
        max_tokens=max(config.llm.max_tokens, 64000),
    )

    # Parse the JSON output
    grammar_def = _parse_grammar_json(resp.content)
    if not grammar_def:
        logger.error("Failed to parse Grammar Planner output")
        return {"status": "failed", "reason": "Grammar Planner produced no valid JSON"}

    scaffold_code = grammar_def.get("scaffold_code", "")
    slot_defs = grammar_def.get("slots", [])
    logger.info(f"Grammar Planner: {len(slot_defs)} slots, scaffold {len(scaffold_code)} chars")

    # --- Step 2: Generate nn.Module class per alternative ---
    logger.info("Step 2: Generating and validating slot modules...")

    executor = CodeExecutor(
        llm_client=llm_client,
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

    validated_slots: dict[str, Slot] = {}
    seed_choices: dict[str, int] = {}

    # Cap number of slots
    if len(slot_defs) > max_slots:
        logger.info(f"Capping slots from {len(slot_defs)} to {max_slots}")
        slot_defs = slot_defs[:max_slots]

    for slot_def in slot_defs:
        slot_name = slot_def["name"]
        class_name = slot_def.get("class_name", f"SLOT_{slot_name}")
        in_shape = tuple(slot_def["input_shape"])
        out_shape = tuple(slot_def["output_shape"])
        description = slot_def["description"]
        seed_desc = slot_def.get("seed_description", "")
        alt_descriptions = slot_def.get("alternatives", [])
        interface = slot_def.get("interface", "single")
        res_shape = tuple(slot_def["residual_shape"]) if slot_def.get("residual_shape") else None

        if seed_desc:
            all_descriptions = [seed_desc] + alt_descriptions
        else:
            all_descriptions = alt_descriptions
            logger.warning(f"Slot '{slot_name}' has no seed_description — first alternative used as seed")
        if len(all_descriptions) > max_alts:
            all_descriptions = all_descriptions[:max_alts]
        logger.info(f"Slot '{slot_name}' ({class_name}, {interface}): {len(all_descriptions)} alternatives, "
                     f"shape {in_shape} → {out_shape}")

        alternatives: list[SlotAlternative] = []
        seed_alt_idx = -1
        for i, alt_desc in enumerate(all_descriptions):
            is_seed = (i == 0 and bool(seed_desc))
            logger.info(f"  Alt {i}: '{alt_desc}' {'(seed)' if is_seed else ''}")

            module_code, status, tokens = executor.generate_slot_module(
                class_name=class_name,
                slot_description=description,
                alternative_description=alt_desc,
                input_shape_at_slot=in_shape,
                output_shape_at_slot=out_shape,
                interface=interface,
                residual_shape=res_shape,
            )

            if status == "success" and module_code:
                if is_seed:
                    seed_alt_idx = len(alternatives)  # Record actual index of seed
                alternatives.append(SlotAlternative(
                    name=f"{slot_name}_alt{i}",
                    description=alt_desc,
                    module_code=module_code,
                ))
            else:
                if is_seed:
                    logger.error(f"  SEED alternative REJECTED for slot '{slot_name}' — seed cannot be expressed!")
                else:
                    logger.warning(f"  Alt {i} REJECTED — dropped")

        if not alternatives:
            logger.error(f"Slot '{slot_name}' has no valid alternatives — skipping")
            continue

        # Step 2a: dedup pass — re-prompt or drop byte-equivalent alternatives.
        alternatives, seed_alt_idx = _dedupe_slot_alternatives(
            slot_name=slot_name,
            class_name=class_name,
            slot_description=description,
            in_shape=in_shape,
            out_shape=out_shape,
            interface=interface,
            residual_shape=res_shape,
            alternatives=alternatives,
            seed_alt_idx=seed_alt_idx,
            executor=executor,
        )

        # Default index: seed if it validated, otherwise 0 (first surviving alternative)
        default_idx = seed_alt_idx if seed_alt_idx >= 0 else 0

        validated_slots[slot_name] = Slot(
            name=slot_name,
            class_name=class_name,
            description=description,
            input_shape_at_slot=list(in_shape),
            output_shape_at_slot=list(out_shape),
            interface=interface,
            residual_shape=list(res_shape) if res_shape else None,
            alternatives=alternatives,
            default_idx=default_idx,
        )
        seed_choices[slot_name] = default_idx

    if not validated_slots:
        return {"status": "failed", "reason": "No valid slots produced"}

    # --- Step 2b: End-to-end validation of realized seed ---
    logger.info("Step 2b: Validating full realized seed architecture...")

    training_config = seed.get("training_config", {})
    # If no training config stored, build a reasonable default from seed's results
    if not training_config and seed.get("epochs_completed"):
        training_config = {"optimizer": "adamw", "lr": 1e-3, "batch_size": 32,
                           "epochs": 50, "scheduler": "cosine"}

    slotted = SlottedArchitecture(
        scaffold_code=scaffold_code,
        slots=validated_slots,
        training_config=training_config,
        seed_choices=seed_choices,
        original_seed_code=seed.get("code", ""),
    )

    # Validate realized seed via subprocess (H-04: safe, no in-process code execution)
    from autospace.agents.executor import _validate_model_code
    import re as _re

    _input_shape = benchmark.input_shape()
    e2e_ok = False
    max_repair = config.budget.max_cfg_retries
    realized_code = slotted.realize_seed()

    for repair_round in range(max_repair):
        ok, validation_output = _validate_model_code(
            realized_code, _input_shape, config.task,
            config.data_root or "", python_bin,
        )

        error_msg: str | None = None
        if not ok:
            error_msg = validation_output[:500]
            logger.warning(f"Realized seed validation failed (repair {repair_round+1}/{max_repair}): {error_msg}")
        else:
            # Subprocess validation passed — also enforce that every slot is wired
            # into the forward path. A scaffold can pass dummy-forward validation
            # while leaving some slots orphaned (instantiated but never called),
            # which silently shrinks the search space.
            slot_class_names = [s.class_name for s in validated_slots.values()]
            dead = _find_dead_slots(scaffold_code, slot_class_names)
            if dead:
                error_msg = (
                    f"These slot classes are either never instantiated in __init__, or "
                    f"are instantiated but bound to attributes that forward() (and methods "
                    f"reachable from it) never call: {dead}. Every slot must be invoked in "
                    f"the data flow — instantiate each as `self.<attr> = SLOT_xxx()` and "
                    f"call it as `x = self.<attr>(x)` (or `self.<attr>(x, residual)` for "
                    f"dual slots) somewhere on the forward path."
                )
                logger.warning(
                    f"Realized seed validation failed (repair {repair_round+1}/{max_repair}): "
                    f"{len(dead)} dead slots: {dead}"
                )
            else:
                logger.info("Realized seed validation OK (subprocess + reachability)")
                e2e_ok = True
                break

        # Ask LLM to fix the scaffold
        fix_resp = llm_client.chat(
            [{"role": "user", "content": (
                f"The realized architecture failed with this error:\n```\n{error_msg}\n```\n\n"
                f"Here is the scaffold code:\n```python\n{scaffold_code}\n```\n\n"
                f"Fix the scaffold_code so that the slot modules chain correctly. "
                f"The slot modules and their output shapes are:\n"
                + "\n".join(f"- {n}: {s.class_name}, in={s.input_shape_at_slot}, out={s.output_shape_at_slot}"
                            for n, s in validated_slots.items())
                + f"\n\nReturn ONLY the fixed scaffold_code in a ```python block."
            )}],
            system="You are fixing a neural architecture's scaffold code. Fix shape mismatches between slot modules.",
            max_tokens=16384,
        )
        fixed = _re.search(r"```python\s*\n(.*?)```", fix_resp.content, _re.DOTALL)
        if fixed:
            scaffold_code = fixed.group(1).strip()
            slotted.scaffold_code = scaffold_code
            realized_code = slotted.realize_seed()
        else:
            logger.error("Could not extract fixed scaffold from LLM response")
            break

    if not e2e_ok:
        logger.error(f"Scaffold repair failed after {max_repair} attempts. "
                      "Returning failed — scaffold is broken.")
        return {"status": "failed", "reason": "Scaffold end-to-end validation failed after repair attempts"}

    return {
        "status": "success",
        "slotted_arch": slotted,
        "validated_slots": validated_slots,
        "seed_choices": seed_choices,
    }


def run_phase2(config: PipelineConfig, resume: bool = True) -> dict[str, Any]:
    """Run Phase 2: Slot-Based Grammar Generation.

    Uses a two-level retry strategy:
    - Level 1: up to ``max_phase2_retries`` fresh Grammar Planner attempts per seed.
    - Level 2: up to ``max_seed_fallbacks`` seeds from Phase 1 (ranked best-first).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    exp_dir = Path(config.experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable
    llm_logger = LLMLogger(exp_dir / "llm_logs")
    llm_client = LLMClient.from_config(config.llm, logger=llm_logger.set_agent("grammar_planner"))  # M-01
    state_mgr = StateManager(exp_dir / "state.json")

    # Resume support: early-exit if completed, otherwise pick up at the next
    # un-tried (seed, attempt) so a mid-loop crash doesn't re-burn LLM tokens.
    seed_start = 0
    attempt_start = 1
    if resume and state_mgr.exists:
        state_mgr.load()
        if state_mgr.get("phase") == "completed":
            logger.info("Phase 2 already completed, returning saved results")
            return {
                "status": "success",
                "slotted_arch_path": state_mgr.get("slotted_arch_path", ""),
                "n_slots": state_mgr.get("n_slots", 0),
                "search_space_size": state_mgr.get("search_space_size", 0),
            }
        prev_seed_idx = state_mgr.get("seed_fallback_idx")
        prev_attempt = state_mgr.get("attempt_number")
        if prev_seed_idx is not None and prev_attempt is not None:
            seed_start = prev_seed_idx
            attempt_start = prev_attempt + 1
            logger.info(
                f"Resuming Phase 2 at seed_idx={seed_start}, attempt={attempt_start} "
                f"(prior failed attempt was seed_idx={prev_seed_idx}, attempt={prev_attempt})"
            )

    # Load Phase 1 state
    phase1_dir = _find_phase1_dir(config)
    if not phase1_dir:
        raise FileNotFoundError("Phase 1 results not found. Run Phase 1 first.")

    phase1_state = StateManager(phase1_dir / "state.json")
    phase1_state.load()
    planner_state = phase1_state.get("planner_state", {})

    # Get ranked seeds (best-first) for the fallback loop
    max_seeds = config.budget.max_seed_fallbacks
    ranked_seeds = _get_ranked_seeds(phase1_state, planner_state, max_seeds)

    benchmark = get_benchmark(config.task, data_root=config.data_root)
    max_retries = config.budget.max_phase2_retries

    # --- Two-level retry: seeds (outer) x attempts (inner) ---
    for seed_idx, (seed, seed_metric, source) in enumerate(ranked_seeds):
        if seed_idx < seed_start:
            continue
        logger.info(
            f"{'=' * 60}\n"
            f"Phase 2 seed {seed_idx + 1}/{len(ranked_seeds)}: "
            f"arch_id={seed['arch_id']}, {source} metric={seed_metric:.4f}\n"
            f"{'=' * 60}"
        )

        attempt_init = attempt_start if seed_idx == seed_start else 1
        for attempt in range(attempt_init, max_retries + 1):
            logger.info(
                f"--- Phase 2 attempt {attempt}/{max_retries} "
                f"(seed {seed_idx + 1}/{len(ranked_seeds)}: {seed['arch_id']}) ---"
            )

            result = _run_phase2_core(
                seed=seed,
                seed_metric=seed_metric,
                config=config,
                benchmark=benchmark,
                llm_client=llm_client,
                planner_state=planner_state,
                python_bin=python_bin,
            )

            if result["status"] == "success":
                slotted = result["slotted_arch"]
                validated_slots = result["validated_slots"]

                # --- Save ---
                arch_path = exp_dir / "slotted_arch.json"
                slotted.save(str(arch_path))

                total_alts = sum(s.n_alternatives for s in validated_slots.values())
                logger.info(
                    f"Phase 2 complete (seed {seed_idx + 1}, attempt {attempt}): "
                    f"{len(validated_slots)} slots, {total_alts} total alternatives, "
                    f"search space = {slotted.search_space_size}"
                )

                state_mgr.update({
                    "slotted_arch_path": str(arch_path),
                    "seed_arch_id": seed["arch_id"],
                    "seed_metric": seed_metric,
                    "n_slots": len(validated_slots),
                    "search_space_size": slotted.search_space_size,
                    "phase": "completed",
                    "seed_fallback_idx": seed_idx,
                    "attempt_number": attempt,
                })
                state_mgr.save()

                return {
                    "status": "success",
                    "slotted_arch_path": str(arch_path),
                    "n_slots": len(validated_slots),
                    "total_alternatives": total_alts,
                    "search_space_size": slotted.search_space_size,
                }

            logger.warning(
                f"Phase 2 attempt {attempt}/{max_retries} failed for seed "
                f"{seed['arch_id']}: {result.get('reason', 'unknown')}"
            )
            # Checkpoint after each failed attempt so resume picks up at the
            # next un-tried (seed, attempt) instead of restarting from (0, 1).
            state_mgr.update({
                "phase": "in_progress",
                "seed_fallback_idx": seed_idx,
                "attempt_number": attempt,
            })
            state_mgr.save()

        logger.warning(
            f"All {max_retries} Phase 2 attempts exhausted for seed "
            f"{seed['arch_id']}. Trying next seed..."
        )

    # All seeds x all retries exhausted
    total_chances = len(ranked_seeds) * max_retries
    logger.error(
        f"Phase 2 FAILED after {total_chances} total attempts "
        f"across {len(ranked_seeds)} seed(s)."
    )
    return {
        "status": "failed",
        "reason": (
            f"All {total_chances} Phase 2 attempts failed "
            f"({len(ranked_seeds)} seeds x {max_retries} retries)"
        ),
    }


def _parse_grammar_json(text: str) -> dict | None:
    """Extract and parse the JSON grammar definition from LLM output.

    The LLM may emit multiple JSON blocks in a single response — e.g. produce a
    first attempt, say "Wait, let me redo…", and emit a second improved block.
    Walks every candidate block (```json fences first, then bare balanced
    `{...}` regions) and accepts the LAST one that parses with the expected
    schema (must contain both ``scaffold_code`` and ``slots`` keys). If none
    match the schema, falls back to the first parseable JSON dict (legacy
    behavior) with a warning.
    """
    candidates: list[str] = []

    # Fenced ```json blocks (in order)
    for m in re.finditer(r"```json\s*\n(.*?)```", text, re.DOTALL):
        candidates.append(m.group(1))

    # Balanced top-level `{...}` regions, non-overlapping (N5 fix preserved)
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        end = -1
        for j in range(i, n):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            break  # unbalanced; no more candidates
        candidates.append(text[i : end + 1])
        i = end + 1

    n_blocks = len(candidates)

    # Walk back-to-front: accept the most recent block matching expected schema.
    for idx in range(len(candidates) - 1, -1, -1):
        try:
            obj = json.loads(candidates[idx])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "scaffold_code" in obj and "slots" in obj:
            if n_blocks >= 2:
                logger.info(
                    f"Grammar Planner emitted {n_blocks} parseable JSON blocks; "
                    f"using block {idx} (most recent matching schema, "
                    f"likely a model rewrite)."
                )
            return obj

    # Legacy fallback: first parseable JSON dict, even with incomplete schema.
    for idx, cand in enumerate(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            logger.warning(
                f"Grammar Planner output: no JSON block matched the expected "
                f"schema; falling back to block {idx}/{max(n_blocks - 1, 0)}."
            )
            return obj
    return None


def _has_planner_state(state_path: Path) -> bool:
    """Check if a state.json contains a COMPLETED Phase 1 (M-05 fix)."""
    try:
        import json as _json
        state = _json.loads(state_path.read_text())
        if "planner_state" not in state:
            return False
        # Prefer completed Phase 1 runs
        if state.get("phase") == "completed":
            return True
        # Accept incomplete if it has successful evaluations
        evals = state["planner_state"].get("evaluations", [])
        return any(e.get("status") == "success" for e in evals)
    except Exception:
        return False


def _find_phase1_dir(config: PipelineConfig) -> Path | None:
    exp_dir = Path(config.experiment_dir)
    # Check sibling "phase1" directory first (Phase 4 round structure)
    sibling = exp_dir.parent / "phase1" / "state.json"
    if sibling.exists() and _has_planner_state(sibling):
        return sibling.parent
    # Search experiments/ — validate that state contains planner_state (B6 fix)
    exp_root = Path("experiments")
    if exp_root.exists():
        for d in sorted(exp_root.iterdir(), reverse=True):
            if d.name.startswith(config.task + "_") or d.name == config.task:  # #9 strict match
                state_path = d / "state.json"
                if state_path.exists() and _has_planner_state(state_path):
                    return d
    return None
