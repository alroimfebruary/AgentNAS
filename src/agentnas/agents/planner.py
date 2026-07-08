"""Planner agent: the main controller for Phase 1 LLM iterative refinement.

The Planner maintains a stateful conversation, choosing actions:
  <EXPLORE>  — launch Data Explorer
  <EXPERIMENT> — propose architectures for parallel evaluation
  <STOP>     — conclude search, submit best for final training

Context is reconstructed (not a simple chat history) to stay within token limits.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from agentnas.llm.client import LLMClient

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = """\
You are the Planner agent for a neural architecture search pipeline. Your goal is to \
design the best possible neural network architecture for the given task.

## Available Actions

At each turn, choose ONE action:

1. **<EXPLORE>** purpose string here **</EXPLORE>**
   Launch the Data Explorer to analyze the dataset. Provide a purpose string.

2. **<EXPERIMENT>**
   Propose up to {n_gpu} architectures for parallel evaluation. For each, provide:
   - **<ARCH id="arch_NNN">**
     - **<DESCRIPTION>** High-level architecture description **</DESCRIPTION>**
     - **<REASONING>** Why this design, what you expect **</REASONING>**
     - **<TRAINING_RECIPE>** Training hyperparameters (optimizer, lr, scheduler, \
batch_size, epochs, augmentations, etc.) **</TRAINING_RECIPE>**
   - **</ARCH>**
   **</EXPERIMENT>**

   **Referencing past architectures (IMPORTANT):** The Executor that turns your \
description into runnable code does NOT see the architectures list — it only sees what \
you write in DESCRIPTION / REASONING / TRAINING_RECIPE plus any arch code you \
explicitly name. If your proposal is a modification of an earlier architecture, name \
it by its literal `arch_NNN` id in the DESCRIPTION (e.g. "arch_006 with wider \
channels (64→256→256)"). The pipeline regex-matches `arch_NNN` tokens and injects the \
referenced architecture's full code into the Executor's prompt as the baseline to \
modify. Without an explicit `arch_NNN` reference, the Executor will invent a new \
architecture from scratch that loosely fits your description — which breaks iterative \
refinement. If the proposal is genuinely new (no baseline), omit any arch reference \
and describe the architecture fully.

3. **<STOP>**
   End the search. Select top architectures for final full training.
   - **<FINAL_SELECTION>** List arch IDs for final training **</FINAL_SELECTION>**
   **</STOP>**

After receiving experiment results, produce:
- **<SUMMARY>** What worked, what failed, what was surprising, next direction **</SUMMARY>**
Then choose your next action.

## Constraints
- {vram_mb} MB VRAM limit (per GPU)
- **Proxy training** (search phase): The pipeline automatically runs only **{proxy_epoch_pct}%** \
of your proposed epochs. A {time_limit_min}-minute GPU time hard cap also applies.
- **Final training**: Uses your full proposed recipe. 24-hour GPU time cap.
- {budget_remaining} architecture evaluations remaining (of {budget_total})
- {patience_info}
- **NO pre-trained weights.** All models must be trained from scratch (random init).

## Performance Baseline
{sota_info}
Your metric is **lower = better**. Scores near 1.0 mean the model learned almost nothing. \
You must significantly beat random chance{sota_aim}.

## Critical: Training Recipe Design
Design your training recipe for **full training** (the final phase). The pipeline will \
automatically scale it to {proxy_epoch_pct}% epochs for proxy evaluation during search. \
The LR scheduler and warmup are also scaled proportionally.
- With ~{train_samples} training samples, propose enough total epochs that \
{proxy_epoch_pct}% is sufficient to see meaningful convergence for architecture ranking \
(e.g., 100 epochs → {proxy_epoch_pct} proxy epochs).

## What the training harness supports — DO NOT propose anything outside this list

This is a neural architecture search pipeline. The training recipe is a **fixed protocol** \
with limited degrees of freedom; your novelty should be in the architecture design, not \
in inventing new optimizers / schedulers / losses / augmentations.

**Optimizer** — choose exactly one of: `sgd`, `adam`, `adamw`. Anything else (lion, \
adafactor, rmsprop, …) will be silently replaced by AdamW.

**Scheduler** — choose exactly one of: `cosine`, `step`, `warmup_cosine`, `plateau`, `none`. \
Anything else (onecycle, exponential, polynomial, cyclic, lambda, …) will be silently \
replaced by cosine. Use `scheduler_params` for standard knobs (T_max, eta_min, step_size, \
gamma, patience, factor). For `warmup_cosine`, set `warmup_epochs` at the top level.

**Loss function** — FIXED per task and enforced by the harness. You cannot swap it \
(focal, class-weighted CE, dice, Tversky, custom regression loss — all silently dropped). \
Only two loss-side knobs actually apply: (a) `label_smoothing` (0.0-0.2) for \
CrossEntropyLoss tasks, (b) `augmentations: ["mixup"|"cutmix"]` for label-space blending. \
Class imbalance should be addressed via architecture/augmentation/recipe choices, \
not custom loss.

**Augmentations (harness-level)** — only `mixup`, `cutmix`, `random_erasing` are honored \
as training-config entries. Any other aug name (time_shift, gaussian_noise, \
horizontal_flip, rotation, random_crop, color_jitter, …) will be silently ignored by the \
harness. If you believe a task-specific aug (e.g., time-shift for EMG/ECG) is essential \
to the architecture's inductive bias, instruct the Executor to implement it inline in \
`Architecture.forward()` gated on `self.training`.

**Not supported AT ALL** — do NOT request any of: mixed precision / AMP / autocast / \
fp16 / bf16 / torch.compile / gradient accumulation / gradient checkpointing (unless \
the model implements it internally with `torch.utils.checkpoint`). The harness runs FP32 \
with standard forward/backward. Asking for these is a waste of recipe slots.

**LLM-tunable (safe knobs)** — `lr`, `weight_decay`, `batch_size`, `epochs`, `grad_clip`, \
`label_smoothing`, `momentum` (SGD only), `betas` (Adam/AdamW), `warmup_epochs`, \
`scheduler_params`, and choice from the approved optimizer/scheduler/aug sets above.

## Task
{task_readme}

## Data Explorer Report
{explorer_report}
"""


@dataclass
class ArchProposal:
    """A single architecture proposal from the Planner."""

    arch_id: str
    description: str
    reasoning: str
    training_recipe: str


@dataclass
class EvalRecord:
    """Record of a single architecture evaluation."""

    arch_id: str
    description: str  # 1-2 line summary
    metric: float
    params: int
    peak_vram_mb: float
    flops: int
    wall_time_s: float
    status: str  # success, rejected, oom, error, timeout
    code: str = ""  # Full model code (may be dropped for older rounds)
    iteration: int = 0
    error_message: str = ""
    training_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerState:
    """Full planner state for checkpoint/resume."""

    iteration: int = 0
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    explorer_report: str = ""
    budget_used: int = 0
    patience_counter: int = 0  # Consecutive evals with no improvement
    best_metric: float = float("inf")
    best_arch_id: str = ""
    arch_counter: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "evaluations": self.evaluations,
            "summaries": self.summaries,
            "explorer_report": self.explorer_report,
            "budget_used": self.budget_used,
            "patience_counter": self.patience_counter,
            "best_metric": self.best_metric,
            "best_arch_id": self.best_arch_id,
            "arch_counter": self.arch_counter,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PlannerState":
        # Filter to known fields only — tolerates extra keys from newer versions (H6 fix)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Planner:
    """The main controller for Phase 1 iterative architecture refinement."""

    def __init__(
        self,
        llm_client: LLMClient,
        task_readme: str,
        n_gpu: int,
        vram_mb: int,
        time_limit_min: int,
        budget_total: int,
        patience: int,
        sota_info: str = "No baseline available.",
        sota_available: bool = True,
        train_samples: str = "unknown",
        state: PlannerState | None = None,
        proxy_epoch_pct: int = 30,
    ):
        self.llm = llm_client
        self.task_readme = task_readme
        self.n_gpu = n_gpu
        self.vram_mb = vram_mb
        self.time_limit_min = time_limit_min
        self.budget_total = budget_total
        self.patience = patience
        self.sota_info = sota_info
        self.sota_available = sota_available
        self.train_samples = train_samples
        self.state = state or PlannerState()
        self.total_tokens = 0
        self.proxy_epoch_pct = proxy_epoch_pct

    def next_arch_id(self) -> str:
        self.state.arch_counter += 1
        return f"arch_{self.state.arch_counter:03d}"

    @property
    def budget_remaining(self) -> int:
        return self.budget_total - self.state.budget_used

    @property
    def should_force_stop(self) -> bool:
        return (
            self.state.patience_counter >= self.patience
            or self.budget_remaining <= 0
        )

    def update_after_eval(self, record: EvalRecord) -> None:
        """Update state after an architecture evaluation."""
        self.state.budget_used += 1
        self.state.evaluations.append({
            "arch_id": record.arch_id,
            "description": record.description,
            "metric": record.metric,
            "params": record.params,
            "peak_vram_mb": record.peak_vram_mb,
            "flops": record.flops,
            "wall_time_s": record.wall_time_s,
            "status": record.status,
            "code": record.code,
            "iteration": record.iteration,
            "error_message": record.error_message,
            "training_config": record.training_config,
        })

        if record.status == "success" and record.metric < self.state.best_metric:
            self.state.best_metric = record.metric
            self.state.best_arch_id = record.arch_id
            self.state.patience_counter = 0
        elif record.status == "success":
            # Only count successful evaluations toward patience (H5 fix)
            self.state.patience_counter += 1
        # Failed/rejected evals don't count toward patience

    def set_explorer_report(self, report: str) -> None:
        self.state.explorer_report = report

    def _build_leaderboard(self) -> str:
        """Build compact leaderboard table from all evaluations."""
        if not self.state.evaluations:
            return "(No architectures evaluated yet)"

        # Sort by metric (lower=better), successful first
        evals = sorted(
            self.state.evaluations,
            key=lambda e: (0 if e["status"] == "success" else 1, e["metric"]),
        )

        lines = ["| Rank | Arch ID | Description | Metric | Params | VRAM MB | Status |"]
        lines.append("|------|---------|-------------|--------|--------|---------|--------|")

        for i, e in enumerate(evals):
            rank = i + 1 if e["status"] == "success" else "-"
            metric_str = f"{e['metric']:.4f}" if e["status"] == "success" else "-"
            desc = e["description"].split("\n")[0]
            lines.append(
                f"| {rank} | {e['arch_id']} | {desc} | {metric_str} | "
                f"{e['params']:,} | {e['peak_vram_mb']:.0f} | {e['status']} |"
            )

        return "\n".join(lines)

    def _build_context_messages(self) -> list[dict[str, str]]:
        """Reconstruct the Planner's conversation context.

        Structure:
        1. Leaderboard (always)
        2. All summaries (always)
        3. Recent round details (last 2-3 iterations of full code)
        4. Prompt for next action
        """
        parts = []

        # Leaderboard
        parts.append(f"## Architecture Leaderboard\n\n{self._build_leaderboard()}")

        # All summaries
        if self.state.summaries:
            parts.append(
                "## Accumulated Summaries\n\n"
                + "\n\n---\n\n".join(
                    f"**Iteration {i+1}:**\n{s}"
                    for i, s in enumerate(self.state.summaries)
                )
            )

        # Recent round details (last 2-3 rounds of architecture code)
        recent_iters = set()
        for e in reversed(self.state.evaluations):
            recent_iters.add(e["iteration"])
            if len(recent_iters) >= 3:
                break

        recent_evals = [e for e in self.state.evaluations if e["iteration"] in recent_iters and e.get("code")]
        if recent_evals:
            detail_parts = []
            for e in recent_evals:
                detail_parts.append(
                    f"### {e['arch_id']} (iter {e['iteration']}, {e['status']})\n"
                    f"```python\n{e['code']}\n```"
                )
            parts.append("## Recent Architecture Code\n\n" + "\n\n".join(detail_parts))

        # Status
        status = (
            f"## Current Status\n"
            f"- Budget: {self.budget_remaining}/{self.budget_total} evaluations remaining\n"
            f"- Best: {self.state.best_arch_id} ({self.state.best_metric:.4f})\n"
            f"- Patience: {self.state.patience_counter}/{self.patience} "
            f"(consecutive no-improvement evaluations)\n"
            f"- Iteration: {self.state.iteration + 1}"
        )
        parts.append(status)

        if self.should_force_stop:
            parts.append(
                "⚠️ **You must STOP now.** "
                f"{'Budget exhausted.' if self.budget_remaining <= 0 else 'Patience exhausted.'} "
                "Issue <STOP> and select top architectures for final training."
            )
        elif not self.state.evaluations:
            parts.append(
                "**Action needed:** You have the data exploration report above. "
                "Now propose your first batch of architectures using <EXPERIMENT>. "
                "Propose up to {n_gpu} architectures.".format(n_gpu=self.n_gpu)
            )
        else:
            parts.append(
                "**Action needed:** Choose <EXPERIMENT> to propose new architectures, "
                "or <STOP> if you believe further iteration is unproductive."
            )

        content = "\n\n".join(parts)
        return [{"role": "user", "content": content}]

    def _build_system_prompt(self) -> str:
        patience_info = (
            f"Patience: {self.state.patience_counter}/{self.patience} "
            f"consecutive no-improvement evaluations. "
            f"{'⚠️ MUST STOP NOW.' if self.should_force_stop else ''}"
        )
        return _PLANNER_SYSTEM_PROMPT.format(
            n_gpu=self.n_gpu,
            vram_mb=self.vram_mb,
            time_limit_min=self.time_limit_min,
            proxy_epoch_pct=self.proxy_epoch_pct,
            budget_remaining=self.budget_remaining,
            budget_total=self.budget_total,
            patience_info=patience_info,
            task_readme=self.task_readme,
            explorer_report=self.state.explorer_report or "(No exploration done yet)",
            sota_info=self.sota_info,
            sota_aim=(
                " and aim toward the SOTA baseline above"
                if self.sota_available else ""
            ),
            train_samples=self.train_samples,
        )

    def get_action(self) -> tuple[str, Any]:
        """Get the Planner's next action.

        Returns:
            (action_type, action_data) where action_type is one of:
            - "explore": action_data = purpose string
            - "experiment": action_data = list[ArchProposal]
            - "stop": action_data = list[str] (arch IDs for final training)
        """
        system = self._build_system_prompt()
        messages = self._build_context_messages()

        resp = self.llm.chat(messages, system=system)
        self.total_tokens += resp.total_tokens

        return self._parse_action(resp.content)

    def feed_results(self, results_text: str) -> str:
        """Feed experiment results to the Planner and get its summary.

        Returns the summary text.
        """
        system = self._build_system_prompt()
        messages = self._build_context_messages()
        # Merge results into the last user message to avoid consecutive user messages (M-02 fix)
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += f"\n\n## Experiment Results\n\n{results_text}\n\nProvide your <SUMMARY>."
        else:
            messages.append({
                "role": "user",
                "content": f"## Experiment Results\n\n{results_text}\n\nProvide your <SUMMARY>.",
            })

        resp = self.llm.chat(messages, system=system)
        self.total_tokens += resp.total_tokens

        # Extract summary
        summary = self._extract_tag(resp.content, "SUMMARY") or resp.content
        self.state.summaries.append(summary)
        self.state.iteration += 1

        return summary

    def _parse_action(self, response: str) -> tuple[str, Any]:
        """Parse the Planner's response into an action."""

        # Check for STOP
        if "<STOP>" in response:
            final_ids = self._extract_tag(response, "FINAL_SELECTION") or ""
            arch_ids = re.findall(r"arch_\d+", final_ids)
            if not arch_ids:
                # Default: top N_GPU by metric
                successful = [
                    e for e in self.state.evaluations if e["status"] == "success"
                ]
                successful.sort(key=lambda e: e["metric"])
                arch_ids = [e["arch_id"] for e in successful[: self.n_gpu]]
            return "stop", arch_ids

        # Check for EXPLORE
        explore = self._extract_tag(response, "EXPLORE")
        if explore:
            return "explore", explore

        # Check for EXPERIMENT
        if "<EXPERIMENT>" in response:
            proposals = self._parse_proposals(response)
            return "experiment", proposals

        # Fallback: treat as experiment request
        logger.warning("Could not parse Planner action, treating as experiment request")
        proposals = self._parse_proposals(response)
        if proposals:
            return "experiment", proposals
        return "explore", "Analyze the dataset to inform architecture design."

    def _parse_proposals(self, response: str) -> list[ArchProposal]:
        """Parse architecture proposals from Planner response."""
        proposals = []

        # Find all <ARCH> blocks
        arch_blocks = re.findall(
            r"<ARCH\s+id=\"(.*?)\">(.*?)</ARCH>", response, re.DOTALL
        )

        if not arch_blocks:
            # Fallback: try to find any architecture descriptions
            desc = self._extract_tag(response, "DESCRIPTION")
            if desc:
                proposals.append(ArchProposal(
                    arch_id=self.next_arch_id(),
                    description=desc,
                    reasoning=self._extract_tag(response, "REASONING") or "",
                    training_recipe=self._extract_tag(response, "TRAINING_RECIPE") or "",
                ))
            return proposals

        for arch_id, block in arch_blocks:
            proposals.append(ArchProposal(
                arch_id=self.next_arch_id(),  # Always auto-generate, ignore LLM IDs (2.2 fix)
                description=self._extract_tag(block, "DESCRIPTION") or block.strip(),
                reasoning=self._extract_tag(block, "REASONING") or "",
                training_recipe=self._extract_tag(block, "TRAINING_RECIPE") or "",
            ))

        # Limit to N_GPU
        return proposals[: self.n_gpu]

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str | None:
        """Extract content between <TAG> and </TAG>."""
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
