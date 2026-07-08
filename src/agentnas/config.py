"""Pydantic configuration models for the AgentNAS pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from agentnas.utils.paths import project_root


class LLMConfig(BaseModel):
    """LLM API configuration."""

    model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.7
    seed: int | None = None
    max_tokens: int = 64000
    max_retries: int = 5  # transient-API-error retries (exponential backoff)
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    @model_validator(mode="after")
    def _load_keys_from_env_or_files(self) -> "LLMConfig":
        # Priority: explicit config value > environment variable > key file
        # at the repo root (CWD-independent).
        root = project_root()
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.anthropic_api_key:
            key_file = root / "anthropic_api_key.txt"
            if key_file.exists():
                self.anthropic_api_key = key_file.read_text().strip()
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.openai_api_key:
            key_file = root / "openai_api_key.txt"
            if key_file.exists():
                self.openai_api_key = key_file.read_text().strip()
        return self

    @property
    def is_claude(self) -> bool:
        return self.model.startswith("claude")

    @property
    def is_openai(self) -> bool:
        return self.model.startswith("gpt")



class TaskConfig(BaseModel):
    """Per-task benchmark metadata, derived from the benchmark class.

    Metric orientation is an invariant, not a knob: every metric in the
    pipeline is normalized to lower=better (report 1 - accuracy, 1 - AUROC,
    ...). Selection, patience, evolution fitness, checkpointing, and prompts
    all assume it; `agentnas validate-task` enforces it for custom tasks.
    """

    name: str
    data_dir: str = ""
    input_shape: list[int]
    num_classes: int | None = None
    output_shape: list[int] | None = None
    loss: str  # cross_entropy, bce_with_logits, mse, lp_loss
    metric: str  # accuracy, map, auroc, f1, mae, relative_l2 — lower=better
    sota_metric: float | None = None  # Human SOTA in our metric scale
    sota_description: str = ""  # Human-readable SOTA description

    @classmethod
    def from_benchmark(cls, benchmark) -> "TaskConfig":
        """Derive task metadata from a BaseBenchmark instance.

        The benchmark class is the single source of truth; this is a plain
        projection of its metadata attributes and shape methods.
        """
        spec = benchmark.output_spec()
        return cls(
            name=benchmark.task_name,
            data_dir=benchmark.data_subdir,
            input_shape=list(benchmark.input_shape()),
            num_classes=spec.get("num_classes"),
            output_shape=(
                list(spec["output_shape"]) if "output_shape" in spec else None
            ),
            loss=benchmark.loss_name,
            metric=benchmark.metric_name,
            sota_metric=benchmark.sota_metric,
            sota_description=benchmark.sota_description,
        )


class BudgetConfig(BaseModel):
    """Evaluation budget and stopping conditions."""

    max_evaluations: int = 160
    patience: int = 64  # Phase 1: no improvement for N evals → forced stop
    nas_patience: int = 64  # Phase 3: no improvement for N evals → stop
    max_executor_retries: int = 10
    max_explorer_steps: int = 30
    # Phase 1 planner-loop guards (previously hardcoded)
    max_consecutive_explores: int = 2  # force an experiment after N explores
    max_empty_iterations: int = 10  # abort after N iterations with no proposals
    # Sandboxed-subprocess timeouts (previously hardcoded)
    code_exec_timeout_s: int = 120  # explorer/modularizer code execution
    executor_validate_timeout_s: int = 60  # executor CPU forward-pass validation
    max_cfg_retries: int = 10
    cfg_validation_samples: int = 50
    top_k_final: int = 8  # Number of top architectures for final training (M-01)
    max_slots: int = 20  # Phase 2: max slots to generate
    max_alternatives_per_slot: int = 8  # Phase 2: max alternatives per slot
    max_phase2_retries: int = 3  # Phase 2: retries per seed before fallback
    max_seed_fallbacks: int = 3  # Phase 2: number of seeds to try


class GPUConfig(BaseModel):
    """GPU and hardware configuration."""

    n_gpu: int = 8  # RTX 2080 Ti — 8 GPUs per station (production default)
    proxy_time_limit_s: int = 3600  # 60 minutes (hard cap; primary limit is epoch-based)
    final_time_limit_s: int = 86400  # 24 hours
    max_vram_mb: int = 11264  # RTX 2080 Ti (11 GB) — production default
    proxy_epoch_fraction: float = 0.3  # Proxy trains 30% of proposed epochs
    # Hard-kill buffer added to a worker's training time limit before the
    # process group is terminated (previously hardcoded at 1 hr).
    worker_timeout_buffer_s: int = 3600
    # Pre-training VRAM gate: estimated training VRAM = forward VRAM × this.
    vram_estimate_multiplier: float = 2.5
    # num_workers controlled via AGENTNAS_NUM_WORKERS env var (default 4)


class EvolutionConfig(BaseModel):
    """Evolutionary search parameters."""

    population_size: int = 16
    max_generations: int = 10
    tournament_size: int = 5
    structural_mutation_prob: float = 0.5
    diversity_threshold: int = 5
    high_structural_prob: float = 0.8  # When diversity < threshold
    search_mode: str = "aggressive"  # "evo" | "random" | "aggressive" | "crossover" | "gdas"


class GDASConfig(BaseModel):
    """GDAS one-shot search hyperparameters (search_mode == "gdas")."""

    epochs: int = 0  # search-time epochs; 0 means "unset, must be supplied"
    warmup_epochs: int = 0  # 0 → derived as max(5, epochs // 10) at runtime
    tau_init: float = 10.0
    tau_final: float = 0.1
    arch_lr: float = 3e-4
    arch_wd: float = 1e-3
    batch_size: int | None = None  # None → use slotted_arch.training_config.batch_size


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    task: str = "fsd50k"
    phase: int = 1  # 1, 2, or 3
    experiment_dir: str = ""
    data_root: str = ""
    # Optional path to a Python file whose @register_benchmark classes define
    # custom tasks. Loaded before the task is resolved.
    benchmark_file: str = ""
    seed: int = 42
    blind_mode: bool = False  # Unseen-NAS ablation: suppress domain knowledge to LLM
    # Fixed-seed-arch ablation (Phase 1 LLM-loop replacement). Empty string = off.
    # Values: "auto" | "resnet18" | "wideresnet_16_4". See grammar/fixed_seeds.py.
    fixed_seed_arch: str = ""
    # Optional epoch-cap for the fixed-seed Wave 1/Wave 2 trainings (smoke-test knob).
    fixed_seed_epochs: int | None = None

    llm: LLMConfig = Field(default_factory=LLMConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    gdas: GDASConfig = Field(default_factory=GDASConfig)

    # Set at runtime
    task_config: TaskConfig | None = None

    @model_validator(mode="after")
    def _resolve_paths(self) -> "PipelineConfig":
        root = project_root()
        if not self.data_root:
            # Check both benchmark directories (M10: also assign unseen)
            nb360 = root / "NAS-Bench-360-Benchmarks"
            unseen = root / "NAS-Unseen"
            if nb360.exists():
                self.data_root = str(nb360)
            elif unseen.exists():
                self.data_root = str(unseen)
        return self


def load_pipeline_config(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Load pipeline config from YAML with optional overrides."""
    data: dict[str, Any] = {}
    if config_path and config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    if overrides:
        # Shallow merge of top-level and nested keys
        for key, val in overrides.items():
            if "." in key:
                parts = key.split(".", 1)
                section = data.setdefault(parts[0], {})
                if isinstance(section, dict):
                    section[parts[1]] = val
            else:
                data[key] = val
    config = PipelineConfig(**data)
    # Register custom tasks before resolving the task name
    if config.benchmark_file:
        from agentnas.benchmarks.registry import load_benchmark_file

        load_benchmark_file(config.benchmark_file)
    # Derive task metadata from the registered benchmark class
    try:
        from agentnas.benchmarks.registry import get_benchmark

        benchmark = get_benchmark(config.task, data_root=config.data_root or None)
        config.task_config = TaskConfig.from_benchmark(benchmark)
    except ValueError:
        import logging as _log
        _log.getLogger(__name__).warning(
            f"Unknown task '{config.task}'. "
            f"SOTA baselines and task metadata will be unavailable."
        )
    return config


def make_experiment_subdir(
    parent: str,
    task: str,
    model: str,
    seed: int,
    phase: int | None = None,
    blind: bool = False,
    fixed_seed: str = "",
    gdas_variant: str = "",
) -> str:
    """Build a deterministic experiment subdirectory under *parent*.

    When ``--experiment-dir`` is specified, this appends a settings-based
    subdirectory to prevent different experiments from colliding.
    The name is deterministic (no datetime) so ``--resume`` can find
    the existing directory on re-invocation.

    If *parent* already contains experiment state files the path is
    returned unchanged (handles resuming a previously generated dir).
    """
    parent_path = Path(parent)

    # If parent already looks like an existing experiment dir, use as-is.
    if phase is not None:
        # cli.py flat layout: experiment dir IS the phase dir
        if (parent_path / "state.json").exists():
            return parent
    else:
        # run_production.py hierarchical layout
        has_phase_subdir = any(
            (parent_path / f"phase{p}" / "state.json").exists()
            for p in range(1, 4)
        )
        if has_phase_subdir or (parent_path / "state.json").exists():
            return parent

    model_tag = model.replace("/", "_")
    name = f"{task}_{model_tag}_seed{seed}"
    if blind:
        name += "_blind"
    if fixed_seed:
        # Family-specific suffix so directory listings self-document the architecture.
        family_tag = "wrn16_4" if fixed_seed == "wideresnet_16_4" else fixed_seed
        name += f"_{family_tag}seed"
    if gdas_variant:
        name += f"_{gdas_variant}"
    if phase is not None:
        name += f"_phase{phase}"
    return str(parent_path / name)
