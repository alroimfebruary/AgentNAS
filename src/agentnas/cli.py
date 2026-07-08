"""CLI entry point for the AgentNAS pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from agentnas.config import load_pipeline_config


@click.group()
@click.version_option(package_name="agentnas")
def main():
    """AgentNAS: LLM + Grammar-Based Neural Architecture Search Pipeline."""


@main.command()
@click.option("--task", required=True, help="Benchmark task name (e.g., fsd50k)")
@click.option("--phase", type=int, required=True, help="Pipeline phase (1-3)")
@click.option("--llm-model", default=None, help="LLM model name")
@click.option("--n-gpu", type=int, default=None, help="Number of GPUs to use")
@click.option("--budget", type=int, default=None, help="Max architecture evaluations")
@click.option("--experiment-dir", default=None, help="Experiment output directory")
@click.option("--data-root", default=None, help="Root directory for benchmark data (absolute path)")
@click.option("--benchmark-file", type=click.Path(exists=True, path_type=Path), default=None,
              help="Python file registering custom tasks via @register_benchmark")
@click.option("--resume/--no-resume", default=True, help="Resume from checkpoint")
@click.option("--seed", type=int, default=None, help="Random seed")
@click.option(
    "--config", "config_path", type=click.Path(exists=True, path_type=Path),
    default=None, help="Path to config YAML file",
)
@click.option("--blind-mode", is_flag=True, default=False,
              help="Blind mode: suppress domain knowledge (Unseen-NAS ablation)")
@click.option("--fixed-seed-arch",
              type=click.Choice(["auto", "resnet18", "wideresnet_16_4"]),
              default=None,
              help="Fixed-seed-arch ablation: skip Phase 1 LLM loop and seed with "
                   "ResNet-18 (Unseen-NAS) or WideResNet-16-4 (NAS-Bench-360). "
                   "Use 'auto' to derive family from the task.")
@click.option("--fixed-seed-epochs", type=int, default=None,
              help="Cap epochs for the fixed-seed final training (smoke-test knob). "
                   "Only effective with --fixed-seed-arch.")
@click.option("--search-mode", type=click.Choice(["evo", "random", "aggressive", "crossover", "gdas"]),
              default="aggressive",
              help="Search strategy: aggressive (default, 50/50 single/multi-slot mutation), "
                   "evo (original tournament+single-slot), random (uniform sampling control), "
                   "crossover (50% 2-parent crossover), gdas (one-shot Gumbel-STE supernet)")
@click.option("--pop-size", type=int, default=None,
              help="Evolution population size (override evolution.population_size)")
@click.option("--patience", type=int, default=None,
              help="NAS patience for early-stopping (override budget.nas_patience)")
@click.option("--gdas-epochs", type=int, default=None,
              help="GDAS only: number of supernet search epochs (required when --search-mode gdas)")
@click.option("--gdas-warmup-epochs", type=int, default=None,
              help="GDAS only: weight-only warmup epochs before arch updates start (default: max(5, epochs//10))")
@click.option("--gdas-tau-init", type=float, default=None,
              help="GDAS only: initial Gumbel-softmax temperature (default 10.0)")
@click.option("--gdas-tau-final", type=float, default=None,
              help="GDAS only: final Gumbel-softmax temperature (default 0.1)")
@click.option("--gdas-arch-lr", type=float, default=None,
              help="GDAS only: arch-parameter Adam learning rate (default 3e-4)")
@click.option("--gdas-arch-wd", type=float, default=None,
              help="GDAS only: arch-parameter Adam weight decay (default 1e-3)")
@click.option("--gdas-batch-size", type=int, default=None,
              help="GDAS only: override search-time batch size (default: from slotted_arch.training_config)")
def run(
    task: str,
    phase: int,
    llm_model: str | None,
    n_gpu: int | None,
    budget: int | None,
    experiment_dir: str | None,
    data_root: str | None,
    benchmark_file: Path | None,
    resume: bool,
    seed: int | None,
    config_path: Path | None,
    blind_mode: bool,
    fixed_seed_arch: str | None,
    fixed_seed_epochs: int | None,
    search_mode: str,
    pop_size: int | None,
    patience: int | None,
    gdas_epochs: int | None,
    gdas_warmup_epochs: int | None,
    gdas_tau_init: float | None,
    gdas_tau_final: float | None,
    gdas_arch_lr: float | None,
    gdas_arch_wd: float | None,
    gdas_batch_size: int | None,
):
    """Run a pipeline phase on a benchmark task."""
    # Build overrides from CLI args
    overrides = {"task": task, "phase": phase}
    if llm_model:
        overrides["llm.model"] = llm_model
    if n_gpu is not None:
        overrides["gpu.n_gpu"] = n_gpu
    if budget is not None:
        overrides["budget.max_evaluations"] = budget
    if experiment_dir:
        overrides["experiment_dir"] = experiment_dir
    if data_root:
        overrides["data_root"] = data_root
    if benchmark_file:
        overrides["benchmark_file"] = str(benchmark_file.resolve())
    if seed is not None:
        overrides["seed"] = seed
    if blind_mode:
        overrides["blind_mode"] = True
    if fixed_seed_arch:
        overrides["fixed_seed_arch"] = fixed_seed_arch
    if fixed_seed_epochs is not None:
        overrides["fixed_seed_epochs"] = fixed_seed_epochs
    overrides["evolution.search_mode"] = search_mode
    if pop_size is not None:
        overrides["evolution.population_size"] = pop_size
    if patience is not None:
        overrides["budget.nas_patience"] = patience

    # GDAS-specific overrides (under the `gdas.*` namespace).
    if search_mode == "gdas":
        if gdas_epochs is None:
            click.echo("Error: --gdas-epochs is required when --search-mode gdas", err=True)
            sys.exit(1)
        overrides["gdas.epochs"] = gdas_epochs
    if gdas_epochs is not None:
        overrides["gdas.epochs"] = gdas_epochs
    if gdas_warmup_epochs is not None:
        overrides["gdas.warmup_epochs"] = gdas_warmup_epochs
    if gdas_tau_init is not None:
        overrides["gdas.tau_init"] = gdas_tau_init
    if gdas_tau_final is not None:
        overrides["gdas.tau_final"] = gdas_tau_final
    if gdas_arch_lr is not None:
        overrides["gdas.arch_lr"] = gdas_arch_lr
    if gdas_arch_wd is not None:
        overrides["gdas.arch_wd"] = gdas_arch_wd
    if gdas_batch_size is not None:
        overrides["gdas.batch_size"] = gdas_batch_size

    config = load_pipeline_config(config_path, overrides)

    if experiment_dir:
        # User dir is a parent; append settings-based subdir to prevent collisions
        from agentnas.config import make_experiment_subdir
        gdas_variant = ""
        if search_mode == "gdas" and config.gdas.epochs > 0:
            gdas_variant = f"gdas_e{config.gdas.epochs}"
        config.experiment_dir = make_experiment_subdir(
            parent=experiment_dir, task=task, model=config.llm.model,
            seed=config.seed, phase=phase, blind=config.blind_mode,
            gdas_variant=gdas_variant,
        )
    elif not config.experiment_dir:
        # Default experiment dir — stable name for resume support (C-05 fix)
        config.experiment_dir = f"experiments/{task}_phase{phase}"

    exp_dir = Path(config.experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Task: {config.task}")
    click.echo(f"Phase: {config.phase}")
    click.echo(f"Model: {config.llm.model}")
    click.echo(f"GPUs: {config.gpu.n_gpu}")
    click.echo(f"Budget: {config.budget.max_evaluations}")
    click.echo(f"Experiment dir: {config.experiment_dir}")
    click.echo()

    # Dispatch to phase
    if phase == 1:
        from agentnas.phases.phase1 import run_phase1
        run_phase1(config, resume=resume)
    elif phase == 2:
        from agentnas.phases.phase2 import run_phase2
        run_phase2(config, resume=resume)
    elif phase == 3:
        from agentnas.phases.phase3 import run_phase3
        run_phase3(config, resume=resume)
    else:
        click.echo(f"Error: invalid phase {phase}. Must be 1-3.", err=True)
        sys.exit(1)


@main.command("clone_and_modularize")
@click.option("--source", required=True,
              help="Existing experiment dir to clone Phase 1 from")
@click.option("--target", required=True,
              help="New experiment dir to create (must not already exist or must be empty)")
@click.option("--task", required=True, help="Benchmark task (e.g., darcyflow)")
@click.option("--modularizer-llm", default="claude-sonnet-4-6",
              help="LLM model for the modularizer agent")
@click.option("--phase2-nudge", is_flag=True, default=False,
              help="Variant C: write a marker that Phase 2 reads to append the "
                   "self-contained-stage-modules nudge to the Grammar Planner prompt")
@click.option("--max-retries", type=int, default=5,
              help="Modularizer LLM retries on validation failure")
@click.option("--data-root", default=None, help="Root directory for benchmark data")
@click.option(
    "--config", "config_path", type=click.Path(exists=True, path_type=Path),
    default=None, help="Path to config YAML file",
)
def clone_and_modularize_cmd(
    source: str,
    target: str,
    task: str,
    modularizer_llm: str,
    phase2_nudge: bool,
    max_retries: int,
    data_root: str | None,
    config_path: Path | None,
):
    """Clone an experiment with a modularized Phase 1 seed.

    Reads the source's Phase 1 best architecture, asks an LLM to refactor it so
    each repeated stage is wrapped in its own nn.Module subclass (forward-pass
    bit-equivalent to the original after weight transplant), and writes a new
    experiment dir whose round_01/phase1/ is a copy of the source's with the
    seed code field replaced. Downstream Phase 2 + Phase 3 then run normally.

    With --phase2-nudge, Phase 2's Grammar Planner prompt also gets a section
    instructing it to emit Identity placeholders for outer norm/act/skip slots
    when stages are self-contained (variant C in MODULARIZATION_STRESS_TEST.md).
    """
    overrides: dict = {"task": task, "llm.model": modularizer_llm}
    if data_root:
        overrides["data_root"] = data_root

    config = load_pipeline_config(config_path, overrides)

    click.echo(f"Source:           {source}")
    click.echo(f"Target:           {target}")
    click.echo(f"Task:             {task}")
    click.echo(f"Modularizer LLM:  {modularizer_llm}")
    click.echo(f"Phase 2 nudge:    {phase2_nudge}")
    click.echo(f"Max retries:      {max_retries}")
    click.echo()

    from agentnas.phases.phase1_5_modularize import run_clone_and_modularize
    result = run_clone_and_modularize(
        config=config,
        source_dir=source,
        target_dir=target,
        phase2_nudge=phase2_nudge,
        max_retries=max_retries,
    )

    click.echo("\n=== clone_and_modularize complete ===")
    for k, v in result.items():
        click.echo(f"  {k}: {v}")


@main.command("init-task")
@click.option("--task", required=True, help="Task name to register")
@click.option("--data-root", required=True,
              help="Root directory; data expected at <data-root>/<data-subdir>/")
@click.option("--data-subdir", default=None,
              help="Task's directory under data-root (default: the task name)")
@click.option("--describe", default="",
              help="What the dataset is: domain, formats, labels, known baselines")
@click.option("--out", "out_path", default=None,
              help="Output benchmark file (default: <task>.py)")
@click.option("--llm-model", default="claude-sonnet-4-6",
              help="Authoring model (a strong model pays for itself here)")
@click.option("--max-attempts", type=int, default=4,
              help="Max file emissions before giving up")
def init_task_cmd(
    task: str,
    data_root: str,
    data_subdir: str | None,
    describe: str,
    out_path: str | None,
    llm_model: str,
    max_attempts: int,
):
    """Author a --benchmark-file for your dataset with an LLM agent.

    The agent inspects the data by executing code, writes the benchmark
    file, and iterates against `agentnas validate-task` until it passes.
    It never starts a search — review the generated file first.
    """
    from agentnas.agents.task_author import TaskAuthor
    from agentnas.config import LLMConfig
    from agentnas.llm.client import LLMClient
    from agentnas.llm.logger import LLMLogger

    data_root_abs = str(Path(data_root).resolve())
    subdir = data_subdir or task
    if not (Path(data_root_abs) / subdir).exists():
        click.echo(f"Error: data directory not found: {data_root_abs}/{subdir}/",
                   err=True)
        sys.exit(1)
    out = Path(out_path or f"{task}.py")
    if out.exists():
        click.echo(f"Error: {out} already exists — pass --out or remove it.",
                   err=True)
        sys.exit(1)

    log_dir = Path("init_task_logs") / task
    llm = LLMClient.from_config(
        LLMConfig(model=llm_model), logger=LLMLogger(log_dir).set_agent("author")
    )
    click.echo(f"Authoring {out} for task '{task}' with {llm_model} "
               f"(transcripts: {log_dir}/) ...")

    author = TaskAuthor(
        llm_client=llm, task=task, data_root=data_root_abs, data_subdir=subdir,
        out_path=out, description=describe, max_attempts=max_attempts,
    )
    result = author.author()

    click.echo(f"\n{'=' * 60}")
    if result.success:
        click.echo(f"VALIDATED: {out} "
                   f"({result.attempts} attempt(s), {result.turns} turns, "
                   f"{result.tokens:,} tokens)")
        click.echo("\nLast validation report:")
        click.echo(result.last_report)
        click.echo(f"\nNEXT STEPS — review before spending GPU/LLM budget:")
        click.echo(f"  1. Read {out}: check split logic and label semantics.")
        click.echo(f"  2. Re-check anytime: agentnas validate-task {out} "
                   f"--data-root {data_root}")
        click.echo(f"  3. Run: agentnas run --task {task} --phase 1 "
                   f"--benchmark-file {out} --data-root {data_root}")
        sys.exit(0)
    click.echo(f"FAILED after {result.attempts} attempt(s) / {result.turns} "
               f"turns ({result.tokens:,} tokens).")
    if result.last_report:
        click.echo("\nLast validation report / reason:")
        click.echo(result.last_report[:3000])
    if out.exists():
        click.echo(f"\nThe last attempt is at {out} — fix it manually "
                   f"(agentnas validate-task {out} --data-root {data_root}) "
                   f"or rerun with a stronger --llm-model.")
    sys.exit(1)


@main.command("validate-task")
@click.argument("benchmark_file", type=click.Path(exists=True, path_type=Path))
@click.option("--task", default=None,
              help="Task to validate (default: the single task the file registers)")
@click.option("--data-root", required=True,
              help="Root directory containing the task's data subdirectory")
def validate_task_cmd(benchmark_file: Path, task: str | None, data_root: str):
    """Validate a custom benchmark file before spending GPU/LLM budget.

    Loads BENCHMARK_FILE, checks the contract, runs the loader smoke test,
    verifies Explorer test-set isolation, checks split integrity and metric
    sanity, and previews exactly what the LLM agents will see.
    """
    from agentnas.validation import validate_custom_task

    ok = validate_custom_task(str(benchmark_file), task, data_root)
    sys.exit(0 if ok else 1)


@main.command("smoke-test")
@click.option("--task", default=None, help="Task to test (or 'all')")
@click.option("--data-root", default=None, help="Data root directory")
@click.option("--benchmark-file", type=click.Path(exists=True, path_type=Path),
              default=None, help="Python file registering custom tasks")
def smoke_test(task: str | None, data_root: str | None, benchmark_file: Path | None):
    """Run smoke tests on benchmark data loaders."""
    from agentnas.benchmarks.registry import (
        get_benchmark,
        list_benchmarks,
        load_benchmark_file,
    )

    if benchmark_file:
        load_benchmark_file(str(benchmark_file))

    tasks = [task] if task and task != "all" else list_benchmarks()

    passed, failed = 0, 0
    for t in tasks:
        try:
            benchmark = get_benchmark(t, data_root=data_root)
            ok = benchmark.smoke_test()
            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            status = f"ERROR: {e}"
            failed += 1
        click.echo(f"  [{status}] {t}")

    click.echo(f"\n{passed} passed, {failed} failed out of {len(tasks)} tasks")
    sys.exit(1 if failed > 0 else 0)


@main.command("test-llm")
@click.option("--model", default="claude-haiku-4-5-20251001", help="Model to test")
def test_llm(model: str):
    """Test LLM client connectivity."""
    from agentnas.config import LLMConfig
    from agentnas.llm.client import LLMClient

    llm_config = LLMConfig(model=model)
    client = LLMClient.from_config(llm_config)

    click.echo(f"Testing {model}...")
    try:
        resp = client.chat(
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        )
        click.echo(f"Response: {resp.content}")
        click.echo(f"Tokens: {resp.input_tokens} in, {resp.output_tokens} out")
        click.echo(f"Latency: {resp.latency_s:.2f}s")
        click.echo("OK")
    except Exception as e:
        click.echo(f"FAILED: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
