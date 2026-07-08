"""Phase 3: Evolutionary Search on SlottedArchitecture.

Regularized evolution by swapping slot alternatives.
All candidates are pre-validated — no code generation at search time.
"""

from __future__ import annotations

import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentnas.config import PipelineConfig
from agentnas.grammar.mutator import (
    choices_signature,
    crossover,
    multi_slot_mutation,
    random_choices,
    slot_mutation,
)
from agentnas.grammar.slots import SlottedArchitecture
from agentnas.phases.phase1 import _parallel_training_limit, _run_training_subprocess
from agentnas.utils.gpu import GPUPool
from agentnas.utils.logging import ExperimentLogger
from agentnas.utils.state import StateManager

logger = logging.getLogger(__name__)


@dataclass
class SlotIndividual:
    """A single individual in slot-based evolution."""

    choices: dict[str, int]  # {slot_name: alternative_index}
    fitness: float = float("inf")
    age: int = 0
    arch_id: str = ""
    parent_id: str | None = None
    mutation_type: str = "init"  # init, single_slot, multi_slot, random
    generation: int = 0
    code: str = ""
    eval_result: dict[str, Any] = field(default_factory=dict)

    @property
    def is_evaluated(self) -> bool:
        # B-01: 1e308 is the JSON sentinel for inf — treat as unevaluated
        return self.fitness < 1e300

    @property
    def signature(self) -> str:
        return choices_signature(self.choices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "choices": self.choices,
            "fitness": self.fitness,
            "age": self.age,
            "arch_id": self.arch_id,
            "parent_id": self.parent_id,
            "mutation_type": self.mutation_type,
            "generation": self.generation,
            "code": self.code,
            "eval_result": self.eval_result,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SlotIndividual":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def run_phase3(config: PipelineConfig, resume: bool = True) -> dict[str, Any]:
    """Run Phase 3: Evolutionary Search on slot-based architecture space."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Seed all random modules for reproducible evolution (B4, H6, #28)
    random.seed(config.seed)
    import numpy as np
    import torch as _torch
    np.random.seed(config.seed)
    _torch.manual_seed(config.seed)

    # Dispatch to GDAS one-shot search if requested. Evolutionary path untouched.
    if config.evolution.search_mode == "gdas":
        from agentnas.phases.phase3_gdas import run_phase3_gdas
        return run_phase3_gdas(config, resume=resume)

    exp_dir = Path(config.experiment_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable
    gpu_pool = GPUPool(n_gpu=config.gpu.n_gpu)
    exp_logger = ExperimentLogger(exp_dir)
    state_mgr = StateManager(exp_dir / "state.json")

    # Load benchmark once (B-05 fix: avoid re-instantiation per wave)
    from agentnas.benchmarks.registry import get_benchmark
    benchmark = get_benchmark(config.task, data_root=config.data_root)

    # RS-only: pre-validation helper. Used in the offspring loop to avoid
    # spending training-pool slots on archs that fail CPU forward (the bulk of
    # "rejected by validation gate" warnings in random mode). Imported once here
    # to avoid the per-call import overhead inside the inner loop.
    if config.evolution.search_mode == "random":
        from agentnas.agents.executor import _validate_model_code as _rs_validate

    # Load SlottedArchitecture
    arch_path = _find_slotted_arch(config)
    if not arch_path:
        raise FileNotFoundError("SlottedArchitecture not found. Run Phase 2 first.")
    slotted = SlottedArchitecture.load(str(arch_path))
    logger.info(
        f"Loaded slotted architecture: {len(slotted.slots)} slots, "
        f"search space = {slotted.search_space_size}"
    )

    # Evolution parameters
    pop_size = config.evolution.population_size
    max_gen = config.evolution.max_generations
    max_evals = config.budget.max_evaluations
    tournament_size = config.evolution.tournament_size
    patience = config.budget.nas_patience

    # State
    population: list[SlotIndividual] = []
    all_history: list[SlotIndividual] = []  # ALL evaluated individuals (for final selection)
    best_ever: SlotIndividual | None = None
    generation = 0
    total_evals = 0
    no_improvement = 0
    arch_counter = 0

    # Resume
    if resume and state_mgr.exists:
        state_mgr.load()
        if "evolution" in state_mgr:
            ev = state_mgr["evolution"]
            population = [SlotIndividual.from_dict(d) for d in ev.get("population", [])]
            all_history = [SlotIndividual.from_dict(d) for d in ev.get("all_history", [])]
            if ev.get("best_ever"):
                best_ever = SlotIndividual.from_dict(ev["best_ever"])
            generation = ev.get("generation", 0)
            total_evals = ev.get("total_evals", 0)
            no_improvement = ev.get("no_improvement", 0)
            arch_counter = ev.get("arch_counter", 0)
            
            if "rng_random" in ev:
                random.setstate(tuple(tuple(x) if isinstance(x, list) else x for x in ev["rng_random"]))
            if "rng_numpy" in ev:
                rng_np = ev["rng_numpy"]
                keys_arr = rng_np[1]
                if isinstance(keys_arr, str):
                    keys_arr = [int(x) for x in keys_arr.strip('[]').split()]
                np.random.set_state((rng_np[0], np.array(keys_arr, dtype=np.uint32), int(rng_np[2]), int(rng_np[3]), float(rng_np[4])))
            if "rng_torch" in ev:
                _torch.random.set_rng_state(_torch.ByteTensor(ev["rng_torch"]))
            logger.info(f"Resumed: gen={generation}, evals={total_evals}, history={len(all_history)}, best={best_ever.fitness if best_ever else 'N/A'}")

    def next_id() -> str:
        nonlocal arch_counter
        arch_counter += 1
        return f"evo_{arch_counter:03d}"

    def save_state():
        state_mgr.update({
            "evolution": {
                "population": [ind.to_dict() for ind in population],
                "all_history": [ind.to_dict() for ind in all_history],
                "best_ever": best_ever.to_dict() if best_ever else None,
                "generation": generation,
                "total_evals": total_evals,
                "no_improvement": no_improvement,
                "arch_counter": arch_counter,
                # C-02: Save RNG states for idempotent resume
                "rng_random": random.getstate(),
                "rng_numpy": np.random.get_state(),
                "rng_torch": _torch.random.get_rng_state().tolist(),
            }
        })
        state_mgr.save()

    # --- Initialize population if needed ---
    if not population:
        if config.evolution.search_mode == "random":
            logger.info("Initializing population (random mode, no seed; pre-validated)")
            init_attempts = 0
            init_max = pop_size * 8
            while len(population) < pop_size and init_attempts < init_max:
                init_attempts += 1
                mutated = random_choices(slotted)
                code = slotted.realize(mutated)
                ok, _err = _rs_validate(
                    code, benchmark.input_shape(), config.task,
                    config.data_root or "", python_bin,
                )
                if not ok:
                    continue
                population.append(SlotIndividual(
                    choices=mutated, arch_id=next_id(), parent_id=None,
                    mutation_type="random", generation=0, code=code,
                ))
            logger.info(
                f"Initial population: {len(population)} validated archs "
                f"in {init_attempts} attempts ({init_attempts - len(population)} rejected)"
            )
        else:
            logger.info("Initializing population from seed + mutations")
            # Seed individual — original Phase 1 code verbatim (not realized scaffold)
            seed_code = slotted.original_seed_code or slotted.realize_seed()
            seed_ind = SlotIndividual(
                choices=dict(slotted.seed_choices),
                arch_id=next_id(),
                mutation_type="seed",
                generation=0,
                code=seed_code,
            )
            population.append(seed_ind)

            # Fill with diverse mutations
            n_remaining = pop_size - 1
            n_single = n_remaining // 3
            n_double = n_remaining // 3
            n_random = n_remaining - n_single - n_double

            for i in range(n_single):
                mutated = slot_mutation(slotted.seed_choices, slotted)
                population.append(SlotIndividual(
                    choices=mutated, arch_id=next_id(), parent_id=seed_ind.arch_id,
                    mutation_type="single_slot", generation=0, code=slotted.realize(mutated),
                ))

            for i in range(n_double):
                mutated = multi_slot_mutation(slotted.seed_choices, slotted, n_slots=2)
                population.append(SlotIndividual(
                    choices=mutated, arch_id=next_id(), parent_id=seed_ind.arch_id,
                    mutation_type="double_slot", generation=0, code=slotted.realize(mutated),
                ))

            for i in range(n_random):
                mutated = random_choices(slotted)
                population.append(SlotIndividual(
                    choices=mutated, arch_id=next_id(), parent_id=seed_ind.arch_id,
                    mutation_type="random", generation=0, code=slotted.realize(mutated),
                ))

        # Evaluate initial population
        _evaluate_batch(population, slotted, config, gpu_pool, exp_logger, python_bin, exp_dir, 0, benchmark=benchmark)
        for ind in population:
            # RS round-robin spirit: don't count train-time-rejected archs (e.g. OOM)
            # toward the budget. Successful trains only.
            if config.evolution.search_mode == "random":
                if ind.is_evaluated:
                    total_evals += 1
            else:
                total_evals += 1
            if ind.is_evaluated:
                all_history.append(ind)
            if ind.is_evaluated and (best_ever is None or ind.fitness < best_ever.fitness):
                best_ever = ind
                no_improvement = 0
            elif ind.is_evaluated:
                no_improvement += 1
            # Failed evals don't count toward patience

        generation = 1
        save_state()

    # --- Evolution loop ---
    # generation starts at 1 (initial population). Loop produces generations 2..max_gen.
    # Total = pop_size (gen 1) + (max_gen - 1) * pop_size = max_gen * pop_size.
    while (total_evals < max_evals
           and no_improvement < patience
           and (config.evolution.search_mode == "random" or generation < max_gen)):
        # Diversity check
        unique_sigs = {ind.signature for ind in population}
        diversity = len(unique_sigs)

        best_str = f"{best_ever.fitness:.4f}" if best_ever else "N/A"
        logger.info(
            f"=== Generation {generation} | evals={total_evals}/{max_evals} | "
            f"diversity={diversity} | best={best_str} | "
            f"no_improve={no_improvement}/{patience} ==="
        )

        # Generate offspring with deduplication (H-02)
        offspring = []
        existing_sigs = {ind.signature for ind in population}
        offspring_sigs = set()
        max_attempts = pop_size * 3  # Avoid infinite loop if search space is small
        attempts = 0

        while len(offspring) < pop_size and attempts < max_attempts:
            if total_evals + len(offspring) >= max_evals:
                break
            attempts += 1

            if config.evolution.search_mode == "random":
                # Random-search control: no tournament, no mutation — uniform sample.
                mutated = random_choices(slotted)
                mut_type = "random"
                parent_id_for_child = None
            elif config.evolution.search_mode == "aggressive":
                # Aggressive: 50/50 single/multi-slot regardless of diversity.
                evaluated = [ind for ind in population if ind.is_evaluated]
                if len(evaluated) < tournament_size:
                    contestants = evaluated
                else:
                    contestants = random.sample(evaluated, tournament_size)
                parent = min(contestants, key=lambda i: i.fitness) if contestants else population[0]
                if random.random() < 0.5:
                    mutated = multi_slot_mutation(parent.choices, slotted, n_slots=2)
                    mut_type = "aggressive_multi"
                else:
                    mutated = slot_mutation(parent.choices, slotted)
                    mut_type = "aggressive_single"
                parent_id_for_child = parent.arch_id
            elif config.evolution.search_mode == "crossover":
                # Crossover: 50% per-slot 2-parent crossover, 50% pure mutation.
                evaluated = [ind for ind in population if ind.is_evaluated]
                if len(evaluated) < tournament_size:
                    contestants1 = evaluated
                    contestants2 = evaluated
                else:
                    contestants1 = random.sample(evaluated, tournament_size)
                    contestants2 = random.sample(evaluated, tournament_size)
                parent1 = min(contestants1, key=lambda i: i.fitness) if contestants1 else population[0]
                parent2 = min(contestants2, key=lambda i: i.fitness) if contestants2 else population[0]
                if random.random() < 0.5 and parent1.arch_id != parent2.arch_id:
                    mutated = crossover(parent1.choices, parent2.choices, slotted)
                    mut_type = "crossover"
                else:
                    mutated = slot_mutation(parent1.choices, slotted)
                    mut_type = "single_slot"
                parent_id_for_child = parent1.arch_id
            else:
                # 'evo': tournament + mutation, multi-slot when diversity drops.
                evaluated = [ind for ind in population if ind.is_evaluated]
                if len(evaluated) < tournament_size:
                    contestants = evaluated
                else:
                    contestants = random.sample(evaluated, tournament_size)
                parent = min(contestants, key=lambda i: i.fitness) if contestants else population[0]

                # Mutation
                if diversity < config.evolution.diversity_threshold:
                    mutated = multi_slot_mutation(parent.choices, slotted, n_slots=2)
                    mut_type = "multi_slot"
                else:
                    mutated = slot_mutation(parent.choices, slotted)
                    mut_type = "single_slot"
                parent_id_for_child = parent.arch_id

            sig = choices_signature(mutated)
            if sig in existing_sigs or sig in offspring_sigs:
                continue  # Skip duplicate
            code = slotted.realize(mutated)
            # RS-only: pre-validate to skip structurally-invalid combos before they
            # enter the training pool. Saves training-slot allocations for archs
            # that can actually run a forward pass.
            if config.evolution.search_mode == "random":
                ok, _err = _rs_validate(
                    code, benchmark.input_shape(), config.task,
                    config.data_root or "", python_bin,
                )
                if not ok:
                    continue
            offspring_sigs.add(sig)

            child = SlotIndividual(
                choices=mutated,
                arch_id=next_id(),
                parent_id=parent_id_for_child,
                mutation_type=mut_type,
                generation=generation,
                code=code,
            )
            offspring.append(child)

        # Evaluate offspring in parallel waves (pass diversity for D6 logging)
        current_diversity = diversity
        wave_size = config.gpu.n_gpu
        for wave_start in range(0, len(offspring), wave_size):
            wave = offspring[wave_start : wave_start + wave_size]
            _evaluate_batch(wave, slotted, config, gpu_pool, exp_logger, python_bin, exp_dir, generation, current_diversity, benchmark=benchmark)

        prev_best = best_ever.fitness if best_ever else float("inf")
        generation_improved = False

        for child in offspring:
            # RS round-robin spirit: only count successful trains toward the budget;
            # OOM/runtime-rejected archs don't burn budget. Evo-mode behavior unchanged
            # since failed evals there still consume an evaluation slot by design.
            if config.evolution.search_mode == "random":
                if child.is_evaluated:
                    total_evals += 1
            else:
                total_evals += 1
            population.append(child)
            if child.is_evaluated:
                all_history.append(child)

            if child.is_evaluated and (best_ever is None or child.fitness < best_ever.fitness):
                best_ever = child
                generation_improved = True

            # Age-based removal (regularized evolution)
            while len(population) > pop_size:
                oldest = max(population, key=lambda i: i.age)
                population.remove(oldest)

        # Patience: count successful evals in this generation that didn't improve
        n_successful = sum(1 for c in offspring if c.is_evaluated)
        if generation_improved:
            no_improvement = 0
        else:
            no_improvement += n_successful  # Only successful evals count

        # Age everyone
        for ind in population:
            ind.age += 1

        generation += 1
        save_state()

    # --- Final training (two waves) for top-N from ALL evaluated history ---
    # Wave 1: full training on train/val split → val metric for all pipeline decisions
    # Wave 2: full training on train+val combined → test metric for analysis only
    logger.info("=== Final training top architectures ===")
    # Select from ALL historically evaluated individuals, not just surviving population
    evaluated = [ind for ind in all_history if ind.is_evaluated and ind.fitness < float("inf")]
    evaluated.sort(key=lambda i: i.fitness)
    # Deduplicate by slot signature to avoid training identical architectures (L3 fix)
    seen_sigs = set()
    unique_evaluated = []
    for ind in evaluated:
        sig = ind.signature
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            unique_evaluated.append(ind)
    top_n = unique_evaluated[: config.budget.top_k_final]  # M-01: use top_k_final (default 8)
    logger.info(f"Selected top {len(top_n)} from {len(all_history)} total evaluated")

    # --- Wave 1: Val-final (full training, train/val split, best checkpoint) ---
    logger.info(f"=== Wave 1: Val-final training {len(top_n)} architectures ===")

    def _val_final_one(ind):
        """Wave 1: Full training on train/val split, best checkpoint by val metric."""
        gpu_id = gpu_pool.allocate_one()
        try:
            save_dir = exp_dir / "val_final" / ind.arch_id
            # Apply per-individual LR multiplier (if lr_mult slot is in grammar).
            config_dict = dict(slotted.training_config)
            config_dict["lr"] = float(config_dict.get("lr", 1e-3)) * slotted.lr_mult_for(ind.choices)
            result = _run_training_subprocess(
                model_code=ind.code,
                config_dict=config_dict,
                task_name=config.task,
                data_root=config.data_root or "",
                gpu_id=gpu_id,
                mode="search",
                time_limit_s=config.gpu.final_time_limit_s,
                seed=config.seed,
                save_dir=save_dir,
                python_bin=python_bin,
                max_vram_mb=config.gpu.max_vram_mb,
                timeout_buffer_s=config.gpu.worker_timeout_buffer_s,
                vram_estimate_multiplier=config.gpu.vram_estimate_multiplier,
                proxy_epoch_fraction=1.0,
            )
            return {
                "arch_id": ind.arch_id,
                "choices": ind.choices,
                "proxy_metric": ind.fitness,
                "val_final_metric": result["metric"],
                "status": result["status"],
                "code": ind.code,
                "training_config": config_dict,
                "params": result.get("params", 0),
            }
        finally:
            gpu_pool.release(gpu_id)

    val_final_results = []
    if resume:
        prev = state_mgr.get("val_final_results", [])
        if prev:
            val_final_results = list(prev)
            logger.info(f"Wave 1 resume: loaded {len(val_final_results)} prior val_final results")
    done_ids = {r["arch_id"] for r in val_final_results if r.get("status") == "success"}
    pending_top_n = [ind for ind in top_n if ind.arch_id not in done_ids]
    if len(pending_top_n) < len(top_n):
        logger.info(f"Wave 1: skipping {len(top_n) - len(pending_top_n)} archs already trained ({sorted(done_ids)})")
    with ThreadPoolExecutor(max_workers=_parallel_training_limit(config.gpu.n_gpu)) as pool:
        futures = {pool.submit(_val_final_one, ind): ind.arch_id for ind in pending_top_n}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                arch_id = futures[future]
                logger.error(f"Val-final training {arch_id} failed: {e}")
                continue
            if result:
                val_final_results.append(result)
                logger.info(f"Val-final {result['arch_id']}: proxy={result['proxy_metric']:.4f}, val_final={result['val_final_metric']:.4f}")
                exp_logger.log_evaluation({
                    "arch_id": result["arch_id"], "phase": 3, "mode": "val_final",
                    "metric": result["val_final_metric"], "status": result["status"],
                    "proxy_metric": result.get("proxy_metric"),
                    "choices": result.get("choices"),
                })
                # Persist after every successful arch — supports mid-Wave-1 resume.
                state_mgr.update({"val_final_results": val_final_results})
                state_mgr.save()

    # Intermediate checkpoint after Wave 1 (resume safety)
    state_mgr.update({"val_final_results": val_final_results})
    state_mgr.save()

    best_val_final = min(
        (r for r in val_final_results if r["status"] == "success"),
        key=lambda r: r["val_final_metric"],
        default=None,
    )

    # --- Wave 2: Test-final (train+val combined, last checkpoint, analysis only) ---
    logger.info(f"=== Wave 2: Test-final training {len(top_n)} architectures (analysis only) ===")

    def _test_final_one(ind):
        """Wave 2: Full training on train+val, last checkpoint, test eval. Analysis only."""
        gpu_id = gpu_pool.allocate_one()
        try:
            save_dir = exp_dir / "test_final" / ind.arch_id
            # Apply per-individual LR multiplier (if lr_mult slot is in grammar).
            config_dict = dict(slotted.training_config)
            config_dict["lr"] = float(config_dict.get("lr", 1e-3)) * slotted.lr_mult_for(ind.choices)
            result = _run_training_subprocess(
                model_code=ind.code,
                config_dict=config_dict,
                task_name=config.task,
                data_root=config.data_root or "",
                gpu_id=gpu_id,
                mode="final",
                time_limit_s=config.gpu.final_time_limit_s,
                seed=config.seed,
                save_dir=save_dir,
                python_bin=python_bin,
                max_vram_mb=config.gpu.max_vram_mb,
                timeout_buffer_s=config.gpu.worker_timeout_buffer_s,
                vram_estimate_multiplier=config.gpu.vram_estimate_multiplier,
            )
            return {
                "arch_id": ind.arch_id,
                "test_metric": result["metric"],
                "status": result["status"],
            }
        finally:
            gpu_pool.release(gpu_id)

    # Resume support for Wave 2 — same shape as Wave 1.
    test_final_results = []
    if resume:
        prev = state_mgr.get("test_final_results", [])
        if prev:
            test_final_results = list(prev)
            logger.info(f"Wave 2 resume: loaded {len(test_final_results)} prior test_final results")
    done_w2 = {r["arch_id"] for r in test_final_results if r.get("status") == "success"}
    pending_wave2 = [ind for ind in top_n if ind.arch_id not in done_w2]
    if len(pending_wave2) < len(top_n):
        logger.info(f"Wave 2: skipping {len(top_n) - len(pending_wave2)} archs already trained ({sorted(done_w2)})")
    with ThreadPoolExecutor(max_workers=_parallel_training_limit(config.gpu.n_gpu)) as pool:
        futures = {pool.submit(_test_final_one, ind): ind.arch_id for ind in pending_wave2}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                arch_id = futures[future]
                logger.error(f"Test-final training {arch_id} failed: {e}")
                continue
            if result:
                test_final_results.append(result)
                logger.info(f"Test-final {result['arch_id']}: test={result['test_metric']:.4f}")
                exp_logger.log_evaluation({
                    "arch_id": result["arch_id"], "phase": 3, "mode": "test_final",
                    "metric": result["test_metric"], "status": result["status"],
                })
                # Persist after every successful arch — supports mid-Wave-2 resume.
                state_mgr.update({"test_final_results": test_final_results})
                state_mgr.save()

    state_mgr.update({
        "val_final_results": val_final_results,
        "test_final_results": test_final_results,
        "phase": "completed",
    })
    state_mgr.save()

    return {
        "phase": 3,
        "total_evaluations": total_evals,
        "generations": generation,
        "best_proxy": {"arch_id": best_ever.arch_id, "fitness": best_ever.fitness} if best_ever else None,
        "best_final": best_val_final,
        "search_space_size": slotted.search_space_size,
        "val_final_results": val_final_results,
        "test_final_results": test_final_results,
    }


def _evaluate_batch(
    individuals: list[SlotIndividual],
    slotted: SlottedArchitecture,
    config: PipelineConfig,
    gpu_pool: GPUPool,
    exp_logger: ExperimentLogger,
    python_bin: str,
    exp_dir: Path,
    generation: int,
    population_diversity: int = 0,
    benchmark=None,
):
    """Evaluate individuals in parallel across GPUs."""
    from agentnas.agents.executor import _validate_model_code
    from agentnas.benchmarks.registry import get_benchmark

    to_eval = [ind for ind in individuals if not ind.is_evaluated and ind.code]

    if not to_eval:
        return

    if benchmark is None:
        benchmark = get_benchmark(config.task, data_root=config.data_root)
    validated = []
    for ind in to_eval:
        ok, error = _validate_model_code(
            ind.code, benchmark.input_shape(), config.task,
            config.data_root or "", python_bin,
        )
        if ok:
            validated.append(ind)
        else:
            ind.fitness = float("inf")
            ind.eval_result = {"status": "rejected", "error_message": f"Validation: {error[:200]}"}
            logger.warning(f"{ind.arch_id} rejected by validation gate")
            exp_logger.log_evaluation({
                "arch_id": ind.arch_id, "phase": 3, "generation": generation,
                "mutation_type": ind.mutation_type, "metric": float("inf"),
                "status": "rejected", "params": 0, "flops": 0,
                "peak_vram_mb": 0, "wall_time_s": 0,
                "population_diversity": population_diversity,
            })
    to_eval = validated

    if not to_eval:
        return

    def _train_one(ind):
        gpu_id = gpu_pool.allocate_one()
        try:
            save_dir = exp_dir / "models" / ind.arch_id
            # Apply per-individual LR multiplier (if lr_mult slot is in grammar).
            config_dict = dict(slotted.training_config)
            config_dict["lr"] = float(config_dict.get("lr", 1e-3)) * slotted.lr_mult_for(ind.choices)
            result = _run_training_subprocess(
                model_code=ind.code,
                config_dict=config_dict,
                task_name=config.task,
                data_root=config.data_root or "",
                gpu_id=gpu_id,
                mode="search",
                time_limit_s=config.gpu.proxy_time_limit_s,
                seed=config.seed,
                save_dir=save_dir,
                python_bin=python_bin,
                max_vram_mb=config.gpu.max_vram_mb,
                timeout_buffer_s=config.gpu.worker_timeout_buffer_s,
                vram_estimate_multiplier=config.gpu.vram_estimate_multiplier,
                proxy_epoch_fraction=config.gpu.proxy_epoch_fraction,
            )
            return ind, result
        finally:
            gpu_pool.release(gpu_id)

    with ThreadPoolExecutor(max_workers=_parallel_training_limit(config.gpu.n_gpu)) as pool:
        futures = {pool.submit(_train_one, ind): ind.arch_id for ind in to_eval}
        for future in as_completed(futures):
            try:
                ind, result = future.result()
            except Exception as e:
                logger.error(f"Eval failed: {e}")
                continue

            ind.fitness = result["metric"]
            ind.eval_result = result

            exp_logger.log_evaluation({
                "arch_id": ind.arch_id,
                "phase": 3,
                "generation": generation,
                "parent_id": ind.parent_id,
                "mutation_type": ind.mutation_type,
                "choices": ind.choices,
                "metric": result["metric"],
                "status": result["status"],
                "params": result.get("params", 0),
                "flops": result.get("flops", 0),
                "peak_vram_mb": result.get("peak_vram_mb", 0),
                "wall_time_s": result.get("wall_time_s", 0),
                "population_diversity": population_diversity,  # D6 fix
            })


def _find_slotted_arch(config: PipelineConfig) -> Path | None:
    """Find the slotted_arch.json from Phase 2."""
    exp_dir = Path(config.experiment_dir)
    if (exp_dir / "slotted_arch.json").exists():
        return exp_dir / "slotted_arch.json"
    sibling = exp_dir.parent / "phase2" / "slotted_arch.json"
    if sibling.exists():
        return sibling
    # Search experiments/ directory
    exp_root = Path("experiments")
    if exp_root.exists():
        for d in sorted(exp_root.iterdir(), reverse=True):
            if (d.name.startswith(config.task + "_") or d.name == config.task) and "phase2" in d.name:  # #9
                p = d / "slotted_arch.json"
                if p.exists():
                    return p
    return None
