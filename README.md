# AgentNAS: LLM + Neural Architecture Search

Reference implementation for the paper *"Agentic Neural Architecture Search"* (2026). AgentNAS combines
LLM-driven architecture design with neural architecture search across 17
benchmarks from NAS-Bench-360 and Unseen NAS.

The `v1.0-reproducible` tag provides the code version for reproducing paper results. To match the paper configuration, use these options: `claude-sonnet-4-6` as the driver LLM, seed 42, RTX 2080 Ti (11 GB) VRAM budget, `configs/production.yaml` defaults.

```bash
git checkout v1.0-reproducible
```

---

## Pipeline

Every run is a four-phase pipeline.

1. **Phase 1** — Planner ↔ Executor ↔ Explorer agent loop. The Planner
   proposes ≤`n_gpu` architectures per round; Executors turn each description
   into runnable code; Explorer analyzes the dataset on demand. Up to 160
   evaluations, early stop on 64-eval no-improvement patience.
2. **Phase 1.5** — Modularize the winning architecture's repeated stages into
   per-stage `nn.Module` classes. Forward-equivalence is validated via a
   state-dict transplant + `torch.allclose(rtol=1e-5, atol=1e-6)`. Runs
   automatically; failures abort the pipeline. Pass `--no-modularize` to skip
   this step — Phase 2 will build a slotted architecture directly from the raw Phase 1 winner. Paper results
   use modularization; skipping is provided as an ablation lever.
3. **Phase 2** — A separate-context LLM rewrites the modularized seed as a
   slotted architecture.
4. **Phase 3** — Regularized evolution over slot choices (default:
   `aggressive` mode, 50/50 single/multi-slot mutation). Top-8 by val-final
   are retrained on train+val for the test-set report.

---

## Quick Start

Reproducing one benchmark end-to-end.

> For the exact paper-reproduction snapshot, `git checkout v1.0-reproducible`
> and follow the README at that tag (the package and CLI there are named
> `autospace`).

```bash
# 1. Environment (Python 3.10+ recommended)
git clone https://github.com/alroimfebruary/AgentNAS && cd AgentNAS
conda create -n agentnas python=3.10 -y
conda activate agentnas
pip install torch torchvision torchaudio # version that matches your CUDA environment
pip install -e ".[dev]"

# 2. Anthropic API key — env var or key file at the repo root
export ANTHROPIC_API_KEY="sk-ant-..."
# (alternative) echo "sk-ant-..." > anthropic_api_key.txt

# 3. Data — download and symlink to the repo root 
ln -s /path/to/NAS-Bench-360-Benchmarks .
ln -s /path/to/NAS-Unseen .
# See the "Expected data layout" section below for the file tree.

# 4. (fsd50k only) precompute mel-spectrograms — may take several hours, one-time
python scripts/precompute_fsd50k.py

# 5. Run one benchmark end-to-end
python scripts/run_production.py --task ninapro --llm-model claude-sonnet-4-6
```

Outputs land in `experiments/<task>_<model>_seed<N>[_blind]/`. The pipeline
runs Phase 1 → Phase 1.5 → Phase 2 → Phase 3 automatically.

### Verify the environment first

Fast checks that don't burn LLM tokens:

```bash
# Loader + explorer-data tests (~1 min; needs data symlinked at repo root)
pytest tests/

# Per-task loader smoke: shape, split, label range, loss finiteness
agentnas smoke-test --task ninapro
agentnas smoke-test --task all         # walks all 17 loaders (needs data)
```

---

## Expected data layout

The two data directories must sit at the repo root (either as symlinks or
copies). Every task lives under its own subdirectory.

- **NAS-Unseen** is available as a data pack from the [Official Unseen NAS Repository](https://github.com/Towers-D/NAS-Unseen-Datasets).
  Its per-task READMEs ship as plain-text `README` files (no `.md`), which differ
  from the curated versions the paper's Planner prompt was written against.
- **NAS-Bench-360** is available from the [Official NAS-Bench-360 Repository](https://github.com/rtu715/NAS-Bench-360).
  It ships no per-task READMEs at all.

For both sources, this repo ships cleaner `README.md` files at
`configs/explorer_readmes/*.md` for **all 17 tasks**, and the pipeline falls
back to them automatically when a raw data dir lacks a `README.md`. No manual
copy step is required to reproduce paper results — but if you prefer to place
them next to the data, copy `configs/explorer_readmes/<task>.md` →
`<data_dir>/<task>/README.md`.

### `./NAS-Bench-360-Benchmarks`

```
NAS-Bench-360-Benchmarks
├── cifar100
│   ├── meta
│   ├── test
│   ├── train
│   └── README.md               # OURS, from configs/explorer_readmes/ (auto-fallback if absent)
├── cosmic
│   ├── cosmic_dataset.py
│   ├── cosmic_test.pt
│   ├── cosmic_train.pt
│   ├── cosmic_valid.pt
│   ├── npy_test
│   ├── npy_train
│   └── README.md               # OURS, from configs/explorer_readmes/ (auto-fallback if absent)
├── darcyflow
│   ├── piececonst_r421_N1024_smooth1.mat
│   ├── piececonst_r421_N1024_smooth2.mat
│   └── README.md               # OURS, from configs/explorer_readmes/ (auto-fallback if absent)
├── deepsea
│   ├── deepsea_filtered.npz
│   └── README.md               # OURS, from configs/explorer_readmes/ (auto-fallback if absent)
├── ecg
│   ├── challenge2017.pkl
│   └── README.md               # OURS, from configs/explorer_readmes/ (auto-fallback if absent)
├── fsd50k
│   ├── chunks
│   ├── dev_22050_chunks
│   ├── eval_22050_chunks
│   ├── noise_22050
│   ├── precomputed             # generated by scripts/precompute_fsd50k.py
│   └── README.md               # OURS, from configs/explorer_readmes/ (auto-fallback if absent)
├── ninapro
│   ├── label_train.npy
│   ├── label_val.npy
│   ├── ninapro_train.npy
│   ├── ninapro_val.npy
│   ├── (test files, held out)
│   └── README.md               # OURS, from configs/explorer_readmes/ (auto-fallback if absent)
├── psicov
│   ├── cameo
│   ├── cameo-hard.lst
│   ├── deepcov
│   ├── deepcov.lst
│   ├── psicov
│   ├── psicov.lst
│   ├── train_ecod.lst
│   └── README.md               # OURS, from configs/explorer_readmes/ (auto-fallback if absent)
├── satellite
│   ├── satellite_train.npy
│   ├── (test files, held out)
│   └── README.md               # OURS, from configs/explorer_readmes/ (auto-fallback if absent)
└── spherical
    ├── s2_cifar100
    └── README.md               # OURS, from configs/explorer_readmes/ (auto-fallback if absent)
```

### `./NAS-Unseen`

```
NAS-Unseen
├── AddNIST
│   ├── metadata
│   ├── train_x.npy
│   ├── train_y.npy
│   ├── valid_x.npy
│   ├── valid_y.npy
│   ├── (test files, held out)
│   ├── README                  # plain text, ships with the Unseen NAS pack — NOT used by the pipeline
│   └── README.md               # OURS, from configs/explorer_readmes/ (auto-fallback if absent)
├── Chesseract
│   ├── ... (same layout as AddNIST)
│   └── README.md
├── CIFARTile
│   ├── ... (same layout)
│   └── README.md
├── GeoClassing
│   ├── ... (same layout, plus BigEarthNetLicense.pdf)
│   └── README.md
├── Gutenberg
│   ├── ... (same layout)
│   └── README.md
├── Language
│   ├── ... (same layout)
│   └── README.md
└── MultNIST
    ├── ... (same layout)
    └── README.md
```

Test files (`test_x.npy` / `test_y.npy` for Unseen; `*_test.*` for NB-360)
are automatically held out from the Explorer's view by
`build_explorer_data_dir` — final training reads from the real data root.

---

## Reproducing Paper Results

Each of the 17 benchmarks is a single command. All use the same defaults from
`configs/production.yaml` (population 16, budget 160, patience 64, proxy
epoch fraction 0.3, 60-min proxy cap, 24-hr final cap). NAS-Unseen tasks add
`--blind-mode`.

```bash
# --- NAS-Bench-360 (10 tasks) ---
python scripts/run_production.py --task fsd50k     --llm-model claude-sonnet-4-6
python scripts/run_production.py --task psicov     --llm-model claude-sonnet-4-6
python scripts/run_production.py --task cifar100   --llm-model claude-sonnet-4-6
python scripts/run_production.py --task spherical  --llm-model claude-sonnet-4-6
python scripts/run_production.py --task ninapro    --llm-model claude-sonnet-4-6
python scripts/run_production.py --task ecg        --llm-model claude-sonnet-4-6
python scripts/run_production.py --task darcyflow  --llm-model claude-sonnet-4-6
python scripts/run_production.py --task satellite  --llm-model claude-sonnet-4-6
python scripts/run_production.py --task deepsea    --llm-model claude-sonnet-4-6
python scripts/run_production.py --task cosmic     --llm-model claude-sonnet-4-6

# --- NAS-Unseen (7 tasks; blind mode strips domain knowledge from prompts) ---
python scripts/run_production.py --task addnist     --llm-model claude-sonnet-4-6 --blind-mode
python scripts/run_production.py --task cifartile   --llm-model claude-sonnet-4-6 --blind-mode
python scripts/run_production.py --task chesseract  --llm-model claude-sonnet-4-6 --blind-mode
python scripts/run_production.py --task geoclassing --llm-model claude-sonnet-4-6 --blind-mode
python scripts/run_production.py --task gutenberg   --llm-model claude-sonnet-4-6 --blind-mode
python scripts/run_production.py --task language    --llm-model claude-sonnet-4-6 --blind-mode
python scripts/run_production.py --task multnist    --llm-model claude-sonnet-4-6 --blind-mode
```

### Modularization

In the original experiments reported in the paper, we optionally applied modularization only when Phase 2 decomposed the architecture into smaller granularities. However, since modularization does not cause any harm even when decomposition does not occur, we now enable it by default. If you specifically want to skip the modularization step, add the `--no-modularize` flag. Below is the full list of reported experiments from the original paper that underwent modularization.

| #  | Task        | LLM Model                      | Seed | Mode                       |
|----|-------------|--------------------------------|------|----------------------------|
| 1  | cifar100    | claude-sonnet-4-6              | 42   | default (non-blind)        |
| 2  | cifartile   | claude-haiku-4-5-20251001      | 42   | blind (Unseen NAS default) |
| 3  | cifartile   | claude-opus-4-6                | 42   | blind (Unseen NAS default) |
| 4  | cifartile   | claude-sonnet-4-6              | 42   | blind (Unseen NAS default) |
| 5  | cifartile   | claude-sonnet-4-6              | 44   | blind (Unseen NAS default) |
| 6  | cosmic      | claude-sonnet-4-6              | 42   | default (non-blind)        |
| 7  | darcyflow   | claude-sonnet-4-6              | 42   | default (non-blind)        |
| 8  | geoclassing | claude-sonnet-4-6              | 42   | blind (Unseen NAS default) |
| 9  | gutenberg   | claude-sonnet-4-6              | 42   | blind (Unseen NAS default) |
| 10 | multnist    | claude-sonnet-4-6              | 42   | blind (Unseen NAS default) |
| 11 | satellite   | claude-sonnet-4-6              | 42   | default (non-blind)        |
| 12 | spherical   | claude-sonnet-4-6              | 44   | default (non-blind)        |


### Resume support

State is atomic-checkpointed after every completed evaluation. Rerunning
the same command picks up at the last saved checkpoint.

```bash
# Kill any run and re-invoke — resumes automatically
python scripts/run_production.py --task fsd50k --llm-model claude-sonnet-4-6

# Force fresh (deletes prior checkpoint)
python scripts/run_production.py --task fsd50k --llm-model claude-sonnet-4-6 --no-resume
```

---

## Ablations

The paper reports four ablations. Each is a CLI flag on top of the standard
command.

### Blind mode (default for Unseen NAS)

Strips the task name and dataset README from LLM prompts. Replicates Unseen NAS competition setting.

```bash
python scripts/run_production.py --task fsd50k --blind-mode
```

Note: `configs/blind_readmes/` only ships curated blind READMEs for the 7
Unseen NAS benchmarks (plus `cifar100` and `geoclassing`). Attempting
`--blind-mode` on other NB-360 tasks will fail with an explicit error.

### Random-search baseline

Replaces Phase 3's evolution with uniform sampling of slot choices under
the same 160-eval budget. Requires a Phase 2 grammar to sample from — run
Phase 1 + Phase 2 first, then invoke Phase 3 in RS mode against that dir.

```bash
python -m agentnas.cli run \
    --task fsd50k --phase 3 --search-mode random \
    --experiment-dir experiments/spherical_claude-sonnet-4-6_seed42
```

### GDAS (one-shot supernet)

One-shot Gumbel-STE supernet in place of discrete evolution.

```bash
python -m agentnas.cli run \
    --task fsd50k --phase 3 --search-mode gdas --gdas-epochs 90 \
    --experiment-dir experiments/spherical_claude-sonnet-4-6_seed42
```

### Fixed-seed architecture (Phase 1 replacement)

Skips Phase 1's LLM loop entirely and seeds with ResNet-18 (Unseen NAS) or
WideResNet-16-4 (NAS-Bench-360). `auto` derives the family from the task.

```bash
python scripts/run_production.py --task ninapro --fixed-seed-arch auto
```

Explicit family: `--fixed-seed-arch resnet18` or `--fixed-seed-arch wideresnet_16_4`.

---

## Configuration

`configs/production.yaml` is used automatically by `run_production.py`.
Paper-reproducible defaults:

```yaml
llm:
  model: claude-sonnet-4-6              # override with --llm-model
  temperature: 0.7
  max_tokens: 64000

budget:
  max_evaluations: 160
  patience: 64                         # Phase 1 stop after N no-improvement evals
  nas_patience: 64                     # Phase 3 same
  max_executor_retries: 10
  max_explorer_steps: 30
  max_slots: 20
  max_alternatives_per_slot: 8

gpu:
  n_gpu: 8                             # RTX 2080 Ti × 8
  max_vram_mb: 11264
  proxy_time_limit_s: 3600             # 60 min proxy training cap
  final_time_limit_s: 86400            # 24 hr final training cap
  proxy_epoch_fraction: 0.3            # proxy runs 30% of proposed epochs

evolution:
  population_size: 16
  max_generations: 10
```

The block above is the paper-relevant subset; `configs/default.yaml` lists
the full knob set (sandbox/validation timeouts, LLM retries, worker kill
buffer, VRAM-gate multiplier, planner-loop guards) — all defaulting to the
paper-run values.

Two conventions worth knowing:

- **Metrics are always lower = better** — a pipeline-wide invariant (report
  `1 - accuracy`, `1 - AUROC`, ...). Selection, patience, evolution fitness,
  and prompts all assume it; `agentnas validate-task` enforces it for custom
  tasks.
- **Extending a stopped Phase 1 search**: resume with a larger budget —
  `agentnas run --task X --phase 1 --budget 300` on the same experiment dir
  continues where it left off (`budget_used` persists; the cap is re-read
  from the flags). A run that already finished its final trainings is
  `completed` and won't reopen — continue with Phase 2/3 instead.

CLI flags override YAML values per run.

### Two CLI entry points

- **`scripts/run_production.py`** — thin wrapper for the standard flow. Exposes
  only the flags a reproduction run needs.
- **`python -m agentnas.cli run`** — full click CLI. Use this when varying
  Phase 3 search strategy (`--search-mode`), population size, patience,
  budget, or GDAS-specific knobs.

Both share the same phase implementations under `src/agentnas/phases/`.

### Useful flags

| Flag | Effect |
|---|---|
| `--task` | Benchmark name (required) |
| `--llm-model` | Anthropic or OpenAI model id (`claude-sonnet-4-6` for paper) |
| `--seed` | Random seed (default 42) |
| `--phase 1\|2\|3` | Single-phase run |
| `--blind-mode` | Strip domain knowledge from prompts |
| `--no-modularize` | Skip Phase 1.5 (stage modularization) |
| `--fixed-seed-arch {auto,resnet18,wideresnet_16_4}` | Skip Phase 1 LLM loop |
| `--search-mode {aggressive,evo,crossover,random,gdas}` | Phase 3 strategy (cli.py) |
| `--pop-size N` | Phase 3 population size (cli.py) |
| `--patience N` | Early-stop patience (cli.py) |
| `--budget N` | Total eval budget (cli.py) |
| `--no-resume` | Force a fresh run |

---

## Supported Benchmarks

| # | Task | Data source | Type | Classical NAS/Expert SOTA (lower = better) | Default mode |
|---|------|-------------|------|-----------------------|--------------|
| 1  | cifar100    | NAS-Bench-360 | 2D classification    | 0.1939 | non-blind |
| 2  | spherical   | NAS-Bench-360 | 2D classification    | 0.4823 | non-blind |
| 3  | ninapro     | NAS-Bench-360 | 2D classification    | 0.0637 | non-blind |
| 4  | fsd50k      | NAS-Bench-360 | 2D multi-label audio | 0.6200 | non-blind |
| 5  | darcyflow   | NAS-Bench-360 | 2D dense regression  | 0.0080 | non-blind |
| 6  | psicov      | NAS-Bench-360 | 2D protein distance  | 2.9400 | non-blind |
| 7  | cosmic      | NAS-Bench-360 | 2D segmentation      | 0.1270 | non-blind |
| 8  | ecg         | NAS-Bench-360 | 1D classification    | 0.2800 | non-blind |
| 9  | satellite   | NAS-Bench-360 | 1D classification    | 0.1251 | non-blind |
| 10 | deepsea     | NAS-Bench-360 | 1D multi-label       | 0.3000 | non-blind |
| 11 | addnist     | NAS-Unseen    | 2D classification    | 0.0209 | blind |
| 12 | language    | NAS-Unseen    | 2D classification    | 0.0208 | blind |
| 13 | multnist    | NAS-Unseen    | 2D classification    | 0.0190 | blind |
| 14 | cifartile   | NAS-Unseen    | 2D classification    | 0.0772 | blind |
| 15 | gutenberg   | NAS-Unseen    | 2D classification    | 0.4598 | blind |
| 16 | geoclassing | NAS-Unseen    | 2D classification    | 0.0397 | blind |
| 17 | chesseract  | NAS-Unseen    | 2D classification    | 0.3117 | blind |

we recommend running a one-time mel-spectrogram precompute for `fsd50k` (`python scripts/precompute_fsd50k.py`).
`spherical` / `deepsea` / `ecg` require a bundled-split preprocess; this runs
automatically on first Explorer invocation, cached thereafter.

## Bring your own dataset

Any dataset you can wrap in one Python file can run through the full
pipeline — see the runnable tutorial in
[`examples/custom_task/`](examples/custom_task/). In short:

```bash
# 1. Define the task: subclass NpyClassificationBenchmark (3 class attrs)
#    or BaseBenchmark (full control), decorated with @register_benchmark —
#    or let an agent write the file for you from the data + a description:
agentnas init-task --task my_task --data-root /path/to/data \
    --describe "ECG windows, 4 classes, npz per split, baseline F1 0.72"

# 2. Validate before spending budget — contract, loaders, test-set
#    isolation, split integrity, metric sanity, and a preview of exactly
#    what the LLM agents will see (init-task already loops on this until
#    it passes; review the generated file yourself regardless):
agentnas validate-task my_task.py --data-root /path/to/data

# 3. Run:
agentnas run --task my_task --phase 1 --benchmark-file my_task.py \
    --data-root /path/to/data
```

Custom tasks declare `explorer_allowed_files` — an allowlist of train/val
files. Anything not listed (your test split) can never reach the LLM, and
the pipeline refuses to run a task with no visibility spec.
