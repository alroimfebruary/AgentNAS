# Bring your own dataset

AgentNAS runs its full LLM + grammar + evolution pipeline on any dataset you
can wrap in one Python file. This directory is a runnable end-to-end example.

## 1. Generate toy data

```bash
python examples/custom_task/make_toy_data.py --out ./toy_data
```

Produces `toy_data/toy_task/{train,valid,test}_{x,y}.npy` — a 4-class
synthetic classification task with (3, 16, 16) inputs.

## 2. Define the task (one file)

Don't want to write it yourself? An LLM agent can inspect the data and
author the file, iterating against the validator until it passes:

```bash
agentnas init-task --task toy_task --data-root ./toy_data \
    --describe "synthetic 4-class images, (3,16,16) float32, pre-split npy"
```

The result is an ordinary benchmark file you review and own. Two manual
authoring styles:

- **[`toy_task.py`](toy_task.py)** — the convenience route. If your data is
  pre-split `.npy` classification arrays, a task is three class attributes on
  `NpyClassificationBenchmark`. Loaders, loss, metric, and Explorer test-set
  isolation come for free.
- **[`full_custom_task.py`](full_custom_task.py)** — the full-control route
  against `BaseBenchmark`, for custom formats, dense outputs, or special
  metrics. You implement the loaders and metric yourself and declare
  `explorer_allowed_files` — the allowlist of files the LLM Explorer may see.

Rules that keep results trustworthy:

- **`__init__` must not read data** — metadata introspection works without
  the dataset present.
- **Metric is lower = better** (report `1 - accuracy`, `1 - AUROC`, ...).
- **Declare `explorer_allowed_files`** listing train/val files only. Files
  not on the list — your test split — can never reach the LLM. The pipeline
  refuses to run a task with no visibility spec.

## 3. Validate before spending budget

```bash
agentnas validate-task examples/custom_task/toy_task.py --data-root ./toy_data
```

Checks the contract, runs the loader smoke test, verifies test-set isolation
and split disjointness, sanity-checks the metric, and prints exactly what the
LLM agents will see (file tree + README). Fix every `[FAIL]` — and eyeball
the previews; label semantics are yours to verify.

## 4. Run a search

```bash
agentnas run --task toy_task --phase 1 \
    --benchmark-file examples/custom_task/toy_task.py \
    --data-root ./toy_data --n-gpu 1 --budget 20
```

or the full pipeline (Phase 1 → 1.5 → 2 → 3):

```bash
python scripts/run_production.py --task toy_task \
    --benchmark-file examples/custom_task/toy_task.py --data-root ./toy_data
```

The `--benchmark-file` registration is forwarded to every training worker
subprocess automatically.
