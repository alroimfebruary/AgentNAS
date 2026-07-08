#!/usr/bin/env python
"""Preprocess bundled-split datasets for the Data Explorer.

Tasks whose benchmark class sets `explorer_preprocessed_name` (currently
spherical, deepsea, ecg) store train+val+test in a single file and need a
train+val-only copy for the Explorer. This script is optional —
`build_explorer_data_dir()` auto-generates these files on first use. Running
this script explicitly just front-loads the work so the first experiment
doesn't pay the ~1-minute cost.

Outputs land in `configs/explorer_data/<task>/`. Idempotent — skips tasks that
are already prepared.

Usage:
    python scripts/prepare_explorer_data.py
    python scripts/prepare_explorer_data.py --force                # overwrite existing
    python scripts/prepare_explorer_data.py --task spherical       # single task
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("prepare")


def main() -> None:
    from agentnas.benchmarks.registry import get_benchmark, list_benchmarks
    from agentnas.utils.explorer_data import ensure_preprocessed

    benchmarks = {t: get_benchmark(t) for t in list_benchmarks()}
    bundled = sorted(
        t for t, b in benchmarks.items() if b.explorer_preprocessed_name
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=bundled, default=None,
        help="Prepare a single task (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate even if output exists",
    )
    args = parser.parse_args()

    real_root = PROJECT_ROOT / "NAS-Bench-360-Benchmarks"
    if not real_root.exists():
        logger.error(f"source data root not found: {real_root}")
        sys.exit(1)

    tasks = [args.task] if args.task else bundled
    for task in tasks:
        out_dir = PROJECT_ROOT / "configs" / "explorer_data" / task
        if args.force and out_dir.exists():
            logger.info(f"[{task}] --force: removing existing {out_dir}")
            shutil.rmtree(out_dir)

        logger.info(f"[{task}] preparing...")
        produced = ensure_preprocessed(benchmarks[task], str(real_root), PROJECT_ROOT)
        logger.info(f"[{task}] ok → {produced}")


if __name__ == "__main__":
    main()
