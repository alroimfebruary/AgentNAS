#!/usr/bin/env python
"""Generate a tiny synthetic classification dataset for the custom-task tutorial.

Writes toy_task/{train,valid,test}_{x,y}.npy under --out (default: ./toy_data).
Four gaussian-blob classes on (3, 16, 16) inputs — learnable in seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_split(
    rng: np.random.Generator, n: int, n_classes: int = 4,
    shape: tuple[int, ...] = (3, 16, 16),
) -> tuple[np.ndarray, np.ndarray]:
    y = rng.integers(0, n_classes, size=n)
    x = rng.normal(0.0, 1.0, size=(n, *shape)).astype(np.float32)
    # Class-dependent mean shift on one channel → learnable signal
    for c in range(n_classes):
        x[y == c, c % shape[0], :, :] += 1.5
    return x, y.astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="toy_data", help="Output data root")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    task_dir = Path(args.out) / "toy_task"
    task_dir.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", 2048), ("valid", 512), ("test", 512)):
        x, y = make_split(rng, n)
        np.save(task_dir / f"{split}_x.npy", x)
        np.save(task_dir / f"{split}_y.npy", y)
        print(f"{split}: x{x.shape} y{y.shape} -> {task_dir}")


if __name__ == "__main__":
    main()
