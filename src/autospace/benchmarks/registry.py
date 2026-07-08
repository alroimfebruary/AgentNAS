"""Benchmark registry: maps task names to loader classes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autospace.benchmarks.base import BaseBenchmark

# Registry of task name → (module_path, class_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    # NAS-Bench-360 tasks
    "cifar100": ("autospace.benchmarks.cifar100", "CIFAR100Benchmark"),
    "spherical": ("autospace.benchmarks.spherical", "SphericalBenchmark"),
    "ninapro": ("autospace.benchmarks.ninapro", "NinaProBenchmark"),
    "satellite": ("autospace.benchmarks.satellite", "SatelliteBenchmark"),
    "deepsea": ("autospace.benchmarks.deepsea", "DeepSEABenchmark"),
    "psicov": ("autospace.benchmarks.psicov", "PsicovBenchmark"),
    "cosmic": ("autospace.benchmarks.cosmic", "CosmicBenchmark"),
    "fsd50k": ("autospace.benchmarks.fsd50k", "FSD50KBenchmark"),
    "ecg": ("autospace.benchmarks.ecg", "ECGBenchmark"),
    "darcyflow": ("autospace.benchmarks.darcyflow", "DarcyFlowBenchmark"),
    # NAS-Unseen tasks
    "addnist": ("autospace.benchmarks.unseen", "UnseenNASBenchmark"),
    "language": ("autospace.benchmarks.unseen", "UnseenNASBenchmark"),
    "multnist": ("autospace.benchmarks.unseen", "UnseenNASBenchmark"),
    "cifartile": ("autospace.benchmarks.unseen", "UnseenNASBenchmark"),
    "gutenberg": ("autospace.benchmarks.unseen", "UnseenNASBenchmark"),
    "geoclassing": ("autospace.benchmarks.unseen", "UnseenNASBenchmark"),
    "chesseract": ("autospace.benchmarks.unseen", "UnseenNASBenchmark"),
}

# Default data roots per task category
_DATA_ROOTS = {
    "nb360": "NAS-Bench-360-Benchmarks",
    "unseen": "NAS-Unseen",
}

# Tasks that use the NAS-Unseen data root
_UNSEEN_TASKS = {
    "addnist", "language", "multnist", "cifartile",
    "gutenberg", "geoclassing", "chesseract",
}


def list_benchmarks() -> list[str]:
    """Return all registered benchmark task names."""
    return sorted(_REGISTRY.keys())


def get_benchmark(task_name: str, data_root: str | None = None) -> "BaseBenchmark":
    """Instantiate a benchmark loader by task name.

    Args:
        task_name: Name of the benchmark task.
        data_root: Root directory containing benchmark data.
                   If None, auto-detects from CWD.
    """
    if task_name not in _REGISTRY:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list_benchmarks()}"
        )

    if data_root is None:
        cwd = Path.cwd()
        if task_name in _UNSEEN_TASKS:
            data_root = str(cwd / _DATA_ROOTS["unseen"])
        else:
            data_root = str(cwd / _DATA_ROOTS["nb360"])
    elif task_name in _UNSEEN_TASKS:
        # Override: unseen tasks always use the NAS-Unseen directory,
        # even if data_root points to NAS-Bench-360-Benchmarks
        nb360_root = _DATA_ROOTS["nb360"]
        if data_root.endswith(nb360_root) or nb360_root in data_root:
            cwd = Path.cwd()
            data_root = str(cwd / _DATA_ROOTS["unseen"])

    module_path, class_name = _REGISTRY[task_name]
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Unseen tasks need task_name to select the specific dataset
    if task_name in _UNSEEN_TASKS:
        return cls(data_root=data_root, task_name=task_name)
    return cls(data_root=data_root)
