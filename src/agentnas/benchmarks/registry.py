"""Benchmark registry: maps task names to loader classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentnas.utils.paths import project_root

if TYPE_CHECKING:
    from agentnas.benchmarks.base import BaseBenchmark

# Registry of task name → (module_path, class_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    # NAS-Bench-360 tasks
    "cifar100": ("agentnas.benchmarks.cifar100", "CIFAR100Benchmark"),
    "spherical": ("agentnas.benchmarks.spherical", "SphericalBenchmark"),
    "ninapro": ("agentnas.benchmarks.ninapro", "NinaProBenchmark"),
    "satellite": ("agentnas.benchmarks.satellite", "SatelliteBenchmark"),
    "deepsea": ("agentnas.benchmarks.deepsea", "DeepSEABenchmark"),
    "psicov": ("agentnas.benchmarks.psicov", "PsicovBenchmark"),
    "cosmic": ("agentnas.benchmarks.cosmic", "CosmicBenchmark"),
    "fsd50k": ("agentnas.benchmarks.fsd50k", "FSD50KBenchmark"),
    "ecg": ("agentnas.benchmarks.ecg", "ECGBenchmark"),
    "darcyflow": ("agentnas.benchmarks.darcyflow", "DarcyFlowBenchmark"),
    # NAS-Unseen tasks
    "addnist": ("agentnas.benchmarks.unseen", "UnseenNASBenchmark"),
    "language": ("agentnas.benchmarks.unseen", "UnseenNASBenchmark"),
    "multnist": ("agentnas.benchmarks.unseen", "UnseenNASBenchmark"),
    "cifartile": ("agentnas.benchmarks.unseen", "UnseenNASBenchmark"),
    "gutenberg": ("agentnas.benchmarks.unseen", "UnseenNASBenchmark"),
    "geoclassing": ("agentnas.benchmarks.unseen", "UnseenNASBenchmark"),
    "chesseract": ("agentnas.benchmarks.unseen", "UnseenNASBenchmark"),
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
                   If None, defaults to the standard directory under the
                   repo root (CWD-independent).
    """
    if task_name not in _REGISTRY:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list_benchmarks()}"
        )

    if data_root is None:
        root = project_root()
        if task_name in _UNSEEN_TASKS:
            data_root = str(root / _DATA_ROOTS["unseen"])
        else:
            data_root = str(root / _DATA_ROOTS["nb360"])
    elif task_name in _UNSEEN_TASKS:
        # Override: unseen tasks always use the NAS-Unseen directory,
        # even if data_root points to NAS-Bench-360-Benchmarks
        nb360_root = _DATA_ROOTS["nb360"]
        if data_root.endswith(nb360_root) or nb360_root in data_root:
            data_root = str(project_root() / _DATA_ROOTS["unseen"])

    module_path, class_name = _REGISTRY[task_name]
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Unseen tasks need task_name to select the specific dataset
    if task_name in _UNSEEN_TASKS:
        return cls(data_root=data_root, task_name=task_name)
    return cls(data_root=data_root)
