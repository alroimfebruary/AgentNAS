"""Benchmark registry: maps task names to loader classes.

Builtin benchmarks are registered lazily as (module_path, class_name) pairs.
Custom benchmarks register a class object at runtime:

    from agentnas.benchmarks.registry import register_benchmark

    @register_benchmark("my_task")
    class MyTaskBenchmark(BaseBenchmark):
        ...

Put the class in its own file and pass it via ``--benchmark-file my_task.py``
(the pipeline calls ``load_benchmark_file()``), or call
``register_benchmark(name, cls)`` from your own code before the pipeline
starts. Training worker subprocesses replay the loaded files (recorded in
``loaded_benchmark_files()`` and carried through the worker's args.json), so
custom tasks resolve there too.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from agentnas.utils.paths import project_root

if TYPE_CHECKING:
    from agentnas.benchmarks.base import BaseBenchmark

# Builtin registry: task name → (module_path, class_name), imported on demand.
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

# Custom registry: task name → class object (populated at runtime).
_CUSTOM_REGISTRY: dict[str, type] = {}

# Files loaded via load_benchmark_file(), resolved, in load order.
_LOADED_BENCHMARK_FILES: list[str] = []

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


def register_benchmark(name: str, cls: type | None = None):
    """Register a custom benchmark class under a task name.

    Usable as a decorator (``@register_benchmark("my_task")``) or called
    directly (``register_benchmark("my_task", MyBenchmark)``). Registering a
    name that already exists — builtin or custom — is an error.
    """

    def _register(c: type) -> type:
        if name in _REGISTRY or name in _CUSTOM_REGISTRY:
            raise ValueError(f"Benchmark '{name}' is already registered")
        _CUSTOM_REGISTRY[name] = c
        return c

    if cls is not None:
        return _register(cls)
    return _register


def load_benchmark_file(path: str | Path) -> list[str]:
    """Import a Python file so its ``@register_benchmark`` classes register.

    Returns the task names newly registered by the file. Loading the same
    file a second time is a no-op (returns []). A file that registers
    nothing is an error.
    """
    resolved = str(Path(path).resolve())
    if resolved in _LOADED_BENCHMARK_FILES:
        return []
    if not Path(resolved).exists():
        raise FileNotFoundError(f"Benchmark file not found: {resolved}")

    before = set(_CUSTOM_REGISTRY)
    module_name = f"agentnas_custom_{len(_LOADED_BENCHMARK_FILES)}_{Path(resolved).stem}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _LOADED_BENCHMARK_FILES.append(resolved)

    new_tasks = sorted(set(_CUSTOM_REGISTRY) - before)
    if not new_tasks:
        raise ValueError(
            f"{resolved} did not register any benchmark. Decorate your class "
            f"with @register_benchmark('<task_name>')."
        )
    return new_tasks


def loaded_benchmark_files() -> list[str]:
    """Files loaded via load_benchmark_file(), for worker subprocess replay."""
    return list(_LOADED_BENCHMARK_FILES)


def list_benchmarks() -> list[str]:
    """Return all registered benchmark task names (builtin + custom)."""
    return sorted(set(_REGISTRY) | set(_CUSTOM_REGISTRY))


def get_benchmark(task_name: str, data_root: str | None = None) -> "BaseBenchmark":
    """Instantiate a benchmark loader by task name.

    Args:
        task_name: Name of the benchmark task.
        data_root: Root directory containing benchmark data. If None,
                   builtin tasks default to the standard directory under the
                   repo root (CWD-independent); custom tasks require an
                   explicit value.
    """
    if task_name in _CUSTOM_REGISTRY:
        if data_root is None:
            raise ValueError(
                f"Custom task '{task_name}' requires an explicit data_root "
                f"(pass --data-root or set data_root in the config)."
            )
        return _CUSTOM_REGISTRY[task_name](data_root=data_root)

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
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # Unseen tasks need task_name to select the specific dataset
    if task_name in _UNSEEN_TASKS:
        return cls(data_root=data_root, task_name=task_name)
    return cls(data_root=data_root)
