"""Minimal custom task: 4-class toy classification from pre-split .npy files.

The convenience base class supplies loaders, loss, metric, and automatic
Explorer test-set isolation — a custom task is three class attributes.

Usage:
    python examples/custom_task/make_toy_data.py --out ./toy_data
    agentnas validate-task examples/custom_task/toy_task.py --data-root ./toy_data
    agentnas run --task toy_task --phase 1 \\
        --benchmark-file examples/custom_task/toy_task.py --data-root ./toy_data
"""

from agentnas.benchmarks.registry import register_benchmark
from agentnas.benchmarks.simple import NpyClassificationBenchmark


@register_benchmark("toy_task")
class ToyTaskBenchmark(NpyClassificationBenchmark):
    task = "toy_task"
    data_shape = (3, 16, 16)
    n_classes = 4
