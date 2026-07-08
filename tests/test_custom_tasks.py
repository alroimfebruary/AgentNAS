"""Unit tests for the custom-task plugin layer.

Covers dynamic registration, --benchmark-file loading, the convenience
NpyClassificationBenchmark, Explorer allowlist isolation, the no-spec
refusal rule, and the README fallback chain. Everything runs on synthetic
data under tmp_path — no benchmark downloads needed.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from agentnas.benchmarks import registry
from agentnas.benchmarks.base import BaseBenchmark
from agentnas.benchmarks.registry import (
    get_benchmark,
    list_benchmarks,
    load_benchmark_file,
    loaded_benchmark_files,
    register_benchmark,
)
from agentnas.benchmarks.simple import NpyClassificationBenchmark
from agentnas.config import TaskConfig
from agentnas.utils.explorer_data import build_explorer_data_dir

N_TRAIN, N_VAL, N_TEST = 32, 16, 16
SHAPE = (3, 8, 8)
N_CLASSES = 4


@pytest.fixture(autouse=True)
def _clean_registry():
    """Snapshot and restore the dynamic-registration state around each test."""
    custom = dict(registry._CUSTOM_REGISTRY)
    loaded = list(registry._LOADED_BENCHMARK_FILES)
    yield
    registry._CUSTOM_REGISTRY.clear()
    registry._CUSTOM_REGISTRY.update(custom)
    registry._LOADED_BENCHMARK_FILES[:] = loaded


def _write_toy_npy(task_dir: Path, extra_files: bool = False) -> None:
    """Write disjoint train/valid/test splits (plus leak-bait extras)."""
    task_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    offset = 0
    for split, n in (("train", N_TRAIN), ("valid", N_VAL), ("test", N_TEST)):
        # Disjoint content across splits via the running offset.
        x = (
            rng.normal(0, 1, size=(n, *SHAPE)).astype(np.float32)
            + np.arange(offset, offset + n, dtype=np.float32).reshape(-1, 1, 1, 1)
        )
        y = (np.arange(n) % N_CLASSES).astype(np.int64)
        np.save(task_dir / f"{split}_x.npy", x)
        np.save(task_dir / f"{split}_y.npy", y)
        offset += n
    if extra_files:
        (task_dir / "secret_notes.txt").write_text("do not show the LLM")
        nested = task_dir / "nested"
        nested.mkdir()
        (nested / "labels_test_meta.json").write_text("{}")


def _toy_class(task_name: str) -> type:
    """A fresh convenience subclass bound to task_name."""

    class _Toy(NpyClassificationBenchmark):
        task = task_name
        data_shape = SHAPE
        n_classes = N_CLASSES

    return _Toy


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_register_decorator_and_direct():
    cls = register_benchmark("t_reg_deco")(_toy_class("t_reg_deco"))
    assert "t_reg_deco" in list_benchmarks()
    register_benchmark("t_reg_direct", _toy_class("t_reg_direct"))
    assert "t_reg_direct" in list_benchmarks()
    assert cls is registry._CUSTOM_REGISTRY["t_reg_deco"]


def test_register_collision_builtin_and_custom():
    with pytest.raises(ValueError, match="already registered"):
        register_benchmark("ninapro", _toy_class("ninapro"))
    register_benchmark("t_collide", _toy_class("t_collide"))
    with pytest.raises(ValueError, match="already registered"):
        register_benchmark("t_collide", _toy_class("t_collide"))


def test_custom_task_requires_explicit_data_root(tmp_path):
    register_benchmark("t_no_root", _toy_class("t_no_root"))
    with pytest.raises(ValueError, match="explicit data_root"):
        get_benchmark("t_no_root")
    _write_toy_npy(tmp_path / "t_no_root")
    bench = get_benchmark("t_no_root", data_root=str(tmp_path))
    assert bench.task_name == "t_no_root"


def test_load_benchmark_file_registers_and_is_idempotent(tmp_path):
    bench_file = tmp_path / "my_task.py"
    bench_file.write_text(textwrap.dedent("""\
        from agentnas.benchmarks.registry import register_benchmark
        from agentnas.benchmarks.simple import NpyClassificationBenchmark


        @register_benchmark("t_from_file")
        class TFromFile(NpyClassificationBenchmark):
            task = "t_from_file"
            data_shape = (3, 8, 8)
            n_classes = 4
    """))
    assert load_benchmark_file(bench_file) == ["t_from_file"]
    assert str(bench_file.resolve()) in loaded_benchmark_files()
    # Second load: no-op, no collision error
    assert load_benchmark_file(bench_file) == []


def test_load_benchmark_file_rejects_no_registration(tmp_path):
    bench_file = tmp_path / "empty.py"
    bench_file.write_text("x = 1\n")
    with pytest.raises(ValueError, match="did not register"):
        load_benchmark_file(bench_file)


# ---------------------------------------------------------------------------
# Convenience class
# ---------------------------------------------------------------------------

def test_convenience_loaders_metric_and_taskconfig(tmp_path):
    _write_toy_npy(tmp_path / "t_conv")
    register_benchmark("t_conv", _toy_class("t_conv"))
    bench = get_benchmark("t_conv", data_root=str(tmp_path))

    tl, vl = bench.get_search_loaders(batch_size=8, num_workers=0)
    assert len(tl.dataset) == N_TRAIN and len(vl.dataset) == N_VAL
    x, y = next(iter(tl))
    assert tuple(x.shape[1:]) == SHAPE and x.dtype == torch.float32
    tvl, tel = bench.get_final_loaders(batch_size=8, num_workers=0)
    assert len(tvl.dataset) == N_TRAIN + N_VAL and len(tel.dataset) == N_TEST

    # Perfect predictions → metric 0; wrong-everywhere → 1
    targets = torch.arange(8) % N_CLASSES
    perfect = torch.nn.functional.one_hot(targets, N_CLASSES).float()
    assert bench.compute_metric([perfect], [targets]) == pytest.approx(0.0)

    tc = TaskConfig.from_benchmark(bench)
    assert tc.name == "t_conv" and tc.num_classes == N_CLASSES
    assert tc.loss == "cross_entropy" and tc.metric == "accuracy"
    assert tc.sota_metric is None

    # Isolation by construction: allowlist covers train/val only
    assert set(bench.explorer_allowed_files) == {
        "train_x.npy", "train_y.npy", "valid_x.npy", "valid_y.npy",
    }


def test_convenience_requires_class_attrs(tmp_path):
    class Incomplete(NpyClassificationBenchmark):
        task = "t_incomplete"

    with pytest.raises(TypeError, match="must set the class attributes"):
        Incomplete(data_root=str(tmp_path))


# ---------------------------------------------------------------------------
# Explorer isolation
# ---------------------------------------------------------------------------

def test_explorer_allowlist_hides_everything_else(tmp_path):
    _write_toy_npy(tmp_path / "t_iso", extra_files=True)
    register_benchmark("t_iso", _toy_class("t_iso"))
    bench = get_benchmark("t_iso", data_root=str(tmp_path))

    temp_dir, sub = build_explorer_data_dir(
        real_data_root=str(tmp_path), benchmark=bench,
        blind_mode=False, project_root=tmp_path,
    )
    try:
        visible = {
            str(p.relative_to(temp_dir / sub))
            for p in (temp_dir / sub).rglob("*")
        }
        assert visible == {
            "train_x.npy", "train_y.npy", "valid_x.npy", "valid_y.npy",
            "README.md",
        }
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_explorer_allowlist_partial_directory(tmp_path):
    task_dir = tmp_path / "t_part"
    (task_dir / "sub").mkdir(parents=True)
    np.save(task_dir / "sub" / "train.npy", np.zeros(4))
    np.save(task_dir / "sub" / "test.npy", np.ones(4))
    np.save(task_dir / "other.npy", np.ones(2))

    class Partial(BaseBenchmark):
        loss_name = "mse"
        metric_name = "mae"
        explorer_allowed_files = ("sub/train.npy",)

        def __init__(self, data_root: str):
            super().__init__(data_root, "t_part")

        def get_search_loaders(self, batch_size, num_workers=4):
            raise NotImplementedError

        def get_final_loaders(self, batch_size, num_workers=4):
            raise NotImplementedError

        def loss_fn(self):
            return torch.nn.MSELoss()

        def compute_metric(self, all_preds, all_targets, all_extras=None):
            return 0.0

        def input_shape(self):
            return (4,)

        def output_spec(self):
            return {"output_shape": (4,)}

    temp_dir, sub = build_explorer_data_dir(
        real_data_root=str(tmp_path), benchmark=Partial(str(tmp_path)),
        blind_mode=False, project_root=tmp_path,
    )
    try:
        visible = {
            str(p.relative_to(temp_dir / sub))
            for p in (temp_dir / sub).rglob("*")
        }
        assert visible == {"sub", "sub/train.npy", "README.md"}
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_explorer_refuses_benchmark_without_spec(tmp_path):
    (tmp_path / "t_nospec").mkdir()

    class NoSpec(BaseBenchmark):
        def __init__(self, data_root: str):
            super().__init__(data_root, "t_nospec")

        def get_search_loaders(self, batch_size, num_workers=4):
            raise NotImplementedError

        def get_final_loaders(self, batch_size, num_workers=4):
            raise NotImplementedError

        def loss_fn(self):
            return torch.nn.MSELoss()

        def compute_metric(self, all_preds, all_targets, all_extras=None):
            return 0.0

        def input_shape(self):
            return (1,)

        def output_spec(self):
            return {"num_classes": 2}

    with pytest.raises(ValueError, match="no Explorer visibility spec"):
        build_explorer_data_dir(
            real_data_root=str(tmp_path), benchmark=NoSpec(str(tmp_path)),
            blind_mode=False, project_root=tmp_path,
        )


# ---------------------------------------------------------------------------
# README fallback chain
# ---------------------------------------------------------------------------

def _build_and_read_readme(tmp_path, bench, fake_project_root) -> str:
    temp_dir, sub = build_explorer_data_dir(
        real_data_root=str(tmp_path), benchmark=bench,
        blind_mode=False, project_root=fake_project_root,
    )
    try:
        return (temp_dir / sub / "README.md").read_text()
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_readme_autogenerated_from_metadata(tmp_path):
    _write_toy_npy(tmp_path / "t_auto")
    register_benchmark("t_auto", _toy_class("t_auto"))
    bench = get_benchmark("t_auto", data_root=str(tmp_path))
    text = _build_and_read_readme(tmp_path, bench, tmp_path)
    assert "# t_auto" in text
    assert "cross_entropy" in text and "accuracy" in text
    assert "train_x.npy" in text and "test_x.npy" not in text


def test_readme_class_hook_beats_autogen(tmp_path):
    _write_toy_npy(tmp_path / "t_hook")

    class Hooked(_toy_class("t_hook")):
        def explorer_readme(self):
            return "# custom readme from the class\n"

    register_benchmark("t_hook", Hooked)
    bench = get_benchmark("t_hook", data_root=str(tmp_path))
    text = _build_and_read_readme(tmp_path, bench, tmp_path)
    assert text == "# custom readme from the class\n"


def test_readme_curated_file_beats_class_hook(tmp_path):
    _write_toy_npy(tmp_path / "t_curated")

    class Hooked(_toy_class("t_curated")):
        def explorer_readme(self):
            return "# from the class\n"

    register_benchmark("t_curated", Hooked)
    bench = get_benchmark("t_curated", data_root=str(tmp_path))

    fake_root = tmp_path / "fake_repo"
    readmes = fake_root / "configs" / "explorer_readmes"
    readmes.mkdir(parents=True)
    (readmes / "t_curated.md").write_text("# curated wins\n")

    text = _build_and_read_readme(tmp_path, bench, fake_root)
    assert text == "# curated wins\n"
