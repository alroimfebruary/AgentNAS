"""Filesystem-level unit tests for the explorer data-directory helper.

Covers all 17 tasks x 2 modes (standard + blind). Asserts that:
- No test files are exposed to the explorer (requirement #1)
- Required train/val files ARE present and resolve to real data
- README source matches the mode (standard vs blind)
- Filename renames are applied (blind) or preserved (standard)
- Bundled tasks (spherical, deepsea, ecg) use preprocessed data
- Domain-leaking files are stripped in blind mode

These tests do not touch the LLM and are fast (<1 min total).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from autospace.utils.explorer_data import (
    BLIND_EXTRA_EXCLUSIONS,
    BLIND_RENAMES,
    BUNDLED_PREPROCESSED_TASKS,
    TEST_EXCLUSIONS,
    build_explorer_data_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NB360_ROOT = PROJECT_ROOT / "NAS-Bench-360-Benchmarks"
UNSEEN_ROOT = PROJECT_ROOT / "NAS-Unseen"
EXPLORER_DATA_ROOT = PROJECT_ROOT / "configs" / "explorer_data"

# Task -> (real_data_root, data_subdir).
# Ordering: all tasks from TEST_EXCLUSIONS.keys().
TASK_LAYOUT: dict[str, tuple[Path, str]] = {
    # Unseen-NAS
    "addnist": (UNSEEN_ROOT, "AddNIST"),
    "language": (UNSEEN_ROOT, "Language"),
    "multnist": (UNSEEN_ROOT, "MultNIST"),
    "cifartile": (UNSEEN_ROOT, "CIFARTile"),
    "gutenberg": (UNSEEN_ROOT, "Gutenberg"),
    "geoclassing": (UNSEEN_ROOT, "GeoClassing"),
    "chesseract": (UNSEEN_ROOT, "Chesseract"),
    # NAS-Bench-360
    "cifar100": (NB360_ROOT, "cifar100"),
    "spherical": (NB360_ROOT, "spherical"),
    "ninapro": (NB360_ROOT, "ninapro"),
    "fsd50k": (NB360_ROOT, "fsd50k"),
    "darcyflow": (NB360_ROOT, "darcyflow"),
    "psicov": (NB360_ROOT, "psicov"),
    "cosmic": (NB360_ROOT, "cosmic"),
    "ecg": (NB360_ROOT, "ecg"),
    "satellite": (NB360_ROOT, "satellite"),
    "deepsea": (NB360_ROOT, "deepsea"),
}

ALL_TASKS = list(TASK_LAYOUT.keys())

# Blind mode is only supported for tasks with a curated blind README.
# Paper ablation uses --blind-mode for the Unseen-NAS benchmarks; a couple of
# NAS-Bench-360 tasks also ship blind READMEs for exploratory ablations.
BLIND_TASKS = sorted(
    p.stem for p in (PROJECT_ROOT / "configs" / "blind_readmes").glob("*.md")
)

# Required train/val filenames per task, in REAL filenames (for standard mode).
# We verify these appear (possibly under a rename in blind mode).
REQUIRED_FILES_STANDARD: dict[str, list[str]] = {
    # Unseen-NAS
    "addnist": ["train_x.npy", "train_y.npy", "valid_x.npy", "valid_y.npy"],
    "language": ["train_x.npy", "train_y.npy", "valid_x.npy", "valid_y.npy"],
    "multnist": ["train_x.npy", "train_y.npy", "valid_x.npy", "valid_y.npy"],
    "cifartile": ["train_x.npy", "train_y.npy", "valid_x.npy", "valid_y.npy"],
    "gutenberg": ["train_x.npy", "train_y.npy", "valid_x.npy", "valid_y.npy"],
    "geoclassing": ["train_x.npy", "train_y.npy", "valid_x.npy", "valid_y.npy"],
    "chesseract": ["train_x.npy", "train_y.npy", "valid_x.npy", "valid_y.npy"],
    # NAS-Bench-360
    "cifar100": ["train", "meta"],
    "spherical": ["s2_cifar100"],
    "ninapro": [
        "ninapro_train.npy", "ninapro_val.npy",
        "label_train.npy", "label_val.npy",
    ],
    "fsd50k": [
        "precomputed/fsd50k_tr.pt",
        "precomputed/fsd50k_val.pt",
    ],
    "darcyflow": ["piececonst_r421_N1024_smooth1.mat"],
    "psicov": ["deepcov", "deepcov.lst", "train_ecod.lst"],
    "cosmic": ["cosmic_train.pt", "cosmic_valid.pt", "cosmic_dataset.py", "npy_train"],
    "ecg": ["challenge2017.pkl"],
    "satellite": ["satellite_train.npy"],
    "deepsea": ["deepsea_filtered.npz"],
}


def _resolve_files(blind_mode: bool, task: str) -> list[str]:
    """Return expected filenames under dataset/ after optional blind rename.

    For blind mode, applies BLIND_RENAMES and BLIND_EXTRA_EXCLUSIONS (cosmic
    drops .pt + .py files entirely and uses raw_train/ from the rename of npy_train/).
    """
    originals = REQUIRED_FILES_STANDARD[task]
    if not blind_mode:
        return originals

    renames = BLIND_RENAMES.get(task, {})
    excluded = set(BLIND_EXTRA_EXCLUSIONS.get(task, []))

    result = []
    for name in originals:
        if name in excluded:
            continue
        result.append(renames.get(name, name))
    return result


@pytest.fixture
def cleanup_tmp():
    """Yield a list; any paths appended get rmtree'd after the test."""
    paths: list[Path] = []
    yield paths
    for p in paths:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


def _skip_if_no_blind_readme(task: str, blind_mode: bool = True) -> None:
    """Skip when the task has no curated blind README (blind-mode is unsupported)."""
    if blind_mode and task not in BLIND_TASKS:
        pytest.skip(
            f"blind mode not supported for task '{task}': "
            f"no configs/blind_readmes/{task}.md"
        )


@pytest.mark.parametrize("task", ALL_TASKS)
@pytest.mark.parametrize("blind_mode", [False, True])
def test_no_test_files_present(task, blind_mode, cleanup_tmp):
    """No test file (per TEST_EXCLUSIONS[task]) appears anywhere in the explorer dir."""
    _skip_if_no_blind_readme(task, blind_mode)
    real_root, subdir = TASK_LAYOUT[task]
    if not real_root.exists():
        pytest.skip(f"data root missing: {real_root}")
    temp_dir, dataset_subdir = build_explorer_data_dir(
        real_data_root=str(real_root),
        task=task,
        task_data_subdir=subdir,
        blind_mode=blind_mode,
        project_root=PROJECT_ROOT,
    )
    cleanup_tmp.append(temp_dir)
    dataset_dir = temp_dir / dataset_subdir

    test_paths = TEST_EXCLUSIONS.get(task, [])
    for rel in test_paths:
        # Exact-path check
        full = dataset_dir / rel
        assert not full.exists(), f"[{task}, blind={blind_mode}] test path present: {rel}"


@pytest.mark.parametrize("task", ALL_TASKS)
@pytest.mark.parametrize("blind_mode", [False, True])
def test_required_files_present_and_valid(task, blind_mode, cleanup_tmp):
    """All expected train/val files are present and resolve to real (existing) paths."""
    _skip_if_no_blind_readme(task, blind_mode)
    real_root, subdir = TASK_LAYOUT[task]
    if not real_root.exists():
        pytest.skip(f"data root missing: {real_root}")
    temp_dir, dataset_subdir = build_explorer_data_dir(
        real_data_root=str(real_root),
        task=task,
        task_data_subdir=subdir,
        blind_mode=blind_mode,
        project_root=PROJECT_ROOT,
    )
    cleanup_tmp.append(temp_dir)
    dataset_dir = temp_dir / dataset_subdir

    for name in _resolve_files(blind_mode, task):
        full = dataset_dir / name
        assert full.exists(), f"[{task}, blind={blind_mode}] missing required file: {name}"
        # Resolve chain must terminate at an existing file or dir
        resolved = full.resolve()
        assert resolved.exists(), (
            f"[{task}, blind={blind_mode}] symlink target missing: {full} -> {resolved}"
        )


@pytest.mark.parametrize("task", ALL_TASKS)
@pytest.mark.parametrize("blind_mode", [False, True])
def test_readme_source_matches_mode(task, blind_mode, cleanup_tmp):
    """The dataset/README.md contents match configs/<mode>_readmes/<task>.md."""
    _skip_if_no_blind_readme(task, blind_mode)
    real_root, subdir = TASK_LAYOUT[task]
    if not real_root.exists():
        pytest.skip(f"data root missing: {real_root}")
    temp_dir, dataset_subdir = build_explorer_data_dir(
        real_data_root=str(real_root),
        task=task,
        task_data_subdir=subdir,
        blind_mode=blind_mode,
        project_root=PROJECT_ROOT,
    )
    cleanup_tmp.append(temp_dir)
    dataset_dir = temp_dir / dataset_subdir

    readme_path = dataset_dir / "README.md"
    assert readme_path.exists(), f"[{task}, blind={blind_mode}] README.md missing"

    expected_subdir = "blind_readmes" if blind_mode else "explorer_readmes"
    expected_src = PROJECT_ROOT / "configs" / expected_subdir / f"{task}.md"
    assert expected_src.exists(), f"Source README missing: {expected_src}"
    assert readme_path.read_text() == expected_src.read_text(), (
        f"[{task}, blind={blind_mode}] README content mismatch"
    )


@pytest.mark.parametrize("task", ALL_TASKS)
def test_blind_renames_applied(task, cleanup_tmp):
    """Blind-mode renames produce renamed entries and suppress originals (when renamed)."""
    _skip_if_no_blind_readme(task)
    real_root, subdir = TASK_LAYOUT[task]
    if not real_root.exists():
        pytest.skip(f"data root missing: {real_root}")
    temp_dir, dataset_subdir = build_explorer_data_dir(
        real_data_root=str(real_root),
        task=task,
        task_data_subdir=subdir,
        blind_mode=True,
        project_root=PROJECT_ROOT,
    )
    cleanup_tmp.append(temp_dir)
    dataset_dir = temp_dir / dataset_subdir

    renames = BLIND_RENAMES.get(task, {})
    excluded = set(BLIND_EXTRA_EXCLUSIONS.get(task, []))

    for original, blind_name in renames.items():
        if original in excluded:
            # Some renames map an excluded->new (e.g., cosmic npy_train→raw_train but
            # the npy_train source is excluded; raw_train shouldn't exist either in
            # that degenerate case). 
            continue
        assert (dataset_dir / blind_name).exists(), (
            f"[{task}] blind name missing: {blind_name}"
        )
        # Original name at top level should not exist UNLESS it happens to also
        # match the blind name (identity rename). The rename table uses distinct
        # keys so this assertion is safe.
        if original != blind_name:
            assert not (dataset_dir / original).exists(), (
                f"[{task}] original name still present (not renamed): {original}"
            )


@pytest.mark.parametrize("task", ALL_TASKS)
def test_standard_mode_no_renames(task, cleanup_tmp):
    """In standard mode, original filenames are preserved (no renames applied)."""
    real_root, subdir = TASK_LAYOUT[task]
    if not real_root.exists():
        pytest.skip(f"data root missing: {real_root}")
    temp_dir, dataset_subdir = build_explorer_data_dir(
        real_data_root=str(real_root),
        task=task,
        task_data_subdir=subdir,
        blind_mode=False,
        project_root=PROJECT_ROOT,
    )
    cleanup_tmp.append(temp_dir)
    dataset_dir = temp_dir / dataset_subdir

    # Renames should NOT be applied in standard mode — original file names must be present
    for original in BLIND_RENAMES.get(task, {}):
        if task in BUNDLED_PREPROCESSED_TASKS:
            # In bundled tasks, source dir is the preprocessed dir; the original
            # real-dir filename is what's inside the preprocessed dir (by convention
            # our preprocessing preserves the original name).
            pass
        # Even bundled tasks: we preserve original name in standard mode
        assert (dataset_dir / original).exists(), (
            f"[{task}] standard mode should keep original name: {original}"
        )


@pytest.mark.parametrize("task", sorted(BUNDLED_PREPROCESSED_TASKS))
@pytest.mark.parametrize("blind_mode", [False, True])
def test_bundled_tasks_use_preprocessed_source(task, blind_mode, cleanup_tmp):
    _skip_if_no_blind_readme(task, blind_mode)
    """For bundled tasks, symlinks resolve into configs/explorer_data/<task>/."""
    real_root, subdir = TASK_LAYOUT[task]
    if not real_root.exists():
        pytest.skip(f"data root missing: {real_root}")
    preprocessed_root = EXPLORER_DATA_ROOT / task
    if not preprocessed_root.exists():
        pytest.skip(
            f"preprocessed dir missing: {preprocessed_root}; "
            f"run: python scripts/prepare_explorer_data.py"
        )
    temp_dir, dataset_subdir = build_explorer_data_dir(
        real_data_root=str(real_root),
        task=task,
        task_data_subdir=subdir,
        blind_mode=blind_mode,
        project_root=PROJECT_ROOT,
    )
    cleanup_tmp.append(temp_dir)
    dataset_dir = temp_dir / dataset_subdir

    # Pick a representative file
    for name in _resolve_files(blind_mode, task):
        # Skip the README copy (not a symlink)
        full = dataset_dir / name
        if full.is_symlink():
            target = full.resolve()
            assert str(target).startswith(str(preprocessed_root.resolve())), (
                f"[{task}, blind={blind_mode}] {name} resolves to {target}, "
                f"not inside {preprocessed_root}"
            )


@pytest.mark.parametrize("task", ALL_TASKS)
def test_blind_strips_domain_files(task, cleanup_tmp):
    _skip_if_no_blind_readme(task)
    """Blind mode excludes README, metadata, SVG, PDF from top level."""
    real_root, subdir = TASK_LAYOUT[task]
    if not real_root.exists():
        pytest.skip(f"data root missing: {real_root}")
    temp_dir, dataset_subdir = build_explorer_data_dir(
        real_data_root=str(real_root),
        task=task,
        task_data_subdir=subdir,
        blind_mode=True,
        project_root=PROJECT_ROOT,
    )
    cleanup_tmp.append(temp_dir)
    dataset_dir = temp_dir / dataset_subdir

    # The only README.md should be the blind one we copy; no metadata/svg/pdf.
    for entry in dataset_dir.iterdir():
        if entry.is_symlink():
            tgt_name = entry.name.lower()
            assert tgt_name != "metadata", f"[{task}] metadata symlink present"
            assert not tgt_name.endswith(".svg"), f"[{task}] SVG symlink: {entry.name}"
            assert not tgt_name.endswith(".pdf"), f"[{task}] PDF symlink: {entry.name}"
