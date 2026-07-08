"""Explorer data directory construction with test-set isolation and optional blinding.

Builds a filtered temporary directory that the Data Explorer sees instead of the real
data directory. Test files are always excluded so the explorer cannot peek at test
data. In blind mode, domain-leaking files are additionally removed and select
filenames are aliased to generic names.

The benchmark loader (`get_benchmark(..., data_root=config.data_root)`) continues to
use the untouched real data directory for training and evaluation — this module only
affects what the explorer agent sees.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

# -----------------------------------------------------------------------------
# Per-task configuration tables
# -----------------------------------------------------------------------------

# Paths (relative to the task's data dir) that hold test data. Always excluded,
# both in standard and blind modes. Supports nested paths (e.g. "a/b.txt").
# Directory entries exclude the entire subtree.
TEST_EXCLUSIONS: dict[str, list[str]] = {
    # Unseen-NAS — identical layout
    "addnist": ["test_x.npy", "test_y.npy"],
    "language": ["test_x.npy", "test_y.npy"],
    "multnist": ["test_x.npy", "test_y.npy"],
    "cifartile": ["test_x.npy", "test_y.npy"],
    "gutenberg": ["test_x.npy", "test_y.npy"],
    "geoclassing": ["test_x.npy", "test_y.npy"],
    "chesseract": ["test_x.npy", "test_y.npy"],
    # NAS-Bench-360 — separable
    "cifar100": ["test"],
    "cosmic": ["cosmic_test.pt", "npy_test"],
    "ninapro": ["ninapro_test.npy", "label_test.npy"],
    "satellite": ["satellite_test.npy"],
    # Nested exclusions
    "fsd50k": [
        "eval_22050_chunks",
        "chunks/eval.csv",
        "precomputed/fsd50k_eval.pt",
    ],
    "psicov": ["psicov", "cameo", "psicov.lst", "cameo-hard.lst"],
    # Whole-file test
    "darcyflow": ["piececonst_r421_N1024_smooth2.mat"],
    # Bundled preprocessed — test content is already filtered out during
    # preprocessing; no additional exclusions needed at walk time.
    "spherical": [],
    "deepsea": [],
    "ecg": [],
}

# Tasks where the explorer's source dir is a preprocessed train+val-only dataset
# instead of the real data dir (because the real file bundles train+val+test).
BUNDLED_PREPROCESSED_TASKS: set[str] = {"spherical", "deepsea", "ecg"}

# Blind-mode-only: map original top-level entry name -> blind alias. Renames
# apply at the top level of the symlink tree only.
BLIND_RENAMES: dict[str, dict[str, str]] = {
    "ninapro": {
        "ninapro_train.npy": "data_train.npy",
        "ninapro_val.npy": "data_val.npy",
        "label_train.npy": "labels_train.npy",
        "label_val.npy": "labels_val.npy",
    },
    "satellite": {"satellite_train.npy": "data_train.npy"},
    "deepsea": {"deepsea_filtered.npz": "data.npz"},
    "spherical": {"s2_cifar100": "data.pkl"},
    "darcyflow": {"piececonst_r421_N1024_smooth1.mat": "data.mat"},
    "cosmic": {"npy_train": "raw_train"},
}

# Blind-mode-only: additional exclusions beyond TEST_EXCLUSIONS. For cosmic,
# the .pt files and their loader module reveal the domain; in blind mode the
# explorer reads the raw .npy files from npy_train/ (renamed to raw_train/).
BLIND_EXTRA_EXCLUSIONS: dict[str, list[str]] = {
    "cosmic": [
        "cosmic_train.pt",
        "cosmic_valid.pt",
        "cosmic_dataset.py",
        "__pycache__",
    ],
}

# File-level patterns always stripped in blind mode (domain visualizations,
# metadata, original README — the last is always replaced with a curated one).
_BLIND_DOMAIN_LEAK_NAMES: frozenset[str] = frozenset({"metadata"})
_BLIND_DOMAIN_LEAK_SUFFIXES: frozenset[str] = frozenset({".svg", ".pdf"})

# Always excluded (both modes). README is always replaced with a curated copy,
# not the real file.
_ALWAYS_EXCLUDE_NAMES: frozenset[str] = frozenset({"readme.md"})


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _is_excluded(rel_path: str, exclusions: list[str]) -> bool:
    """Return True if rel_path is an exact exclusion or lives under one."""
    for exc in exclusions:
        if rel_path == exc:
            return True
        if rel_path.startswith(exc + "/"):
            return True
    return False


def _has_nested_exclusion(dir_rel_path: str, exclusions: list[str]) -> bool:
    """True if any exclusion is strictly inside this directory."""
    prefix = dir_rel_path + "/"
    return any(e.startswith(prefix) for e in exclusions)


def _walk_and_symlink(
    src_dir: Path,
    dst_dir: Path,
    rel_base: str,
    exclusions: list[str],
    blind_mode: bool,
    top_level_renames: dict[str, str],
) -> None:
    """Recursively mirror src_dir into dst_dir via symlinks, skipping exclusions.

    Renames apply only at the top level (rel_base == "").
    """
    for item in src_dir.iterdir():
        rel_path = f"{rel_base}/{item.name}" if rel_base else item.name

        # Always-on exclusions
        if item.name.lower() in _ALWAYS_EXCLUDE_NAMES:
            continue

        # Per-task / test-set exclusions (both modes)
        if _is_excluded(rel_path, exclusions):
            continue

        # Blind-mode domain-leak exclusions
        if blind_mode:
            if item.name.lower() in _BLIND_DOMAIN_LEAK_NAMES:
                continue
            if item.suffix.lower() in _BLIND_DOMAIN_LEAK_SUFFIXES:
                continue

        # Top-level rename (blind only)
        target_name = item.name
        if blind_mode and rel_base == "":
            target_name = top_level_renames.get(item.name, item.name)

        target = dst_dir / target_name

        if item.is_dir():
            if _has_nested_exclusion(rel_path, exclusions):
                target.mkdir()
                _walk_and_symlink(
                    src_dir=item,
                    dst_dir=target,
                    rel_base=rel_path,
                    exclusions=exclusions,
                    blind_mode=blind_mode,
                    top_level_renames={},
                )
            else:
                target.symlink_to(item.resolve())
        else:
            target.symlink_to(item.resolve())


# -----------------------------------------------------------------------------
# On-demand preprocessing for bundled-split tasks
#
# Three tasks store train+val+test in a single file and cannot be filtered by
# symlinking alone. We preprocess train+val-only copies at
# configs/explorer_data/<task>/<original_filename>. The files are produced once
# and cached on disk (~1 GB total). build_explorer_data_dir() calls
# ensure_preprocessed() which is idempotent and fast on repeat calls.
# -----------------------------------------------------------------------------

def _prepare_spherical(out_dir: Path, real_root: Path) -> None:
    """Save a copy of s2_cifar100 with only the 'train' entry (no 'test')."""
    import pickle as _pickle

    out_path = out_dir / "s2_cifar100"
    src_path = real_root / "spherical" / "s2_cifar100"
    if not src_path.exists():
        raise FileNotFoundError(f"spherical source missing: {src_path}")
    with open(src_path, "rb") as f:
        data = _pickle.load(f, encoding="bytes")
    filtered = {"train": data["train"]}
    with open(out_path, "wb") as f:
        _pickle.dump(filtered, f, protocol=_pickle.HIGHEST_PROTOCOL)


def _prepare_deepsea(out_dir: Path, real_root: Path) -> None:
    """Save a copy of deepsea_filtered.npz with only train/val arrays."""
    import numpy as _np

    out_path = out_dir / "deepsea_filtered.npz"
    src_path = real_root / "deepsea" / "deepsea_filtered.npz"
    if not src_path.exists():
        raise FileNotFoundError(f"deepsea source missing: {src_path}")
    data = _np.load(src_path)
    _np.savez(
        out_path,
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_val=data["x_val"],
        y_val=data["y_val"],
    )


def _prepare_ecg(out_dir: Path, real_root: Path) -> None:
    """Save a copy of challenge2017.pkl with the final-mode test split removed.

    Replicates the benchmark loader's split (train_test_split(indices,
    test_size=0.1, random_state=0)) to identify and drop the ~10% real-test
    recordings, keeping the ~90% train+val portion.
    """
    import pickle as _pickle
    import numpy as _np
    from sklearn.model_selection import train_test_split as _tts

    out_path = out_dir / "challenge2017.pkl"
    src_path = real_root / "ecg" / "challenge2017.pkl"
    if not src_path.exists():
        raise FileNotFoundError(f"ecg source missing: {src_path}")
    with open(src_path, "rb") as f:
        data = _pickle.load(f, encoding="bytes")
    signals = data["data"]
    labels = data["label"]
    n = len(signals)
    indices = _np.arange(n)
    train_idx, _test_idx = _tts(indices, test_size=0.1, random_state=0)
    keep = set(train_idx.tolist())
    filtered = {
        "data": [signals[i] for i in range(n) if i in keep],
        "label": [labels[i] for i in range(n) if i in keep],
    }
    with open(out_path, "wb") as f:
        _pickle.dump(filtered, f, protocol=_pickle.HIGHEST_PROTOCOL)


_PREP_FNS = {
    "spherical": _prepare_spherical,
    "deepsea": _prepare_deepsea,
    "ecg": _prepare_ecg,
}


def ensure_preprocessed(task: str, real_data_root: str, project_root: Path) -> Path:
    """Ensure configs/explorer_data/<task>/ is populated; generate if missing.

    Returns the path to the preprocessed task directory. Idempotent — a no-op
    if the expected output file already exists. For non-bundled tasks this
    raises KeyError.
    """
    if task not in BUNDLED_PREPROCESSED_TASKS:
        raise KeyError(f"task '{task}' is not a bundled-preprocessed task")

    out_dir = project_root / "configs" / "explorer_data" / task
    # The output file name matches the real source file name
    expected_names = {
        "spherical": "s2_cifar100",
        "deepsea": "deepsea_filtered.npz",
        "ecg": "challenge2017.pkl",
    }
    expected = out_dir / expected_names[task]
    if expected.exists():
        return out_dir

    # Resolve the real root. real_data_root might be the task-specific
    # subdirectory root (e.g., NAS-Bench-360-Benchmarks) — strip trailing
    # task name if present.
    real_root = Path(real_data_root)
    if real_root.name == task:  # pointed directly at <root>/spherical
        real_root = real_root.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    _PREP_FNS[task](out_dir, real_root)
    return out_dir


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_explorer_data_dir(
    real_data_root: str,
    task: str,
    task_data_subdir: str,
    blind_mode: bool,
    project_root: Path,
) -> tuple[Path, str]:
    """Construct a filtered temp directory for the Data Explorer.

    Args:
        real_data_root: Filesystem root containing the task's real data dir.
        task: Task name key (e.g., "chesseract", "cosmic").
        task_data_subdir: Subdirectory name inside real_data_root (from task YAML).
        blind_mode: If True, apply domain-knowledge blinding on top of test isolation.
        project_root: Repo root, used to locate configs/explorer_readmes,
            configs/blind_readmes, configs/explorer_data.

    Returns:
        (temp_dir, "dataset") — pass these as (data_root, data_subdir) to
        DataExplorer. The caller must clean up temp_dir with shutil.rmtree().
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="agentnas_explorer_"))
    dataset_dir = temp_dir / "dataset"
    dataset_dir.mkdir()

    # Determine source directory
    if task in BUNDLED_PREPROCESSED_TASKS:
        # Auto-generate preprocessed copy if missing (idempotent; cached on disk).
        try:
            source_dir = ensure_preprocessed(task, real_data_root, project_root)
        except FileNotFoundError as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise FileNotFoundError(
                f"Cannot preprocess '{task}': {e}. "
                f"Ensure the real data exists under real_data_root."
            ) from e
    else:
        source_dir = Path(real_data_root) / task_data_subdir
        if not source_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise FileNotFoundError(f"Data directory not found: {source_dir}")

    # Assemble exclusion list
    exclusions = list(TEST_EXCLUSIONS.get(task, []))
    if blind_mode:
        exclusions.extend(BLIND_EXTRA_EXCLUSIONS.get(task, []))

    top_level_renames = BLIND_RENAMES.get(task, {}) if blind_mode else {}

    # Mirror the source dir
    _walk_and_symlink(
        src_dir=source_dir,
        dst_dir=dataset_dir,
        rel_base="",
        exclusions=exclusions,
        blind_mode=blind_mode,
        top_level_renames=top_level_renames,
    )

    # Copy the mode-appropriate curated README
    readme_subdir = "blind_readmes" if blind_mode else "explorer_readmes"
    readme_src = project_root / "configs" / readme_subdir / f"{task}.md"
    if not readme_src.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        mode_str = "blind" if blind_mode else "standard"
        raise FileNotFoundError(
            f"Explorer README not found: {readme_src}. "
            f"Expected a {mode_str}-mode README for task '{task}'."
        )
    (dataset_dir / "README.md").write_text(readme_src.read_text())

    return temp_dir, "dataset"
