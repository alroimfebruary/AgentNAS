"""Explorer data directory construction with test-set isolation and optional blinding.

Builds a filtered temporary directory that the Data Explorer sees instead of the real
data directory. Test files are always excluded so the explorer cannot peek at test
data. In blind mode, domain-leaking files are additionally removed and select
filenames are aliased to generic names.

The per-task visibility spec — test exclusions, blind renames/exclusions, and
the preprocessed-copy hook for bundled-split datasets — lives on each benchmark
class (see BaseBenchmark). This module is the generic machinery that applies it.

The benchmark loader (`get_benchmark(..., data_root=config.data_root)`) continues to
use the untouched real data directory for training and evaluation — this module only
affects what the explorer agent sees.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentnas.benchmarks.base import BaseBenchmark

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


def _is_allowed(rel_path: str, allowlist: tuple[str, ...]) -> bool:
    """True if rel_path is an allowlist entry or lives under an allowed dir."""
    for allowed in allowlist:
        if rel_path == allowed:
            return True
        if rel_path.startswith(allowed + "/"):
            return True
    return False


def _has_nested_allowed(dir_rel_path: str, allowlist: tuple[str, ...]) -> bool:
    """True if any allowlist entry is strictly inside this directory."""
    prefix = dir_rel_path + "/"
    return any(a.startswith(prefix) for a in allowlist)


def _walk_and_symlink(
    src_dir: Path,
    dst_dir: Path,
    rel_base: str,
    exclusions: list[str],
    blind_mode: bool,
    top_level_renames: dict[str, str],
    allowlist: tuple[str, ...] | None = None,
) -> None:
    """Recursively mirror src_dir into dst_dir via symlinks, skipping exclusions.

    With an allowlist (not None), only allowlisted paths — and anything
    beneath an allowlisted directory — are mirrored; exclusions still apply
    on top. Renames apply only at the top level (rel_base == "").
    """
    for item in src_dir.iterdir():
        rel_path = f"{rel_base}/{item.name}" if rel_base else item.name

        # Always-on exclusions
        if item.name.lower() in _ALWAYS_EXCLUDE_NAMES:
            continue

        # Per-task / test-set exclusions (both modes)
        if _is_excluded(rel_path, exclusions):
            continue

        # Allowlist filter: hide anything not allowed. Directories that are
        # not fully allowed but contain allowed entries are recursed into.
        fully_allowed = True
        if allowlist is not None:
            fully_allowed = _is_allowed(rel_path, allowlist)
            if item.is_dir():
                if not fully_allowed and not _has_nested_allowed(rel_path, allowlist):
                    continue
            elif not fully_allowed:
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
            if _has_nested_exclusion(rel_path, exclusions) or not fully_allowed:
                target.mkdir()
                _walk_and_symlink(
                    src_dir=item,
                    dst_dir=target,
                    rel_base=rel_path,
                    exclusions=exclusions,
                    blind_mode=blind_mode,
                    top_level_renames={},
                    allowlist=allowlist,
                )
            else:
                target.symlink_to(item.resolve())
        else:
            target.symlink_to(item.resolve())


def _generate_readme(benchmark: "BaseBenchmark", dataset_dir: Path) -> str:
    """Auto-generate an Explorer README from benchmark metadata + visible files."""
    spec = benchmark.output_spec()
    if "num_classes" in spec:
        output_desc = f"{spec['num_classes']} classes"
    else:
        output_desc = f"dense output of shape {tuple(spec['output_shape'])}"

    listing = "\n".join(
        f"- `{p.name}{'/' if p.is_dir() else ''}`"
        for p in sorted(dataset_dir.iterdir())
    )
    return (
        f"# {benchmark.task_name}\n\n"
        f"- Input shape (per sample): {tuple(benchmark.input_shape())}\n"
        f"- Output: {output_desc}\n"
        f"- Loss: {benchmark.loss_name}\n"
        f"- Metric: {benchmark.metric_name} (lower = better)\n\n"
        f"## Files\n\n"
        f"Only the training and validation portion of the dataset is visible "
        f"here; test data is withheld.\n\n"
        f"{listing}\n"
    )


# -----------------------------------------------------------------------------
# On-demand preprocessing for bundled-split tasks
#
# Some datasets store train+val+test in a single file and cannot be filtered by
# symlinking alone. Their benchmark classes set `explorer_preprocessed_name` and
# implement `prepare_explorer_source()`, which writes a train+val-only copy to
# configs/explorer_data/<task>/. The files are produced once and cached on disk
# (~1 GB total). build_explorer_data_dir() calls ensure_preprocessed() which is
# idempotent and fast on repeat calls.
# -----------------------------------------------------------------------------

def ensure_preprocessed(
    benchmark: "BaseBenchmark", real_data_root: str, project_root: Path
) -> Path:
    """Ensure configs/explorer_data/<task>/ is populated; generate if missing.

    Returns the path to the preprocessed task directory. Idempotent — a no-op
    if the expected output file already exists. For benchmarks that don't set
    ``explorer_preprocessed_name`` this raises ValueError.
    """
    if benchmark.explorer_preprocessed_name is None:
        raise ValueError(
            f"task '{benchmark.task_name}' is not a bundled-preprocessed task"
        )

    out_dir = project_root / "configs" / "explorer_data" / benchmark.task_name
    # The output file name matches the real source file name
    expected = out_dir / benchmark.explorer_preprocessed_name
    if expected.exists():
        return out_dir

    # Resolve the real root. real_data_root might be the task-specific
    # subdirectory root (e.g., NAS-Bench-360-Benchmarks) — strip trailing
    # task name if present.
    real_root = Path(real_data_root)
    if real_root.name == benchmark.task_name:  # pointed directly at <root>/spherical
        real_root = real_root.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark.prepare_explorer_source(out_dir, real_root)
    return out_dir


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def build_explorer_data_dir(
    real_data_root: str,
    benchmark: "BaseBenchmark",
    blind_mode: bool,
    project_root: Path,
) -> tuple[Path, str]:
    """Construct a filtered temp directory for the Data Explorer.

    Args:
        real_data_root: Filesystem root containing the task's real data dir.
        benchmark: Benchmark instance supplying the per-task visibility spec
            (test exclusions, blind renames/exclusions, preprocessed hook).
        blind_mode: If True, apply domain-knowledge blinding on top of test isolation.
        project_root: Repo root, used to locate configs/explorer_readmes,
            configs/blind_readmes, configs/explorer_data.

    Returns:
        (temp_dir, "dataset") — pass these as (data_root, data_subdir) to
        DataExplorer. The caller must clean up temp_dir with shutil.rmtree().
    """
    task = benchmark.task_name

    # Refuse benchmarks with no visibility spec: exposing the raw data dir
    # would hand the Explorer the test set.
    if (
        benchmark.explorer_preprocessed_name is None
        and benchmark.explorer_allowed_files is None
        and not benchmark.explorer_test_exclusions
    ):
        raise ValueError(
            f"Benchmark '{task}' declares no Explorer visibility spec. Set "
            f"explorer_allowed_files (recommended for custom tasks), "
            f"explorer_test_exclusions, or a preprocessed source — refusing "
            f"to expose the raw data dir to the Explorer."
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="agentnas_explorer_"))
    dataset_dir = temp_dir / "dataset"
    dataset_dir.mkdir()

    # Determine source directory
    if benchmark.explorer_preprocessed_name is not None:
        # Auto-generate preprocessed copy if missing (idempotent; cached on disk).
        try:
            source_dir = ensure_preprocessed(benchmark, real_data_root, project_root)
        except FileNotFoundError as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise FileNotFoundError(
                f"Cannot preprocess '{task}': {e}. "
                f"Ensure the real data exists under real_data_root."
            ) from e
    else:
        source_dir = Path(real_data_root) / benchmark.data_subdir
        if not source_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise FileNotFoundError(f"Data directory not found: {source_dir}")

    # Assemble exclusion list
    exclusions = list(benchmark.explorer_test_exclusions)
    if blind_mode:
        exclusions.extend(benchmark.explorer_blind_exclusions)

    top_level_renames = dict(benchmark.explorer_blind_renames) if blind_mode else {}

    # Mirror the source dir
    _walk_and_symlink(
        src_dir=source_dir,
        dst_dir=dataset_dir,
        rel_base="",
        exclusions=exclusions,
        blind_mode=blind_mode,
        top_level_renames=top_level_renames,
        allowlist=benchmark.explorer_allowed_files,
    )

    # README: curated repo copy > benchmark-provided > auto-generated.
    # Blind mode requires the curated blind README (the whole point of the
    # ablation is a controlled description), so it stays a hard failure.
    readme_subdir = "blind_readmes" if blind_mode else "explorer_readmes"
    readme_src = project_root / "configs" / readme_subdir / f"{task}.md"
    if readme_src.exists():
        readme_text = readme_src.read_text()
    elif blind_mode:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise FileNotFoundError(
            f"Explorer README not found: {readme_src}. "
            f"Expected a blind-mode README for task '{task}'."
        )
    else:
        readme_text = benchmark.explorer_readme() or _generate_readme(
            benchmark, dataset_dir
        )
    (dataset_dir / "README.md").write_text(readme_text)

    return temp_dir, "dataset"
