"""Custom-task validation: check a --benchmark-file before spending budget.

Run via ``agentnas validate-task my_task.py --data-root ...``. Performs:

1. Registration    — the file loads and registers the task
2. Contract        — I/O-free __init__, metadata attrs, shape/output spec
3. Loaders         — the base smoke test (shapes, splits, loss finiteness)
4. Split integrity — sampled train/test disjointness by content hash
5. Explorer view   — isolation spec enforced; prints the exact visible tree
6. Metric sanity   — perfect predictions ≈ optimum, shuffled clearly worse
7. LLM preview     — the README and metadata lines the agents will see

Only machine-checkable properties are asserted; semantic label alignment and
split conventions still need the human eye — that is what the previews are
for. Review them before starting a paid search.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

import torch

from agentnas.benchmarks.registry import get_benchmark, load_benchmark_file
from agentnas.config import TaskConfig
from agentnas.utils.explorer_data import _is_allowed, build_explorer_data_dir
from agentnas.utils.paths import project_root


class _Report:
    """Collects [PASS]/[FAIL]/[WARN] lines and the final verdict."""

    def __init__(self, echo=print):
        self.echo = echo
        self.failures = 0
        self.warnings = 0

    def ok(self, msg: str) -> None:
        self.echo(f"  [PASS] {msg}")

    def fail(self, msg: str) -> None:
        self.failures += 1
        self.echo(f"  [FAIL] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings += 1
        self.echo(f"  [WARN] {msg}")

    def info(self, msg: str) -> None:
        self.echo(f"  [info] {msg}")

    def section(self, title: str) -> None:
        self.echo(f"\n== {title}")


def _sample_hashes(loader, limit: int = 256) -> set[str]:
    """Content hashes of up to `limit` input samples from a loader."""
    hashes: set[str] = set()
    seen = 0
    for batch in loader:
        x = batch[0]
        for i in range(x.shape[0]):
            raw = x[i].contiguous().cpu().numpy().tobytes()
            hashes.add(hashlib.sha1(raw).hexdigest())
            seen += 1
            if seen >= limit:
                return hashes
    return hashes


def _check_metric_sanity(benchmark, report: _Report, batch_limit: int = 8) -> None:
    """metric(perfect) must be ~optimal and beat metric(shuffled)."""
    try:
        _, val_loader = benchmark.get_search_loaders(batch_size=64, num_workers=0)
        target_batches = []
        for j, batch in enumerate(val_loader):
            target_batches.append(batch[1])
            if j + 1 >= batch_limit:
                break
        y = torch.cat(target_batches, dim=0)

        spec = benchmark.output_spec()
        is_classification = (
            "num_classes" in spec
            and y.dim() == 1
            and not torch.is_floating_point(y)
        )
        if is_classification:
            perfect = (
                torch.nn.functional.one_hot(y.long(), spec["num_classes"]).float()
                * 10.0
            )
        else:
            perfect = y.clone().float()
        shuffled = perfect[torch.randperm(perfect.shape[0])]

        m_perfect = benchmark.compute_metric([perfect], [y])
        m_shuffled = benchmark.compute_metric([shuffled], [y])
        report.info(
            f"metric(perfect) = {m_perfect:.6f}, metric(shuffled) = {m_shuffled:.6f}"
        )
        if m_perfect > 0.01:
            report.warn(
                "metric(perfect predictions) is not near 0 — check that the "
                "metric is normalized to lower=better"
            )
        elif m_shuffled == m_perfect:
            report.warn(
                "metric(shuffled) == metric(perfect) — sampled targets look "
                "degenerate (single class?); check the metric on more data"
            )
        elif m_perfect < m_shuffled:
            report.ok("metric sanity: perfect ≈ optimum and beats shuffled")
        else:
            report.fail(
                "metric(shuffled) < metric(perfect): the metric does not "
                "reward correct predictions (orientation bug?)"
            )
    except Exception as e:  # noqa: BLE001 — diagnostic path, keep going
        report.warn(f"metric sanity check skipped ({type(e).__name__}: {e})")


def _check_split_integrity(benchmark, report: _Report) -> None:
    """Sampled train/test disjointness by input-content hash."""
    try:
        train_loader, _ = benchmark.get_search_loaders(batch_size=64, num_workers=0)
        _, test_loader = benchmark.get_final_loaders(batch_size=64, num_workers=0)
        train_hashes = _sample_hashes(train_loader)
        test_hashes = _sample_hashes(test_loader)
        overlap = train_hashes & test_hashes
        if overlap:
            report.fail(
                f"train/test leak: {len(overlap)} sampled training inputs "
                f"appear verbatim in the test split"
            )
        else:
            report.ok(
                f"train/test disjoint in sample ({len(train_hashes)} train vs "
                f"{len(test_hashes)} test inputs compared)"
            )
        # One-directional caveat: augmented/transformed loaders can hide leaks.
        report.info(
            "note: content-hash check is sampled and defeated by train-time "
            "augmentation — it catches gross split bugs, not all leaks"
        )
    except Exception as e:  # noqa: BLE001 — diagnostic path, keep going
        report.warn(f"split integrity check skipped ({type(e).__name__}: {e})")


def _preview_explorer_view(benchmark, data_root: str, report: _Report) -> None:
    """Build the explorer dir, verify the allowlist, print the visible tree."""
    tmp_dir = None
    try:
        tmp_dir, sub = build_explorer_data_dir(
            real_data_root=data_root,
            benchmark=benchmark,
            blind_mode=False,
            project_root=project_root(),
        )
        dataset_dir = tmp_dir / sub
        entries = sorted(dataset_dir.rglob("*"))
        report.ok(f"explorer tree built ({len(entries)} visible entries)")

        allow = benchmark.explorer_allowed_files
        if allow is not None:
            leaked = [
                str(p.relative_to(dataset_dir))
                for p in entries
                if not p.is_dir()
                and p.name != "README.md"
                and not _is_allowed(str(p.relative_to(dataset_dir)), allow)
            ]
            if leaked:
                report.fail(f"files outside the allowlist are visible: {leaked}")
            else:
                report.ok("every visible file is on the allowlist")
        else:
            report.warn(
                "no explorer_allowed_files declared — exclusion lists are "
                "vetted for builtins, but custom tasks should use the "
                "allowlist so unlisted files can never leak"
            )

        report.info("the Data Explorer LLM will see exactly:")
        for p in entries:
            rel = p.relative_to(dataset_dir)
            suffix = "/" if p.is_dir() else (" -> " + str(p.resolve()) if p.is_symlink() else "")
            report.echo(f"      dataset/{rel}{suffix}")

        curated = project_root() / "configs" / "explorer_readmes" / f"{benchmark.task_name}.md"
        if curated.exists():
            source = f"curated repo file ({curated})"
        elif benchmark.explorer_readme() is not None:
            source = "benchmark.explorer_readme()"
        else:
            source = "auto-generated from metadata"
        report.info(f"README source: {source}; first lines:")
        for line in (dataset_dir / "README.md").read_text().splitlines()[:12]:
            report.echo(f"      | {line}")
    except Exception as e:  # noqa: BLE001 — surface as a finding
        report.fail(f"explorer view failed: {type(e).__name__}: {e}")
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def validate_custom_task(
    benchmark_file: str,
    task: str | None,
    data_root: str,
    echo=print,
) -> bool:
    """Run the full validation report for a custom benchmark file.

    Returns True when no [FAIL] findings were raised.
    """
    torch.manual_seed(0)
    report = _Report(echo=echo)

    report.section("Registration")
    new_tasks = load_benchmark_file(benchmark_file)
    if new_tasks:
        report.ok(f"{Path(benchmark_file).name} registered: {', '.join(new_tasks)}")
    if task is None:
        if len(new_tasks) == 1:
            task = new_tasks[0]
        else:
            report.fail(
                f"file registers {len(new_tasks)} tasks ({new_tasks}); pass "
                f"--task to pick one"
            )
            return False

    report.section("Contract")
    # __init__ must work without any data present (metadata introspection).
    fake_root = str(Path(tempfile.mkdtemp(prefix="agentnas_validate_")) / "missing")
    try:
        probe = get_benchmark(task, data_root=fake_root)
        report.ok("__init__ is I/O-free (works without data on disk)")
    except Exception as e:  # noqa: BLE001 — the finding itself
        report.fail(f"__init__ requires data on disk ({type(e).__name__}: {e})")
        probe = None

    if probe is not None:
        if probe.task_name != task:
            report.warn(
                f"benchmark.task_name ({probe.task_name!r}) != registered "
                f"name ({task!r}) — keep them identical to avoid confusion"
            )
        if not probe.loss_name or not probe.metric_name:
            report.fail("loss_name and metric_name must be set (LLM prompts use them)")
        else:
            report.ok(f"metadata: loss={probe.loss_name}, metric={probe.metric_name}")
        try:
            shape = tuple(probe.input_shape())
            spec = probe.output_spec()
            assert shape and all(int(d) > 0 for d in shape)
            assert "num_classes" in spec or "output_shape" in spec
            report.ok(f"shapes: input {shape}, output {spec}")
        except Exception as e:  # noqa: BLE001 — the finding itself
            report.fail(f"input_shape()/output_spec() broken without data: {e}")
        tc = TaskConfig.from_benchmark(probe)
        if tc.sota_metric is None:
            report.info("no SOTA declared — Planner prompt will omit the SOTA baseline")
        else:
            report.info(f"SOTA: {tc.sota_metric} — {tc.sota_description}")

    report.section("Loaders (smoke test)")
    benchmark = get_benchmark(task, data_root=data_root)
    if benchmark.smoke_test():
        report.ok("smoke test: shapes, split sizes, loss finiteness")
    else:
        report.fail("smoke test failed (see message above)")

    report.section("Split integrity")
    _check_split_integrity(benchmark, report)

    report.section("Explorer view (what the LLM sees)")
    _preview_explorer_view(benchmark, data_root, report)

    report.section("Metric sanity")
    _check_metric_sanity(benchmark, report)

    report.section("Verdict")
    if report.failures:
        report.echo(f"  {report.failures} failure(s), {report.warnings} warning(s) — fix before running a search.")
    else:
        report.echo(
            f"  All checks passed ({report.warnings} warning(s)). Review the "
            f"explorer tree and README above, then run a search."
        )
    return report.failures == 0
