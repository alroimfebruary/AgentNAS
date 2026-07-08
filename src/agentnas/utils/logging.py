"""Structured experiment logging (JSONL format)."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from threading import Lock
from typing import Any


def _sanitize_floats(obj):
    """Replace inf/-inf/NaN with JSON-safe values."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return 1e308 if obj > 0 else -1e308
        if math.isnan(obj):
            return None
    elif isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_sanitize_floats(v) for v in obj)
    return obj


class ExperimentLogger:
    """Appends one JSON line per architecture evaluation to a JSONL file.

    Also maintains a human-readable CSV leaderboard.

    Usage:
        logger = ExperimentLogger(experiment_dir)
        logger.log_evaluation({
            "arch_id": "arch_001",
            "phase": 1,
            "metric": 0.38,
            ...
        })
    """

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.experiment_dir / "evaluations.jsonl"
        self.leaderboard_path = self.experiment_dir / "leaderboard.csv"
        self._lock = Lock()

    def log_evaluation(self, record: dict[str, Any]) -> None:
        """Append a single evaluation record to the JSONL log."""
        record.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))

        with self._lock:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(_sanitize_floats(record), default=str) + "\n")
            # Update leaderboard inside the lock for thread safety
            self._update_leaderboard(record)

    def _update_leaderboard(self, record: dict[str, Any]) -> None:
        """Append to CSV leaderboard (human-readable summary)."""
        fields = [
            "arch_id", "phase", "iteration", "metric", "status",
            "params", "flops", "peak_vram_mb", "wall_time_s", "timestamp",
        ]
        row = {k: record.get(k, "") for k in fields}

        write_header = not self.leaderboard_path.exists()
        with open(self.leaderboard_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def load_evaluations(self) -> list[dict[str, Any]]:
        """Load all evaluation records from the JSONL file (thread-safe)."""
        with self._lock:  # H-04: protect against concurrent writes
            if not self.jsonl_path.exists():
                return []
            records = []
            with open(self.jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # Skip malformed lines from crash
            return records

    def get_best(self, key: str = "metric") -> dict[str, Any] | None:
        """Return the evaluation record with the lowest metric (lower=better)."""
        records = self.load_evaluations()
        successful = [r for r in records if r.get("status") == "success"]
        if not successful:
            return None
        return min(successful, key=lambda r: r.get(key, float("inf")))

    def count_evaluations(self) -> int:
        """Count total evaluations (including failed)."""
        return len(self.load_evaluations())
