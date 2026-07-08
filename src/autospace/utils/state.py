"""Checkpoint/resume state management with atomic writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically: write to temp file, then rename.

    This ensures the file is never in a partial/corrupt state,
    which is critical for crash-safe resume.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in the same directory (same filesystem for rename)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=".state_"
    )
    try:
        def _sanitize(obj):
            """Replace inf/-inf/NaN with JSON-safe values (B2 fix).
            Also convert numpy arrays/integers to native Python types
            so they serialize as JSON arrays/numbers, not str().
            """
            if isinstance(obj, float):
                if obj == float("inf"): return 1e308  # M07: close to max float, always > real metrics
                if obj == float("-inf"): return -1e308
                if obj != obj: return None  # NaN
            elif isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_sanitize(v) for v in obj)
            # Handle numpy types that default=str would mangle
            try:
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
            except ImportError:
                pass
            return obj

        with os.fdopen(fd, "w") as f:
            json.dump(_sanitize(data), f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_state(path: Path) -> dict[str, Any] | None:
    """Load state from a JSON checkpoint file. Returns None if not found."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


class StateManager:
    """Manages checkpoint state for a pipeline phase.

    Provides get/set/save/load with automatic dirty tracking.

    Usage:
        state = StateManager(experiment_dir / "state.json")
        state.load()  # Load existing state or start fresh
        state["evaluated_archs"].append(result)
        state["budget_remaining"] -= 1
        state.save()  # Atomic write
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, Any] = {}

    def load(self) -> bool:
        """Load state from disk. Returns True if state was found."""
        loaded = load_state(self.path)
        if loaded is not None:
            self._data = loaded
            return True
        return False

    def save(self) -> None:
        """Save current state to disk atomically."""
        atomic_write_json(self.path, self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def setdefault(self, key: str, default: Any) -> Any:
        return self._data.setdefault(key, default)

    def update(self, data: dict[str, Any]) -> None:
        self._data.update(data)

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def __repr__(self) -> str:
        return f"StateManager(path={self.path}, keys={list(self._data.keys())})"
