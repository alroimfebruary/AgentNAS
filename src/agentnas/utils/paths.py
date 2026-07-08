"""Filesystem anchors — CWD-independent resolution of repo-rooted resources."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root, anchored to the package location.

    Resolves ``<root>/src/agentnas/utils/paths.py`` upward, so it is valid
    from any working directory under an editable install (``pip install -e .``),
    which is the documented setup. Repo-rooted resources (``configs/``, key
    files, default data directories) are resolved against this.
    """
    return Path(__file__).resolve().parents[3]
