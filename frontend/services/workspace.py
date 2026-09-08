"""Workspace root handling and path confinement.

Vendored/adapted from ``ui/backend/workspace.py`` so the frontend stands
alone (``ui/`` can be deleted without affecting it).
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from frontend.config import get_workspace

__all__ = ["get_workspace", "safe_path", "list_files", "cwd"]


@contextlib.contextmanager
def cwd():
    """Temporarily ``chdir`` into the workspace.

    ``integrate`` reads linked files (``f5_data`` / ``f5_prior``) by bare
    relative name, so any ``ig.*`` call that touches them must run with the
    workspace as the current directory.
    """
    prev = os.getcwd()
    root = get_workspace()
    try:
        if root.is_dir():
            os.chdir(root)
        yield
    finally:
        os.chdir(prev)


def safe_path(name: str) -> Path:
    """Resolve *name* under the workspace, refusing escapes.

    Raises ``ValueError`` if the resolved path is outside the workspace.
    """
    root = get_workspace()
    p = (root / name).resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"Path escapes workspace: {name}")
    return p


def list_files(*exts: str) -> list[Path]:
    """Top-level files in the workspace matching any of *exts* (e.g. ``".h5"``).

    No recursion — the workspace is expected to be the folder holding the
    project files. Sorted case-insensitively by name.
    """
    root = get_workspace()
    if not root.is_dir():
        return []
    want = {e.lower() for e in exts}
    return sorted(
        (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in want),
        key=lambda p: p.name.lower(),
    )
