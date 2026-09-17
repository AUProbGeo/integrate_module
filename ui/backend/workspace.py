"""Workspace root handling.

The UI operates on a single workspace directory containing the project's
HDF5 files. It is the process working directory by default, overridable
with the INTEGRATE_WORKSPACE environment variable.
"""

import os
from pathlib import Path


def get_workspace() -> Path:
    """Return the workspace root directory."""
    return Path(os.environ.get("INTEGRATE_WORKSPACE", os.getcwd())).resolve()


def safe_path(name: str) -> Path:
    """Resolve *name* relative to the workspace, refusing escapes.

    Raises ValueError if the resolved path is outside the workspace.
    """
    root = get_workspace()
    p = (root / name).resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"Path escapes workspace: {name}")
    return p
