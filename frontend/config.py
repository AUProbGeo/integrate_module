"""Paths and runtime configuration for the frontend.

The one knob that matters is the *workspace* — the directory holding the
project's ``.h5`` (and ``.gex`` / ``.xlsx``) files. Everything the UI reads or
writes is resolved relative to it.
"""

from __future__ import annotations

import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
STATIC_DIR = PKG_DIR / "static"
FIGURES_DIR = STATIC_DIR / "figures"

APP_TITLE = "INTEGRATE Workbench"
DEFAULT_PORT = 8051
DEFAULT_HOST = "127.0.0.1"


def resolve_workspace(explicit: str | os.PathLike | None = None) -> Path:
    """Resolve the workspace directory.

    Priority: explicit argument > ``WB_WORKSPACE`` > ``INTEGRATE_WORKSPACE`` >
    current working directory.
    """
    for cand in (explicit, os.environ.get("WB_WORKSPACE"), os.environ.get("INTEGRATE_WORKSPACE")):
        if cand:
            return Path(cand).expanduser().resolve()
    return Path.cwd().resolve()


# Set once by app.main() / create_app(); read via get_workspace().
_WORKSPACE: Path | None = None


def set_workspace(path: str | os.PathLike | None) -> Path:
    global _WORKSPACE
    _WORKSPACE = resolve_workspace(path)
    return _WORKSPACE


def get_workspace() -> Path:
    return _WORKSPACE if _WORKSPACE is not None else resolve_workspace(None)
