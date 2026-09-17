"""Render an ``integrate`` plotting call to a PNG under ``static/figures/``.

matplotlib is not thread-safe, so every render holds a module-global lock.
Filenames are content-hashed (call name + args + source-file mtime) so a
repeated request is a cache hit and the browser can cache aggressively.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Callable

from frontend.config import FIGURES_DIR

_LOCK = threading.Lock()
_KEEP = 200  # prune oldest beyond this many PNGs

os.environ.setdefault("MPLBACKEND", "Agg")


def _prune() -> None:
    pngs = sorted(FIGURES_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in pngs[_KEEP:]:
        try:
            p.unlink()
        except OSError:
            pass


def figure_key(tag: str, *parts: object, salt: Path | str | None = None) -> str:
    """Stable hash for a figure identity."""
    h = hashlib.sha1(tag.encode())
    for part in parts:
        h.update(repr(part).encode())
    if salt is not None:
        try:
            h.update(str(Path(salt).stat().st_mtime_ns).encode())
        except OSError:
            pass
    return h.hexdigest()[:16]


def render(fn: Callable, *args, key: str, **kwargs) -> str:
    """Call ``fn(*args, **kwargs)``, save the current figure, return a URL path.

    Returns ``/static/figures/<key>.png``. If that file already exists the
    call is skipped.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"{key}.png"
    url = f"/static/figures/{key}.png"
    if out.exists():
        return url

    from frontend.config import get_workspace

    with _LOCK:
        if out.exists():
            return url
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # integrate_plot resolves linked files (f5_data / f5_prior) by bare
        # relative name -> run with the workspace as CWD.
        prev_cwd = os.getcwd()
        try:
            ws = get_workspace()
            if ws.is_dir():
                os.chdir(ws)
            plt.close("all")
            fn(*args, **kwargs)
            plt.gcf().savefig(out, dpi=110, bbox_inches="tight")
        finally:
            plt.close("all")
            os.chdir(prev_cwd)
        _prune()
    return url


def render_all(key: str, draw: Callable) -> list[str]:
    """Run ``draw()`` and save **every** matplotlib figure it opened.

    Returns ``[/static/figures/<key>-0.png, …]``. Not cached (the caller's
    ``key`` should already encode the inputs; re-render is cheap enough and
    always correct). Used for ``ig.query_plot`` / ``query_percentile_plot``
    which may emit several figures.
    """
    from frontend.config import get_workspace

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    with _LOCK:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        prev_cwd = os.getcwd()
        try:
            ws = get_workspace()
            if ws.is_dir():
                os.chdir(ws)
            plt.close("all")
            before = set(plt.get_fignums())
            draw()
            new = sorted(set(plt.get_fignums()) - before) or plt.get_fignums()
            for i, num in enumerate(new):
                out = FIGURES_DIR / f"{key}-{i}.png"
                plt.figure(num).savefig(out, dpi=110, bbox_inches="tight")
                urls.append(f"/static/figures/{key}-{i}.png")
        finally:
            plt.close("all")
            os.chdir(prev_cwd)
        _prune()
    return urls


def render_interactive(key: str, draw: Callable, *, figsize=(7.6, 6.6), dpi: int = 110) -> tuple[str, dict]:
    """Render a single-axes figure and also return its axes geometry.

    ``draw(ax)`` populates one Axes. Returns ``(url, meta)`` where ``meta`` has
    the axes box as figure fractions (``ax_l/ax_r/ax_b/ax_t``, y from bottom)
    and the data limits (``x0/x1/y0/y1``) — enough for a browser overlay to map
    clicks back to data coordinates. Cached alongside a ``<key>.json``.
    """
    from frontend.config import get_workspace

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURES_DIR / f"{key}.png"
    meta_path = FIGURES_DIR / f"{key}.json"
    url = f"/static/figures/{key}.png"
    if png.exists() and meta_path.exists():
        try:
            return url, json.loads(meta_path.read_text())
        except (OSError, ValueError):
            pass

    with _LOCK:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        prev_cwd = os.getcwd()
        try:
            ws = get_workspace()
            if ws.is_dir():
                os.chdir(ws)
            plt.close("all")
            fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")
            ax = fig.add_subplot(111)
            extra = draw(ax) or {}
            fig.canvas.draw()
            fig.canvas.draw()  # let constrained layout settle before reading the box
            pos = ax.get_position()
            xl, yl = ax.get_xlim(), ax.get_ylim()
            fig.savefig(png, dpi=dpi)
            meta = {
                "ax_l": float(pos.x0), "ax_r": float(pos.x1),
                "ax_b": float(pos.y0), "ax_t": float(pos.y1),
                "x0": float(xl[0]), "x1": float(xl[1]),
                "y0": float(yl[0]), "y1": float(yl[1]),
                **{k: v for k, v in extra.items()},
            }
            meta_path.write_text(json.dumps(meta))
        finally:
            plt.close("all")
            os.chdir(prev_cwd)
        _prune()
    return url, meta
