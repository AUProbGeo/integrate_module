"""Result visualization endpoints for POST.h5 files.

Stats series are served as JSON for inline charts; profile rendering reuses
integrate's matplotlib plot_profile, rendered to PNG via the Agg backend.
matplotlib is not thread-safe, so rendering is serialized under a lock.
"""

from __future__ import annotations

import asyncio
import io
import os
import threading

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from ui.backend.workspace import safe_path

router = APIRouter(prefix="/api/results", tags=["results"])

_mpl_lock = threading.Lock()

MAX_SERIES_POINTS = 2000


def _resolve_post(name: str):
    try:
        p = safe_path(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"No such file: {name}")
    return p


@router.get("/{name:path}/stats")
def post_stats(name: str):
    """Downsampled EV / T / CHI2 / N_UNIQUE series plus geometry."""
    import h5py

    p = _resolve_post(name)
    with h5py.File(p, "r") as f:
        if "i_use" not in f:
            raise HTTPException(status_code=400, detail="Not a POSTERIOR file")
        n_points = int(f["i_use"].shape[0])
        stride = max(1, n_points // MAX_SERIES_POINTS)
        idx = np.arange(0, n_points, stride)

        series: dict = {"index": idx.tolist(), "n_points": n_points}
        for key in ("EV", "EV_post", "T", "CHI2", "N_UNIQUE", "UTMX", "UTMY"):
            if key in f:
                arr = f[key][:].ravel()[idx].astype(float)
                series[key.lower()] = np.where(np.isfinite(arr), arr, None).tolist()
        return series


def _render_profile(path, im: int, xaxis: str, panels: list[str] | None) -> bytes:
    import integrate as ig

    from ui.backend.workspace import get_workspace

    buf = io.BytesIO()
    cwd = os.getcwd()
    with _mpl_lock:
        os.chdir(get_workspace())  # plot_profile opens referenced files relatively
        try:
            before = set(plt.get_fignums())
            ig.plot_profile(str(path), im=im, xaxis=xaxis, panels=panels)
        finally:
            os.chdir(cwd)
        new_figs = [n for n in plt.get_fignums() if n not in before]
        if not new_figs:
            raise RuntimeError("plot_profile produced no figure")
        fig = plt.figure(new_figs[0])
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        for n in new_figs:
            plt.close(n)
    return buf.getvalue()


@router.get("/{name:path}/profile.png")
async def profile_png(name: str, im: int = 1, xaxis: str = "index",
                      panels: str | None = None):
    p = _resolve_post(name)
    panel_list = [s for s in (panels or "").split(",") if s] or None
    try:
        png = await asyncio.to_thread(_render_profile, p, im, xaxis, panel_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=png, media_type="image/png")
