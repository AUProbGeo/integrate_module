"""Query Volume: probability map + interactive coherent-area growing + volumes."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import threading
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ui.backend.workspace import get_workspace, safe_path

router = APIRouter(prefix="/api/volume", tags=["volume"])

_mpl_lock = threading.Lock()

_MAX_CACHE = 4                              # LRU cap of scaffold entries
_scaffold_lock = threading.Lock()
_scaffold_cache: OrderedDict[tuple, dict] = OrderedDict()


def _resolve_post(name: str) -> Path:
    try:
        p = safe_path(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"No such file: {name}")
    return p


def _posterior_prior_posterior(post: Path) -> tuple[Path, str]:
    import h5py

    with h5py.File(post, "r") as hf:
        if "i_use" not in hf:
            raise HTTPException(status_code=400, detail="Not a POSTERIOR file (no 'i_use' dataset)")
        f_prior = str(hf.attrs.get("f5_prior", "") or "")
    return _resolve_prior(f_prior), f_prior


def _resolve_prior(f_prior: str) -> Path:
    """Resolve the prior file path stored in a posterior's f5_prior attribute.

    Relative paths resolve against the workspace; a bare-name fallback covers
    files moved between machines since the inversion ran.
    """
    if not f_prior:
        raise HTTPException(
            status_code=400,
            detail="Posterior file has no 'f5_prior' attribute — cannot run a query",
        )
    p = Path(f_prior)
    candidates = [p if p.is_absolute() else get_workspace() / p]
    if Path(f_prior).is_absolute():
        candidates.append(get_workspace() / p.name)
    for cand in candidates:
        if cand.is_file():
            return cand
    raise HTTPException(status_code=404, detail=f"Prior file not found: {f_prior}")


def _load_xy(post: Path):
    """(N,) float X/Y sounding coordinates; None handling mirrored from ig.query.

    Same source priority as integrate_query._load_query_inputs: posterior file
    first, then the linked data file (f5_data attribute). 400 if unavailable.
    """
    import h5py

    with h5py.File(post, "r") as f:
        X = f["UTMX"][:].ravel().astype(float) if "UTMX" in f else None
        Y = f["UTMY"][:].ravel().astype(float) if "UTMY" in f else None
        f_data = str(f.attrs.get("f5_data", "") or "")
    if (X is None or Y is None) and f_data:
        p = Path(f_data)
        cand = p if p.is_absolute() else get_workspace() / p
        if not cand.is_file():
            cand = get_workspace() / Path(f_data).name
        if cand.is_file():
            with h5py.File(cand, "r") as f:
                if X is None and "UTMX" in f:
                    X = f["UTMX"][:].ravel().astype(float)
                if Y is None and "UTMY" in f:
                    Y = f["UTMY"][:].ravel().astype(float)
    if X is None or Y is None:
        raise HTTPException(status_code=400,
                            detail=f"No UTMX/UTMY coordinates in {post.name} or its data file")
    return X, Y


# ---------------------------------------------------------------------------
# Voronoi scaffold cache: geometry depends only on (X, Y, hull/edge knobs).
# ---------------------------------------------------------------------------

def _scaffold_key(f: str, post: Path, hull_ratio, edge_buffer, cell_area_k, elong_max):
    return (f, post.stat().st_mtime_ns, float(hull_ratio),
            None if edge_buffer is None else float(edge_buffer),
            float(cell_area_k),
            None if elong_max is None else float(elong_max))


def _build_scaffold(X, Y, hull_ratio, edge_buffer, cell_area_k, elong_max) -> dict:
    """Pure geometry — no probabilities. Runs in a worker thread."""
    import numpy as np
    from scipy.spatial import ConvexHull
    from shapely import Polygon, MultiPoint, concave_hull
    from integrate.integrate_query import voronoi_graph, voronoi_cells_ordered, flag_edge_cells

    vor, neighbors = voronoi_graph(X, Y)
    XY = np.column_stack([X, Y])
    boundary = concave_hull(MultiPoint(XY), ratio=hull_ratio)
    if boundary.geom_type == "MultiPolygon":
        boundary = max(boundary.geoms, key=lambda g: g.area)
    if boundary.is_empty or boundary.geom_type != "Polygon":
        boundary = Polygon(XY[ConvexHull(XY).vertices])
    cells = voronoi_cells_ordered(X, Y, boundary)
    cell_area = np.array([c.area if (c is not None and not c.is_empty) else 0.0
                          for c in cells])
    good, _, _ = flag_edge_cells(X, Y, vor, cells, boundary,
                                 edge_buffer=edge_buffer, k=cell_area_k,
                                 elong_max=elong_max)
    return {"neighbors": neighbors, "cells": cells, "cell_area": cell_area,
            "boundary": boundary, "good": good, "X": X, "Y": Y}


def _get_scaffold(key, X, Y, hull_ratio, edge_buffer, cell_area_k, elong_max) -> dict:
    with _scaffold_lock:
        sc = _scaffold_cache.get(key)
        if sc is not None:
            _scaffold_cache.move_to_end(key)
            return sc
    sc = _build_scaffold(X, Y, hull_ratio, edge_buffer, cell_area_k, elong_max)
    with _scaffold_lock:
        _scaffold_cache[key] = sc
        while len(_scaffold_cache) > _MAX_CACHE:
            _scaffold_cache.popitem(last=False)
    return sc


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class GeoParams(BaseModel):
    hull_ratio: float = 0.10
    edge_buffer: float | None = None        # None → auto (see flag_edge_cells)
    cell_area_k: float = 6.0
    elong_max: float | None = 4.0           # None skips the elongation test


class VolumeProbParams(BaseModel):
    f: str                                  # posterior file (workspace-relative)
    query_dict: dict                        # LLM/hand-edited probability query dict
    geo: GeoParams = GeoParams()


class VolumeGrowParams(BaseModel):
    f: str
    p: list[float]                          # per-sounding P, same order as /prob
    p_min: float
    x_center: float | None = None           # must be given together
    y_center: float | None = None
    max_area_m2: float | None = None
    geo: GeoParams = GeoParams()


class AreaMask(BaseModel):
    name: str
    indices: list[int]                      # sounding indices in the area


class VolumeVolumesParams(BaseModel):
    f: str
    query_dict: dict                        # must contain "metric" (percentile query)
    areas: list[AreaMask]
    geo: GeoParams = GeoParams()
    text: str = ""                          # NL prompt → figure title
    interpretation: str | None = None


# ---------------------------------------------------------------------------
# POST /prob — probability map
# ---------------------------------------------------------------------------

@router.post("/prob")
async def volume_prob(params: VolumeProbParams):
    """Evaluate a probability query; return per-sounding X/Y/P + Voronoi scaffold."""
    if not params.query_dict:
        raise HTTPException(status_code=400, detail="Query JSON is empty.")
    if "metric" in params.query_dict:
        raise HTTPException(status_code=400,
                            detail="Expected a probability query (no 'metric' key).")
    post = _resolve_post(params.f)
    _posterior_prior_posterior(post)        # validates posterior/prior link
    try:
        return await asyncio.to_thread(_volume_prob, post, params)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


def _volume_prob(post: Path, params: VolumeProbParams) -> dict:
    import os

    import numpy as np
    import integrate as ig

    cwd = os.getcwd()
    os.chdir(get_workspace())  # f5_prior / f5_data attrs resolve relative to cwd
    try:
        result, meta = ig.query(str(post), params.query_dict)
        if result is None:
            raise RuntimeError(f"ig.query returned no result for {post.name}")
    finally:
        os.chdir(cwd)

    X, Y = meta.get("X"), meta.get("Y")
    if X is None or Y is None:
        X, Y = _load_xy(post)
    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    P = np.asarray(result, dtype=float).ravel()

    g = params.geo
    key = _scaffold_key(params.f, post, g.hull_ratio, g.edge_buffer, g.cell_area_k, g.elong_max)
    sc = _get_scaffold(key, X, Y, g.hull_ratio, g.edge_buffer, g.cell_area_k, g.elong_max)

    finite = np.isfinite(P)
    return {
        "x": np.where(finite, X, 0.0).tolist(),
        "y": np.where(finite, Y, 0.0).tolist(),
        "p": np.where(finite, P, 0.0).tolist(),
        "good": sc["good"].tolist(),
        "cell_area": sc["cell_area"].tolist(),
        "boundary": list(map(list, np.asarray(sc["boundary"].exterior.coords))),
        "n": int(len(X)),
        "n_dropped": int((~sc["good"]).sum()),
        "mean_probability": float(np.mean(P[finite])),
    }


# ---------------------------------------------------------------------------
# POST /grow — grow one area
# ---------------------------------------------------------------------------

@router.post("/grow")
async def volume_grow(params: VolumeGrowParams):
    """Grow one connected region from a clicked center using the cached scaffold."""
    post = _resolve_post(params.f)
    if (params.x_center is None) != (params.y_center is None):
        raise HTTPException(status_code=400,
                            detail="x_center and y_center must be given together (or both omitted).")
    try:
        return await asyncio.to_thread(_volume_grow, params.f, post, params)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


def _volume_grow(f: str, post: Path, params: VolumeGrowParams) -> dict:
    import numpy as np
    from integrate.integrate_query import grow_connected_region, cells_to_polygon

    X, Y = _load_xy(post)                   # same arrays as /prob used (same file)
    g = params.geo
    key = _scaffold_key(f, post, g.hull_ratio, g.edge_buffer, g.cell_area_k, g.elong_max)
    sc = _get_scaffold(key, X, Y, g.hull_ratio, g.edge_buffer, g.cell_area_k, g.elong_max)

    P = np.asarray(params.p, dtype=float).ravel()
    if P.size != X.size:
        raise HTTPException(status_code=400,
                            detail=f"P has length {P.size}, expected {X.size}")
    P_eff = np.where(sc["good"], P, -np.inf)
    P_eff = np.where(np.isfinite(P_eff), P_eff, -np.inf)

    good = sc["good"]
    if params.x_center is not None:
        cand = np.where(good)[0]
        seed = int(cand[np.argmin((X[cand] - params.x_center) ** 2
                                  + (Y[cand] - params.y_center) ** 2)])
    else:
        seed = int(np.argmax(P_eff))        # all-finite by construction

    idx, area_m2, _order = grow_connected_region(
        P_eff, sc["neighbors"], sc["cell_area"],
        p_min=params.p_min, max_area_m2=params.max_area_m2, seed=seed)

    mask = np.zeros(len(P), dtype=bool)
    mask[idx] = True
    if idx.size:
        polygon = cells_to_polygon(sc["cells"], mask)
    else:                                   # empty region (cannot happen with a seed)
        polygon = None

    return {
        "seed": seed,
        "center": [float(X[seed]), float(Y[seed])],
        "indices": idx.astype(int).tolist(),
        "n_soundings": int(idx.size),
        "area_m2": float(area_m2),
        "polygon": (list(map(list, np.asarray(polygon.exterior.coords)))
                    if (polygon is not None and not polygon.is_empty
                        and polygon.geom_type == "Polygon") else []),
        "p_seed": float(P[seed]) if np.isfinite(P[seed]) else None,
    }


# ---------------------------------------------------------------------------
# POST /volumes — per-area volumes + bar chart
# ---------------------------------------------------------------------------

@router.post("/volumes")
async def volume_volumes(params: VolumeVolumesParams):
    """Percentile thickness per area → m³ volumes + P50/P5-P95 bar chart (PNG b64)."""
    import os

    if not params.query_dict or "metric" not in params.query_dict:
        raise HTTPException(status_code=400,
                            detail="Expected a percentile query dict (with a 'metric' key).")
    if not params.areas:
        raise HTTPException(status_code=400, detail="No areas defined.")
    post = _resolve_post(params.f)
    _posterior_prior_posterior(post)
    try:
        return await asyncio.to_thread(_volume_volumes, params.f, post, params)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


def _volume_volumes(f: str, post: Path, params: VolumeVolumesParams) -> dict:
    import os

    import numpy as np
    import integrate as ig

    pct_dict = dict(params.query_dict)
    pct_dict.setdefault("percentiles", [5, 50, 95])

    cwd = os.getcwd()
    os.chdir(get_workspace())
    try:
        result, meta = ig.query(str(post), pct_dict)
        if result is None:
            raise RuntimeError(f"ig.query returned no result for {post.name}")
    finally:
        os.chdir(cwd)
    pct = np.asarray(result, dtype=float)   # (N, n_pct)
    percentiles = [int(p) for p in meta.get("percentiles", pct_dict["percentiles"])]

    X, Y = _load_xy(post)
    g = params.geo
    key = _scaffold_key(f, post, g.hull_ratio, g.edge_buffer, g.cell_area_k, g.elong_max)
    sc = _get_scaffold(key, X, Y, g.hull_ratio, g.edge_buffer, g.cell_area_k, g.elong_max)

    per_area = []
    vols = []
    for a in params.areas:
        mask = np.zeros(len(X), dtype=bool)
        idx = np.asarray(a.indices, dtype=int)
        if idx.size:
            if idx.min() < 0 or idx.max() >= len(X):
                raise HTTPException(status_code=400, detail="Area index out of range.")
            mask[idx] = True
        v = np.asarray(ig.region_volumes(pct, {"mask": mask, "cell_area": sc["cell_area"]}),
                       dtype=float)
        vols.append(v)
        per_area.append({"name": a.name, "n_soundings": int(idx.size),
                         "volumes": v.tolist()})

    V = np.vstack(vols)                     # (n_areas, n_pct)
    figure_b64 = _volume_barchart(per_area, percentiles, V, params.text, params.interpretation)

    return {"percentiles": percentiles, "areas": per_area,
            "figure": figure_b64, "n_locations": int(meta.get("N_data", len(X)))}


def _volume_barchart(per_area, percentiles, V, text, interpretation) -> str:
    import numpy as np

    with _mpl_lock:
        before = set(plt.get_fignums())
        try:
            fig, ax = plt.subplots(figsize=(3 + 1.5 * len(per_area), 5))
            xg = np.arange(len(per_area))
            ax.bar(xg, V[:, 1],
                   yerr=[V[:, 1] - V[:, 0], V[:, 2] - V[:, 1]],
                   capsize=5, color="C0")
            ax.set_xticks(xg)
            ax.set_xticklabels([a["name"] for a in per_area])
            ax.set_ylabel("Volume (m$^3$)")
            title = text or "Volume per grown area"
            ax.set_title(f"{title}\n(bar = P{percentiles[1]}, whiskers = "
                         f"P{percentiles[0]}–P{percentiles[2]})"
                         + (f"\n{interpretation}" if interpretation else ""), fontsize=9)
            ax.grid(True, axis="y", ls="--", alpha=0.4)
        except Exception:
            plt.close("all")
            raise
        figs = sorted(set(plt.get_fignums()) - before)
        buf = io.BytesIO()
        plt.figure(figs[0]).savefig(buf, format="png", dpi=110, bbox_inches="tight")
        for num in figs:
            plt.close(num)
    return base64.b64encode(buf.getvalue()).decode("ascii")