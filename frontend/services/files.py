"""Lightweight HDF5 structural scan + classification for the file browser.

Vendored from ``ui/backend/h5inspect.py`` (which mirrors ``doc/format.rst``).
This is deliberately a *structural* view only — realization counts, geometry,
statistics and plots come from ``integrate`` via
``frontend.services.integrate_api``, not from here. Do not grow this into a
data layer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np

MAX_TREE_NODES = 500
MAX_ATTR_ITEMS = 32

__all__ = ["classify", "tree", "summary"]


def _json_safe(value: Any) -> Any:
    """Convert numpy/h5py attribute values to plain python objects."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size > MAX_ATTR_ITEMS:
            return f"[array shape={value.shape} dtype={value.dtype}]"
        return [_json_safe(v) for v in value.tolist()] if value.ndim else value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _attrs_to_dict(attrs) -> dict:
    return {k: _json_safe(v) for k, v in attrs.items()}


def classify(path: Path | str) -> str:
    """Classify an HDF5 file as POSTERIOR, PRIOR, DATA, UNKNOWN or UNREADABLE.

    - POSTERIOR: has an ``/i_use`` dataset.
    - DATA:      has a ``/D<n>`` *group* (observed data).
    - PRIOR:     has model realizations ``/M<n>`` datasets.
    """
    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            if "i_use" in f and isinstance(f["i_use"], h5py.Dataset):
                return "POSTERIOR"
            if any(re.fullmatch(r"D\d+", k) and isinstance(f[k], h5py.Group) for k in keys):
                return "DATA"
            if any(re.fullmatch(r"M\d+", k) and isinstance(f[k], h5py.Dataset) for k in keys):
                return "PRIOR"
            return "UNKNOWN"
    except OSError:
        return "UNREADABLE"


def _dset_ids(f, pattern: str, cls) -> list[str]:
    rx = re.compile(pattern)
    return sorted(
        (k for k in f.keys() if rx.fullmatch(k) and isinstance(f[k], cls)),
        key=lambda k: int(k[1:]),
    )


def tree(path: Path | str) -> dict:
    """JSON-safe tree of groups/datasets with shapes, dtypes and attributes."""
    counter = {"n": 0}

    def build(name: str, obj) -> dict | None:
        if counter["n"] >= MAX_TREE_NODES:
            return None
        counter["n"] += 1
        node = {
            "name": name.split("/")[-1] or "/",
            "path": "/" + name,
            "kind": "dataset" if isinstance(obj, h5py.Dataset) else "group",
            "attrs": _attrs_to_dict(obj.attrs),
        }
        if isinstance(obj, h5py.Dataset):
            node["shape"] = list(obj.shape)
            node["dtype"] = str(obj.dtype)
            node["size_mb"] = round(obj.size * obj.dtype.itemsize / 1e6, 3)
        else:
            node["children"] = [
                c for key in obj.keys()
                if (c := build(f"{name}/{key}".strip("/"), obj[key])) is not None
            ]
        return node

    with h5py.File(path, "r") as f:
        root = {
            "name": "/", "path": "/", "kind": "group",
            "attrs": _attrs_to_dict(f.attrs), "children": [],
        }
        for key in f.keys():
            child = build(key, f[key])
            if child is not None:
                root["children"].append(child)
        root["truncated"] = counter["n"] >= MAX_TREE_NODES
        return root


def flat_datasets(path: Path | str) -> list[dict]:
    """Flat list of every dataset: ``{"path": "/D1/d_obs", "shape": [...]}``."""
    out: list[dict] = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append({"path": "/" + name, "shape": list(obj.shape), "dtype": str(obj.dtype)})

    with h5py.File(path, "r") as f:
        f.visititems(visit)
    return out


def summary(path: Path | str) -> dict:
    """Compact, class-specific structural summary of an INTEGRATE HDF5 file."""
    path = Path(path)
    cls = classify(path)
    out: dict[str, Any] = {"class": cls, "file": path.name}
    if cls in ("UNKNOWN", "UNREADABLE"):
        return out

    with h5py.File(path, "r") as f:
        out["root_attrs"] = _attrs_to_dict(f.attrs)

        if cls == "PRIOR":
            models = [
                {
                    "id": k,
                    "shape": list(f[k].shape),
                    "name": _json_safe(f[k].attrs.get("name", "")),
                    "is_discrete": bool(f[k].attrs.get("is_discrete", 0)),
                }
                for k in _dset_ids(f, r"M\d+", h5py.Dataset)
            ]
            data = [{"id": k, "shape": list(f[k].shape)} for k in _dset_ids(f, r"D\d+", h5py.Dataset)]
            out["n_realizations"] = int(f[models[0]["id"]].shape[0]) if models else 0
            out["models"] = models
            out["data"] = data
            out["has_forward_data"] = bool(data)

        elif cls == "DATA":
            datasets = []
            n_points = 0
            for k in _dset_ids(f, r"D\d+", h5py.Group):
                g = f[k]
                entry: dict[str, Any] = {
                    "id": k,
                    "noise_model": _json_safe(g.attrs.get("noise_model", "")),
                    "name": _json_safe(g.attrs.get("name", "")),
                    "keys": sorted(g.keys()),
                }
                if "d_obs" in g:
                    entry["shape"] = list(g["d_obs"].shape)
                    n_points = int(g["d_obs"].shape[0])
                if "i_use" in g:
                    i_use = g["i_use"][:].ravel()
                    entry["n_used"] = int((i_use != 0).sum())
                datasets.append(entry)
            out["n_points"] = n_points
            out["datasets"] = datasets
            out["has_geometry"] = all(g in f for g in ("UTMX", "UTMY"))

        elif cls == "POSTERIOR":
            i_use = f["i_use"]
            out["n_points"], out["n_realizations"] = (
                (int(i_use.shape[0]), int(i_use.shape[1])) if i_use.ndim == 2
                else (1, int(i_use.shape[0]))
            )
            for key in ("T", "EV", "EV_post", "CHI2", "N_UNIQUE"):
                if key in f:
                    arr = f[key][:].ravel().astype(float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size:
                        out[key.lower()] = {
                            "min": float(arr.min()), "max": float(arr.max()),
                            "mean": float(arr.mean()), "n": int(arr.size),
                        }
            out["models"] = [k for k in f.keys() if re.fullmatch(r"M\d+", k)]
            out["f5_data"] = _json_safe(f.attrs.get("f5_data", ""))
            out["f5_prior"] = _json_safe(f.attrs.get("f5_prior", ""))
            for key in ("inv_time", "date_start", "date_end", "N_use"):
                if key in f.attrs:
                    out[key] = _json_safe(f.attrs[key])
    return out
