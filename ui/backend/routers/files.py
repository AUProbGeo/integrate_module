"""File browsing / inspection / upload endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from ui.backend import h5inspect
from ui.backend.workspace import get_workspace, safe_path

router = APIRouter(prefix="/api/files", tags=["files"])


@router.get("")
def list_files():
    """List .h5 files in the workspace with classification."""
    ws = get_workspace()
    items = []
    for p in sorted(ws.glob("*.h5"), key=lambda p: p.stat().st_mtime, reverse=True):
        st = p.stat()
        items.append({
            "name": p.name,
            "size_mb": round(st.st_size / 1e6, 2),
            "mtime": st.st_mtime,
            "class": h5inspect.classify(p),
        })
    return {"workspace": str(ws), "files": items}


def _resolve(name: str) -> Path:
    try:
        p = safe_path(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"No such file: {name}")
    return p


@router.get("/{name:path}/tree")
def file_tree(name: str):
    return h5inspect.tree(_resolve(name))


@router.get("/{name:path}/summary")
def file_summary(name: str):
    return h5inspect.summary(_resolve(name))


@router.post("/upload")
async def upload(file: UploadFile):
    if not file.filename or not file.filename.endswith(".h5"):
        raise HTTPException(status_code=400, detail="Only .h5 files are accepted")
    dest = safe_path(Path(file.filename).name)
    data = await file.read()
    dest.write_bytes(data)
    return {"name": dest.name, "size_mb": round(len(data) / 1e6, 2)}
