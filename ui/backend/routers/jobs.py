"""Job endpoints: start/stop/query jobs, WebSocket event stream."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ui.backend.jobmanager import manager

router = APIRouter(tags=["jobs"])


class RejectionParams(BaseModel):
    f_prior_h5: str
    f_data_h5: str
    f_post_h5: str | None = None
    N_use: int | None = None
    nr: int | None = None
    autoT: int | None = None
    T_base: float | None = None
    Ncpu: int | None = None
    parallel: bool | None = None
    use_N_best: int | None = None
    T_N_above: int | None = None
    T_P_acc_level: float | None = None
    backend: str | None = None
    id_use: list[int] | None = None
    ip_range: list[int] | None = None
    normalize_likelihood: bool | None = None
    updatePostStat: bool | None = None


@router.post("/api/jobs/rejection")
def start_rejection(params: RejectionParams):
    payload = params.model_dump(exclude_none=True)
    # integrate treats empty lists as "use all"; dropping them keeps parity.
    for key in ("id_use", "ip_range"):
        if not payload.get(key):
            payload.pop(key, None)
    job = manager.start_rejection(payload)
    return job.snapshot()


@router.get("/api/jobs")
def list_jobs():
    return {"jobs": manager.list()}


@router.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    snap = job.snapshot()
    snap["logs"] = list(job.logs)
    return snap


@router.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: str):
    job = manager.stop(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job.snapshot()


@router.websocket("/api/ws")
async def jobs_ws(ws: WebSocket):
    """Streams every job event: {type: job|progress|log, ...}."""
    await ws.accept()
    manager.attach_loop(asyncio.get_running_loop())
    q = manager.subscribe()
    try:
        await ws.send_json({"type": "jobs", "jobs": manager.list()})
        while True:
            event = await q.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        manager.unsubscribe(q)
