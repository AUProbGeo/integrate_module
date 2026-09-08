"""Tiny job runner: one spawned child process per long task, a daemon thread
draining its ``multiprocessing.Queue`` into an in-memory :class:`Job`.

The web layer never blocks on a job — pages poll ``jobs.get(id)`` over HTMX.
State is process-wide and lost on restart (single-user desktop tool).
"""

from __future__ import annotations

import multiprocessing
import queue as queue_mod
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

MAX_LOG_LINES = 400

# integrate's progress info_dict['phase'] -> label (from streamlit/ig_progress.py)
PHASE_LABELS = {
    "initializing": "Initializing",
    "generating": "Generating",
    "computing": "Computing",
    "sampling": "Sampling",
    "saving": "Saving",
    "post_processing": "Post-processing",
    "completed": "Completed",
}


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "running"  # running | done | error | cancelled
    created_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    progress: dict = field(default_factory=lambda: {"current": 0, "total": 0, "info": {}})
    result: dict = field(default_factory=dict)
    error: str | None = None
    logs: deque = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    process: multiprocessing.Process | None = field(default=None, repr=False)

    @property
    def elapsed(self) -> float:
        return (self.ended_at or time.time()) - self.created_at

    @property
    def pct(self) -> int | None:
        t = self.progress.get("total") or 0
        c = self.progress.get("current") or 0
        return int(100 * c / t) if t else None

    @property
    def phase(self) -> str:
        ph = (self.progress.get("info") or {}).get("phase", "")
        return PHASE_LABELS.get(ph, ph or "")


_CTX = multiprocessing.get_context("spawn")
_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


def start(kind: str, target, params: dict) -> Job:
    """Spawn ``target(params, queue)`` in a child process and track it."""
    from frontend.config import get_workspace

    params = {**params, "_workspace": str(get_workspace())}
    job = Job(id=uuid.uuid4().hex[:8], kind=kind, params=params)
    q = _CTX.Queue()
    proc = _CTX.Process(target=target, args=(params, q), daemon=False)
    job.process = proc
    with _LOCK:
        _JOBS[job.id] = job
    proc.start()
    threading.Thread(target=_pump, args=(job, q, proc), daemon=True).start()
    return job


def get(job_id: str) -> Job | None:
    return _JOBS.get(job_id)


def cancel(job_id: str) -> Job | None:
    job = _JOBS.get(job_id)
    if job and job.process and job.process.is_alive():
        job.process.terminate()
        job.status = "cancelled"
        job.ended_at = time.time()
    return job


def _pump(job: Job, q, proc) -> None:
    exited = False
    while not exited:
        try:
            ev = q.get(timeout=0.25)
        except queue_mod.Empty:
            if not proc.is_alive():
                break
            continue
        etype = ev.get("type")
        if etype == "progress":
            job.progress = {"current": ev["current"], "total": ev["total"], "info": ev.get("info", {})}
        elif etype == "log":
            job.logs.append(ev["line"])
        elif etype == "done":
            job.result = {k: v for k, v in ev.items() if k != "type"}
        elif etype == "error":
            job.error = ev.get("traceback", "unknown error")
            job.logs.append(job.error)
        elif etype == "exit":
            exited = True

    proc.join(timeout=5)
    job.ended_at = time.time()
    if job.status == "running":
        if job.error or (proc.exitcode not in (0, None)):
            job.status = "error"
            job.error = job.error or f"worker exited with code {proc.exitcode}"
        else:
            job.status = "done"
