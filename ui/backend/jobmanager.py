"""In-process job manager.

Jobs run in spawned child processes (see worker.py). A pump thread per job
forwards queue events into the asyncio event loop, which fans them out to
WebSocket subscribers.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import queue as queue_mod
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

MAX_LOG_LINES = 1000


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = "pending"  # pending|running|done|error|cancelled
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    progress: dict = field(default_factory=lambda: {"current": 0, "total": 0, "info": {}})
    result: dict = field(default_factory=dict)
    error: str | None = None
    logs: deque = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    process: multiprocessing.Process | None = field(default=None, repr=False)

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "params": {k: v for k, v in self.params.items() if not k.startswith("_")},
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._ctx = multiprocessing.get_context("spawn")

    # -- asyncio side ----------------------------------------------------

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        with self._lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def _broadcast(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop log spam for slow consumers; never block the pump.
                if event.get("type") != "log":
                    try:
                        q.get_nowait()
                        q.put_nowait(event)
                    except asyncio.QueueEmpty:
                        pass

    def _emit(self, event: dict) -> None:
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._broadcast, event)

    # -- job lifecycle ---------------------------------------------------

    def start_rejection(self, params: dict) -> Job:
        from ui.backend.workspace import get_workspace
        params = {**params, "_workspace": str(get_workspace())}
        job = Job(id=uuid.uuid4().hex[:8], kind="rejection", params=dict(params))
        with self._lock:
            self.jobs[job.id] = job

        q = self._ctx.Queue()
        from ui.backend.worker import run_rejection_job

        proc = self._ctx.Process(target=run_rejection_job, args=(params, q), daemon=False)
        job.process = proc
        job.status = "running"
        job.started_at = time.time()
        proc.start()

        threading.Thread(target=self._pump, args=(job, q, proc), daemon=True).start()
        self._emit({"type": "job", "job": job.snapshot()})
        return job

    def _pump(self, job: Job, q, proc) -> None:
        """Consume worker events until the worker signals exit."""
        exited = False
        while not exited:
            try:
                ev = q.get(timeout=0.25)
            except queue_mod.Empty:
                if not proc.is_alive():
                    exited = True
                continue
            ev.setdefault("job_id", job.id)
            etype = ev.pop("type", None)

            if etype == "progress":
                job.progress = {"current": ev["current"], "total": ev["total"], "info": ev["info"]}
                self._emit({"type": "progress", **ev, "info": ev["info"]})
            elif etype == "log":
                job.logs.append(ev["line"])
                self._emit({"type": "log", **ev})
            elif etype == "done":
                job.result = {"f_post_h5": ev.get("f_post_h5")}
                self._emit({"type": "job", "job": job.snapshot()})
            elif etype == "error":
                job.error = ev.get("traceback", "unknown error")
                job.logs.append(job.error)
                self._emit({"type": "job", "job": job.snapshot()})
            elif etype == "exit":
                exited = True

        proc.join(timeout=5)
        job.ended_at = time.time()
        if job.status == "running":
            if job.error:
                job.status = "error"
            elif proc.exitcode not in (0, None):
                job.status = "error"
                job.error = job.error or f"worker exited with code {proc.exitcode}"
            elif job.result.get("f_post_h5") or proc.exitcode == 0:
                job.status = "done"
        self._emit({"type": "job", "job": job.snapshot()})

    def stop(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if job and job.process and job.process.is_alive():
            job.process.terminate()
            job.status = "cancelled"
            job.ended_at = time.time()
        return job

    def list(self) -> list[dict]:
        with self._lock:
            return [j.snapshot() for j in sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)]

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)


manager = JobManager()
