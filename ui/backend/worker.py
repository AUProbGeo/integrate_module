"""Worker entry points executed in a spawned child process.

Each job runs isolated from the web server so that long inversions can be
terminated without killing the API, and so integrate's own multiprocessing
pools are unaffected by the server event loop. Progress and console output
are streamed back through a multiprocessing queue.
"""

from __future__ import annotations

import contextlib
import io
import multiprocessing
import os
import traceback


class _QueueWriter(io.TextIOBase):
    """File-like object forwarding written lines to the job queue."""

    def __init__(self, queue):
        self._queue = queue
        self._buf = ""

    def writable(self):
        return True

    def write(self, s):
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line.strip():
                try:
                    self._queue.put({"type": "log", "line": line})
                except Exception:
                    pass
        return len(s)

    def flush(self):
        if self._buf.strip():
            try:
                self._queue.put({"type": "log", "line": self._buf.strip()})
            except Exception:
                pass
        self._buf = ""


def _clean_params(params: dict) -> dict:
    """Drop None values so integrate defaults kick in."""
    return {k: v for k, v in params.items() if v is not None}


def run_rejection_job(params: dict, queue) -> None:
    """Run integrate_rejection in this (child) process, streaming events."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    writer = _QueueWriter(queue)
    params = dict(params)
    workspace = params.pop("_workspace", None)
    if workspace:
        os.chdir(workspace)
    # integrate_rejection refuses to run unless the current process is named
    # "MainProcess" (guard against its own pool workers). This job process IS
    # the main process of the inversion, so claim the name.
    multiprocessing.current_process().name = "MainProcess"

    def progress_callback(current, total, info_dict=None):
        try:
            queue.put({
                "type": "progress",
                "current": int(current),
                "total": int(total),
                "info": dict(info_dict or {}),
            })
        except Exception:
            pass

    try:
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            import integrate as ig

            kwargs = _clean_params(params)
            kwargs.setdefault("progress_callback", progress_callback)
            f_post = ig.integrate_rejection(**kwargs)
        if f_post and workspace:
            f_post = os.path.relpath(f_post, workspace)
        if f_post:
            queue.put({"type": "done", "f_post_h5": f_post})
        else:
            queue.put({"type": "error", "traceback": "integrate_rejection returned no output file"})
    except BaseException:
        queue.put({"type": "error", "traceback": traceback.format_exc()})
    finally:
        queue.put({"type": "exit"})
