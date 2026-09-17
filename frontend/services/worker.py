"""Child-process entry points for long tasks.

Each runs in a spawned process (see ``jobs.start``): isolated from the web
server, free to use ``integrate`` / ``geoprior1d`` multiprocessing pools, and
killable. Progress + stdout/stderr stream back over a ``multiprocessing.Queue``.

This is the *only* module besides ``integrate_api`` that imports ``integrate``
— and it does so inside the child process, never in the web process.
"""

from __future__ import annotations

import contextlib
import io
import multiprocessing
import os
import traceback


class _QueueWriter(io.TextIOBase):
    """File-like object forwarding written lines to the job queue as logs."""

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
                with contextlib.suppress(Exception):
                    self._queue.put({"type": "log", "line": line})
        return len(s)

    def flush(self):
        if self._buf.strip():
            with contextlib.suppress(Exception):
                self._queue.put({"type": "log", "line": self._buf.strip()})
        self._buf = ""


def _prep(params: dict) -> str:
    """Common child-process setup. Returns the workspace path."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    workspace = params.pop("_workspace", None)
    if workspace:
        os.chdir(workspace)
    # prior_model_* / integrate_rejection refuse to run unless the current
    # process is named "MainProcess" (a guard against their own pool workers).
    multiprocessing.current_process().name = "MainProcess"
    return workspace or os.getcwd()


def _clean(params: dict) -> dict:
    return {k: v for k, v in params.items() if v is not None and not k.startswith("_")}


def _relpath(path, workspace) -> str | None:
    if not path:
        return None
    try:
        return os.path.relpath(path, workspace)
    except ValueError:
        return str(path)


def _progress_cb(queue):
    def progress_callback(current, total, info_dict=None):
        with contextlib.suppress(Exception):
            queue.put({
                "type": "progress",
                "current": int(current),
                "total": int(total),
                "info": dict(info_dict or {}),
            })

    return progress_callback


def _run(queue, workspace, out_key: str, label: str, call):
    """Common wrapper: redirect I/O, run ``call()``, emit done/error/exit."""
    writer = _QueueWriter(queue)
    try:
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            out = call()
        writer.flush()
        rel = _relpath(out, workspace)
        if rel:
            queue.put({"type": "done", out_key: rel})
        else:
            queue.put({"type": "error", "traceback": f"{label} returned no output file"})
    except BaseException:  # noqa: BLE001 — report everything to the UI
        queue.put({"type": "error", "traceback": traceback.format_exc()})
    finally:
        queue.put({"type": "exit"})


def run_prior_job(params: dict, queue) -> None:
    """Dispatch to ig.prior_model_{layered,workbench,workbench_direct,smooth,blocky,sharp}."""
    workspace = _prep(params)
    params = dict(params)
    model = params.pop("model", "layered")

    def call():
        import integrate as ig

        fn = {
            "layered": ig.prior_model_layered,
            "workbench": ig.prior_model_workbench,
            "workbench_direct": ig.prior_model_workbench_direct,
            "smooth": ig.prior_model_smooth,
            "blocky": ig.prior_model_blocky,
            "sharp": ig.prior_model_sharp,
        }[model]
        return fn(**_clean(params), progress_callback=_progress_cb(queue))

    _run(queue, workspace, "f_prior_h5", model, call)


_FORWARD_BACKENDS = ("ga-aem", "anemone", "simpeg")


def run_forward_job(params: dict, queue) -> None:
    """ig.prior_data_{gaaem,anemone,simpeg}(f_prior_h5, file_gex, …) -> prior-data .h5."""
    workspace = _prep(params)
    params = dict(params)
    backend = params.pop("backend", "ga-aem")

    # Validate backend early, before _run
    if backend not in _FORWARD_BACKENDS:
        raise ValueError(f"unknown forward backend: {backend!r}")
    name = {"ga-aem": "gaaem", "anemone": "anemone", "simpeg": "simpeg"}[backend]

    def call():
        import integrate as ig

        fn = getattr(ig, f"prior_data_{name}")
        return fn(**_clean(params), progress_callback=_progress_cb(queue))

    _run(queue, workspace, "f_prior_data_h5", f"prior_data_{name}", call)


def run_rejection_job(params: dict, queue) -> None:
    """ig.integrate_rejection(f_prior_h5, f_data_h5, …) -> posterior .h5."""
    workspace = _prep(params)
    params = dict(params)

    def call():
        import integrate as ig

        return ig.integrate_rejection(**_clean(params), progress_callback=_progress_cb(queue))

    _run(queue, workspace, "f_post_h5", "integrate_rejection", call)


def run_borehole_job(params: dict, queue) -> None:
    """Add borehole data types onto a copy of a prior-data file.

    ``ig.copy_hdf5_file`` -> ``ig.read_borehole`` -> ``ig.save_borehole_data``.
    """
    workspace = _prep(params)
    p = dict(params)
    writer = _QueueWriter(queue)
    try:
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            import integrate as ig

            src = p["f_prior_h5"]
            out = (p.get("output_file") or "").strip()
            if not out:
                base, ext = os.path.splitext(os.path.basename(src))
                out = f"{base}_BH{ext or '.h5'}"
            ig.copy_hdf5_file(src, out)
            bh = ig.read_borehole(p["borehole_json"], showInfo=int(p.get("showInfo", 0)))
            kw = {
                k: p[k]
                for k in ("im_prior", "range_xyz", "range_data", "showInfo")
                if p.get(k) not in (None, "")
            }
            _id_prior, id_bh = ig.save_borehole_data(
                out, p["f_data_h5"], bh, doPlot=False, **kw
            )
            try:
                id_bh = [int(x) for x in (id_bh if isinstance(id_bh, (list, tuple)) else [id_bh])]
            except (TypeError, ValueError):
                id_bh = []
            print(f"borehole /D ids added: {id_bh}")
        writer.flush()
        rel = _relpath(out, workspace)
        queue.put({"type": "done", "f_prior_data_bh_h5": rel, "id_bh": id_bh})
    except BaseException:  # noqa: BLE001
        queue.put({"type": "error", "traceback": traceback.format_exc()})
    finally:
        queue.put({"type": "exit"})


def run_geoprior_job(params: dict, queue) -> None:
    """Run geoprior1d(file_xlsx, Nreals, dmax, dz, ...) -> PRIOR .h5."""
    workspace = _prep(params)
    writer = _QueueWriter(queue)
    params = dict(params)
    try:
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            from geoprior1d import geoprior1d

            name, flags = geoprior1d(
                params["file_xlsx"],
                Nreals=int(params["Nreals"]),
                dmax=float(params["dmax"]),
                dz=float(params["dz"]),
                n_processes=int(params.get("n_processes", -1)),
                output_file=params.get("output_file") or None,
            )
        writer.flush()
        rel = _relpath(name, workspace)
        if rel:
            queue.put({"type": "done", "f_prior_h5": rel, "flags": list(flags or [])})
        else:
            queue.put({"type": "error", "traceback": "geoprior1d returned no output file"})
    except BaseException:  # noqa: BLE001
        queue.put({"type": "error", "traceback": traceback.format_exc()})
    finally:
        queue.put({"type": "exit"})


# NOTE: geoprior1d live-preview runs inline in a worker thread
# (``integrate_api.geoprior_preview_run``), not here — the child-process spawn
# cost (~2 s of interpreter + imports) dwarfs the <1 s generation itself.


# --------------------------------------------------------------------------- #
# Simple workflow: prior -> prior_data_em -> integrate_rejection in one process
# --------------------------------------------------------------------------- #
WORKFLOW_STEPS = ("Sampling prior", "Computing prior data", "Inversion")


def _step_progress_cb(queue, *, step: int, n_steps: int, label: str):
    """Like ``_progress_cb`` but tags every event with its workflow step so
    the run panel can draw a step bar above the per-step 0–100 % bar."""
    def progress_callback(current, total, info_dict=None):
        info = dict(info_dict or {})
        info.update(step=step, n_steps=n_steps, step_label=label)
        with contextlib.suppress(Exception):
            queue.put({
                "type": "progress",
                "current": int(current),
                "total": int(total),
                "info": info,
            })

    return progress_callback


def run_workflow_job(params: dict, queue) -> None:
    """``params = {"prior": {...}, "forward": {...}, "inversion": {...}}``.

    * ``prior["kind"]`` ∈ {"layered", "geoprior"}; the rest are that
      generator's kwargs (``geoprior1d`` takes ``file_xlsx/Nreals/dmax/dz/
      n_processes``).
    * ``forward["method"]`` ∈ {"ga-aem", "anemone", "simpeg"} + ``prior_data_em`` kwargs
      (``file_gex`` or ``stmfiles``, ``im``, ``id``, …).
    * ``inversion`` = ``integrate_rejection`` kwargs minus ``f_prior_h5``.

    Each step's output file feeds the next; the done event carries all three.
    """
    workspace = _prep(params)
    prior = dict(params["prior"])
    forward = dict(params["forward"])
    inversion = dict(params["inversion"])
    n = len(WORKFLOW_STEPS)

    def cb(step):
        return _step_progress_cb(queue, step=step, n_steps=n, label=WORKFLOW_STEPS[step - 1])

    writer = _QueueWriter(queue)
    try:
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            import integrate as ig

            # a) prior ------------------------------------------------------
            cb(1)(0, 0, {"phase": "initializing"})
            kind = prior.pop("kind", "layered")
            if kind == "geoprior":
                from geoprior1d import geoprior1d

                f_prior_h5, _flags = geoprior1d(
                    prior["file_xlsx"],
                    Nreals=int(prior["Nreals"]),
                    dmax=float(prior["dmax"]),
                    dz=float(prior["dz"]),
                    n_processes=int(prior.get("n_processes", -1)),
                )
            else:
                f_prior_h5 = ig.prior_model_layered(**_clean(prior), progress_callback=cb(1))
            print(f"[workflow] prior -> {f_prior_h5}")

            # b) forward ----------------------------------------------------
            cb(2)(0, 0, {"phase": "initializing"})
            method = forward.pop("method", "ga-aem")
            f_prior_data_h5 = ig.prior_data_em(
                f_prior_h5, method=method, **_clean(forward), progress_callback=cb(2))
            print(f"[workflow] prior data ({method}) -> {f_prior_data_h5}")

            # c) inversion --------------------------------------------------
            cb(3)(0, 0, {"phase": "initializing"})
            f_post_h5 = ig.integrate_rejection(
                f_prior_h5=f_prior_data_h5, **_clean(inversion), progress_callback=cb(3))
            print(f"[workflow] posterior -> {f_post_h5}")
        writer.flush()
        rel = _relpath(f_post_h5, workspace)
        if rel:
            queue.put({
                "type": "done",
                "f_prior_h5": _relpath(f_prior_h5, workspace),
                "f_prior_data_h5": _relpath(f_prior_data_h5, workspace),
                "f_post_h5": rel,
            })
        else:
            queue.put({"type": "error", "traceback": "integrate_rejection returned no output file"})
    except BaseException:  # noqa: BLE001 — report everything to the UI
        queue.put({"type": "error", "traceback": traceback.format_exc()})
    finally:
        queue.put({"type": "exit"})
