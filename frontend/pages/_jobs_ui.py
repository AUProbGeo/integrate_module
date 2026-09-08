"""Shared run/progress panel for child-process jobs (services/jobs).

A page calls ``register_job_routes(rt, base, done_extra=...)`` once and renders
``run_panel(job, base)`` into a ``#wb-run`` container. While the job runs the
panel self-polls ``{base}/progress/{id}`` every second; the terminal panel
carries no trigger, so polling stops on its own.
"""

from __future__ import annotations

from typing import Callable

from fasthtml.common import A, Div, I, Pre, Span

from frontend.components import btn, error_box, eyebrow
from frontend.services import jobs

_PANELS: dict[str, dict] = {}


def register_job_routes(rt, base: str, *, out_key: str = "f_prior_h5",
                        done_extra: Callable | None = None) -> None:
    _PANELS[base] = {"out_key": out_key, "done_extra": done_extra}

    @rt(f"{base}/progress/{{job_id}}", name=f"job_progress_{base.strip('/')}")
    def _progress(job_id: str):
        return run_panel(jobs.get(job_id), base)

    @rt(f"{base}/cancel/{{job_id}}", methods=["POST"], name=f"job_cancel_{base.strip('/')}")
    def _cancel(job_id: str):
        return run_panel(jobs.cancel(job_id), base)

    @rt(f"{base}/clear", name=f"job_clear_{base.strip('/')}")
    def _clear():
        return Span()


def run_panel(job, base: str):
    if job is None:
        return Span()
    opts = _PANELS.get(base, {})
    out_key = opts.get("out_key", "f_prior_h5")
    done_extra: Callable | None = opts.get("done_extra")

    running = job.status == "running"
    pct = job.pct
    big = f"{pct}%" if pct is not None else f"{job.elapsed:.0f} s"
    bar_cls = "wb-bar" + ("" if pct is not None else (" indet" if running else ""))
    width = f"{pct}%" if pct is not None else ("100%" if job.status == "done" else "0%")

    cur, tot = job.progress.get("current"), job.progress.get("total")
    if running:
        status = f"{job.kind} — {cur}/{tot}" if tot else f"{job.kind} — running…"
    else:
        status = f"{job.status} in {job.elapsed:.0f} s"

    body = [
        Div(Span(big, cls="big"), Span(job.phase or job.kind, cls="status"),
            style="display:flex;align-items:baseline;gap:12px;"),
        Div(I(style=f"width:{width}"), cls=bar_cls),
        Div(status, cls="status"),
    ]
    if job.logs:
        body.append(Pre("\n".join(list(job.logs)[-16:]), cls="wb-log"))
    if job.status == "error":
        body.append(error_box("Failed — see log above."))
    if job.status == "done":
        out = job.result.get(out_key) or "—"
        done = [
            eyebrow("Output"),
            Div(Span(out), A("Open in Data files →", href="/"), cls="wb-kv"),
        ]
        if done_extra is not None:
            extra = done_extra(job)
            if extra is not None:
                done.append(extra)
        body.append(Div(*done, style="margin-top:10px;"))

    if running:
        body.append(Div(
            btn("Cancel", kind="secondary",
                hx_post=f"{base}/cancel/{job.id}", hx_target="#wb-run", hx_swap="innerHTML"),
            cls="actions",
        ))
        poll = dict(hx_get=f"{base}/progress/{job.id}", hx_trigger="every 1s",
                    hx_target="this", hx_swap="outerHTML")
    else:
        body.append(Div(
            btn("Clear", kind="secondary",
                hx_get=f"{base}/clear", hx_target="#wb-run", hx_swap="innerHTML"),
            cls="actions",
        ))
        poll = {}
    return Div(*body, cls="wb-run", id="wb-run-panel", **poll)
