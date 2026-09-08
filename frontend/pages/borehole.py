"""Module 03 (Boreholes) — Forward borehole data.

The second forward-model type: take a prior-data file that already carries the
EM prior data (``/D1`` from GA-AEM), a DATA file for the survey XY grid, and a
JSON borehole spec, and append the boreholes as extra jointly-invertible data
types (``ig.save_borehole_data``) onto a fresh copy — the B4 step of
``examples/integrate_rawmaterial_daugaard.py``.
"""

from __future__ import annotations

from fasthtml.common import Details, Div, Form, P, Span, Summary

from frontend.components import btn, error_box, field, shell, text_input
from frontend.pages._forms import F, cast, field_grid
from frontend.pages._jobs_ui import register_job_routes, run_panel
from frontend.services import integrate_api as api
from frontend.services import jobs
from frontend.services.workspace import safe_path

_ESSENTIAL = [
    F("f_prior_h5", "Prior-data file (has /D1 from GA-AEM)", "priordata"),
    F("f_data_h5", "Survey DATA file (XY grid)", ("h5", ["DATA"])),
    F("borehole_json", "Borehole spec (.json)", "json"),
    F("im_prior", "im_prior — lithology model index", "int", "2"),
]
_ADVANCED = [
    F("range_xyz", "range_xyz (m, XY fade-out)", "int", "300"),
    F("range_data", "range_data (blank = default)", "str", ""),
    F("showInfo", "showInfo", "int", "0"),
]
_NUMERIC = {f.name: f for f in (_ESSENTIAL + _ADVANCED) if f.kind in ("int", "float")}


def _form():
    return Form(
        field_grid(_ESSENTIAL),
        Details(Summary("Advanced options"), field_grid(_ADVANCED), cls="wb-details"),
        Div(field("Output .h5 (blank = <prior>_BH.h5)", text_input("output_file", "")),
            style="margin-top:12px;max-width:420px;"),
        btn("Run borehole forward", kind="primary",
            hx_post="/forward-bh/run", hx_target="#wb-run", hx_swap="innerHTML",
            hx_include="closest form", style="margin-top:16px;"),
        id="wb-bh-form",
    )


def _done_extra(job):
    ids = [str(x) for x in (job.result.get("id_bh") or []) if x is not None]
    return Div(
        Span("borehole data ids: ", cls="text-muted"),
        Span(", ".join(ids) if ids else "see log"),
        Span("  — add these to id_use on the Inversion page for a joint inversion.",
             cls="text-muted"),
        cls="status", style="margin-top:8px;",
    )


def register(rt) -> None:
    register_job_routes(rt, "/forward-bh", out_key="f_prior_data_bh_h5", done_extra=_done_extra)

    @rt("/forward-bh", name="page_forward-bh")
    def page():
        return shell(
            "forward-bh",
            P("Append boreholes as extra jointly-invertible data types onto a "
              "copy of a prior-data file (ig.save_borehole_data).", cls="text-muted"),
            Div(cls="hr"),
            _form(),
            Div(id="wb-run"),
        )

    @rt("/forward-bh/run", methods=["POST"])
    async def bh_run(request):
        form = await request.form()
        prior = (form.get("f_prior_h5") or "").strip()
        data = (form.get("f_data_h5") or "").strip()
        bh = (form.get("borehole_json") or "").strip()
        for label, v in (("PRIOR .h5", prior), ("DATA .h5", data), (".json borehole spec", bh)):
            if not v or v.startswith("("):
                return error_box(f"No {label} in the working folder.")
        try:
            kw = {
                "f_prior_h5": str(safe_path(prior)),
                "f_data_h5": str(safe_path(data)),
                "borehole_json": str(safe_path(bh)),
                "output_file": (form.get("output_file") or "").strip(),
                "range_data": (form.get("range_data") or "").strip(),
                **{name: cast(f, form.get(name, f.default)) for name, f in _NUMERIC.items()},
            }
        except ValueError as e:
            return error_box(str(e))
        job_id = api.start_borehole_job(kw)
        return run_panel(jobs.get(job_id), "/forward-bh")
