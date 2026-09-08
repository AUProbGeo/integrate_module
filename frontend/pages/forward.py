"""Module 03 (GA-AEM) — Forward modelling of prior EM data.

``ig.prior_data_gaaem(f_prior_h5, file_gex|stmfiles, im, …)``. Only three
inputs matter (prior file, system file, resistivity model index ``im``); the
rest live in a collapsed **Advanced options** block.
"""

from __future__ import annotations

from fasthtml.common import Details, Div, Form, P, Summary

from frontend.components import btn, error_box, shell
from frontend.pages._forms import F, cast, field_grid
from frontend.pages._jobs_ui import register_job_routes, run_panel
from frontend.services import integrate_api as api
from frontend.services import jobs
from frontend.services.workspace import safe_path

_ESSENTIAL = [
    F("f_prior_h5", "Prior model f_prior_h5", ("h5", ["PRIOR"])),
    F("file_sys", "System file (.gex / .stm)", "system"),
    F("im", "im — resistivity model index", "int", "1"),
]
_ADVANCED = [
    F("id", "id (data index)", "int", "1"),
    F("im_height", "im_height", "int", "0"),
    F("N", "N (0 = all)", "int", "0"),
    F("Nhank", "Nhank", "int", "280"),
    F("Nfreq", "Nfreq", "int", "12"),
    F("Ncpu", "Ncpu (0 = auto)", "int", "0"),
    F("showInfo", "showInfo", "int", "0"),
    F("parallel", "Parallel processing", "bool", "1"),
    F("doMakePriorCopy", "Make prior copy", "bool", "1"),
    F("is_log", "Log-scale data (is_log)", "bool", "0"),
]
_NUMERIC = {f.name: f for f in (_ESSENTIAL + _ADVANCED) if f.kind in ("int", "float")}
_BOOLS = [f.name for f in _ADVANCED if f.kind == "bool"]


def _form():
    return Form(
        field_grid(_ESSENTIAL),
        Details(
            Summary("Advanced options"),
            field_grid(_ADVANCED),
            cls="wb-details",
        ),
        btn("Run forward modelling", kind="primary",
            hx_post="/forward/run", hx_target="#wb-run", hx_swap="innerHTML",
            hx_include="closest form", style="margin-top:16px;"),
        id="wb-forward-form",
    )


def register(rt) -> None:
    register_job_routes(rt, "/forward", out_key="f_prior_data_h5")

    @rt("/forward", name="page_forward")
    def page():
        return shell(
            "forward",
            P("Compute prior data with GA-AEM (prior_data_gaaem) so every prior "
              "realization carries a synthetic EM response.", cls="text-muted"),
            Div(cls="hr"),
            _form(),
            Div(id="wb-run"),
        )

    @rt("/forward/run", methods=["POST"])
    async def forward_run(request):
        form = await request.form()
        prior = (form.get("f_prior_h5") or "").strip()
        sysf = (form.get("file_sys") or "").strip()
        if not prior or prior.startswith("("):
            return error_box("No PRIOR .h5 file in the working folder.")
        if not sysf or sysf.startswith("("):
            return error_box("No .gex / .stm system file in the working folder.")
        try:
            kw = {
                "f_prior_h5": str(safe_path(prior)),
                **{name: cast(f, form.get(name, f.default)) for name, f in _NUMERIC.items()},
                **{name: (name in form) for name in _BOOLS},
            }
            if sysf.lower().endswith(".stm"):
                kw["stmfiles"] = [str(safe_path(sysf))]
            else:
                kw["file_gex"] = str(safe_path(sysf))
        except ValueError as e:
            return error_box(str(e))
        job_id = api.start_forward_job(kw)
        return run_panel(jobs.get(job_id), "/forward")
