"""Module 04 — Inversion (localized rejection sampling).

``ig.integrate_rejection(f_prior_h5, f_data_h5, …)`` — one posterior ensemble
per sounding. Runs as a child-process job; the run panel (right column) shows
the live phase / data-point counter and streamed log.
"""

from __future__ import annotations

from fasthtml.common import Details, Div, Form, Input, Label, P, Span, Summary

from frontend.components import btn, error_box, field, select, shell, text_input
from frontend.pages._forms import F, cast, field_grid
from frontend.pages._jobs_ui import register_job_routes, run_panel
from frontend.services import integrate_api as api
from frontend.services import jobs
from frontend.services.workspace import safe_path

# N_use and T_base are rendered separately (N_use default = prior realization
# count; T_base belongs under the Temperature radios).
_ADV = [
    F("nr", "nr — retained / sounding", "int", "100"),
    F("id_use", "id_use (e.g. 1,2)", "str", ""),
    F("ip_range", "ip_range (e.g. 0:500)", "str", ""),
    F("Ncpu", "Ncpu (0 = auto)", "int", "0"),
    F("Nchunks", "Nchunks (0 = auto)", "int", "0"),
    F("use_N_best", "use_N_best (0 = off)", "int", "0"),
]
_N_USE = F("N_use", "N_use — prior samples used", "int", "100000")
_T_BASE = F("T_base", "T_base (fixed-T only)", "float", "1")
_NUMERIC = {f.name: f for f in ([_N_USE, _T_BASE] + _ADV) if f.kind in ("int", "float")}


def _nuse_field(prior_name: str | None):
    n = api.prior_n_realizations(prior_name) if prior_name else None
    return Div(field(_N_USE.label, text_input("N_use", str(n) if n else _N_USE.default)),
              id="inv-nuse")


def _essential_grid():
    priors = api.list_prior_with_data() or ["(no prior-data files — run Forward first)"]
    datas = api.list_h5_by_class("DATA") or ["(no DATA files)"]
    first_prior = priors[0] if not priors[0].startswith("(") else None
    return Div(
        Div(field("Prior model f_prior_h5", select(
            "f_prior_h5", priors, value=priors[0],
            hx_get="/inversion/nuse", hx_trigger="change",
            hx_target="#inv-nuse", hx_swap="outerHTML")),
            cls="wb-row"),
        Div(field("Observed data f_data_h5", select("f_data_h5", datas, value=datas[0])),
            cls="wb-row"),
        Div(_nuse_field(first_prior), cls="wb-row", style="max-width:260px;"),
    )


def _temp_radio():
    return Div(
        Div("Temperature", cls="wb-eyebrow"),
        Div(
            Label(Input(type="radio", name="autoT", value="auto", checked=True),
                  Span(cls="dot"), " autoT — estimated", cls="radio"),
            Label(Input(type="radio", name="autoT", value="fixed"),
                  Span(cls="dot"), " fixed T_base", cls="radio"),
            style="display:flex;gap:20px;margin:6px 0 10px;",
        ),
        Div(field(_T_BASE.label, text_input("T_base", _T_BASE.default)),
            style="max-width:220px;"),
        style="margin:12px 0;",
    )


def _form():
    return Form(
        _essential_grid(),
        Details(
            Summary("Advanced options"),
            Div(field("f_post_h5 (blank = auto)", text_input("f_post_h5", "")),
                style="max-width:420px;"),
            field_grid(_ADV),
            _temp_radio(),
            Div(
                field("backend", select("backend", ["numpy", "jax"], value="numpy")),
                Div(Label(Input(type="checkbox", name="parallel", value="1", checked=True),
                          Span(cls="dot"), " Parallel processing", cls="radio"),
                    style="margin-top:8px;"),
                style="margin-top:12px;max-width:420px;",
            ),
            cls="wb-details",
        ),
        btn("Run inversion", kind="primary",
            hx_post="/inversion/run", hx_target="#wb-run", hx_swap="innerHTML",
            hx_include="closest form", style="margin-top:20px;"),
        id="wb-inv-form",
    )


def register(rt) -> None:
    register_job_routes(rt, "/inversion", out_key="f_post_h5")

    @rt("/inversion/nuse")
    def nuse(f_prior_h5: str = ""):
        return _nuse_field(f_prior_h5 or None)

    @rt("/inversion", name="page_inversion")
    def page():
        return shell(
            "inversion",
            P("Localized rejection sampling of the prior against the observed "
              "data — one posterior ensemble per sounding.", cls="text-muted"),
            Div(cls="hr"),
            Div(_form(), Div(id="wb-run"), cls="wb-cols",
                style="grid-template-columns:minmax(0,1.7fr) minmax(0,1fr);"),
        )

    @rt("/inversion/run", methods=["POST"])
    async def inversion_run(request):
        form = await request.form()
        prior = (form.get("f_prior_h5") or "").strip()
        data = (form.get("f_data_h5") or "").strip()
        if not prior or prior.startswith("("):
            return error_box("No PRIOR .h5 file in the working folder.")
        if not data or data.startswith("("):
            return error_box("No DATA .h5 file in the working folder.")
        try:
            kw = {
                "f_prior_h5": str(safe_path(prior)),
                "f_data_h5": str(safe_path(data)),
                "f_post_h5": (form.get("f_post_h5") or "").strip(),
                "id_use": api.parse_int_list(form.get("id_use", "")),
                "ip_range": api.parse_range(form.get("ip_range", "")),
                "autoT": 1 if form.get("autoT", "auto") == "auto" else 0,
                "backend": form.get("backend", "numpy"),
                "parallel": "parallel" in form,
                **{name: cast(f, form.get(name, f.default)) for name, f in _NUMERIC.items()},
            }
        except ValueError as e:
            return error_box(str(e))
        job_id = api.start_inversion_job(kw)
        return run_panel(jobs.get(job_id), "/inversion")
