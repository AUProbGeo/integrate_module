"""Module 02 — Prior model (generic resistivity priors).

``ig.prior_model_layered`` / ``prior_model_workbench`` /
``prior_model_workbench_direct`` / ``prior_model_smooth`` /
``prior_model_blocky`` / ``prior_model_sharp``. Runs in a child process
(``services/jobs`` + ``services/worker``); the run panel polls for progress.

geoprior1d lives on its own page (``pages/geoprior.py``) because it needs an
Excel-spec editor rather than a parameter form.
"""

from __future__ import annotations

from dataclasses import dataclass

from fasthtml.common import Div, Form, P

from frontend.components import btn, error_box, field, figure_panel, select, shell, text_input
from frontend.pages._jobs_ui import register_job_routes, run_panel
from frontend.services import integrate_api as api
from frontend.services import jobs


@dataclass(frozen=True)
class F:
    name: str
    label: str
    kind: object  # "int" | "float" | "str" | ("select", [opts])
    default: str


_RHO = [
    F("RHO_min", "RHO_min", "float", "1"),
    F("RHO_max", "RHO_max", "float", "300"),
    F("RHO_mean", "RHO_mean (normal)", "float", "180"),
    F("RHO_std", "RHO_std (normal)", "float", "80"),
]
_RHO_DIST = ("select", ["log-uniform", "uniform", "normal", "lognormal"])
_RHO_DIST_CHI2 = ("select", ["log-uniform", "uniform", "normal", "lognormal", "chi2"])

MODELS: dict[str, dict] = {
    "layered": {
        "label": "prior_model_layered",
        "fields": [
            F("N", "N (realizations)", "int", "100000"),
            F("lay_dist", "lay_dist", ("select", ["uniform", "chi2"]), "uniform"),
            F("dz", "dz (m)", "float", "1.0"),
            F("z_max", "z_max (m)", "float", "90.0"),
            F("NLAY_min", "NLAY_min", "int", "3"),
            F("NLAY_max", "NLAY_max", "int", "6"),
            F("NLAY_deg", "NLAY_deg (chi2)", "int", "6"),
            F("RHO_dist", "RHO_dist", _RHO_DIST, "log-uniform"),
            F("RHO_min", "RHO_min", "float", "0.1"),
            F("RHO_max", "RHO_max", "float", "100.0"),
            F("RHO_mean", "RHO_mean (normal)", "float", "100.0"),
            F("RHO_std", "RHO_std (normal)", "float", "80.0"),
        ],
    },
    "workbench": {
        "label": "prior_model_workbench",
        "fields": [
            F("N", "N (realizations)", "int", "100000"),
            F("p", "p (thickness power)", "int", "2"),
            F("z1", "z1 (m)", "float", "0"),
            F("z_max", "z_max (m)", "float", "100"),
            F("dz", "dz (m)", "float", "1"),
            F("lay_dist", "lay_dist", ("select", ["uniform", "chi2"]), "uniform"),
            F("nlayers", "nlayers (0 = use range)", "int", "0"),
            F("NLAY_min", "NLAY_min", "int", "3"),
            F("NLAY_max", "NLAY_max", "int", "6"),
            F("NLAY_deg", "NLAY_deg (chi2)", "int", "5"),
            F("RHO_dist", "RHO_dist", _RHO_DIST_CHI2, "log-uniform"),
            *_RHO,
            F("chi2_deg", "chi2_deg", "int", "100"),
        ],
    },
    "workbench_direct": {
        "label": "prior_model_workbench_direct",
        "fields": [
            F("N", "N (realizations)", "int", "100000"),
            F("p", "p (thickness power)", "int", "2"),
            F("z1", "z1 (m)", "float", "0"),
            F("z_max", "z_max (m)", "float", "100"),
            F("nlayers", "nlayers (0 → 30)", "int", "0"),
            F("NLAY_min", "NLAY_min", "int", "3"),
            F("NLAY_max", "NLAY_max", "int", "6"),
            F("RHO_dist", "RHO_dist", _RHO_DIST_CHI2, "log-uniform"),
            *_RHO,
            F("chi2_deg", "chi2_deg", "int", "100"),
        ],
    },
    "smooth": {
        "label": "prior_model_smooth (Workbench Smooth / L2)",
        "fields": [
            F("N", "N (realizations)", "int", "100000"),
            F("z1", "z1 (m)", "float", "0"),
            F("z_max", "z_max (m)", "float", "100"),
            F("dz", "dz (M1 grid, m)", "float", "1"),
            F("nlayers", "nlayers (0 → 30)", "int", "0"),
            F("p", "p (thickness power)", "int", "2"),
            F("corr_length", "corr_length (m)", "float", "15.0"),
            F("sigma_logrho", "sigma_logrho (BetaV)", "float", "0.25"),
            F("RHO_ref", "RHO_ref (process mean)", "float", "100.0"),
            F("RHO_min", "RHO_min", "float", "1"),
            F("RHO_max", "RHO_max", "float", "300"),
        ],
    },
    "blocky": {
        "label": "prior_model_blocky (Workbench Blocky / L1)",
        "fields": [
            F("N", "N (realizations)", "int", "100000"),
            F("z1", "z1 (m)", "float", "0"),
            F("z_max", "z_max (m)", "float", "100"),
            F("dz", "dz (M1 grid, m)", "float", "1"),
            F("nlayers", "nlayers (0 → 30)", "int", "0"),
            F("p", "p (thickness power)", "int", "2"),
            F("blocky_scale", "blocky_scale (Laplace)", "float", "0.25"),
            F("RHO_ref", "RHO_ref (process mean)", "float", "100.0"),
            F("RHO_min", "RHO_min", "float", "1"),
            F("RHO_max", "RHO_max", "float", "300"),
        ],
    },
    "sharp": {
        "label": "prior_model_sharp (Workbench Sharp / MGS)",
        "fields": [
            F("N", "N (realizations)", "int", "100000"),
            F("z1", "z1 (m)", "float", "0"),
            F("z_max", "z_max (m)", "float", "100"),
            F("dz", "dz (M1 grid, m)", "float", "1"),
            F("nlayers", "nlayers (0 → 30, native)", "int", "0"),
            F("p", "p (thickness power)", "int", "2"),
            F("n_jumps_mean", "n_jumps_mean (Poisson)", "float", "3.0"),
            F("RHO_dist", "RHO_dist", _RHO_DIST, "log-uniform"),
            *_RHO,
        ],
    },
}
_SEG = [(k, v["label"]) for k, v in MODELS.items()]


def _control(f: F, value: str):
    if isinstance(f.kind, tuple) and f.kind[0] == "select":
        return select(f.name, list(f.kind[1]), value=value)
    return text_input(f.name, value)


def _fieldset(model: str):
    return Div(
        *[field(f.label, _control(f, f.default)) for f in MODELS[model]["fields"]],
        cls="wb-fields",
    )


def _prior_form(model: str):
    return Form(
        Div(
            field(
                "Prior model",
                select(
                    "model", _SEG, value=model,
                    hx_get="/prior/form", hx_target="#wb-prior-form", hx_swap="outerHTML",
                    hx_trigger="change",
                ),
            ),
            style="max-width:420px;margin-bottom:20px;",
        ),
        _fieldset(model),
        Div(field("Output f_prior_h5 (blank = auto)", text_input("out_name", "")),
            style="margin-top:16px;max-width:420px;"),
        btn(
            f"Run {MODELS[model]['label']}", kind="primary",
            hx_post="/prior/run", hx_target="#wb-run", hx_swap="innerHTML",
            hx_include="closest form", style="margin-top:16px;",
        ),
        id="wb-prior-form",
    )


def _cast(f: F, raw):
    s = str(raw).strip()
    try:
        if f.kind == "int":
            return int(float(s or f.default))
        if f.kind == "float":
            return float(s or f.default)
    except ValueError:
        return f.default
    return s


def _done_extra(job):
    out = job.result.get("f_prior_h5") or "—"
    return Div(
        btn("Plot prior stats", kind="secondary",
            hx_get=f"/prior/figure/{out}", hx_target="#wb-fig", hx_swap="innerHTML",
            style="margin-top:10px;"),
        Div(id="wb-fig", style="margin-top:12px;"),
    )


def register(rt) -> None:
    register_job_routes(rt, "/prior", out_key="f_prior_h5", done_extra=_done_extra)

    @rt("/prior", name="page_prior")
    def page():
        return shell(
            "prior",
            P("Draw an ensemble of layered-earth realizations from your geological "
              "assumptions. Written to a PRIOR*.h5 file.", cls="text-muted"),
            Div(cls="hr"),
            _prior_form("layered"),
            Div(id="wb-run"),
        )

    @rt("/prior/form")
    def prior_form(model: str = "layered"):
        return _prior_form(model if model in MODELS else "layered")

    @rt("/prior/run", methods=["POST"])
    async def prior_run(request):
        form = await request.form()
        model = form.get("model", "layered")
        if model not in MODELS:
            return error_box(f"Unknown model: {model}")
        kwargs = {f.name: _cast(f, form.get(f.name, f.default)) for f in MODELS[model]["fields"]}
        job_id = api.start_prior_job(model, kwargs, form.get("out_name", ""))
        return run_panel(jobs.get(job_id), "/prior")

    @rt("/prior/figure/{name}")
    def prior_figure(name: str):
        return figure_panel("ig.plot_prior_stats()", api.prior_stats_figure(name),
                            "no figure produced")
