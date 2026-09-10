"""Module 02 — Prior model.

Two pages share one form/run/figure machinery:

* **Generic** (``/prior``)  — ``ig.prior_model_layered``.
* **WB** (``/prior-wb``)    — the Aarhus Workbench family
  ``ig.prior_model_smooth`` (L2) / ``prior_model_blocky`` (L1) /
  ``prior_model_sharp`` (MGS).

``ig.prior_model_workbench`` / ``prior_model_workbench_direct`` stay in the core
module for reference but are no longer surfaced here — the smooth/blocky/sharp
generators supersede them (and ``smooth`` with ``corr_length<=0`` reproduces an
uncorrelated i.i.d.-per-layer prior).

Runs in a child process (``services/jobs`` + ``services/worker``); the run
panel polls for progress. geoprior1d lives on its own page.
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
# smooth (L2): 'lognormal' keeps the historical correlated-GP-in-log output;
# 'log-uniform'/'uniform' swap the per-layer marginal via a Gaussian copula.
_RHO_DIST_L2 = ("select", ["lognormal", "log-uniform", "uniform", "normal"])


# --------------------------------------------------------------------------- #
# model catalogues — one page per catalogue
# --------------------------------------------------------------------------- #
GENERIC_MODELS: dict[str, dict] = {
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
}

WB_MODELS: dict[str, dict] = {
    "smooth": {
        "label": "prior_model_smooth (Workbench Smooth / L2)",
        "fields": [
            F("N", "N (realizations)", "int", "100000"),
            F("z1", "z1 (m)", "float", "0"),
            F("z_max", "z_max (m)", "float", "100"),
            F("dz", "dz (M1 grid, m)", "float", "1"),
            F("nlayers", "nlayers (0 → 30)", "int", "0"),
            F("p", "p (thickness power)", "int", "2"),
            F("corr_length", "corr_length (m, ≤0 = i.i.d.)", "float", "15.0"),
            F("sigma_logrho", "sigma_logrho (BetaV)", "float", "0.25"),
            F("RHO_dist", "RHO_dist", _RHO_DIST_L2, "lognormal"),
            F("RHO_ref", "RHO_ref (lognormal mean)", "float", "100.0"),
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


# --------------------------------------------------------------------------- #
# shared rendering
# --------------------------------------------------------------------------- #
def _control(f: F, value: str):
    if isinstance(f.kind, tuple) and f.kind[0] == "select":
        return select(f.name, list(f.kind[1]), value=value)
    return text_input(f.name, value)


def _fieldset(models: dict, model: str):
    return Div(
        *[field(f.label, _control(f, f.default)) for f in models[model]["fields"]],
        cls="wb-fields",
    )


def _prior_form(base: str, models: dict, model: str):
    seg = [(k, v["label"]) for k, v in models.items()]
    swap = dict(hx_get=f"{base}/form", hx_target="#wb-prior-form",
                hx_swap="outerHTML", hx_trigger="change") if len(seg) > 1 else {}
    return Form(
        Div(
            field("Prior model", select("model", seg, value=model, **swap)),
            style="max-width:480px;margin-bottom:20px;"
                  + ("" if len(seg) > 1 else "display:none;"),
        ),
        _fieldset(models, model),
        Div(field("Output f_prior_h5 (blank = auto)", text_input("out_name", "")),
            style="margin-top:16px;max-width:420px;"),
        btn(
            f"Run {models[model]['label']}", kind="primary",
            hx_post=f"{base}/run", hx_target="#wb-run", hx_swap="innerHTML",
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


def _done_extra(base: str):
    def _inner(job):
        out = job.result.get("f_prior_h5") or ""
        ims = api.prior_model_ims(out) if out else []
        picker = (select("im", [(str(i), lbl) for i, lbl in ims], value=str(ims[0][0]))
                  if ims else text_input("im", "1"))
        return Div(
            Form(
                field("Model parameter", picker),
                btn("Plot prior stats", kind="secondary",
                    hx_get=f"{base}/figure/{out}", hx_target="#wb-fig", hx_swap="innerHTML",
                    hx_include="closest form"),
                style="display:flex;gap:12px;align-items:end;margin-top:10px;",
            ),
            Div(id="wb-fig", style="margin-top:12px;"),
        )

    return _inner


def _register(rt, *, base: str, slug: str, models: dict, default_model: str,
              intro: str, page_name: str):
    register_job_routes(rt, base, out_key="f_prior_h5", done_extra=_done_extra(base))

    @rt(base, name=page_name)
    def page():
        return shell(
            slug,
            P(intro, cls="text-muted"),
            Div(cls="hr"),
            _prior_form(base, models, default_model),
            Div(id="wb-run"),
        )

    @rt(f"{base}/form")
    def prior_form(model: str = ""):
        return _prior_form(base, models, model if model in models else default_model)

    @rt(f"{base}/run", methods=["POST"])
    async def prior_run(request):
        form = await request.form()
        model = form.get("model", default_model)
        if model not in models:
            return error_box(f"Unknown model: {model}")
        kwargs = {f.name: _cast(f, form.get(f.name, f.default)) for f in models[model]["fields"]}
        job_id = api.start_prior_job(model, kwargs, form.get("out_name", ""))
        return run_panel(jobs.get(job_id), base)

    @rt(f"{base}/figure/{{name}}")
    def prior_figure(name: str, im: int = 1):
        return figure_panel(f"ig.plot_prior_stats(im={im})",
                            api.prior_stats_figure(name, im),
                            "no figure produced", tall=True)


def register(rt) -> None:
    _register(
        rt, base="/prior", slug="prior", models=GENERIC_MODELS,
        default_model="layered", page_name="page_prior",
        intro=("Draw an ensemble of layered-earth realizations from your geological "
               "assumptions. Written to a PRIOR*.h5 file."),
    )
    _register(
        rt, base="/prior-wb", slug="prior-wb", models=WB_MODELS,
        default_model="smooth", page_name="page_prior_wb",
        intro=("Aarhus Workbench 1D model types as prior ensembles — Smooth (L2), "
               "Blocky (L1) and Sharp (MGS). Written to a PRIOR*.h5 file."),
    )
