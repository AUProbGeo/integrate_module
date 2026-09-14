"""Workflow — Simple Workflow (``/workflow``).

One page, one button: **a)** sample a prior, **b)** compute prior EM data
(``ig.prior_data_em``), **c)** invert (``ig.integrate_rejection``) — all three
run back-to-back in a single child process (``worker.run_workflow_job``).
Only the essentials are visible (prior source + N, system file, observed
data); everything else sits in collapsed *options* blocks. The shared run
panel draws a step bar (a/b/c) above the per-step 0–100 % bar.
"""

from __future__ import annotations

from fasthtml.common import A, Details, Div, Form, Input, Label, P, Span, Summary

from frontend.components import btn, error_box, eyebrow, field, select, shell, text_input
from frontend.pages._forms import F, cast, field_grid
from frontend.pages._jobs_ui import register_job_routes, run_panel
from frontend.pages.forward import _ADVANCED as _FWD_ADV
from frontend.pages.inversion import _ADV as _INV_ADV
from frontend.pages.inversion import _T_BASE
from frontend.pages.prior import GENERIC_MODELS
from frontend.services import integrate_api as api
from frontend.services import jobs
from frontend.services.workspace import safe_path

BASE = "/workflow"

# a) prior ------------------------------------------------------------------
_N = F("N", "N (prior realizations)", "int", "100000")
# layered's options minus N (rendered separately as the visible field)
_LAYERED_OPTS = [F(f.name, f.label, f.kind, f.default)
                 for f in GENERIC_MODELS["layered"]["fields"] if f.name != "N"]
_GEOPRIOR_OPTS = [
    F("dmax", "dmax (m)", "float", "90"),
    F("dz", "dz (m)", "float", "1"),
    F("n_processes", "n_processes (-1 = all cores)", "int", "-1"),
]
_SOURCES = [("layered", "Default prior (prior_model_layered)"),
            ("geoprior", "geoprior1d (.xlsx)")]



def _ns(specs, prefix: str) -> list[F]:
    """Copy field specs under a form-name prefix. The three sections reuse the
    forward/inversion pages' specs, which share names (``N``, ``Ncpu``,
    ``parallel``) — and a form keeps only the last value per name."""
    return [F(prefix + f.name, f.label, f.kind, f.default) for f in specs]


# b) forward — form names ``fwd_*`` ------------------------------------------
_FWD = "fwd_"
_METHOD = F(_FWD + "method", "Forward method", ("select", ["ga-aem", "anemone"]), "ga-aem")
_FWD_FIELDS = _ns(_FWD_ADV, _FWD)
_FWD_NUM = [f for f in _FWD_FIELDS if f.kind in ("int", "float")]
_FWD_BOOL = [f for f in _FWD_FIELDS if f.kind == "bool"]

# c) inversion — form names ``inv_*`` ----------------------------------------
_INV = "inv_"
_N_USE = F(_INV + "N_use", "N_use (blank = all)", "int", "")
_INV_FIELDS = _ns(_INV_ADV, _INV)
_INV_T_BASE = F(_INV + _T_BASE.name, _T_BASE.label, _T_BASE.kind, _T_BASE.default)
_INV_NUM = [f for f in _INV_FIELDS + [_INV_T_BASE] if f.kind in ("int", "float")]


def _radio(name: str, value: str, label: str, checked: bool, **kw):
    return Label(Input(type="radio", name=name, value=value, checked=checked, **kw),
                 Span(cls="dot"), " " + label, cls="radio")


def _prior_block(source: str, xlsx: str = "", error: str | None = None):
    swap = dict(hx_get=f"{BASE}/prior-form", hx_target="#wf-prior", hx_swap="outerHTML",
                hx_include="closest form", hx_trigger="change")
    radios = Div(*[_radio("source", v, lbl, v == source, **swap) for v, lbl in _SOURCES],
                 style="display:flex;gap:20px;flex-wrap:wrap;margin:6px 0 12px;")
    if source == "geoprior":
        xs = api.list_xlsx() or ["(no .xlsx in the working folder)"]
        picker = Div(
            Div(field("geoprior1d spec (.xlsx)", select("file_xlsx", xs, value=xlsx or xs[0])),
                cls="wb-row"),
            Div(Label("Upload .xlsx", cls="wb-eyebrow"),
                Input(type="file", name="xlsx_upload", accept=".xlsx", cls="input",
                      hx_post=f"{BASE}/upload", hx_encoding="multipart/form-data",
                      hx_trigger="change", hx_target="#wf-prior", hx_swap="outerHTML"),
                cls="wb-row", style="max-width:420px;"),
        )
        opts = _GEOPRIOR_OPTS
    else:
        picker = Span()
        opts = _LAYERED_OPTS
    return Div(
        eyebrow("a) Prior"),
        error_box(error) if error else Span(),
        radios,
        picker,
        Div(field(_N.label, text_input(_N.name, _N.default)), cls="wb-row", style="max-width:260px;"),
        Details(Summary("Prior options"), field_grid(opts), cls="wb-details"),
        id="wf-prior",
    )


def _forward_block():
    gex = api.list_gex() or ["(no .gex system file in the working folder)"]
    return Div(
        eyebrow("b) Forward"),
        Div(field("System file (.gex)", select("file_gex", gex, value=gex[0])), cls="wb-row"),
        Details(
            Summary("Forward options"),
            Div(field(_METHOD.label, select(_METHOD.name, list(_METHOD.kind[1]), value=_METHOD.default)),
                style="max-width:260px;margin-bottom:12px;"),
            field_grid(_FWD_FIELDS),
            cls="wb-details",
        ),
    )


def _inversion_block():
    datas = api.list_h5_by_class("DATA") or ["(no DATA .h5 in the working folder)"]
    return Div(
        eyebrow("c) Inversion"),
        Div(field("Observed data f_data_h5", select("f_data_h5", datas, value=datas[0])), cls="wb-row"),
        Details(
            Summary("Inversion options"),
            Div(field("f_post_h5 (blank = auto)", text_input("f_post_h5", "")), style="max-width:420px;"),
            field_grid([_N_USE] + _INV_FIELDS),
            Div(
                Div("Temperature", cls="wb-eyebrow"),
                Div(_radio("autoT", "auto", "autoT — estimated", True),
                    _radio("autoT", "fixed", "fixed T_base", False),
                    style="display:flex;gap:20px;margin:6px 0 10px;"),
                Div(field(_INV_T_BASE.label, text_input(_INV_T_BASE.name, _INV_T_BASE.default)),
                    style="max-width:220px;"),
                style="margin:12px 0;",
            ),
            Div(field("backend", select(_INV + "backend", ["numpy", "jax"], value="numpy")),
                Div(Label(Input(type="checkbox", name=_INV + "parallel", value="1", checked=True),
                          Span(cls="dot"), " Parallel processing", cls="radio"), style="margin-top:8px;"),
                style="margin-top:12px;max-width:420px;"),
            cls="wb-details",
        ),
        btn("Start inversion workflow", kind="primary",
            hx_post=f"{BASE}/run", hx_target="#wb-run", hx_swap="innerHTML",
            hx_include="closest form", style="margin-top:20px;"),
    )


def _form(source: str = "layered"):
    return Form(
        _prior_block(source),
        Div(cls="hr"),
        _forward_block(),
        Div(cls="hr"),
        _inversion_block(),
        id="wf-form",
    )


def _done_extra(job):
    rows = [("Prior", job.result.get("f_prior_h5")),
            ("Prior data", job.result.get("f_prior_data_h5")),
            ("Posterior", job.result.get("f_post_h5"))]
    return Div(
        eyebrow("Workflow outputs"),
        *[Div(Span(f"{k}: {v or '—'}"), cls="wb-kv") for k, v in rows],
        Div(A("Open in Results →", href="/results"), style="margin-top:6px;"),
    )


def _bad(v: str) -> bool:
    return not v or v.startswith("(")


def _parse(form) -> dict:
    """Form -> ``{"prior", "forward", "inversion"}`` for ``run_workflow_job``.
    Raises ``ValueError`` with a user-facing message."""
    source = form.get("source", "layered")
    n = cast(_N, form.get("N", _N.default))
    if source == "geoprior":
        xlsx = (form.get("file_xlsx") or "").strip()
        if _bad(xlsx):
            raise ValueError("No geoprior1d .xlsx file selected.")
        prior = {"kind": "geoprior", "file_xlsx": str(safe_path(xlsx)), "Nreals": n,
                 **{f.name: cast(f, form.get(f.name, f.default)) for f in _GEOPRIOR_OPTS}}
    else:
        prior = {"kind": "layered", "N": n,
                 **{f.name: cast(f, form.get(f.name, f.default)) for f in _LAYERED_OPTS}}

    gex = (form.get("file_gex") or "").strip()
    if _bad(gex):
        raise ValueError("No .gex system file in the working folder.")
    forward = {
        "method": form.get(_METHOD.name, _METHOD.default),
        "file_gex": str(safe_path(gex)),
        **{f.name[len(_FWD):]: cast(f, form.get(f.name, f.default)) for f in _FWD_NUM},
        **{f.name[len(_FWD):]: (f.name in form) for f in _FWD_BOOL},
    }

    data = (form.get("f_data_h5") or "").strip()
    if _bad(data):
        raise ValueError("No DATA .h5 file in the working folder.")
    inversion = {
        "f_data_h5": str(safe_path(data)),
        "f_post_h5": (form.get("f_post_h5") or "").strip(),
        "id_use": api.parse_int_list(form.get(_INV + "id_use", "")),
        "ip_range": api.parse_range(form.get(_INV + "ip_range", "")),
        "autoT": 1 if form.get("autoT", "auto") == "auto" else 0,
        "backend": form.get(_INV + "backend", "numpy"),
        "parallel": (_INV + "parallel") in form,
        **{f.name[len(_INV):]: cast(f, form.get(f.name, f.default)) for f in _INV_NUM},
    }
    n_use = (form.get(_N_USE.name) or "").strip()
    if n_use:
        inversion["N_use"] = int(float(n_use))
    return {"prior": prior, "forward": forward, "inversion": inversion}


def register(rt) -> None:
    register_job_routes(rt, BASE, out_key="f_post_h5", done_extra=_done_extra)

    @rt(BASE, name="page_workflow")
    def page():
        return shell(
            "workflow",
            P("Sample a prior, compute its EM response and invert the observed data — "
              "in one go. Open the options blocks only if you need to change defaults.",
              cls="text-muted"),
            Div(cls="hr"),
            Div(_form(), Div(id="wb-run"), cls="wb-cols",
                style="grid-template-columns:minmax(0,1.7fr) minmax(0,1fr);"),
        )

    @rt(f"{BASE}/prior-form")
    def prior_form(source: str = "layered", file_xlsx: str = ""):
        return _prior_block(source if source in dict(_SOURCES) else "layered", file_xlsx)

    @rt(f"{BASE}/upload", methods=["POST"])
    async def upload(request):
        form = await request.form()
        up = form.get("xlsx_upload")
        try:
            if up is None or not getattr(up, "filename", ""):
                raise ValueError("No file chosen.")
            name = api.save_upload(up.filename, await up.read(), suffix=".xlsx")
        except ValueError as e:
            return _prior_block("geoprior", error=str(e))
        return _prior_block("geoprior", name)

    @rt(f"{BASE}/run", methods=["POST"])
    async def run(request):
        form = await request.form()
        try:
            params = _parse(form)
        except ValueError as e:
            return error_box(str(e))
        job_id = api.start_workflow_job(params)
        return run_panel(jobs.get(job_id), BASE)
