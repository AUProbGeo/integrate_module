"""geoprior1d — build a lithology/resistivity PRIOR from an Excel spec.

Unlike the generic resistivity priors (module 02) this one is driven by an
``.xlsx`` file. The page loads that workbook, lets you edit every sheet in
place (save-on-change into a session copy), Save it back to disk, then run
``geoprior1d(file_xlsx, Nreals, dmax, dz, ...)`` as a child-process job.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass

from fasthtml.common import (
    Div, Form, HttpHeader, Img, Input, Label, P, Span, Table, Tbody, Td, Th, Thead, Tr,
)

from frontend.components import btn, eyebrow, error_box, field, figure_panel, select, shell, text_input
from frontend.config import SCRATCH_DIR
from frontend.pages._jobs_ui import register_job_routes, run_panel
from frontend.services import integrate_api as api
from frontend.services import jobs, xlsx
from frontend.services.session import state_for
from frontend.services.workspace import safe_path

# live-preview: realization counts offered, and a hard cap
_PREVIEW_N_CHOICES = ["50", "100", "200", "500", "1000"]
_PREVIEW_MAX = 2000
_PREVIEW_DEBOUNCE_MS = 800


@dataclass(frozen=True)
class F:
    name: str
    label: str
    kind: str  # "int" | "float"
    default: str


_RUN_FIELDS = [
    F("Nreals", "Nreals", "int", "100000"),
    F("dmax", "dmax (m)", "float", "90"),
    F("dz", "dz (m)", "float", "1"),
    F("n_processes", "n_processes (-1 = all cores)", "int", "-1"),
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _lock_present(name: str) -> bool:
    try:
        p = safe_path(name)
        return (p.parent / f".~lock.{p.name}#").exists()
    except ValueError:
        return False


def _cast(f: F, raw) -> object:
    s = str(raw).strip()
    try:
        return int(float(s or f.default)) if f.kind == "int" else float(s or f.default)
    except ValueError:
        return f.default


def _status_span(st: dict, oob: bool = False):
    if st.get("gp_dirty"):
        txt, cls = "● unsaved edits", "gp-dirty"
    elif st.get("gp_saved_at"):
        txt, cls = f"saved {time.strftime('%H:%M:%S', time.localtime(st['gp_saved_at']))}", "gp-clean"
    else:
        txt, cls = "loaded", "gp-clean"
    return Span(txt, id="gp-status", cls=cls, hx_swap_oob="true" if oob else None)


def _tabs(st: dict):
    wb = st["gp_book"]
    active = st["gp_sheet"]
    return Div(
        *[
            btn(name, kind="primary" if name == active else "secondary",
                hx_get=f"/geoprior/sheet?name={name}", hx_target="#gp-grid", hx_swap="innerHTML")
            for name in xlsx.sheet_names(wb)
        ],
        cls="gp-tabs",
    )


def _grid(st: dict):
    wb, sheet = st["gp_book"], st["gp_sheet"]
    rows = xlsx.grid(wb, sheet)
    ncols = max((len(r) for r in rows), default=1)
    header = Tr(Th(""), *[Th(lbl) for lbl in xlsx.col_labels(ncols)])
    body = []
    for ri, row in enumerate(rows):
        tds = [Td(str(ri + 1), cls="rownum")]
        for ci in range(ncols):
            val = row[ci] if ci < len(row) else ""
            tds.append(Td(Input(
                name="v", value=val, cls="cell",
                hx_post="/geoprior/cell", hx_trigger="change", hx_swap="none",
                hx_vals=json.dumps({"sheet": sheet, "r": ri, "c": ci}),
            )))
        body.append(Tr(*tds))
    return Div(
        _tabs(st),
        Div(Table(Thead(header), Tbody(*body), cls="table wb-sheet"), cls="wb-tablewrap"),
        Div(
            btn("+ Row", hx_post="/geoprior/addrow",
                hx_vals=json.dumps({"sheet": sheet}), hx_target="#gp-grid", hx_swap="innerHTML"),
            btn("+ Col", hx_post="/geoprior/addcol",
                hx_vals=json.dumps({"sheet": sheet}), hx_target="#gp-grid", hx_swap="innerHTML"),
            cls="actions",
        ),
        id="gp-grid",
    )


def _savebar(st: dict, error: str | None = None):
    tgt = st.get("gp_target", st["gp_file"])
    row = Div(
        Span("SAVE TO", cls="wb-eyebrow"),
        Input(name="target", id="gp-target", value=tgt, cls="input",
              spellcheck="false", style="max-width:280px;"),
        btn("Save", kind="primary",
            hx_post="/geoprior/save", hx_include="#gp-target",
            hx_target="#gp-savebar", hx_swap="outerHTML"),
        btn("Reload original", kind="secondary",
            hx_post="/geoprior/reload", hx_target="#gp-editor", hx_swap="innerHTML"),
        _status_span(st),
        (Span(f"(loaded from {st['gp_file']})", cls="gp-clean", style="font-size:11px;")
         if tgt != st["gp_file"] else None),
        cls="gp-savebar-row",
    )
    children = ([error_box(error)] if error else []) + [row]
    return Div(*children, id="gp-savebar", cls="gp-savebar")


def _editor(st: dict):
    parts = []
    tgt = st.get("gp_target", st["gp_file"])
    if _lock_present(tgt):
        parts.append(error_box(
            f"{tgt} looks open in another app (.~lock file present) — "
            "saving may clobber unsaved changes there."
        ))
    parts += [_savebar(st), _grid(st), _preview_block(st)]
    return Div(*parts)


# --------------------------------------------------------------------------- #
# live summary-stats preview
# --------------------------------------------------------------------------- #
def _preview_n(st: dict) -> int:
    try:
        return max(1, min(_PREVIEW_MAX, int(st.get("gp_preview_n", 100))))
    except (TypeError, ValueError):
        return 100


def _preview_listener():
    """Always-present element that refreshes the preview when it hears the
    ``gp-changed`` body event (dispatched via an ``HX-Trigger`` response
    header from the cell / +row / +col handlers while auto-update is on).

    ``delay:`` debounces — each new event resets the timer, so a burst of
    edits yields one run. Being persistent (never swapped) it dodges the
    unreliable ``load``-on-OOB-content path.
    """
    return Div(
        id="gp-preview-trigger",
        hx_post="/geoprior/preview", hx_target="#gp-preview-figs", hx_swap="innerHTML",
        hx_include="#gp-preview-form, #gp-runform", hx_indicator="#gp-preview-spin",
        hx_trigger=f"gp-changed from:body delay:{_PREVIEW_DEBOUNCE_MS}ms",
    )


def _changed(st: dict):
    """`(HttpHeader,)` firing the client events an edit should trigger — splat
    into a route's return tuple.

    ``gp-cond`` always (the analytic ρ|lithology panel is instant, so it
    tracks every edit); ``gp-changed`` only while auto-update is on (that one
    runs geoprior1d).
    """
    events = ["gp-cond"] + (["gp-changed"] if st.get("gp_autopreview") else [])
    return (HttpHeader("HX-Trigger", ", ".join(events)),)


def _cond_listener():
    """Persistent element that redraws the ρ|lithology panel on the
    ``gp-cond`` body event (and once on load)."""
    return Div(
        id="gp-cond-trigger",
        hx_post="/geoprior/cond", hx_target="#gp-cond-fig", hx_swap="innerHTML",
        hx_include="#gp-cond-form", hx_indicator="#gp-cond-spin",
        hx_trigger="load, gp-cond from:body delay:400ms",
    )


def _cond_block(st: dict):
    ov = bool(st.get("gp_cond_overlay"))
    return Div(
        eyebrow("ρ | lithology — conditional resistivity priors"),
        Form(
            Label(
                Input(type="checkbox", name="cond_overlay", checked=ov,
                      hx_post="/geoprior/cond/toggle", hx_trigger="change",
                      hx_include="#gp-cond-form", hx_swap="none"),
                Span(cls="dot"), " overlay sampled (needs a preview run)", cls="radio",
            ),
            id="gp-cond-form", style="margin-bottom:10px;",
        ),
        Div(_cond_listener()),
        Div(P(Span("● ", style="color:var(--color-accent);"), "Drawing…",
              cls="wb-wait-msg", style="font-size:13px;"),
            id="gp-cond-spin", cls="htmx-indicator wb-wait"),
        Div(P("Analytic log-normal ρ priors, one per lithology.", cls="wb-empty"),
            id="gp-cond-fig", cls="gp-cond-fig"),
        id="gp-cond", style="margin-bottom:24px;",
    )


def _preview_block(st: dict):
    on = bool(st.get("gp_autopreview"))
    n = _preview_n(st)
    toggle_hx = dict(hx_post="/geoprior/preview/toggle", hx_trigger="change",
                     hx_include="#gp-preview-form", hx_swap="none")
    form = Form(
        Label(
            Input(type="checkbox", name="autopreview", checked=on, **toggle_hx),
            Span(cls="dot"), " Auto-update summary stats", cls="radio",
        ),
        field("realizations", select("preview_n", _PREVIEW_N_CHOICES, value=str(n), **toggle_hx)),
        btn("Refresh preview", kind="secondary",
            hx_post="/geoprior/preview", hx_target="#gp-preview-figs", hx_swap="innerHTML",
            hx_include="#gp-preview-form, #gp-runform", hx_indicator="#gp-preview-spin"),
        id="gp-preview-form",
        style="display:flex;gap:16px;align-items:end;flex-wrap:wrap;margin-bottom:12px;",
    )
    return Div(
        _cond_block(st),
        eyebrow("Summary statistics — prior realizations"),
        form,
        Div(_preview_listener()),
        Div(P(Span("● ", style="color:var(--color-accent);"),
              "Generating realizations…", cls="wb-wait-msg", style="font-size:13px;"),
            id="gp-preview-spin", cls="htmx-indicator wb-wait"),
        Div(P("Enable auto-update, or hit Refresh preview, to generate a small "
              "sample and see the realization panels.", cls="wb-empty"),
            id="gp-preview-figs", cls="gp-preview-figs"),
        id="gp-preview", style="margin-top:24px;border-top:2px solid var(--color-divider);padding-top:16px;",
    )


def _preview_figs(h5: str, n: int):
    figs = []
    for im, lbl in ((2, "Lithology (M2)"), (1, "Resistivity (M1)")):
        url = api.geoprior_preview_figure(h5, im=im, nr=n)
        if url:
            figs.append(Div(
                Div(f"M{im} — {lbl}  ·  {n} realizations", cls="wb-eyebrow"),
                Div(Img(src=url, alt=lbl), cls="gp-preview-fig"),
                cls="wb-panel",
            ))
    if not figs:
        return P("Preview ran but produced no /M1 or /M2 realizations.", cls="wb-empty")
    return Div(*figs, cls="gp-preview-grid")


def _run_form(st: dict):
    tgt = st.get("gp_target", st.get("gp_file", "?"))
    return Form(
        Div(*[field(f.label, text_input(f.name, f.default)) for f in _RUN_FIELDS], cls="wb-fields"),
        Div(field("Output .h5 (blank = auto)", text_input("output_file", "")),
            style="margin-top:16px;max-width:420px;"),
        P(Span("Runs on ", cls="text-muted"), Span(tgt),
          Span(" — the current edits are saved there first.", cls="text-muted"),
          style="font-size:12px;margin-top:12px;"),
        btn("Save & run geoprior1d", kind="primary",
            hx_post="/geoprior/run", hx_target="#wb-run", hx_swap="innerHTML",
            hx_include="closest form", style="margin-top:8px;"),
        id="gp-runform",
    )


def _picker(selected: str | None = None):
    files = api.list_xlsx()
    if not files:
        return P("No .xlsx files in the working folder. Add one, or change the "
                 "working folder above.", cls="wb-empty")
    return Form(
        field("Excel spec", select("file", files, value=selected or files[0])),
        btn("Load", kind="secondary",
            hx_post="/geoprior/load", hx_target="#gp-editor", hx_swap="innerHTML",
            hx_include="closest form"),
        style="display:flex;gap:12px;align-items:end;max-width:520px;",
    )


def _done_extra(job):
    out = job.result.get("f_prior_h5") or "—"
    flags = job.result.get("flags")
    parts = []
    if flags:
        parts.append(Div("flags: " + ", ".join(map(str, flags)), cls="status"))
    parts.append(btn("Plot prior stats", kind="secondary",
                     hx_get=f"/geoprior/figure/{out}", hx_target="#wb-fig", hx_swap="innerHTML",
                     style="margin-top:10px;"))
    parts.append(Div(id="wb-fig", style="margin-top:12px;"))
    return Div(*parts)


# --------------------------------------------------------------------------- #
def _load_into(st: dict, name: str) -> None:
    st["gp_file"] = name          # the on-disk source (Reload original reads this)
    st["gp_target"] = name        # where Save / Run writes (editable "Save to")
    st["gp_book"] = xlsx.open_book(safe_path(name))
    st["gp_sheet"] = xlsx.sheet_names(st["gp_book"])[0]
    st["gp_dirty"] = False
    st["gp_saved_at"] = None


def _norm_xlsx_name(name: str) -> str:
    name = name.strip().lstrip("/")
    if not name:
        raise ValueError("Enter a filename.")
    if not name.lower().endswith(".xlsx"):
        name += ".xlsx"
    return name


def _save_to(st: dict, target: str) -> str:
    """Write the workbook to *target* (workspace-relative). Returns the name."""
    name = _norm_xlsx_name(target)
    xlsx.save_book(st["gp_book"], safe_path(name))
    st["gp_target"] = name
    st["gp_dirty"] = False
    st["gp_saved_at"] = time.time()
    return name


def _scratch_paths(session: dict):
    """Per-session throwaway (xlsx, h5) under the frontend scratch dir."""
    sid = session.get("wb_sid", "anon")
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    return SCRATCH_DIR / f"gp_{sid}.xlsx", SCRATCH_DIR / f"gp_{sid}.h5"


def _preview_lock(st: dict):
    lk = st.get("gp_preview_lock")
    if lk is None:
        import threading

        lk = st["gp_preview_lock"] = threading.Lock()
    return lk


def _reset_preview(st: dict, session: dict | None = None) -> None:
    if session is not None:
        for p in _scratch_paths(session):
            with contextlib.suppress(OSError):
                p.unlink()


def register(rt) -> None:
    register_job_routes(rt, "/geoprior", out_key="f_prior_h5", done_extra=_done_extra)

    @rt("/geoprior", name="page_geoprior")
    def page(session):
        st = state_for(session)
        loaded = "gp_book" in st and st["gp_file"] in set(api.list_xlsx())
        return shell(
            "geoprior",
            P("Build a lithology/resistivity PRIOR with geoprior1d from an Excel "
              "spec. Edit the sheets below, save, then run.", cls="text-muted"),
            Div(cls="hr"),
            _picker(st.get("gp_file")),
            Div(_editor(st) if loaded else Span(), id="gp-editor", style="margin-top:20px;"),
            Div(_run_form(st) if loaded else Span(), id="gp-runwrap", style="margin-top:20px;"),
            Div(id="wb-run"),
        )

    @rt("/geoprior/load", methods=["POST"])
    def load(session, file: str = ""):
        st = state_for(session)
        if file not in set(api.list_xlsx()):
            return error_box("Pick an .xlsx in the working folder first.")
        _reset_preview(st, session)
        _load_into(st, file)
        return _editor(st), _run_form_oob(st)

    @rt("/geoprior/reload", methods=["POST"])
    def reload(session):
        st = state_for(session)
        if "gp_file" not in st:
            return Span()
        _reset_preview(st, session)
        _load_into(st, st["gp_file"])
        return _editor(st)

    @rt("/geoprior/sheet")
    def sheet(session, name: str):
        st = state_for(session)
        if "gp_book" in st and name in xlsx.sheet_names(st["gp_book"]):
            st["gp_sheet"] = name
        return _grid(st)

    @rt("/geoprior/cell", methods=["POST"])
    def cell(session, sheet: str, r: int, c: int, v: str = ""):
        st = state_for(session)
        if "gp_book" not in st:
            return Span()
        try:
            xlsx.set_cell(st["gp_book"], sheet, r, c, v)
            st["gp_dirty"] = True
        except (ValueError, KeyError) as e:
            return _status_span(st, oob=True), Span(str(e), hx_swap_oob="false")
        return _status_span(st, oob=True), *_changed(st)

    @rt("/geoprior/addrow", methods=["POST"])
    def addrow(session, sheet: str):
        st = state_for(session)
        if "gp_book" in st:
            xlsx.add_row(st["gp_book"], sheet)
            st["gp_dirty"] = True
        return _grid(st), *_changed(st)

    @rt("/geoprior/addcol", methods=["POST"])
    def addcol(session, sheet: str):
        st = state_for(session)
        if "gp_book" in st:
            xlsx.add_col(st["gp_book"], sheet)
            st["gp_dirty"] = True
        return _grid(st), *_changed(st)

    @rt("/geoprior/save", methods=["POST"])
    def save(session, target: str = ""):
        st = state_for(session)
        if "gp_book" not in st:
            return Div(id="gp-savebar", cls="gp-savebar")
        try:
            _save_to(st, target or st.get("gp_target") or st["gp_file"])
        except (OSError, ValueError) as e:
            return _savebar(st, error=f"Save failed: {e}")
        return _savebar(st)

    @rt("/geoprior/run", methods=["POST"])
    async def run(session, request):
        st = state_for(session)
        if "gp_book" not in st:
            return error_box("Load an .xlsx first.")
        form = await request.form()
        # always write the current edits to the chosen target before running
        try:
            name = _save_to(st, st.get("gp_target") or st["gp_file"])
            path = safe_path(name)
        except (OSError, ValueError) as e:
            return error_box(f"Could not save {st.get('gp_target')}: {e}")
        kw = {
            "file_xlsx": str(path),
            **{f.name: _cast(f, form.get(f.name, f.default)) for f in _RUN_FIELDS},
            "output_file": (form.get("output_file") or "").strip(),
        }
        job_id = api.start_geoprior_job(kw)
        return run_panel(jobs.get(job_id), "/geoprior")

    @rt("/geoprior/figure/{name}")
    def figure(name: str, im: int = 1):
        return figure_panel(f"ig.plot_prior_stats(im={im})",
                            api.prior_stats_figure(name, im),
                            "no figure produced", tall=True)

    # ----- live summary-stats preview -----------------------------------
    @rt("/geoprior/preview/toggle", methods=["POST"])
    def preview_toggle(session, autopreview: str = "", preview_n: str = ""):
        st = state_for(session)
        st["gp_autopreview"] = bool(autopreview)
        if preview_n:
            st["gp_preview_n"] = preview_n
        # turning it on (or changing N while on) kicks an immediate refresh
        return "", *_changed(st)

    @rt("/geoprior/preview", methods=["POST"])
    async def preview(session, request):
        import asyncio

        st = state_for(session)
        if "gp_book" not in st:
            return P("Load an .xlsx first.", cls="wb-empty")
        form = await request.form()
        if form.get("preview_n"):
            st["gp_preview_n"] = form.get("preview_n")
        n = _preview_n(st)
        try:
            dmax = float(form.get("dmax") or 90)
            dz = float(form.get("dz") or 1)
        except ValueError:
            dmax, dz = 90.0, 1.0

        xlsx_path, h5_path = _scratch_paths(session)
        try:
            xlsx.save_book(st["gp_book"], xlsx_path)
        except OSError as e:
            return error_box(f"Could not write preview spec: {e}")

        def _work():
            with _preview_lock(st):  # serialise overlapping refreshes
                return api.geoprior_preview_run(str(xlsx_path), str(h5_path),
                                                Nreals=n, dmax=dmax, dz=dz)

        try:
            h5 = await asyncio.to_thread(_work)
        except Exception as e:  # noqa: BLE001
            return error_box(f"Preview failed: {str(e).splitlines()[-1]}")
        return _preview_figs(h5, n)

    # ----- ρ | lithology conditional-prior panel (analytic, always live) --
    @rt("/geoprior/cond/toggle", methods=["POST"])
    def cond_toggle(session, cond_overlay: str = ""):
        st = state_for(session)
        st["gp_cond_overlay"] = bool(cond_overlay)
        return "", HttpHeader("HX-Trigger", "gp-cond")

    @rt("/geoprior/cond", methods=["POST"])
    def cond(session):
        st = state_for(session)
        if "gp_book" not in st:
            return P("Load an .xlsx first.", cls="wb-empty")
        xlsx_path, h5_path = _scratch_paths(session)
        with _preview_lock(st):  # don't collide with a running preview save
            try:
                xlsx.save_book(st["gp_book"], xlsx_path)
            except OSError as e:
                return error_box(f"Could not write spec: {e}")
            overlay = bool(st.get("gp_cond_overlay"))
            url = api.cond_resistivity_figure(
                str(xlsx_path),
                h5_path=str(h5_path) if h5_path.exists() else None,
                overlay=overlay,
            )
        if not url:
            return error_box("Could not read the Resistivity / Geology1 sheets — "
                             "check the class names and numbers line up.")
        note = (None if not overlay or h5_path.exists()
                else P("Run a preview to see the sampled histograms.",
                       cls="wb-empty", style="margin-top:6px;"))
        return Div(Img(src=url, alt="rho | lithology priors"), note, cls="gp-cond-img")


def _run_form_oob(st):
    f = _run_form(st)
    return Div(f, id="gp-runwrap", hx_swap_oob="true", style="margin-top:20px;")
