"""geoprior1d — build a lithology/resistivity PRIOR from an Excel spec.

Unlike the generic resistivity priors (module 02) this one is driven by an
``.xlsx`` file. The page loads that workbook, lets you edit every sheet in
place (save-on-change into a session copy), Save it back to disk, then run
``geoprior1d(file_xlsx, Nreals, dmax, dz, ...)`` as a child-process job.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from fasthtml.common import Div, Form, Input, P, Span, Table, Tbody, Td, Th, Thead, Tr

from frontend.components import btn, error_box, field, figure_panel, select, shell, text_input
from frontend.pages._jobs_ui import register_job_routes, run_panel
from frontend.services import integrate_api as api
from frontend.services import jobs, xlsx
from frontend.services.session import state_for
from frontend.services.workspace import safe_path


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
    parts += [_savebar(st), _grid(st)]
    return Div(*parts)


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
        _load_into(st, file)
        return _editor(st), _run_form_oob(st)

    @rt("/geoprior/reload", methods=["POST"])
    def reload(session):
        st = state_for(session)
        if "gp_file" not in st:
            return Span()
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
        return _status_span(st, oob=True)

    @rt("/geoprior/addrow", methods=["POST"])
    def addrow(session, sheet: str):
        st = state_for(session)
        if "gp_book" in st:
            xlsx.add_row(st["gp_book"], sheet)
            st["gp_dirty"] = True
        return _grid(st)

    @rt("/geoprior/addcol", methods=["POST"])
    def addcol(session, sheet: str):
        st = state_for(session)
        if "gp_book" in st:
            xlsx.add_col(st["gp_book"], sheet)
            st["gp_dirty"] = True
        return _grid(st)

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
    def figure(name: str):
        return figure_panel("ig.plot_prior_stats()", api.prior_stats_figure(name),
                            "no figure produced")


def _run_form_oob(st):
    f = _run_form(st)
    return Div(f, id="gp-runwrap", hx_swap_oob="true", style="margin-top:20px;")
