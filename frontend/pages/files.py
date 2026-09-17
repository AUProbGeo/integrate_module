"""Module 01 — Data files.

Lists the workspace ``*.h5`` files, classifies each (DATA / PRIOR / POSTERIOR
/ UNKNOWN), and — on selection — shows a structural detail panel plus a
rendered ``ig.plot_geometry`` figure.
"""

from __future__ import annotations

from fasthtml.common import Div, P, Span, Table, Tbody, Td, Th, Thead, Tr

from frontend.components import (
    btn, error_box, eyebrow, field, figure_panel, seg, shell, stat, stat_grid, tag,
)
from frontend.services import integrate_api as api

_FILTERS = [("all", "All files"), ("data", "Hide PRIOR* / POST*")]


def _visible(files: list[dict], ffilter: str) -> list[dict]:
    """``data`` filter hides everything classified PRIOR or POSTERIOR."""
    if ffilter != "data":
        return files
    return [f for f in files if f["kind"] not in ("PRIOR", "POSTERIOR")]


def _row(f: dict, selected: str | None):
    return Tr(
        Td(f["name"], style="font-variant-numeric:tabular-nums;overflow-wrap:anywhere;"),
        Td(tag(f["kind"])),
        Td(f["size"], style="text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;"),
        cls="sel" if f["name"] == selected else None,
        hx_get=f"/files/inspect/{f['name']}",
        hx_target="#wb-detail",
        hx_swap="innerHTML",
    )


def _file_table(ffilter: str, selected: str | None = None):
    files = api.list_h5()
    shown = _visible(files, ffilter)
    body = (
        Tbody(*[_row(f, selected) for f in shown])
        if shown
        else Tbody(Tr(Td("No .h5 files in the workspace.", colspan="3", cls="wb-empty")))
    )
    return Div(
        Table(
            Thead(Tr(Th("File"), Th("Type"), Th("Size", style="text-align:right;"))),
            body,
            cls="table",
        ),
        cls="wb-tablewrap",
        id="wb-filelist",
    )


def _count_label(ffilter: str) -> str:
    files = api.list_h5()
    shown = _visible(files, ffilter)
    return f"{len(shown)} / {len(files)} shown" if len(shown) != len(files) else f"{len(files)} files"


def _toolbar(ffilter: str):
    hx = dict(
        hx_get="/files/list",
        hx_target="#wb-filelist",
        hx_swap="outerHTML",
        hx_include="[name='ffilter']",
    )
    seg_ctl = seg("ffilter", _FILTERS, ffilter, hx_trigger="change", **hx)
    return Div(
        seg_ctl,
        _count(ffilter),
        btn("Rescan", **hx),
        cls="wb-toolbar",
    )


def _count(ffilter: str, oob: bool = False):
    return Span(
        _count_label(ffilter), id="wb-count",
        cls="text-muted",
        style="margin-left:auto;font-size:12px;font-variant-numeric:tabular-nums;",
        hx_swap_oob="true" if oob else None,
    )


def _detail_placeholder():
    return P("Select a file to inspect.", cls="wb-empty")


def _kv_rows(pairs):
    return [Div(Span(k), Span(str(v), cls="text-muted"), cls="wb-kv") for k, v in pairs]


def _types_table(t: dict):
    return Div(
        Table(
            Thead(Tr(*[Th(c) for c in t["columns"]])),
            Tbody(*[
                Tr(*[Td(str(x), style="font-variant-numeric:tabular-nums;") for x in row])
                for row in t["rows"]
            ]),
            cls="table",
        ),
        cls="wb-tablewrap",
        style="margin-bottom:16px;",
    )


def _detail(name: str):
    try:
        d = api.file_detail(name)
    except Exception as e:  # noqa: BLE001 — surface any read error in the panel
        return error_box(f"Could not read {name}: {e}")

    blocks = [
        Div(
            Span(d["name"], style="font-family:var(--font-heading);font-weight:800;font-size:20px;overflow-wrap:anywhere;"),
            " ",
            tag(d["kind"]),
            style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:12px;",
        ),
    ]
    if d["stats"]:
        cells = [stat(lbl, val) for lbl, val in d["stats"]]
        blocks.append(stat_grid(*cells, cols=min(len(cells), 5)))
    if d.get("types") and d["types"]["rows"]:
        blocks += [eyebrow(d["types"]["title"]), _types_table(d["types"])]
    if d.get("meta"):
        blocks += [eyebrow("Run details"), *_kv_rows(d["meta"])]
    blocks.append(eyebrow("Datasets"))
    blocks.append(
        Div(*_kv_rows((ds["path"], ds["shape"]) for ds in d["datasets"][:40]))
        if d["datasets"] else P("—", cls="wb-empty")
    )
    return Div(Div(*blocks, cls="wb-panel"), _geometry_panel(name, url=None))


def _geometry_panel(name: str, url: str | None):
    action = btn(
        "Open figure", kind="ghost",
        hx_get=f"/files/geometry/{name}",
        hx_target="closest .wb-panel",
        hx_swap="outerHTML",
    )
    return figure_panel(
        "ig.plot_geometry()",
        url,
        "Survey geometry — UTMX / UTMY by LINE" if url is None else "",
        action=action,
    )


# --------------------------------------------------------------------------- #
def register(rt) -> None:
    @rt("/", name="page_files")
    def page():
        ffilter = "all"
        return shell(
            "files",
            P("HDF5 files in the working folder, classified as DATA, PRIOR, "
              "FORWARD or POSTERIOR from their dataset structure.", cls="text-muted"),
            Div(cls="hr"),
            _toolbar(ffilter),
            Div(
                _file_table(ffilter),
                Div(_detail_placeholder(), id="wb-detail"),
                cls="wb-cols",
            ),
        )

    @rt("/files/list")
    def file_list(ffilter: str = "all"):
        # swap the table (outerHTML); refresh the count via an OOB span
        return _file_table(ffilter), _count(ffilter, oob=True)

    @rt("/files/inspect/{name}")
    def inspect(name: str):
        return _detail(name)

    @rt("/files/geometry/{name}")
    def geometry(name: str):
        url = api.geometry_figure(name)
        panel = _geometry_panel(name, url=url)
        if url is None:
            return Div(panel, error_box("No usable geometry in this file (needs /UTMX, /UTMY)."))
        return panel
