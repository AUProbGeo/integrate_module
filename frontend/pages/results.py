"""Module 05 — Results.

Posterior statistics for a POSTERIOR ``.h5``: a KPI row, a chosen
``integrate_plot`` figure, and a per-sounding table.
"""

from __future__ import annotations

from fasthtml.common import Div, Form, Input, P, Span, Table, Tbody, Td, Th, Thead, Tr

from frontend.components import (
    btn, error_box, eyebrow, field, figure_panel, select, shell, stat, stat_grid, text_input,
)
from frontend.services import integrate_api as api

_PLOTS = [
    ("plot_profile", "posterior profile vs depth (auto discrete/continuous)"),
    ("plot_profile_continuous", "continuous parameter, 4-panel"),
    ("plot_profile_discrete", "discrete parameter, 3-panel"),
    ("plot_T_EV", "T / EV / #data across the survey"),
    ("plot_post_stats", "posterior distributions at one sounding"),
]


def _picker(selected: str | None):
    files = api.list_h5_by_class("POSTERIOR")
    if not files:
        return P("No POSTERIOR .h5 files in the working folder.", cls="wb-empty")
    return Form(
        field("Posterior file", select(
            "file", files, value=selected or files[0],
            hx_get="/results/view", hx_target="#results", hx_swap="innerHTML",
            hx_trigger="change",
        )),
        style="max-width:520px;",
    )


def _kpis(k: dict):
    return stat_grid(
        stat("Soundings", k["soundings"]),
        stat("Realizations / sounding", k["nr"]),
        stat("Mean T", k["mean_t"]),
        stat("Mean EV", k["mean_ev"]),
        stat("Mean N_UNIQUE", k["n_unique"]),
        stat("Inversion time", k["inv_time"]),
        cols=6,
    )


def _fig_form(file: str):
    return Form(
        Input(type="hidden", name="file", value=file),
        Div(
            field("Plot function", select("fn", _PLOTS, value="plot_profile")),
            field("im", text_input("im", "1")),
            field("i1", text_input("i1", "1")),
            field("i2 (0 = all)", text_input("i2", "0")),
            cls="wb-fields",
        ),
        btn("Render figure", kind="primary",
            hx_get="/results/figure", hx_target="#results-fig", hx_swap="innerHTML",
            hx_include="closest form", style="margin-top:12px;"),
        id="results-figform",
    )


def _table(rows: list[dict]):
    if not rows:
        return P("No per-sounding T / EV / N_UNIQUE datasets in this file.", cls="wb-empty")
    head = Tr(Th("ip"), Th("LINE"), Th("T", style="text-align:right;"),
              Th("EV", style="text-align:right;"), Th("N_UNIQUE", style="text-align:right;"))
    body = [
        Tr(
            Td(str(r["ip"]), style="font-variant-numeric:tabular-nums;"),
            Td(r["line"], style="font-variant-numeric:tabular-nums;"),
            Td(r["t"], style="text-align:right;font-variant-numeric:tabular-nums;"),
            Td(r["ev"], style="text-align:right;font-variant-numeric:tabular-nums;"),
            Td(r["n_unique"], style="text-align:right;font-variant-numeric:tabular-nums;"),
        )
        for r in rows
    ]
    return Div(Table(Thead(head), Tbody(*body), cls="table"), cls="wb-tablewrap",
              style="max-height:420px;overflow:auto;")


def _view(file: str):
    try:
        k = api.posterior_kpis(file)
        rows = api.posterior_table(file)
    except Exception as e:  # noqa: BLE001
        return error_box(f"Could not read {file}: {e}")
    linked = []
    if k.get("f5_data"):
        linked.append(Span(f"data: {k['f5_data']}", cls="text-muted", style="font-size:12px;"))
    if k.get("f5_prior"):
        linked.append(Span(f"  ·  prior: {k['f5_prior']}", cls="text-muted", style="font-size:12px;"))
    return Div(
        P(*linked) if linked else Span(),
        _kpis(k),
        Div(cls="hr"),
        Div(
            Div(_fig_form(file),
                Div(figure_panel(f"integrate_plot on {file}", None, "pick a function and render"),
                    id="results-fig", style="margin-top:12px;")),
            Div(eyebrow("Per-sounding summary"), _table(rows)),
            cls="wb-cols", style="grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);",
        ),
    )


def register(rt) -> None:
    @rt("/results", name="page_results")
    def page():
        files = api.list_h5_by_class("POSTERIOR")
        first = files[0] if files else None
        return shell(
            "results",
            P("Posterior statistics for a POSTERIOR .h5 — KPIs, a chosen "
              "integrate_plot figure, and a per-sounding table.", cls="text-muted"),
            Div(cls="hr"),
            _picker(first),
            Div(_view(first) if first else Span(), id="results", style="margin-top:20px;"),
        )

    @rt("/results/view")
    def view(file: str):
        return _view(file)

    @rt("/results/figure")
    def figure(file: str, fn: str = "plot_profile", im: str = "", i1: str = "", i2: str = ""):
        def _int(s):
            try:
                return int(float(s))
            except (TypeError, ValueError):
                return None

        params = {"im": _int(im), "i1": _int(i1)}
        if _int(i2):
            params["i2"] = _int(i2)
        url = api.render_plot(fn, file, params)
        return figure_panel(f"ig.{fn}()", url, "no figure produced (check im / i1 / i2)")
