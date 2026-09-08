"""Module 06 — Plotting.

Pick a source class (DATA / PRIOR / POSTERIOR), a file, and one of a curated
set of ``integrate.integrate_plot`` functions; render it to a figure.
"""

from __future__ import annotations

from dataclasses import dataclass

from fasthtml.common import Div, Form, P, Span

from frontend.components import btn, error_box, field, figure_panel, seg, select, shell, text_input
from frontend.services import integrate_api as api


@dataclass(frozen=True)
class Arg:
    name: str
    label: str
    default: str = ""


@dataclass(frozen=True)
class Plot:
    fn: str
    note: str
    args: tuple[Arg, ...] = ()


PLOTS: dict[str, list[Plot]] = {
    "DATA": [
        Plot("plot_geometry", "survey layout coloured by LINE / elevation / id",
             (Arg("pl", "pl (LINE|ELEVATION|id|NDATA)", "LINE"),)),
        Plot("plot_data_xy", "one data channel across the survey",
             (Arg("Dkey", "Dkey", "D1"), Arg("data_channel", "data_channel", "0"))),
        Plot("plot_data", "data as imshow / time-series", (Arg("id", "id", "1"),)),
    ],
    "PRIOR": [
        Plot("plot_prior_stats", "parameter histograms + sample realizations",
             (Arg("im", "im (blank = all)", ""),)),
    ],
    "POSTERIOR": [
        Plot("plot_T_EV", "T / EV / #data across the survey", ()),
        Plot("plot_profile", "posterior profile vs depth (auto discrete/continuous)",
             (Arg("im", "im", "1"), Arg("i1", "i1", "1"), Arg("i2", "i2 (0 = all)", "0"))),
        Plot("plot_post_stats", "posterior distributions at one sounding",
             (Arg("i_plot", "i_plot (sounding)", "0"), Arg("im", "im (blank = all)", ""))),
    ],
}
_CLASS_OF = {"DATA": "DATA", "PRIOR": "PRIOR", "POSTERIOR": "POSTERIOR"}


def _plot(source: str, fn: str) -> Plot | None:
    return next((p for p in PLOTS.get(source, []) if p.fn == fn), None)


def _files(source: str) -> list[str]:
    return api.list_h5_by_class(_CLASS_OF.get(source, source))


def _form(source: str, fn: str | None = None):
    plots = PLOTS[source]
    fn = fn or plots[0].fn
    plot = _plot(source, fn)
    files = _files(source)
    file_ctl = (
        select("file", files, value=files[0]) if files
        else Span("(no files of this class)", cls="wb-empty")
    )
    arg_cells = [field(a.label, text_input(a.name, a.default)) for a in (plot.args if plot else ())]
    return Div(
        Form(
            Div(
                seg("source", [(s, s) for s in PLOTS], source,
                    hx_get="/plotting/form", hx_target="#plot-form", hx_swap="outerHTML",
                    hx_include="[name='source']", hx_trigger="change"),
                style="margin-bottom:16px;",
            ),
            Div(field("File", file_ctl), style="max-width:520px;"),
            Div(
                field("Plot function", select(
                    "fn", [(p.fn, f"{p.fn} — {p.note}") for p in plots], value=fn,
                    hx_get="/plotting/form", hx_target="#plot-form", hx_swap="outerHTML",
                    hx_include="[name='source'],[name='fn']", hx_trigger="change",
                )),
                style="max-width:640px;margin-top:12px;",
            ),
            Div(*arg_cells, cls="wb-fields", style="margin-top:12px;") if arg_cells else Span(),
            btn("Render figure", kind="primary",
                hx_get="/plotting/figure", hx_target="#plot-fig", hx_swap="innerHTML",
                hx_include="closest form", style="margin-top:16px;"),
            id="plot-form",
        ),
        Div(figure_panel("integrate_plot", None, "pick a file + function and render"),
            id="plot-fig", style="margin-top:16px;"),
    )


def register(rt) -> None:
    @rt("/plotting", name="page_plotting")
    def page():
        return shell(
            "plotting",
            P("Every figure in integrate_plot, driven from one place — pick a "
              "source file, a function, and its arguments.", cls="text-muted"),
            Div(cls="hr"),
            _form("POSTERIOR"),
        )

    @rt("/plotting/form")
    def form(source: str = "POSTERIOR", fn: str = ""):
        if source not in PLOTS:
            source = "POSTERIOR"
        valid = {p.fn for p in PLOTS[source]}
        return _form(source, fn if fn in valid else None).children[0]

    @rt("/plotting/figure")
    async def figure(request):
        q = dict(request.query_params)
        source = q.pop("source", "POSTERIOR")
        fn = q.pop("fn", "")
        file = q.pop("file", "")
        if not file or file.startswith("("):
            return error_box("No file of this class in the working folder.")
        plot = _plot(source, fn)
        if plot is None:
            return error_box(f"Unknown plot: {fn}")

        def _coerce(v: str):
            try:
                return int(v)
            except ValueError:
                try:
                    return float(v)
                except ValueError:
                    return v

        params = {a.name: _coerce(q.get(a.name, a.default)) for a in plot.args}
        params = {k: v for k, v in params.items() if v not in ("", 0, "0")}
        url = api.render_plot(fn, file, params)
        return figure_panel(f"ig.{fn}()", url, "no figure produced (check arguments)")
