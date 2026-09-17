"""Module 06 (Profile line) — interactive profile picking.

Show a scatter map of a posterior file's soundings, click to lay down a
polyline, and plot ``ig.plot_profile`` along it — the
``ig.find_points_along_line_segments`` → ``plot_profile(..., ii=…, xaxis='x')``
pattern from ``examples/integrate_rawmaterial_daugaard.py``.

The click capture + coordinate math is the one bit of client JS the frontend
owns (``static/picker.js``); everything numeric is delegated to ``integrate``.
"""

from __future__ import annotations

import json

from fasthtml.common import Div, Form, Input, NotStr, P, Span

from frontend.components import btn, error_box, field, figure_panel, select, shell, text_input
from frontend.services import integrate_api as api

_SVG = NotStr('<svg viewBox="0 0 1000 1000" preserveAspectRatio="none"></svg>')


def _picker(file: str, color: str):
    m = api.geometry_map(file, color)
    if m is None:
        return Div(error_box("No survey geometry (need /UTMX, /UTMY)."), id="profile-map")
    from fasthtml.common import Img

    return Div(
        Div(
            btn("Plot profile along line", kind="primary",
                hx_post="/profile/plot", hx_target="#profile-fig", hx_swap="innerHTML",
                hx_include="closest form"),
            btn("Undo", kind="secondary", **{"data-picker-undo": "1"}),
            btn("Clear", kind="secondary", **{"data-picker-clear": "1"}),
            Span("0 points", cls="wb-picker-count"),
            Span(f"{m['n']} soundings", cls="text-muted",
                 style="margin-left:auto;font-size:12px;"),
            cls="wb-picker-bar",
        ),
        Div(Img(src=m["url"], alt="survey map"), _SVG, cls="wb-picker-plot"),
        cls="wb-picker", id="profile-map",
        **{
            "data-target": "profile-points",
            "data-ax-l": m["ax_l"], "data-ax-r": m["ax_r"],
            "data-ax-b": m["ax_b"], "data-ax-t": m["ax_t"],
            "data-x0": m["x0"], "data-x1": m["x1"],
            "data-y0": m["y0"], "data-y1": m["y1"],
        },
    )


def _form(file: str | None, color: str = "elevation"):
    posts = api.list_h5_by_class("POSTERIOR") or ["(no POSTERIOR files)"]
    file = file or posts[0]
    reload_hx = dict(hx_get="/profile/map", hx_target="#profile-map",
                     hx_swap="outerHTML", hx_include="[name='file'],[name='color']")
    return Form(
        Div(field("Posterior file", select("file", posts, value=file, **reload_hx)),
            cls="wb-row"),
        Div(
            field("Colour by", select("color", [("elevation", "elevation"), ("line", "LINE")],
                                      value=color, **reload_hx)),
            field("x-axis", select("xaxis", [
                ("auto", "auto (x / y by line)"),
                ("x", "x — easting"),
                ("y", "y — northing"),
                ("index", "index — along the line"),
                ("id", "id — data id"),
            ], value="auto")),
            field("im — model index", text_input("im", "1")),
            field("tolerance (m, band width)", text_input("tolerance", "10")),
            field("gap_threshold (m)", text_input("gap_threshold", "100")),
            cls="wb-fields",
        ),
        P("Click on the map to add points along the profile line, then plot.",
          cls="text-muted", style="font-size:12px;margin:12px 0 6px;"),
        _picker(file, color),
        Input(type="hidden", name="points", id="profile-points", value="[]"),
        id="profile-form",
    )


def register(rt) -> None:
    @rt("/profile", name="page_profile")
    def page():
        return shell(
            "profile",
            P("Pick a line on the survey map and plot ig.plot_profile along it.",
              cls="text-muted"),
            Div(cls="hr"),
            _form(None),
            Div(figure_panel("ig.plot_profile()", None, "pick a line and plot", tall=True),
                id="profile-fig", style="margin-top:16px;"),
        )

    @rt("/profile/map")
    def profile_map(file: str = "", color: str = "elevation"):
        posts = api.list_h5_by_class("POSTERIOR")
        if file not in posts:
            file = posts[0] if posts else ""
        return _picker(file, color if color in ("elevation", "line") else "elevation")

    @rt("/profile/plot", methods=["POST"])
    def profile_plot(file: str = "", points: str = "", im: str = "1",
                     gap_threshold: str = "100", xaxis: str = "auto", tolerance: str = "10"):
        try:
            pts = json.loads(points or "[]")
        except ValueError:
            pts = []
        if not isinstance(pts, list) or len(pts) < 2:
            return error_box("Click at least two points on the map first.")
        try:
            imv = int(float(im))
        except ValueError:
            imv = 1
        try:
            gap = float(gap_threshold)
        except ValueError:
            gap = 100.0
        try:
            tol = float(tolerance)
        except ValueError:
            tol = 10.0
        if xaxis == "auto":
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            xaxis = "x" if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else "y"
        res = api.profile_along_line(file, pts, im=imv, gap_threshold=gap, xaxis=xaxis, tolerance=tol)
        if res is None:
            return figure_panel(
                "ig.plot_profile()", None,
                f"no soundings within {tol:g} m of that line — widen tolerance or move the points",
                tall=True,
            )
        return figure_panel(
            f"ig.plot_profile(xaxis='{res['xaxis']}')  ·  {res['n']} soundings "
            f"within {res['tolerance']:g} m of the line",
            res["url"], "", tall=True,
        )
