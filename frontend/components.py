"""Design-system component helpers.

Thin wrappers that emit the classes defined in ``static/modernist.css``
(and a little chrome from ``static/app.css``). Pages compose these; they do
not hand-write class strings.
"""

from __future__ import annotations

from pathlib import Path

from fasthtml.common import (
    A, B, Button, Div, Form, H2, H4, Img, Input, Label, Main, Nav, Option, P,
    Script, Select, Span, Table, Tbody, Td, Th, Thead, Title, Tr,
)

from frontend.config import APP_TITLE, get_workspace

# (slug, number, label, children) — order defines the nav and the URL map.
# `children` = ((slug, label), …); a parent with children is a section label,
# not itself a link.
PAGES: list[tuple[str, str, str, tuple]] = [
    ("files", "01", "Data files", ()),
    ("prior", "02", "Prior model",
     (("prior", "Generic"), ("prior-wb", "WB"), ("geoprior", "geoprior1d"))),
    ("forward", "03", "Forward", (("forward", "GA-AEM"), ("forward-bh", "Boreholes"))),
    ("inversion", "04", "Inversion", ()),
    ("results", "05", "Results", ()),
    ("plotting", "06", "Plotting", (("plotting", "Library"), ("profile", "Profile line"))),
    ("query", "—", "Query", ()),
]

# every reachable leaf slug -> (number, header label)
_SLUG_META: dict[str, tuple[str, str]] = {}
for _slug, _num, _label, _children in PAGES:
    if _children:
        for _cslug, _clabel in _children:
            _SLUG_META[_cslug] = (_num, f"{_label} — {_clabel}")
    else:
        _SLUG_META[_slug] = (_num, _label)


def leaf_slugs() -> list[str]:
    out: list[str] = []
    for slug, _num, _label, children in PAGES:
        out.extend(c[0] for c in children) if children else out.append(slug)
    return out


def _href(slug: str) -> str:
    return "/" if slug == "files" else f"/{slug}"


def nav(active: str):
    items = []
    for slug, num, label, children in PAGES:
        if not children:
            items.append(A(Span(num, cls="num"), Span(label), href=_href(slug),
                           cls="active" if slug == active else None))
            continue
        active_here = any(active == c[0] for c in children)
        items.append(Div(
            Div(Span(num, cls="num"), Span(label),
                cls="wb-navgroup" + (" open" if active_here else "")),
            *[A(Span(clabel), href=_href(cslug),
                cls="wb-navchild" + (" active" if active == cslug else ""))
              for cslug, clabel in children],
        ))
    return Nav(
        Span("INTEGRATE", cls="wordmark"),
        *items,
        Div(cls="spacer"),
        Span(APP_TITLE, cls="ws"),
        cls="wb-nav",
    )


def _subdirs(path: Path) -> list[Path]:
    try:
        return sorted(
            (d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")),
            key=lambda d: d.name.lower(),
        )[:16]
    except OSError:
        return []


def _has_h5(path: Path) -> bool:
    try:
        return any(p.suffix.lower() == ".h5" for p in path.iterdir())
    except OSError:
        return False


def workspace_bar(*, editing: bool = False, error: str | None = None):
    """Top strip: current working folder + a Change control.

    Not editing -> path + "Change…". Editing -> a plain POST form to
    ``/workspace`` (full reload on success) with a text field and quick-pick
    buttons for the parent and sub-folders.
    """
    cur = get_workspace()

    if not editing:
        return Div(
            Span("WORKING FOLDER", cls="wb-eyebrow"),
            Span(str(cur), cls="path"),
            btn("Change…", kind="ghost",
                hx_get="/workspace/edit", hx_target="#wb-ws-bar", hx_swap="outerHTML"),
            id="wb-ws-bar", cls="wb-wsbar",
        )

    picks = []
    parent = cur.parent
    if parent != cur:
        picks.append(Button("../  " + parent.name, name="path", value=str(parent),
                            type="submit", cls="pick", title=str(parent)))
    for d in _subdirs(cur):
        mark = " ●" if _has_h5(d) else ""
        picks.append(Button(d.name + mark, name="path", value=str(d),
                            type="submit", cls="pick", title=str(d)))

    return Div(
        Form(
            Span("WORKING FOLDER", cls="wb-eyebrow"),
            Input(name="path_text", value=str(cur), cls="input", type="text",
                  placeholder="/absolute/path/to/folder"),
            btn("Use folder", kind="primary", type="submit"),
            btn("Cancel", kind="secondary",
                hx_get="/workspace/cancel", hx_target="#wb-ws-bar", hx_swap="outerHTML"),
            (error_box(error) if error else Span()),
            Div(*picks, cls="picks") if picks else Span(),
            method="post", action="/workspace",
            hx_post="/workspace", hx_target="#wb-ws-bar", hx_swap="outerHTML",
        ),
        id="wb-ws-bar", cls="wb-wsbar editing",
    )


def page_header(slug: str):
    num, label = _SLUG_META.get(slug, ("—", slug.title()))
    counter = label if num == "—" else f"{num} — {label}"
    return Div(
        Span("INTEGRATE", cls="mark"),
        Span(counter, cls="counter"),
        cls="wb-head",
    )


def shell(active: str, *body):
    """Full page: (Title, shell div). Return this from a GET route."""
    _slug_num, label = _SLUG_META.get(active, ("—", active.title()))
    return (
        Title(f"{label} · {APP_TITLE}"),
        Div(
            nav(active),
            Main(workspace_bar(), page_header(active), *body, cls="wb-main"),
            cls="wb-shell",
        ),
    )


# --------------------------------------------------------------------------- #
# atoms
# --------------------------------------------------------------------------- #
def btn(label, *, kind: str = "secondary", block: bool = False, **kw):
    cls = f"btn btn-{kind}" + (" btn-block" if block else "")
    return Button(label, cls=cls, type=kw.pop("type", "button"), **kw)


def tag(text: str):
    kind = {
        "DATA": "tag-neutral",
        "PRIOR": "tag-accent-2",
        "POSTERIOR": "tag-accent",
        "UNKNOWN": "tag-outline",
        "UNREADABLE": "tag-outline",
    }.get(str(text).upper(), "tag-neutral")
    return Span(text, cls=f"tag {kind}")


def field(label: str, control):
    return Div(Label(label), control, cls="field")


def text_input(name: str, value: str = "", **kw):
    return Input(name=name, value=value, cls="input", type="text", **kw)


def select(name: str, options, value: str | None = None, **kw):
    opts = [
        Option(o, value=o, selected=(o == value)) if not isinstance(o, tuple)
        else Option(o[1], value=o[0], selected=(o[0] == value))
        for o in options
    ]
    return Select(*opts, name=name, cls="input", **kw)


def seg(name: str, options: list[tuple[str, str]], value: str, **kw):
    """Segmented control. ``options`` = [(value, label), ...]."""
    parts = []
    for val, lbl in options:
        parts.append(
            Label(
                Input(type="radio", name=name, value=val, checked=(val == value)),
                Span(lbl),
                cls="seg-opt",
            )
        )
    return Div(*parts, cls="seg", **kw)


def stat(label: str, value):
    return Div(Div(label, cls="l"), Div(str(value), cls="v"), cls="wb-stat")


def stat_grid(*stats, cols: int | None = None):
    n = cols or len(stats) or 1
    return Div(*stats, cls="wb-stats",
               style=f"grid-template-columns:repeat({n},minmax(0,1fr));")


def eyebrow(text: str):
    return Div(text, cls="wb-eyebrow")


def figure_panel(eyebrow_text: str, img_url: str | None, placeholder: str, action=None, *, tall: bool = False):
    head = Div(
        eyebrow(eyebrow_text),
        action if action is not None else Span(),
        style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:10px;",
    )
    from fasthtml.common import Img

    # `tall`: show the image at its natural size (scrolls) rather than fitting
    # a fixed 260px box — for multi-panel plots like plot_profile.
    body = Img(src=img_url, alt=eyebrow_text) if img_url else Span(placeholder, cls="wb-empty")
    return Div(head, Div(body, cls="wb-figure-tall" if tall else "wb-figure"), cls="wb-panel")


def error_box(message: str):
    return Div(message, cls="wb-err")


__all__ = [
    "PAGES", "shell", "page_header", "nav", "workspace_bar", "btn", "tag", "field",
    "text_input", "select", "seg", "stat", "stat_grid", "eyebrow", "figure_panel", "error_box",
    # re-export common FT for pages
    "Div", "Span", "P", "H2", "H4", "Form", "Table", "Thead", "Tbody", "Tr",
    "Th", "Td", "A", "Script",
]
