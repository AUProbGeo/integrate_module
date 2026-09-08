"""Tiny declarative parameter-form helper shared by the job pages.

``F(name, label, kind, default)`` where ``kind`` is one of:
  "int" | "float" | "str" | "bool" | ("select", [options]) | ("h5", [classes]) | "gex"
``field_grid`` renders a ``.wb-fields`` grid; ``cast`` coerces one submitted value.
"""

from __future__ import annotations

from dataclasses import dataclass

from fasthtml.common import Div, Input, Label, Span

from frontend.components import field, select, text_input
from frontend.services import integrate_api as api


@dataclass(frozen=True)
class F:
    name: str
    label: str
    kind: object
    default: str = ""


def _control(f: F, value: str):
    k = f.kind
    if isinstance(k, tuple) and k[0] == "select":
        return select(f.name, list(k[1]), value=value)
    if isinstance(k, tuple) and k[0] == "h5":
        opts = api.list_h5_by_class(*k[1]) or ["(none)"]
        return select(f.name, opts, value=value or opts[0])
    if k in ("gex", "system", "json", "priordata"):
        opts = {
            "gex": api.list_gex,
            "system": api.list_system_files,
            "json": api.list_json,
            "priordata": api.list_prior_with_data,
        }[k]() or ["(none)"]
        return select(f.name, opts, value=value or opts[0])
    if k == "bool":
        checked = str(value).lower() in ("1", "true", "on", "yes")
        return Label(
            Input(type="checkbox", name=f.name, value="1", checked=checked),
            Span(cls="dot"),
            " " + f.label, cls="radio",
        )
    return text_input(f.name, value)


def field_grid(specs: list[F], values: dict | None = None):
    values = values or {}
    cells = []
    for f in specs:
        v = values.get(f.name, f.default)
        if f.kind == "bool":
            cells.append(Div(_control(f, v), cls="wb-boolfield"))
        else:
            cells.append(field(f.label, _control(f, v)))
    return Div(*cells, cls="wb-fields")


def cast(f: F, raw) -> object:
    s = str(raw).strip()
    try:
        if f.kind == "int":
            return int(float(s or f.default or "0"))
        if f.kind == "float":
            return float(s or f.default or "0")
    except ValueError:
        return f.default
    return s
