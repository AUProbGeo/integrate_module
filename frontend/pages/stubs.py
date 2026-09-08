"""Placeholder pages for modules not yet built. Replace one at a time."""

from __future__ import annotations

from fasthtml.common import Div, P

from frontend.components import leaf_slugs, shell

_TODO = {
    "prior": "Generate prior ensembles — prior_model_layered / workbench / workbench_direct.",
    "forward": "Compute prior data with GA-AEM (prior_data_gaaem).",
    "inversion": "Localized rejection sampling (integrate_rejection).",
    "results": "Posterior statistics and profile plots.",
    "plotting": "Every figure in integrate_plot, driven from one place.",
    "query": "Natural-language queries over posterior realizations (deferred).",
}


def _stub(slug: str):
    def handler():
        return shell(
            slug,
            Div(
                P(_TODO.get(slug, "Not implemented yet."), cls="text-muted"),
                P("This module is planned in FRONTEND.md and not built yet.", cls="wb-empty"),
                cls="wb-panel",
            ),
        )

    return handler


def register(rt, skip: set[str] | None = None) -> None:
    skip = skip or {"files"}
    for slug in leaf_slugs():
        if slug in skip:
            continue
        rt(f"/{slug}", name=f"page_{slug}")(_stub(slug))
