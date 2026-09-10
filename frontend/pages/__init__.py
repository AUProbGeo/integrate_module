"""Page modules. Each exposes ``register(rt)`` which adds its routes.

Order and slugs are defined by ``frontend.components.PAGES``.
"""

from __future__ import annotations

from frontend.pages import (
    borehole, files, forward, geoprior, inversion, plotting, prior, profile,
    query, results, stubs, workspace_ui,
)

# slugs with a real page module (stubs.py covers the rest)
IMPLEMENTED = {
    "files", "prior", "prior-wb", "geoprior", "forward", "forward-bh", "inversion",
    "results", "plotting", "profile", "query",
}


def register_all(rt) -> None:
    workspace_ui.register(rt)
    files.register(rt)
    prior.register(rt)
    geoprior.register(rt)
    forward.register(rt)
    borehole.register(rt)
    inversion.register(rt)
    results.register(rt)
    plotting.register(rt)
    profile.register(rt)
    query.register(rt)
    stubs.register(rt, skip=IMPLEMENTED)
