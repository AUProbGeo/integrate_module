"""Working-folder selector (the strip at the top of every page).

Process-wide: changing it affects the whole app (single-user desktop tool).
Not persisted across restarts — pass ``--workspace`` / ``WB_WORKSPACE`` for
a fixed default.
"""

from __future__ import annotations

from pathlib import Path

from starlette.responses import RedirectResponse, Response

from frontend.components import workspace_bar
from frontend.config import set_workspace


def register(rt) -> None:
    @rt("/workspace/edit")
    def ws_edit():
        return workspace_bar(editing=True)

    @rt("/workspace/cancel")
    def ws_cancel():
        return workspace_bar(editing=False)

    @rt("/workspace", methods=["POST"])
    def ws_change(path: str = "", path_text: str = ""):
        raw = (path or path_text or "").strip()
        target = Path(raw).expanduser() if raw else None
        if target is None or not target.is_dir():
            return workspace_bar(editing=True, error=f"Not a folder: {raw or '(empty)'}")
        set_workspace(str(target.resolve()))
        # htmx submit -> tell the client to reload the whole page; plain submit -> 303.
        return Response(status_code=204, headers={"HX-Redirect": "/"})
