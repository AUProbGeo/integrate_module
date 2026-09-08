"""FastHTML application factory + ``integrate_workbench`` entry point."""

from __future__ import annotations

import secrets

from fasthtml.common import FastHTML, Link, Meta, Script
from starlette.staticfiles import StaticFiles

from frontend.config import (
    APP_TITLE, DEFAULT_HOST, DEFAULT_PORT, STATIC_DIR, get_workspace, set_workspace,
)
from frontend.pages import register_all

_HDRS = (
    Meta(charset="utf-8"),
    Meta(name="viewport", content="width=device-width, initial-scale=1"),
    Link(rel="stylesheet", href="/static/modernist.css"),
    Link(rel="stylesheet", href="/static/app.css"),
    Script(src="/static/htmx.min.js"),
    Script(src="/static/picker.js", defer=True),
)


def create_app() -> FastHTML:
    app = FastHTML(
        hdrs=_HDRS,
        default_hdrs=False,
        pico=False,
        surreal=False,
        htmx=False,
        title=APP_TITLE,
        secret_key=secrets.token_hex(16),
    )
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    register_all(app.route)
    return app


app = create_app()


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(prog="integrate_workbench", description=APP_TITLE)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--workspace", default=None, help="Folder with the project .h5 files (default: CWD).")
    args = p.parse_args(argv)

    set_workspace(args.workspace)
    print(f"{APP_TITLE}  —  workspace: {get_workspace()}")
    print(f"  http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
