"""INTEGRATE UI — FastAPI backend.

Serves the REST/WebSocket API and, when built, the React frontend from
``ui/frontend/dist``. Run from the directory that holds the project .h5
files (or set INTEGRATE_WORKSPACE):

    python -m ui.backend.main [--port 8000]
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from ui.backend.jobmanager import manager
from ui.backend.routers import files, jobs, query, results

DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"


class SPAStaticFiles(StaticFiles):
    """Static files with index.html fallback and no-cache on HTML entry points.

    The hashed asset files are immutable; index.html must always revalidate so
    users never run a stale frontend against a changed API.
    """

    async def get_response(self, path: str, scope):
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as e:
            if e.status_code == 404 and not path.startswith("api/"):
                response = await super().get_response("index.html", scope)
            else:
                raise
        if response.headers.get("content-type", "").startswith("text/html"):
            response.headers["Cache-Control"] = "no-cache"
        return response


def create_app() -> FastAPI:
    app = FastAPI(title="INTEGRATE UI", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(files.router)
    app.include_router(jobs.router)
    app.include_router(query.router)
    app.include_router(results.router)

    @app.on_event("startup")
    async def _attach_loop():
        manager.attach_loop(asyncio.get_running_loop())

    if DIST_DIR.is_dir():
        app.mount("/", SPAStaticFiles(directory=str(DIST_DIR), html=True), name="spa")

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="INTEGRATE UI backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "ui.backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
