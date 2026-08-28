"""Natural-language query endpoints (LLM translation + posterior evaluation).

Mirrors the streamlit ig_query pane: translate a plain-English query into a
query dict via ``ig.query_from_text()``, evaluate it over the posterior with
``ig.query()``, and render the resulting figures to base64 PNGs.

LLM credentials resolve with server-side environment variables taking
precedence (ANTHROPIC_API_KEY / OLLAMA_API_KEY); when none are set the client
must supply a provider + key/model per request. Keys are never returned to
or logged by the server.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import os
import threading
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ui.backend.workspace import get_workspace, safe_path

router = APIRouter(prefix="/api/query", tags=["query"])

# matplotlib is not thread-safe; serialize figure capture with other views.
_mpl_lock = threading.Lock()

CLAUDE_MODEL = "anthropic/claude-sonnet-4-6"
OLLAMA_DEFAULT_MODEL = "ollama_chat/qwen3:latest"


def _env_llm() -> dict | None:
    """LLM config from the environment, if any provider key is set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return {
            "provider": "claude",
            "model": os.environ.get("INTEGRATE_CLAUDE_MODEL", CLAUDE_MODEL),
            "api_key": os.environ["ANTHROPIC_API_KEY"],
        }
    if os.environ.get("OLLAMA_API_KEY"):
        return {
            "provider": "ollama",
            "model": os.environ.get("INTEGRATE_OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL),
            "api_key": os.environ["OLLAMA_API_KEY"],
        }
    return None


@router.get("/config")
def llm_config():
    """Report the server-side LLM configuration (never the key itself)."""
    cfg = _env_llm()
    if cfg is None:
        return {"configured": False, "provider": None, "model": None}
    return {"configured": True, "provider": cfg["provider"], "model": cfg["model"]}


def _resolve_post(name: str) -> Path:
    try:
        p = safe_path(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"No such file: {name}")
    return p


def _resolve_prior(f_prior: str) -> Path:
    """Resolve the prior file path stored in a posterior's f5_prior attribute.

    Relative paths resolve against the workspace; a bare-name fallback covers
    files moved between machines since the inversion ran.
    """
    if not f_prior:
        raise HTTPException(
            status_code=400,
            detail="Posterior file has no 'f5_prior' attribute — cannot list models or run a query",
        )
    p = Path(f_prior)
    candidates = [p if p.is_absolute() else get_workspace() / p]
    if Path(f_prior).is_absolute():
        candidates.append(get_workspace() / p.name)
    for cand in candidates:
        if cand.is_file():
            return cand
    raise HTTPException(status_code=404, detail=f"Prior file not found: {f_prior}")


def _posterior_prior_posterior(post: Path) -> tuple[Path, str]:
    import h5py

    with h5py.File(post, "r") as hf:
        if "i_use" not in hf:
            raise HTTPException(status_code=400, detail="Not a POSTERIOR file (no 'i_use' dataset)")
        f_prior = str(hf.attrs.get("f5_prior", "") or "")
    return _resolve_prior(f_prior), f_prior


@router.get("/ollama-models")
async def ollama_models():
    """List model names from the Ollama server, if one is reachable locally."""
    import httpx

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            r = await client.get(f"{host}/api/tags")
            r.raise_for_status()
        names = sorted(m.get("name", "") for m in r.json().get("models", []))
        return {"running": True, "models": [n for n in names if n]}
    except Exception:
        return {"running": False, "models": []}


@router.get("/models")
def prior_models(f: str):
    """Prior-model table for the prior file linked from posterior file *f*."""
    import h5py

    import integrate as ig

    prior, f_prior = _posterior_prior_posterior(_resolve_post(f))
    with h5py.File(prior, "r") as hf:
        model_keys = sorted(
            (k for k in hf.keys() if k.startswith("M") and k[1:].isdigit()),
            key=lambda k: int(k[1:]),
        )

    models = []
    for key in model_keys:
        im = int(key[1:])
        info = ig.get_prior_model_info(str(prior), im)
        z = info["z"]
        n_layers = max(0, len(z) - 1)
        is_scalar = (float(z[-1]) - float(z[0])) == 0 or n_layers == 0
        if is_scalar:
            kind = "SCALAR-DISCRETE" if info["is_discrete"] else "SCALAR"
        else:
            kind = "DISCRETE" if info["is_discrete"] else "CONTINUOUS"
        classes = None
        if info["is_discrete"] and info["class_id"] is not None and info["class_name"] is not None:
            classes = [
                {"id": int(cid), "name": str(cname)}
                for cid, cname in zip(info["class_id"].flatten(), info["class_name"].flatten())
            ]
        name = str(info["name"])
        models.append({
            "im": im,
            "name": name if name != key else key,
            "kind": kind,
            "depth_min": float(z[0]),
            "depth_max": float(z[-1]),
            "n_layers": n_layers,
            "classes": classes,
        })
    describe_buf = io.StringIO()
    with contextlib.redirect_stdout(describe_buf):
        ig.prior_describe(str(prior))
    return {"f_prior_h5": f_prior, "models": models, "describe": describe_buf.getvalue().strip()}


class QueryRunParams(BaseModel):
    f: str                                    # posterior file (workspace-relative)
    text: str                                 # natural-language query
    provider: Literal["claude", "ollama"] | None = None  # None → server config
    api_key: str | None = None
    model: str | None = None


def _resolve_llm(params: QueryRunParams) -> dict:
    """Decide provider/model/key: explicit request beats server env config."""
    env = _env_llm()
    if params.provider is None:
        if env is None:
            raise HTTPException(
                status_code=400,
                detail="No LLM configured. Set ANTHROPIC_API_KEY or OLLAMA_API_KEY "
                       "on the server, or enter an API key / choose Ollama in the UI.",
            )
        return env
    if params.provider == "claude":
        key = params.api_key or (env["api_key"] if env and env["provider"] == "claude" else None)
        if not key:
            raise HTTPException(status_code=400, detail="Enter an Anthropic API key to use Claude.")
        return {"provider": "claude", "model": params.model or CLAUDE_MODEL, "api_key": key}
    # ollama: keys optional (remote Ollama servers only)
    return {
        "provider": "ollama",
        "model": params.model or (env["model"] if env and env["provider"] == "ollama" else OLLAMA_DEFAULT_MODEL),
        "api_key": params.api_key or (env["api_key"] if env and env["provider"] == "ollama" else None),
    }


class _StageError(Exception):
    """Error tagged with the pipeline stage that produced it."""

    def __init__(self, stage: str, cause: BaseException):
        super().__init__(f"{stage} failed: {cause}")
        self.stage = stage
        self.cause = cause


def _execute_query(post: Path, prior: Path, text: str, model: str, api_key: str | None) -> dict:
    """Translate, evaluate, and render one query. Runs in a worker thread."""
    import integrate as ig

    cwd = os.getcwd()
    os.chdir(get_workspace())  # f5_prior / f5_data attrs resolve relative to cwd
    try:
        try:
            query_dict, interpretation, system_prompt = ig.query_from_text(
                text, str(prior), model=model, api_key=api_key
            )
        except Exception as e:
            raise _StageError("LLM query translation", e)

        kind = "percentile" if "metric" in query_dict else "probability"
        try:
            result, meta = ig.query(str(post), query_dict)
            if result is None:
                raise RuntimeError(f"ig.query returned no result for {post.name}")
        except Exception as e:
            raise _StageError("Query evaluation", e)

        figures = []
        with _mpl_lock:
            before = set(plt.get_fignums())
            try:
                if kind == "percentile":
                    ig.query_percentile_plot(result, meta, query_text=text,
                                             interpretation=interpretation, text_panel=True)
                else:
                    ig.query_plot(result, meta, query_text=text,
                                  interpretation=interpretation, text_panel=True)
            except Exception as e:
                plt.close("all")
                raise _StageError("Plotting", e)
            for num in sorted(set(plt.get_fignums()) - before):
                buf = io.BytesIO()
                plt.figure(num).savefig(buf, format="png", dpi=110, bbox_inches="tight")
                figures.append(base64.b64encode(buf.getvalue()).decode("ascii"))
                plt.close(num)
    finally:
        os.chdir(cwd)

    import numpy as np

    out = {
        "kind": kind,
        "interpretation": interpretation,
        "query_dict": query_dict,
        "system_prompt": system_prompt,
        "n_locations": int(meta.get("N_data", 0)),
        "figures": figures,
    }
    if kind == "probability":
        out["mean_probability"] = float(np.mean(result))
    else:
        out["percentiles"] = [int(p) for p in meta.get("percentiles", [])]
    return out


@router.post("/run")
async def run_query(params: QueryRunParams):
    """Translate *text* with the LLM and evaluate it over the posterior file."""
    if not params.text.strip():
        raise HTTPException(status_code=400, detail="Please enter a query.")
    post = _resolve_post(params.f)
    prior, _ = _posterior_prior_posterior(post)
    llm = _resolve_llm(params)
    try:
        return await asyncio.to_thread(
            _execute_query, post, prior, params.text, llm["model"], llm["api_key"]
        )
    except _StageError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
