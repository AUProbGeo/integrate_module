"""Natural-language query endpoints (LLM translation + posterior evaluation).

Mirrors the streamlit ig_query pane: translate a plain-English query into a
query dict via ``ig.query_from_text()``, evaluate it over the posterior with
``ig.query()``, and render the resulting figures to base64 PNGs.

LLM credentials resolve with server-side environment variables taking
precedence (``INTEGRATE_LLM_MODEL`` plus the provider's own key var, or the
back-compat ``ANTHROPIC_API_KEY`` / ``OLLAMA_API_KEY``); when none are set the
client supplies a LiteLLM model string + key per request. Any LiteLLM-supported
provider works. Keys are never returned to or logged by the server.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import os
import threading
from pathlib import Path

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
    """LLM config from the environment, if the server operator set one up.

    ``INTEGRATE_LLM_MODEL`` is the general knob: a full LiteLLM model string
    (e.g. ``openai/gpt-4o``, ``openrouter/anthropic/claude-3.5-sonnet``); LiteLLM
    then reads the provider's own key var. ``ANTHROPIC_API_KEY`` / ``OLLAMA_API_KEY``
    stay supported for back-compat.
    """
    model = os.environ.get("INTEGRATE_LLM_MODEL")
    if model:
        return {"provider": model.split("/", 1)[0], "model": model, "api_key": None}
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


# provider id -> env var(s) that may hold its API key (first hit wins)
_PROVIDER_ENV_KEYS = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
}

# OpenAI-compatible /models base URLs LiteLLM doesn't supply (or supplies wrong)
_OPENAI_COMPAT_BASE = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",   # get_llm_provider gives .../beta
}


async def _ollama_tags() -> dict:
    """Models on the reachable Ollama server (``OLLAMA_API_BASE``/``OLLAMA_HOST``)."""
    import httpx

    host = (
        os.environ.get("OLLAMA_API_BASE")
        or os.environ.get("OLLAMA_HOST")
        or "http://localhost:11434"
    ).rstrip("/")
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


def _provider_key(provider: str, explicit: str | None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    for var in _PROVIDER_ENV_KEYS.get(provider, []):
        if os.environ.get(var):
            return os.environ[var]
    return None


def _openai_compat_base(provider: str) -> str | None:
    if provider in _OPENAI_COMPAT_BASE:
        return _OPENAI_COMPAT_BASE[provider]
    try:
        import litellm

        litellm.suppress_debug_info = True  # silence the "Provider List" banner
        _, _, _, base = litellm.get_llm_provider(f"{provider}/_")
        if base:
            return base.rstrip("/")
    except Exception:
        pass
    return None


def _short_err(e: Exception) -> str:
    msg = str(e).strip() or e.__class__.__name__
    return msg[:200]


async def _list_provider_models(provider: str, api_key: str | None) -> dict:
    """Best-effort live model list for *provider*. Never raises; degrades to
    ``{models: [], live: False, error: ...}`` so the UI can fall back to a
    free-text model id."""
    provider = provider.strip().lower()
    if provider in ("ollama", "ollama_chat"):
        tags = await _ollama_tags()
        return {
            "models": tags["models"],
            "live": tags["running"],
            "error": None if tags["running"] else "No reachable Ollama server.",
        }

    key = _provider_key(provider, api_key)
    if not key:
        return {"models": [], "live": False, "error": "Enter an API key to load models."}

    import httpx

    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            if provider == "anthropic":
                r = await client.get(
                    "https://api.anthropic.com/v1/models",
                    headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                )
                r.raise_for_status()
                ids = [m["id"] for m in r.json().get("data", []) if m.get("id")]
            elif provider == "gemini":
                r = await client.get(
                    "https://generativelanguage.googleapis.com/v1beta/models",
                    params={"key": key},
                )
                r.raise_for_status()
                ids = [
                    m["name"].split("/", 1)[-1]
                    for m in r.json().get("models", [])
                    if m.get("name")
                    and "generateContent" in m.get("supportedGenerationMethods", [])
                ]
            else:
                base = _openai_compat_base(provider)
                if not base:
                    return {
                        "models": [],
                        "live": False,
                        "error": f"Live model listing isn't supported for '{provider}'. "
                                 "Type the model id.",
                    }
                r = await client.get(
                    f"{base}/models", headers={"Authorization": f"Bearer {key}"}
                )
                r.raise_for_status()
                ids = [m["id"] for m in r.json().get("data", []) if m.get("id")]
    except Exception as e:
        return {"models": [], "live": False, "error": _short_err(e)}

    return {"models": sorted(dict.fromkeys(ids)), "live": True, "error": None}


class ProviderModelsParams(BaseModel):
    provider: str
    api_key: str | None = None


@router.post("/provider-models")
async def provider_models(params: ProviderModelsParams):
    """List models for a provider via a live call to its API (best-effort).

    The key comes from the request body or, failing that, the provider's env
    var. It is used only for the upstream call and never stored or echoed.
    """
    return await _list_provider_models(params.provider, params.api_key)


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
    provider: str | None = None       # advisory label; backend keys off `model`
    api_key: str | None = None
    model: str | None = None          # full LiteLLM model string
    system_prompt: str | None = None   # custom system prompt (hand-edited in UI)


def _resolve_llm(params: QueryRunParams) -> dict:
    """Decide model + key. Server env config, when present, wins outright — the
    UI hides its LLM picker in that case, so client LLM fields are ignored."""
    env = _env_llm()
    if env is not None:
        return env

    model = (params.model or "").strip()
    if not model:
        raise HTTPException(
            status_code=400,
            detail="No LLM configured. Choose a provider and model in the UI, or set "
                   "INTEGRATE_LLM_MODEL / ANTHROPIC_API_KEY / OLLAMA_API_KEY on the server.",
        )
    api_key = (params.api_key or "").strip() or None
    provider = model.split("/", 1)[0]
    # `validate_environment` falsely flags ollama as needing OLLAMA_API_BASE — skip it there.
    if api_key is None and provider not in ("ollama", "ollama_chat"):
        try:
            import litellm

            info = litellm.validate_environment(model)
        except Exception:
            info = {"keys_in_environment": True, "missing_keys": []}
        if not info.get("keys_in_environment") and info.get("missing_keys"):
            raise HTTPException(
                status_code=400,
                detail=f"Enter an API key for '{provider}' "
                       f"(needs {', '.join(info['missing_keys'])}).",
            )
    return {"provider": provider, "model": model, "api_key": api_key}


class _StageError(Exception):
    """Error tagged with the pipeline stage that produced it."""

    def __init__(self, stage: str, cause: BaseException):
        super().__init__(f"{stage} failed: {cause}")
        self.stage = stage
        self.cause = cause


def _translate_query(prior: Path, text: str, model: str, api_key: str | None,
                     system_prompt: str | None = None) -> dict:
    """Stage 1: translate *text* into a query dict via the LLM. Runs in a worker thread."""
    import integrate as ig

    cwd = os.getcwd()
    os.chdir(get_workspace())  # f5_prior attrs resolve relative to cwd
    try:
        try:
            query_dict, interpretation, system_prompt = ig.query_from_text(
                text, str(prior), model=model, api_key=api_key, system_prompt=system_prompt
            )
        except Exception as e:
            raise _StageError("LLM query translation", e)
    finally:
        os.chdir(cwd)
    return {"query_dict": query_dict, "interpretation": interpretation, "system_prompt": system_prompt}


def _evaluate_query(post: Path, text: str, query_dict: dict, interpretation: str | None) -> dict:
    """Stage 2: evaluate *query_dict* over the posterior file and render. Worker thread."""
    import integrate as ig

    cwd = os.getcwd()
    os.chdir(get_workspace())  # f5_prior / f5_data attrs resolve relative to cwd
    try:
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
        "n_locations": int(meta.get("N_data", 0)),
        "figures": figures,
    }
    if kind == "probability":
        out["mean_probability"] = float(np.mean(result))
    else:
        out["percentiles"] = [int(p) for p in meta.get("percentiles", [])]
    return out


class QueryEvaluateParams(BaseModel):
    f: str                                    # posterior file (workspace-relative)
    text: str                                 # natural-language query (for the plot caption)
    query_dict: dict                          # LLM-produced / hand-edited query spec
    interpretation: str | None = None         # LLM interpretation (optional caption)


@router.get("/system-prompt")
def get_system_prompt(f: str):
    """Default LLM system prompt for the prior file linked from posterior *f*."""
    from integrate.integrate_query import _build_llm_system_prompt

    prior, _ = _posterior_prior_posterior(_resolve_post(f))
    try:
        prompt = _build_llm_system_prompt(str(prior))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build system prompt: {e}")
    return {"system_prompt": prompt}


def _friendly_llm_error(msg: str) -> str:
    """Append actionable hints to common LLM provider errors."""
    lowered = msg.lower()
    if any(s in lowered for s in ("authenticat", "invalid x-api-key", "unauthorized", "401",
                                  "api key", "api_key")):
        return (
            msg
            + "\n\nThe API key was rejected. Check the key entered in the UI, or the provider's "
            "key set on the server (restart the backend after changing an environment variable)."
        )
    if "rate limit" in lowered or "429" in lowered:
        return msg + "\n\nRate limited by the provider — wait a moment and run the query again."
    return msg



@router.post("/translate")
async def translate_query(params: QueryRunParams):
    """Stage 1: translate *text* into a query dict with the LLM. JSON is returned
    unexecuted so the UI can show and optionally hand-edit it before evaluation."""
    if not params.text.strip():
        raise HTTPException(status_code=400, detail="Please enter a query.")
    _resolve_post(params.f)
    prior, _ = _posterior_prior_posterior(_resolve_post(params.f))
    llm = _resolve_llm(params)
    try:
        return await asyncio.to_thread(
            _translate_query, prior, params.text, llm["model"], llm["api_key"],
            params.system_prompt,
        )
    except _StageError as e:
        raise HTTPException(status_code=502, detail=_friendly_llm_error(str(e)))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def evaluate_query(params: QueryEvaluateParams):
    """Stage 2: evaluate a (possibly hand-edited) query dict over the posterior file."""
    if not params.query_dict:
        raise HTTPException(status_code=400, detail="Query JSON is empty.")
    post = _resolve_post(params.f)
    prior, _ = _posterior_prior_posterior(post)  # validates posterior + prior link
    try:
        return await asyncio.to_thread(
            _evaluate_query, post, params.text, params.query_dict, params.interpretation
        )
    except _StageError as e:
        raise HTTPException(status_code=502, detail=_friendly_llm_error(str(e)))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))
