"""The single boundary to the ``integrate`` package.

Every function here is: coerce arguments -> call ``ig.*`` -> shape a plain
return. No numerics, HDF5 poking, statistics or plotting logic of its own —
that all lives in ``integrate``. ``import integrate`` happens lazily inside
each function so the rest of the app (and its tests) can load without the
heavy scientific stack.
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path

from frontend.services import figures, files
from frontend.services.workspace import cwd as _wcwd
from frontend.services.workspace import safe_path


# --------------------------------------------------------------------------- #
# argument parsing helpers
# --------------------------------------------------------------------------- #
def parse_int_list(text: str) -> list[int]:
    """``"1, 2 ,3"`` -> ``[1, 2, 3]``; empty -> ``[]``."""
    return [int(t) for t in str(text).replace(";", ",").split(",") if t.strip()]


def parse_range(text: str) -> list[int]:
    """``"0:500"`` -> ``[0, 500]``; ``"1,2,3"`` -> ``[1, 2, 3]``; empty -> ``[]``."""
    text = str(text).strip()
    if not text:
        return []
    if ":" in text:
        lo, _, hi = text.partition(":")
        return [int(lo), int(hi)]
    return parse_int_list(text)


# --------------------------------------------------------------------------- #
# file-list helpers (shared by modules 02–06)
# --------------------------------------------------------------------------- #
def list_xlsx() -> list[str]:
    """Workspace ``*.xlsx`` file names (geoprior1d input specs)."""
    from frontend.services.workspace import list_files

    return [p.name for p in list_files(".xlsx")]


def list_gex() -> list[str]:
    """Workspace ``*.gex`` file names (GA-AEM system files)."""
    from frontend.services.workspace import list_files

    return [p.name for p in list_files(".gex")]


def list_system_files() -> list[str]:
    """Workspace GA-AEM system files: ``*.gex`` and ``*.stm``."""
    from frontend.services.workspace import list_files

    return [p.name for p in list_files(".gex", ".stm")]


def list_json() -> list[str]:
    """Workspace ``*.json`` file names (borehole specs)."""
    from frontend.services.workspace import list_files

    return [p.name for p in list_files(".json")]


def list_h5_by_class(*classes: str) -> list[str]:
    """Workspace ``*.h5`` names whose classification is one of *classes*."""
    from frontend.services.workspace import list_files

    want = set(classes)
    return [p.name for p in list_files(".h5") if files.classify(p) in want]


def prior_n_realizations(name: str) -> int | None:
    """Number of realizations in a PRIOR ``.h5`` (``/M1.shape[0]``), or ``None``."""
    try:
        n = files.summary(safe_path(name)).get("n_realizations")
        return int(n) if n else None
    except Exception:
        return None


def list_prior_with_data() -> list[str]:
    """PRIOR ``.h5`` files that also carry prior forward data (``/D<n>``
    datasets). Only these can be inverted / used for borehole forward — a
    plain prior (models only) must be forward-modelled first.
    """
    from frontend.services.workspace import list_files

    out = []
    for p in list_files(".h5"):
        try:
            summ = files.summary(p)
            if summ.get("class") == "PRIOR" and summ.get("data"):
                out.append(p.name)
        except Exception:
            pass
    return out


def start_prior_job(model: str, kwargs: dict, out_name: str = "") -> str:
    """Kick off ``ig.prior_model_<model>`` in a child process. Returns job id.

    ``model`` ∈ {"layered", "workbench", "workbench_direct"}. ``kwargs`` are the
    already-coerced model parameters; ``out_name`` (optional) becomes
    ``f_prior_h5`` (blank ⇒ integrate auto-names it).
    """
    from frontend.services import jobs, worker

    params = {"model": model, **kwargs}
    if out_name and out_name.strip():
        params["f_prior_h5"] = out_name.strip()
    return jobs.start("prior", worker.run_prior_job, params).id


def start_geoprior_job(kwargs: dict) -> str:
    """Kick off ``geoprior1d(file_xlsx, ...)`` in a child process. Returns job id."""
    from frontend.services import jobs, worker

    return jobs.start("geoprior", worker.run_geoprior_job, kwargs).id


def geoprior_preview_run(xlsx_path: str, h5_path: str, *, Nreals: int,
                         dmax: float, dz: float) -> str:
    """Run ``geoprior1d`` synchronously into a scratch ``.h5`` for the preview.

    Fast enough (<1 s for a few hundred realizations with ``n_processes=1``)
    to run in a worker thread instead of a spawned process — the ~2 s the
    child-process path costs is almost all interpreter spawn + imports. The
    web process is ``MainProcess`` and ``n_processes=1`` means geoprior1d
    starts no pool of its own. Returns the abs path; raises on failure.
    """
    import os

    from geoprior1d import geoprior1d

    with contextlib.suppress(OSError):
        os.unlink(h5_path)
    name, _flags = geoprior1d(xlsx_path, Nreals=int(Nreals), dmax=float(dmax),
                              dz=float(dz), n_processes=1, output_file=h5_path)
    if not name or not os.path.exists(name):
        raise RuntimeError("geoprior1d produced no output file")
    return os.path.abspath(name)


def geoprior_preview_figure(h5_abspath: str, im: int, nr: int = 100) -> str | None:
    """The right-hand 'realizations' panel of ``ig.plot_prior_stats`` for /M<im>.

    ``h5_abspath`` is an absolute scratch prior file (not workspace-relative).
    Returns a cached PNG URL, or ``None`` if that /M<im> is absent / on error.
    """
    from pathlib import Path

    p = Path(h5_abspath)
    if not p.is_file():
        return None
    key = figures.figure_key("gp-preview", str(p), int(im), int(nr), salt=p)

    def _call():
        import integrate.integrate_plot as igp

        igp.plot_prior_stats(str(p), Mkey=f"M{int(im)}", nr=int(nr),
                             panels="reals", hardcopy=False, title="")

    try:
        return figures.render(_call, key=key)
    except Exception:
        return None


def cond_resistivity_figure(xlsx_path: str, h5_path: str | None = None,
                            overlay: bool = False) -> str | None:
    """Overlaid analytic ρ|lithology priors from a geoprior1d ``.xlsx`` spec.

    One log-normal PDF per lithology on a shared log-ρ axis, coloured by the
    class RGB from the spec (via ``geoprior1d.io.extract_prior_info``). The
    curve is analytic — median from the *Resistivity* sheet, σ (in log10
    space) = log10(uncertainty factor) / 3, matching what geoprior1d samples.

    ``overlay`` + a scratch ``h5_path`` (with ``/M1`` resistivity, ``/M2``
    class code) adds a step-histogram of the sampled values per class.
    Returns a cached PNG URL, or ``None`` on any parse / render failure.
    """
    import os

    import numpy as np

    try:
        from geoprior1d.io import extract_prior_info

        info, cmaps = extract_prior_info(xlsx_path)
        res = np.asarray(info["Resistivity"]["res"], dtype=float)
        sig = np.asarray(info["Resistivity"]["res_unc"], dtype=float)
        names = list(info["Classes"]["names"])
        codes = list(info["Classes"]["codes"])
        colors = np.asarray(cmaps["Classes"], dtype=float)
        if res.size == 0 or res.size != sig.size:
            return None
    except Exception:
        return None

    use_overlay = bool(overlay) and bool(h5_path) and os.path.exists(h5_path)

    def _mt(p: str | None) -> int:
        try:
            return os.stat(p).st_mtime_ns  # type: ignore[arg-type]
        except (OSError, TypeError):
            return 0

    key = figures.figure_key("gp-cond", str(xlsx_path), _mt(xlsx_path),
                             _mt(h5_path) if use_overlay else 0, use_overlay)

    lo = float(np.log10(res).min() - 4.0 * max(sig.max(), 1e-3))
    hi = float(np.log10(res).max() + 4.0 * max(sig.max(), 1e-3))
    x = np.linspace(lo, hi, 400)

    def _draw():
        import h5py
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        fig = plt.figure(figsize=(9.0, 4.2))
        ax = fig.add_subplot(111)
        for i, _code in enumerate(codes):
            col = colors[i] if i < len(colors) else (0.5, 0.5, 0.5)
            s = max(float(sig[i]), 1e-3)
            pdf = norm.pdf(x, float(np.log10(res[i])), s)
            ax.fill_between(10.0 ** x, pdf, color=col, alpha=0.12)
            ax.plot(10.0 ** x, pdf, color=col, lw=1.6, label=names[i])

        if use_overlay:
            with h5py.File(h5_path, "r") as f:
                m1 = np.asarray(f["M1"][:]).ravel()
                m2 = np.asarray(f["M2"][:]).ravel()
            # histogram log10(ρ) with linear bins over the SAME log10 range as
            # the analytic curves, density=True -> density per log10-unit, so
            # the step-hist and the norm.pdf share one amplitude scale.
            lbins = np.linspace(lo, hi, 60)
            for i, code in enumerate(codes):
                v = m1[(m2 == code) & np.isfinite(m1) & (m1 > 0)]
                if v.size:
                    col = colors[i] if i < len(colors) else (0.5, 0.5, 0.5)
                    h, e = np.histogram(np.log10(v), bins=lbins, density=True)
                    ax.step(10.0 ** e, np.append(h, h[-1]), where="post",
                            color=col, alpha=0.8, lw=1.3)

        ax.set_xscale("log")
        ax.set_xlabel("Resistivity [Ω·m]")
        ax.set_ylabel("density (over log₁₀ ρ)")
        ax.set_title("ρ | lithology — assumed prior"
                     + ("  ·  sampled overlaid" if use_overlay else ""))
        ax.legend(fontsize=8, ncol=2, loc="upper right")
        ax.margins(x=0)
        fig.tight_layout()

    try:
        return figures.render(_draw, key=key)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Module 03 — Forward / Module 04 — Inversion
# --------------------------------------------------------------------------- #
def start_forward_job(kwargs: dict) -> str:
    """Kick off ``ig.prior_data_gaaem(...)`` in a child process. Returns job id."""
    from frontend.services import jobs, worker

    kwargs = {k: v for k, v in kwargs.items() if v is not None and v != ""}
    return jobs.start("forward", worker.run_forward_job, kwargs).id


def start_inversion_job(kwargs: dict) -> str:
    """Kick off ``ig.integrate_rejection(...)`` in a child process. Returns job id."""
    from frontend.services import jobs, worker

    kwargs = {k: v for k, v in kwargs.items() if v is not None and v != ""}
    return jobs.start("rejection", worker.run_rejection_job, kwargs).id


def start_borehole_job(kwargs: dict) -> str:
    """Kick off borehole forward-data (``ig.save_borehole_data``) in a child process."""
    from frontend.services import jobs, worker

    kwargs = {k: v for k, v in kwargs.items() if v is not None and v != ""}
    return jobs.start("borehole", worker.run_borehole_job, kwargs).id


def prior_model_ims(name: str) -> list[tuple[int, str]]:
    """``[(im, "M<im> — <name>")]`` for the ``/M<n>`` datasets in a PRIOR file,
    for a model-parameter picker. Empty on any read failure."""
    try:
        models = files.summary(safe_path(name)).get("models", [])
    except Exception:
        return []
    out: list[tuple[int, str]] = []
    for m in models:
        mo = re.match(r"M(\d+)$", str(m.get("id", "")))
        if mo:
            im = int(mo.group(1))
            nm = str(m.get("name") or "").strip()
            out.append((im, f"M{im} — {nm}" if nm else f"M{im}"))
    return out


def prior_stats_figure(name: str, im: int | None = None) -> str | None:
    """Render ``ig.plot_prior_stats(<file>, im=<im>)`` -> PNG URL, or ``None``.

    ``im`` selects one ``/M<im>`` model parameter; ``None`` lets
    ``plot_prior_stats`` recurse over all of them (only the last is saved).
    """
    params = {} if im is None else {"im": int(im)}
    return render_plot("plot_prior_stats", name, params)


# --------------------------------------------------------------------------- #
# Modules 05 (Results) / 06 (Plotting) — figures + posterior tables
# --------------------------------------------------------------------------- #
def render_plot(fn: str, file: str, params: dict) -> str | None:
    """Render ``integrate.integrate_plot.<fn>(<file>, **params)`` -> PNG URL.

    ``params`` values that are ``""`` / ``None`` are dropped so the plot
    function's own defaults apply. Returns ``None`` on any failure.
    """
    path = safe_path(file)
    kw = {k: v for k, v in params.items() if v not in ("", None)}
    kw.setdefault("hardcopy", False)  # ask integrate_plot not to drop its own PNGs
    try:
        import integrate.integrate_plot as igp

        plot_fn = getattr(igp, fn)
        key = figures.figure_key(f"plot:{fn}", str(path), sorted(kw.items()), salt=path)

        def _call():
            plot_fn(str(path), **kw)

        return figures.render(_call, key=key)
    except Exception:
        return None


def geometry_map(file: str, color: str = "elevation") -> dict | None:
    """Scatter map of a file's survey (X, Y) for interactive line picking.

    Returns ``{url, ax_l, ax_r, ax_b, ax_t, x0, x1, y0, y1, n}`` (see
    ``figures.render_interactive``) or ``None`` if the file has no geometry.
    """
    path = safe_path(file)
    label = "LINE" if color == "line" else "Elevation (m)"
    key = figures.figure_key("geomap", str(path), color, salt=path)

    try:
        import integrate as ig

        with _wcwd():
            X, Y, line, elev = ig.get_geometry(str(path))
    except Exception:
        return None
    c = line if color == "line" else elev

    # Size the figure to the survey aspect so equal-aspect doesn't leave a
    # slab of whitespace. Axes box ≈ 68 % of width (ylabel + colorbar), 80 %
    # of height (xlabel).
    dx = float(X.max() - X.min()) or 1.0
    dy = float(Y.max() - Y.min()) or 1.0
    w = 8.0
    h = max(3.2, min(w * (0.68 / 0.80) * (dy / dx), 8.5))

    def draw(ax):
        sc = ax.scatter(X, Y, c=c, s=6, cmap="viridis")
        ax.set_xlabel("UTM X (m)")
        ax.set_ylabel("UTM Y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.figure.colorbar(sc, ax=ax, label=label, shrink=0.85)
        return {"n": int(len(X))}

    try:
        url, meta = figures.render_interactive(key, draw, figsize=(w, h))
    except Exception:
        return None
    meta["url"] = url
    return meta


def profile_along_line(file: str, points: list[list[float]], im: int = 1,
                       gap_threshold: float = 100.0, xaxis: str = "x",
                       tolerance: float | None = None) -> dict | None:
    """Pick soundings near the polyline *points* and plot a profile along it.

    ``points`` are UTM ``[x, y]`` pairs. Uses
    ``ig.find_points_along_line_segments`` → ``ig.plot_profile(..., ii=…,
    xaxis='x')`` — the pattern from
    ``examples/integrate_rawmaterial_daugaard.py``. Returns
    ``{url, n}`` or ``None``.
    """
    if not points or len(points) < 2:
        return None
    path = safe_path(file)
    try:
        import numpy as np

        import integrate as ig
        import integrate.integrate_plot as igp

        with _wcwd():
            X, Y, *_ = ig.get_geometry(str(path))
            Xl = np.asarray([p[0] for p in points], float)
            Yl = np.asarray([p[1] for p in points], float)
            # Match examples/integrate_rawmaterial_daugaard.py: a *tight* band
            # around the drawn line (a wide tolerance grabs a swath of
            # off-line soundings and smears the section).
            if tolerance is None or float(tolerance) <= 0:
                tolerance = 10.0
            idx, _dist, _seg = ig.find_points_along_line_segments(
                X, Y, Xl, Yl, tolerance=float(tolerance))
        idx = np.asarray(idx)
        if idx.size == 0:
            return None

        xaxis = xaxis if xaxis in ("x", "y", "index", "id") else "x"
        key = figures.figure_key(
            "profline", str(path), int(im), float(gap_threshold), xaxis, float(tolerance),
            tuple((round(x, 2), round(y, 2)) for x, y in points), salt=path,
        )

        def _call():
            igp.plot_profile(str(path), ii=idx, im=int(im), xaxis=xaxis,
                             gap_threshold=float(gap_threshold), hardcopy=False)

        return {"url": figures.render(_call, key=key), "n": int(idx.size),
                "xaxis": xaxis, "tolerance": float(tolerance)}
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Query — natural-language questions over a posterior (mirrors ui/backend/routers/query.py)
# --------------------------------------------------------------------------- #
_CLAUDE_MODEL = "anthropic/claude-sonnet-4-6"
_OLLAMA_MODEL = "ollama_chat/qwen3:latest"


def llm_env_config() -> dict | None:
    """LLM model configured on the server via env vars, or ``None``.

    ``INTEGRATE_LLM_MODEL`` (full LiteLLM string) wins; ``ANTHROPIC_API_KEY`` /
    ``OLLAMA_API_KEY`` stay supported for back-compat. The key is never returned.
    """
    import os

    model = os.environ.get("INTEGRATE_LLM_MODEL")
    if model:
        return {"model": model, "source": "INTEGRATE_LLM_MODEL"}
    if os.environ.get("ANTHROPIC_API_KEY"):
        return {"model": os.environ.get("INTEGRATE_CLAUDE_MODEL", _CLAUDE_MODEL),
                "source": "ANTHROPIC_API_KEY"}
    if os.environ.get("OLLAMA_API_KEY"):
        return {"model": os.environ.get("INTEGRATE_OLLAMA_MODEL", _OLLAMA_MODEL),
                "source": "OLLAMA_API_KEY"}
    return None


# provider id -> (LiteLLM model prefix, env-var names that may hold its key)
_PROVIDERS = {
    "openai":     ("openai/",      ["OPENAI_API_KEY"]),
    "anthropic":  ("anthropic/",   ["ANTHROPIC_API_KEY"]),
    "gemini":     ("gemini/",      ["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
    "groq":       ("groq/",        ["GROQ_API_KEY"]),
    "mistral":    ("mistral/",     ["MISTRAL_API_KEY"]),
    "deepseek":   ("deepseek/",    ["DEEPSEEK_API_KEY"]),
    "xai":        ("xai/",         ["XAI_API_KEY"]),
    "openrouter": ("openrouter/",  ["OPENROUTER_API_KEY"]),
    "ollama":     ("ollama_chat/", ["OLLAMA_API_KEY"]),
}
_OPENAI_COMPAT_BASE = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
}


def provider_model_prefix(provider: str) -> str:
    return _PROVIDERS.get(provider, ("", []))[0]


def _ollama_host() -> str:
    import os

    host = (os.environ.get("OLLAMA_API_BASE") or os.environ.get("OLLAMA_HOST")
            or "http://localhost:11434").rstrip("/")
    return host if host.startswith(("http://", "https://")) else f"http://{host}"


def _provider_key(provider: str, explicit: str | None) -> str | None:
    import os

    if explicit and explicit.strip():
        return explicit.strip()
    for var in _PROVIDERS.get(provider, ("", []))[1]:
        if os.environ.get(var):
            return os.environ[var]
    return None


def _openai_compat_base(provider: str) -> str | None:
    if provider in _OPENAI_COMPAT_BASE:
        return _OPENAI_COMPAT_BASE[provider]
    try:
        import litellm

        litellm.suppress_debug_info = True
        _n, _p, _k, base = litellm.get_llm_provider(f"{provider}/_")
        return base.rstrip("/") if base else None
    except Exception:
        return None


def list_provider_models(provider: str, api_key: str | None) -> dict:
    """Best-effort live model list for *provider*. Never raises.

    Returns ``{"models": [<raw ids>], "prefix": "<litellm prefix>", "live": bool,
    "error": str|None}``. Port of ``ui/backend/routers/query.py``.
    """
    import os

    provider = (provider or "").strip().lower()
    prefix = provider_model_prefix(provider)

    if provider in ("ollama", "ollama_chat"):
        host = _ollama_host()
        try:
            import httpx

            r = httpx.get(f"{host}/api/tags", timeout=3.0)
            r.raise_for_status()
            names = sorted(m.get("name", "") for m in r.json().get("models", []))
            return {"models": [n for n in names if n], "prefix": "ollama_chat/",
                    "live": True, "error": None}
        except Exception as e:  # noqa: BLE001
            return {"models": [], "prefix": "ollama_chat/", "live": False,
                    "error": f"No reachable Ollama server at {host} ({e})."}

    key = _provider_key(provider, api_key)
    if not key:
        return {"models": [], "prefix": prefix, "live": False,
                "error": "Enter an API key to load models."}

    try:
        import httpx

        with httpx.Client(timeout=8.0) as c:
            if provider == "anthropic":
                r = c.get("https://api.anthropic.com/v1/models",
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
                r.raise_for_status()
                ids = [m["id"] for m in r.json().get("data", []) if m.get("id")]
            elif provider == "gemini":
                r = c.get("https://generativelanguage.googleapis.com/v1beta/models",
                          params={"key": key})
                r.raise_for_status()
                ids = [m["name"].split("/", 1)[-1] for m in r.json().get("models", [])
                       if m.get("name") and "generateContent" in m.get("supportedGenerationMethods", [])]
            else:
                base = _openai_compat_base(provider)
                if not base:
                    return {"models": [], "prefix": prefix, "live": False,
                            "error": f"Live model listing not supported for '{provider}' — type the id."}
                r = c.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"})
                r.raise_for_status()
                ids = [m["id"] for m in r.json().get("data", []) if m.get("id")]
    except Exception as e:  # noqa: BLE001
        msg = (str(e) or e.__class__.__name__)[:200]
        return {"models": [], "prefix": prefix, "live": False, "error": msg}

    return {"models": sorted(dict.fromkeys(ids)), "prefix": prefix, "live": True, "error": None}


def _resolve_prior_link(f_prior: str) -> Path | None:
    if not f_prior:
        return None
    from frontend.config import get_workspace

    p = Path(f_prior)
    for cand in ([p] if p.is_absolute() else []) + [get_workspace() / p.name]:
        if cand.is_file():
            return cand
    return None


def posterior_prior(name: str) -> tuple[Path | None, str]:
    """(resolved prior path or None, raw ``f5_prior`` attr) for a POSTERIOR file."""
    import h5py

    try:
        with h5py.File(safe_path(name), "r") as f:
            raw = str(f.attrs.get("f5_prior", "") or "")
    except Exception:
        raw = ""
    return _resolve_prior_link(raw), raw


def list_posteriors_with_prior() -> list[str]:
    """POSTERIOR ``.h5`` files whose linked prior file is present in the workspace."""
    return [n for n in list_h5_by_class("POSTERIOR") if posterior_prior(n)[0] is not None]


def query_models(name: str) -> dict:
    """Prior-model table + ``prior_describe`` text for the prior linked from *name*."""
    import contextlib
    import io

    prior, raw = posterior_prior(name)
    if prior is None:
        return {"error": f"linked prior file not found ({raw or 'no f5_prior attr'})", "models": []}
    try:
        import integrate as ig

        with _wcwd():
            summ = files.summary(prior)
            models = []
            for m in summ.get("models", []):
                im = int(m["id"][1:])
                info = ig.get_prior_model_info(str(prior), im)
                z = info.get("z")
                rng = "—"
                if z is not None and len(z) >= 2 and float(z[-1]) != float(z[0]):
                    rng = f"{float(z[0]):g} – {float(z[-1]):g}"
                typ = "DISCRETE" if info.get("is_discrete") else ("SCALAR" if rng == "—" else "CONTINUOUS")
                classes = ""
                cid, cname = info.get("class_id"), info.get("class_name")
                if info.get("is_discrete") and cid is not None and cname is not None:
                    classes = ", ".join(
                        f"{int(i)}={str(n)}" for i, n in zip(list(cid), list(cname))
                    )
                models.append({
                    "im": im, "name": info.get("name") or m["id"], "type": typ,
                    "range": rng, "classes": classes,
                })
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ig.prior_describe(str(prior))
        return {"prior": prior.name, "models": models, "describe": buf.getvalue().strip()}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "models": []}


def query_translate(name: str, text: str, model: str | None, api_key: str | None) -> dict:
    """Stage 1: ``ig.query_from_text`` → ``{query_dict, interpretation}`` or ``{error}``."""
    prior, raw = posterior_prior(name)
    if prior is None:
        return {"error": f"linked prior file not found ({raw or 'no f5_prior attr'})"}
    env = llm_env_config()
    mdl = env["model"] if env else (model or "").strip()
    key = None if env else ((api_key or "").strip() or None)
    if not mdl:
        return {"error": "No LLM configured — enter a model id (and key), or set "
                         "INTEGRATE_LLM_MODEL / ANTHROPIC_API_KEY on the server."}
    try:
        import json as _json
        import os

        import integrate as ig

        # ig._litellm_extra passes OLLAMA_API_BASE / OLLAMA_HOST straight to
        # litellm without normalising — but OLLAMA_HOST is often a bare
        # "host:port" (litellm then errors "missing http:// protocol"). Force
        # a normalised OLLAMA_API_BASE (same value the model dropdown used) for
        # the call, then restore.
        _restore = object()
        prev_base = os.environ.get("OLLAMA_API_BASE", _restore)
        if mdl.startswith("ollama"):
            os.environ["OLLAMA_API_BASE"] = _ollama_host()
        try:
            with _wcwd():
                query_dict, interpretation, _sp = ig.query_from_text(
                    text, str(prior), model=mdl, api_key=key
                )
        finally:
            if mdl.startswith("ollama"):
                if prev_base is _restore:
                    os.environ.pop("OLLAMA_API_BASE", None)
                else:
                    os.environ["OLLAMA_API_BASE"] = prev_base
        return {
            "query_dict": query_dict,
            "query_json": _json.dumps(query_dict, indent=2),
            "interpretation": str(interpretation or ""),
            "model": mdl,
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"LLM translation failed: {e}"}


def query_evaluate(name: str, text: str, query_dict: dict, interpretation: str = "") -> dict:
    """Stage 2: ``ig.query`` + ``ig.query_plot`` / ``query_percentile_plot`` → figures + stats."""
    try:
        import numpy as np

        import integrate as ig

        with _wcwd():
            result, meta = ig.query(str(safe_path(name)), query_dict)
        if result is None:
            return {"error": "ig.query returned no result"}
        kind = "percentile" if "metric" in query_dict else "probability"

        key = figures.figure_key("query", str(safe_path(name)), kind,
                                 repr(sorted(query_dict.items())), salt=safe_path(name))

        def _draw():
            if kind == "percentile":
                ig.query_percentile_plot(result, meta, query_text=text,
                                         interpretation=interpretation or None, text_panel=True)
            else:
                ig.query_plot(result, meta, query_text=text,
                              interpretation=interpretation or None, text_panel=True)

        urls = figures.render_all(key, _draw)
        out = {"kind": kind, "figures": urls,
               "n_locations": int(meta.get("N_data", 0) or 0)}
        if kind == "probability":
            out["mean_probability"] = float(np.mean(np.asarray(result, float)))
        else:
            out["percentiles"] = [int(p) for p in meta.get("percentiles", [])]
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": f"Query evaluation failed: {e}"}


def posterior_kpis(name: str) -> dict:
    """Headline numbers for a POSTERIOR file (structural summary + N_UNIQUE)."""
    path = safe_path(name)
    summ = files.summary(path)
    out = {
        "soundings": summ.get("n_points", "—"),
        "nr": summ.get("n_realizations", "—"),
        "mean_t": f"{summ['t']['mean']:.3g}" if isinstance(summ.get("t"), dict) else "—",
        "mean_ev": f"{summ['ev']['mean']:.3g}" if isinstance(summ.get("ev"), dict) else "—",
        "inv_time": (f"{float(summ['inv_time']):.1f} s" if summ.get("inv_time") else "—"),
        "n_unique": "—",
        "f5_data": summ.get("f5_data", ""),
        "f5_prior": summ.get("f5_prior", ""),
    }
    try:
        import h5py
        import numpy as np

        with h5py.File(path, "r") as f:
            if "N_UNIQUE" in f:
                a = np.asarray(f["N_UNIQUE"][:], float).ravel()
                a = a[np.isfinite(a)]
                if a.size:
                    out["n_unique"] = f"{a.mean():.0f}"
    except Exception:
        pass
    return out


def posterior_table(name: str, limit: int = 200) -> list[dict]:
    """Per-sounding rows: ``ip | line | t | ev | n_unique`` (first *limit*)."""
    path = safe_path(name)
    try:
        import h5py
        import numpy as np

        import integrate as ig

        try:
            _X, _Y, line, _E = ig.get_geometry(str(path))
        except Exception:
            line = None
        with h5py.File(path, "r") as f:
            def col(key):
                return np.asarray(f[key][:], float).ravel() if key in f else None

            t, ev, nu = col("T"), col("EV"), col("N_UNIQUE")
            if line is None and "LINE" in f:      # posterior often carries its own LINE
                line = col("LINE")
        n = max((len(a) for a in (t, ev, nu) if a is not None), default=0)
        rows = []
        for i in range(min(n, limit)):
            rows.append({
                "ip": i,
                "line": _fmt(line[i]) if line is not None and i < len(line) else "—",
                "t": _fmt(t[i]) if t is not None and i < len(t) else "—",
                "ev": _fmt(ev[i]) if ev is not None and i < len(ev) else "—",
                "n_unique": _fmt(nu[i], 0) if nu is not None and i < len(nu) else "—",
            })
        return rows
    except Exception:
        return []


def _fmt(v, digits: int = 3) -> str:
    try:
        fv = float(v)
        return f"{fv:.0f}" if digits == 0 else f"{fv:.{digits}g}"
    except (TypeError, ValueError):
        return str(v)


# --------------------------------------------------------------------------- #
# Module 01 — Data files
# --------------------------------------------------------------------------- #
def list_h5() -> list[dict]:
    """Workspace ``*.h5`` files with class + size."""
    out = []
    for p in _list_workspace_h5():
        out.append(
            {
                "name": p.name,
                "kind": files.classify(p),
                "size": _human_size(p.stat().st_size),
                "bytes": p.stat().st_size,
            }
        )
    return out


def file_detail(name: str) -> dict:
    """Class-specific headline stats + a detail table + the raw dataset list.

    ``stats``  — ``[(label, value), ...]`` headline numbers, named per class
                 (DATA: Soundings / Data types / Continuous / Discrete;
                  PRIOR: Realizations / Model types / Prior-data types;
                  POSTERIOR: Soundings / Realizations per sounding / Model types).
    ``types``  — ``{"title", "columns", "rows"}`` or ``None``: one row per
                 ``/D{i}`` data type or ``/M{i}`` model parameter.
    ``meta``   — ``[(label, value), ...]`` extra key/values (linked files, runtime).
    """
    path = safe_path(name)
    summ = files.summary(path)
    cls = summ.get("class", "UNKNOWN")
    builder = {
        "DATA": _data_detail,
        "PRIOR": _prior_detail,
        "POSTERIOR": _post_detail,
    }.get(cls)
    stats, types, meta = builder(path, summ) if builder else ([], None, [])
    return {
        "name": path.name,
        "kind": cls,
        "stats": stats,
        "types": types,
        "meta": meta,
        "datasets": [
            {"path": d["path"], "shape": "×".join(map(str, d["shape"])) or "scalar"}
            for d in files.flat_datasets(path)
        ],
        "summary": summ,
    }


def geometry_figure(name: str) -> str | None:
    """Render ``ig.plot_geometry(file, pl='LINE')`` -> ``/static/figures/*.png`` URL.

    Returns ``None`` if the file has no usable geometry.
    """
    path = safe_path(name)
    try:
        import integrate as ig

        key = figures.figure_key("plot_geometry:LINE", str(path), salt=path)

        def _call():
            ig.plot_geometry(str(path), pl="LINE")

        return figures.render(_call, key=key)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _list_workspace_h5() -> list[Path]:
    from frontend.services.workspace import list_files

    return list_files(".h5")


_Stats = list[tuple[str, object]]
_Detail = tuple[_Stats, "dict | None", _Stats]


def _n_soundings(path: Path, fallback) -> object:
    """Number of data locations, via ``ig.get_geometry`` (falls back to summary)."""
    try:
        import integrate as ig

        X, *_ = ig.get_geometry(str(path))
        return len(X)
    except Exception:
        return fallback


def _data_detail(path: Path, summ: dict) -> _Detail:
    """DATA: continuous vs discrete data types (``/D{i}/noise_model``)."""
    dsets = summ.get("datasets", [])
    n_cont = n_disc = 0
    rows = []
    for d in dsets:
        nm = (d.get("noise_model") or "").lower()
        shape = d.get("shape") or []
        discrete = nm.startswith("multinomial") or (not nm and len(shape) >= 3)
        if discrete:
            n_disc += 1
            kind = "discrete"
            channels = "×".join(map(str, shape[1:])) if len(shape) > 1 else "—"
        else:
            n_cont += 1
            kind = "continuous"
            channels = str(shape[1]) if len(shape) > 1 else "—"
        used = d.get("n_used")
        rows.append([
            d["id"], d.get("name") or "—", d.get("noise_model") or "—",
            kind, channels, str(used) if used is not None else "all",
        ])
    stats: _Stats = [
        ("Soundings", _n_soundings(path, summ.get("n_points", "—"))),
        ("Data types", len(dsets)),
        ("Continuous", n_cont),
        ("Discrete", n_disc),
    ]
    types = {
        "title": "Data types",
        "columns": ["#", "Name", "Noise model", "Kind", "Channels", "Used"],
        "rows": rows,
    }
    return stats, types, []


def _prior_detail(path: Path, summ: dict) -> _Detail:
    """PRIOR: per model parameter — type + dimension (Nm = len of ``/M{i}/x``)."""
    models = summ.get("models", [])
    pdata = summ.get("data", [])
    rows = []
    for m in models:
        im = int(m["id"][1:])
        nm = m["shape"][1] if len(m["shape"]) > 1 else 1
        name = m.get("name") or m["id"]
        typ, rng, nclass = "CONTINUOUS", "—", "—"
        try:
            import integrate as ig

            info = ig.get_prior_model_info(str(path), im)
            name = info.get("name") or name
            z = info.get("z")
            if z is not None and len(z) >= 2 and float(z[-1]) != float(z[0]):
                rng = f"{float(z[0]):g} – {float(z[-1]):g}"
            if info.get("is_discrete"):
                typ = "DISCRETE"
                cid = info.get("class_id")
                nclass = str(len(cid)) if cid is not None else "—"
            elif rng == "—" or nm <= 1:
                typ = "SCALAR"
        except Exception:
            if m.get("is_discrete"):
                typ = "DISCRETE"
            elif nm <= 1:
                typ = "SCALAR"
        rows.append([f"M{im}", name, typ, str(nm), rng, nclass])
    stats: _Stats = [
        ("Realizations", summ.get("n_realizations", "—")),
        ("Model types", len(models)),
        ("Prior-data types", len(pdata)),
    ]
    types = {
        "title": "Model parameters",
        "columns": ["im", "Name", "Type", "Dimension", "Depth range (m)", "Classes"],
        "rows": rows,
    }
    return stats, types, []


def _post_detail(path: Path, summ: dict) -> _Detail:
    """POSTERIOR: soundings + realizations-per-sounding (``/i_use`` is [Np, Nr])."""
    models = summ.get("models", [])
    stats: _Stats = [
        ("Soundings", summ.get("n_points", "—")),
        ("Realizations / sounding", summ.get("n_realizations", "—")),
        ("Model types", len(models)),
    ]
    if isinstance(summ.get("t"), dict):
        stats.append(("Mean T", f"{summ['t']['mean']:.3g}"))
    if isinstance(summ.get("ev"), dict):
        stats.append(("Mean EV", f"{summ['ev']['mean']:.3g}"))

    # per-model posterior statistics present (/M{i}/<Stat>)
    stats_by_m: dict[str, list[str]] = {}
    for ds in files.flat_datasets(path):
        parts = ds["path"].strip("/").split("/")
        if len(parts) == 2 and re.fullmatch(r"M\d+", parts[0]):
            stats_by_m.setdefault(parts[0], []).append(parts[1])
    types = None
    if models:
        types = {
            "title": "Posterior model types",
            "columns": ["im", "Statistics"],
            "rows": [[m, ", ".join(stats_by_m.get(m, [])) or "—"] for m in models],
        }

    meta: _Stats = []
    for label, key, fmt in (
        ("Data file", "f5_data", str),
        ("Prior file", "f5_prior", str),
        ("Inversion time", "inv_time", lambda v: f"{float(v):.1f} s"),
        ("N_use", "N_use", str),
        ("Started", "date_start", str),
        ("Finished", "date_end", str),
    ):
        v = summ.get(key)
        if v not in (None, "", 0):
            meta.append((label, fmt(v)))
    return stats, types, meta


def _human_size(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.0f} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} TB"
