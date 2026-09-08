"""Query — natural-language questions over a posterior.

Mirrors ``ui/backend/routers/query.py`` / the streamlit ``ig_query`` pane:
``ig.query_from_text`` (LLM → query dict) → optional hand-edit → ``ig.query``
→ ``ig.query_plot`` / ``ig.query_percentile_plot``.
"""

from __future__ import annotations

import asyncio
import json

from fasthtml.common import (
    Details, Div, Form, Input, P, Pre, Span, Summary, Table, Tbody, Td, Textarea, Th, Thead, Tr,
)

from frontend.components import btn, error_box, eyebrow, field, figure_panel, select, shell
from frontend.services import integrate_api as api

_EXAMPLE = ("What is the probability that the cumulative thickness of sand and gravel "
            "exceeds 10 m within the top 30 m of depth?")


def _WAIT(msg: str):
    return P(Span("● ", style="color:var(--color-accent);"), msg,
             cls="wb-wait-msg", style="font-size:13px;")


_PROVIDERS = [
    ("openai", "OpenAI"), ("anthropic", "Anthropic"), ("gemini", "Gemini"),
    ("groq", "Groq"), ("mistral", "Mistral"), ("deepseek", "DeepSeek"),
    ("xai", "xAI"), ("openrouter", "OpenRouter"), ("ollama", "Ollama"),
    ("other", "Other…"),
]


def _model_selector(provider: str = "", api_key: str = ""):
    """`#query-modelsel`: a <select name=model> of full LiteLLM ids, or a text
    fallback + the provider error."""
    if not provider or provider == "other":
        return Div(
            field("Model (full LiteLLM id)", Input(
                name="model", cls="input",
                placeholder="e.g. openrouter/anthropic/claude-3.5-sonnet")),
            id="query-modelsel",
        )
    res = api.list_provider_models(provider, api_key or None)
    pfx = res["prefix"]
    opts = [(f"{pfx}{mid}", mid) for mid in res["models"]]
    if opts:
        return Div(
            field(f"Model ({len(opts)} from {provider})", select("model", opts, value=opts[0][0])),
            id="query-modelsel",
        )
    return Div(
        field("Model (full LiteLLM id)", Input(
            name="model", cls="input", value=pfx,
            placeholder=f"{pfx}<model-id>")),
        P(res.get("error") or "No models returned.", cls="wb-empty", style="font-size:12px;"),
        id="query-modelsel",
    )


def _llm_block():
    env = api.llm_env_config()
    if env:
        return Div(
            eyebrow("LLM"),
            P(Span(env["model"]), Span(f"  (server-configured via {env['source']})", cls="text-muted"),
              style="font-size:13px;"),
            cls="wb-panel",
        )
    reload_hx = dict(hx_post="/query/provider-models", hx_target="#query-modelsel",
                     hx_swap="outerHTML", hx_include="[name='provider'],[name='api_key']")
    return Div(
        eyebrow("LLM"),
        Div(
            field("Provider", select("provider", [("", "— select —"), *_PROVIDERS], value="",
                                     hx_trigger="change", **reload_hx)),
            field("API key", Input(name="api_key", type="password", cls="input",
                  placeholder="sent per request, never stored server-side", autocomplete="off",
                  hx_trigger="change delay:400ms", **reload_hx)),
            cls="wb-fields", style="grid-template-columns:1fr 2fr;",
        ),
        _model_selector(),
        btn("Load models", kind="secondary", style="margin-top:8px;", **reload_hx),
        P("Ollama needs no key (models come from the reachable OLLAMA_HOST). "
          "For others, pick a provider and paste its key, then Load models.",
          cls="text-muted", style="font-size:12px;margin-top:8px;"),
        cls="wb-panel",
    )


def _models_table(file: str):
    info = api.query_models(file)
    if info.get("error"):
        return Div(error_box(info["error"]), id="query-models")
    rows = [
        Tr(Td(str(m["im"])), Td(m["name"]), Td(m["type"]),
           Td(m["range"], style="font-variant-numeric:tabular-nums;"), Td(m["classes"] or "—"))
        for m in info["models"]
    ]
    return Div(
        eyebrow(f"Prior models — {info['prior']}"),
        Div(Table(
            Thead(Tr(Th("im"), Th("Name"), Th("Type"), Th("Depth range (m)"), Th("Classes"))),
            Tbody(*rows), cls="table"), cls="wb-tablewrap"),
        Details(Summary("prior_describe()"),
                Pre(info.get("describe", ""), cls="wb-log", style="max-height:240px;"),
                cls="wb-details"),
        id="query-models",
    )


def _translated(file: str, text: str, res: dict):
    if res.get("error"):
        return error_box(res["error"])
    return Div(
        eyebrow("Interpretation"),
        P(res["interpretation"] or "(none)", cls="text-muted", style="font-size:13px;"),
        eyebrow("Query JSON  (edit before evaluating if needed)"),
        Form(
            Input(type="hidden", name="file", value=file),
            Input(type="hidden", name="text", value=text),
            Textarea(res["query_json"], name="query_json", rows="12", cls="input",
                     style="font-family:ui-monospace,Menlo,monospace;font-size:12px;"),
            btn("Evaluate over the posterior", kind="primary",
                hx_post="/query/evaluate", hx_target="#query-result", hx_swap="innerHTML",
                hx_include="closest form", hx_indicator="#query-eval-wait",
                style="margin-top:10px;",
                **{"hx-on::before-request": "htmx.find('#query-result').innerHTML=''"}),
        ),
        Div(_WAIT("Evaluating the query over the posterior…"),
            id="query-eval-wait", cls="htmx-indicator wb-wait"),
        Div(id="query-result", style="margin-top:16px;"),
        style="margin-top:8px;",
    )


def _evaluated(res: dict):
    if res.get("error"):
        return error_box(res["error"])
    if res["kind"] == "probability":
        stat = f"mean probability = {res['mean_probability']:.3f}"
    else:
        stat = f"percentiles = {res['percentiles']}"
    figs = [figure_panel(f"figure {i + 1}", u, "", tall=True) for i, u in enumerate(res["figures"])]
    return Div(
        P(f"{stat}   ·   {res['n_locations']} locations", style="font-size:13px;"),
        *(figs or [P("no figure produced", cls="wb-empty")]),
    )


def _form(file: str | None):
    posts = api.list_posteriors_with_prior() or ["(no POSTERIOR files with a linked prior)"]
    file = file or posts[0]
    return Form(
        Div(field("Posterior file", select(
            "file", posts, value=file,
            hx_get="/query/models", hx_target="#query-models", hx_swap="outerHTML",
            hx_trigger="change")), cls="wb-row"),
        _llm_block(),
        _models_table(file),
        Div(field("Question", Textarea(name="text", rows="3", cls="input", placeholder=_EXAMPLE)),
            style="margin-top:14px;max-width:760px;"),
        btn("Run query", kind="primary",
            hx_post="/query/translate", hx_target="#query-out", hx_swap="innerHTML",
            hx_include="closest form", hx_indicator="#query-wait",
            style="margin-top:12px;",
            **{"hx-on::before-request": "htmx.find('#query-out').innerHTML=''"}),
        Div(_WAIT("Request sent to the LLM — waiting for a reply…"),
            id="query-wait", cls="htmx-indicator wb-wait"),
        Div(id="query-out", style="margin-top:16px;"),
        id="query-form",
    )


def register(rt) -> None:
    @rt("/query", name="page_query")
    def page():
        return shell(
            "query",
            P("Ask a plain-English question about a posterior — the LLM translates it into "
              "an ig.query spec, which you can edit before it runs.", cls="text-muted"),
            Div(cls="hr"),
            _form(None),
        )

    @rt("/query/models")
    def models(file: str = ""):
        posts = api.list_posteriors_with_prior()
        if file not in posts:
            file = posts[0] if posts else ""
        return _models_table(file)

    @rt("/query/provider-models", methods=["POST"])
    async def provider_models(provider: str = "", api_key: str = ""):
        return await asyncio.to_thread(_model_selector, provider, api_key)

    @rt("/query/translate", methods=["POST"])
    async def translate(file: str = "", text: str = "", model: str = "", api_key: str = ""):
        if not text.strip():
            return error_box("Enter a question first.")
        res = await asyncio.to_thread(api.query_translate, file, text.strip(),
                                      model or None, api_key or None)
        return _translated(file, text.strip(), res)

    @rt("/query/evaluate", methods=["POST"])
    async def evaluate(file: str = "", text: str = "", query_json: str = ""):
        try:
            qd = json.loads(query_json or "{}")
        except ValueError as e:
            return error_box(f"Query JSON is not valid: {e}")
        if not isinstance(qd, dict) or not qd:
            return error_box("Query JSON is empty.")
        res = await asyncio.to_thread(api.query_evaluate, file, text.strip(), qd)
        return _evaluated(res)
