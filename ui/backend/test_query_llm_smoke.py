"""Smoke test for the generic LLM provider wiring in routers/query.py.

Network-free by default: only the no-key / bad-input paths are exercised. Set a
real provider key in the environment to also hit the live model-listing path.
"""
import os

os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("OLLAMA_API_KEY", None)
os.environ.pop("INTEGRATE_LLM_MODEL", None)
os.environ["INTEGRATE_WORKSPACE"] = "examples"

from fastapi.testclient import TestClient  # noqa: E402

from ui.backend.main import app  # noqa: E402

client = TestClient(app)

# A posterior file that ships in examples/ (adjust if the fixtures change).
F = next(
    (n for n in sorted(os.listdir("examples"))
     if n.startswith("DAUGAARD_POSTERIOR") and n.endswith(".h5")),
    None,
)

# --- /config with no server LLM env -------------------------------------------
r = client.get("/api/query/config")
assert r.status_code == 200, r.text
assert r.json() == {"configured": False, "provider": None, "model": None}, r.json()
print("config (no env) ok:", r.json())

# --- /provider-models: no key -> not live, no network ------------------------
r = client.post("/api/query/provider-models", json={"provider": "openai"})
assert r.status_code == 200, r.text
body = r.json()
assert body["live"] is False and body["models"] == [] and "API key" in (body["error"] or "")
print("provider-models openai (no key) ok:", body)

# --- /provider-models: unknown provider -> graceful ------------------------
r = client.post("/api/query/provider-models", json={"provider": "nope", "api_key": "x"})
assert r.status_code == 200, r.text
assert r.json()["live"] is False and "nope" in (r.json()["error"] or "")
print("provider-models unknown ok:", r.json())

# --- /provider-models: ollama (best-effort, no key) --------------------------
r = client.post("/api/query/provider-models", json={"provider": "ollama"})
assert r.status_code == 200, r.text
assert "models" in r.json() and "live" in r.json()
print("provider-models ollama ok: live=%s n=%d" % (r.json()["live"], len(r.json()["models"])))

# --- /translate: keyed provider, no key -> 400 -------------------------------
if F:
    r = client.post("/api/query/translate", json={
        "f": F, "text": "probability that sand thickness exceeds 5 m",
        "provider": "openai", "model": "openai/gpt-4o",
    })
    assert r.status_code == 400, r.text
    assert "API key" in r.json()["detail"], r.json()
    print("translate (no key) -> 400 ok:", r.json()["detail"])

    # --- /translate: nothing configured at all -> 400 ----------------------
    r = client.post("/api/query/translate", json={"f": F, "text": "x"})
    assert r.status_code == 400, r.text
    assert "No LLM configured" in r.json()["detail"], r.json()
    print("translate (no model) -> 400 ok")
else:
    print("SKIP translate checks — no DAUGAARD_POSTERIOR*.h5 in examples/")

# --- server-configured path: INTEGRATE_LLM_MODEL --------------------------
os.environ["INTEGRATE_LLM_MODEL"] = "openai/gpt-4o"
try:
    from ui.backend.routers import query as qmod

    assert qmod._env_llm() == {"provider": "openai", "model": "openai/gpt-4o", "api_key": None}
    r = client.get("/api/query/config")
    assert r.json() == {"configured": True, "provider": "openai", "model": "openai/gpt-4o"}, r.json()
    print("config (INTEGRATE_LLM_MODEL) ok:", r.json())
finally:
    os.environ.pop("INTEGRATE_LLM_MODEL", None)

# --- optional: live listing when a real key is present ---------------------
for prov, var in [("openai", "OPENAI_API_KEY"), ("openrouter", "OPENROUTER_API_KEY"),
                  ("anthropic", "ANTHROPIC_API_KEY"), ("gemini", "GEMINI_API_KEY")]:
    if os.environ.get(var):
        r = client.post("/api/query/provider-models", json={"provider": prov})
        b = r.json()
        assert r.status_code == 200 and b["live"] and b["models"], b
        print(f"live provider-models {prov} ok: {len(b['models'])} models")

print("ALL SMOKE TESTS PASSED")
