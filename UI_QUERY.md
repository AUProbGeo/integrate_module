# Natural-language query UI: how it was built

Notes on the Streamlit **Query Tool** (`streamlit/ig_query.py`), so the same idea
can be reused in another GUI (Qt, FastAPI/React, Jupyter widgets, CLI, ...).

- Deleted in commit `f38d834` (2026-09-29, *"Clean up UI, and fixed issued #47"*).
- Get the original back with `git show f38d834^:streamlit/ig_query.py`.

## Key idea

The GUI file was only about 200 lines and did almost nothing itself. All the real
work is in `integrate/integrate_query.py`, which is still in the module. A GUI only
has to:

1. Get the LLM settings (remote or local).
2. Pick a posterior file, and from it find the prior file.
3. Show the user which prior models and classes they can ask about.
4. Send the plain-English text to `ig.query_from_text()`, which returns a JSON query dict.
5. Run `ig.query()` on the posterior.
6. Plot with `ig.query_plot()`, and show the interpretation, the JSON and the system prompt.

```
 user text ──► ig.query_from_text(text, f_prior_h5, model, api_key)
                     │   (LLM via litellm; system prompt built from the prior file)
                     ▼
               query_dict, interpretation, system_prompt
                     │
                     ▼
               ig.query(f_post_h5, query_dict) ──► P (N_data,), meta
                     │
                     ▼
               ig.query_plot(P, meta, query_text=..., interpretation=..., text_panel=True)
```

## Module functions the GUI relies on

| Function | Purpose |
|---|---|
| `ig.query_from_text(text, f_prior_h5, model=..., api_key=None, system_prompt=None)` | Translates text into `(query_dict, interpretation, system_prompt)`. Raises `ValueError` if the LLM answers `UNSUPPORTED: ...` or returns invalid JSON (it retries once). Raises `ImportError` if `litellm` is missing. |
| `ig.query(f_post_h5, query_dict)` | Dispatcher. If the dict has `"constraints"` it calls `query_probability()` and returns `P, meta`. If it has `"metric"` it calls `query_percentile()`. |
| `ig.query_plot(P, meta, query_text=, interpretation=, text_panel=True)` | Plots the XY probability map. With `ip=` plus `query_dict`/`f_prior_h5`/`f_post_h5` it shows the realizations for one data point instead. |
| `ig.query_percentile_plot(values, meta, ...)` | Plot for percentile queries. The old GUI did not handle this case. |
| `ig.get_prior_model_info(f_prior_h5, im)` | Returns `{name, is_discrete, z, class_id, class_name}`. Used for the "available models" table. |
| `ig.prior_describe(f_prior_h5)` | Text version of the same table. |
| `ig.query_test_llm(model, api_key)` | Checks a model and key by asking for `{"status": "ok"}`. Returns `{ok, model, response, error}`. Good for a "Test connection" button. |
| `ig.save_query(q, path)` / `ig.load_query(path)` | Save and load query JSON. |

## Step by step (as implemented in Streamlit)

### 1. LLM configuration (always shown first)

- If `ANTHROPIC_API_KEY` is set in the environment, it used it directly with
  `model = 'anthropic/claude-sonnet-4-6'`.
- Otherwise it showed a radio button: **Claude** or **Ollama**.
  - **Claude (remote):** a password field for the API key. Model fixed to
    `anthropic/claude-sonnet-4-6`.
  - **Ollama (local):** no key. A text field for the model, default
    `ollama_chat/qwen3:latest`.
- The page stopped here until it had either a key or Ollama selected.

The model string is a **LiteLLM** model string, so any provider works
(`openai/gpt-4o`, `gemini/...`, `ollama_chat/...`). A new GUI could just offer a
free-text "model" field plus an optional "API key" field.

Ollama details are handled in `_litellm_extra(model)`:
- `think: False` is sent, so thinking models (Qwen3, DeepSeek-R1) don't return empty content.
- `OLLAMA_API_BASE` or `OLLAMA_HOST` is passed through as `api_base`, so a
  **remote Ollama server** works with no GUI changes.

### 2. Posterior file selection

- It listed `*.h5` files in the current directory that contain an `i_use`
  dataset (that marks a posterior file).
- A filter text box narrowed the list by substring, and a select box picked the file.
- The prior path comes from the posterior's attribute
  `h5py.File(f_post_h5).attrs['f5_prior']`. If that file is missing, the page stopped.

### 3. "Available prior models" table

For each `M<im>` dataset in the prior it showed: `im | Name | DISCRETE/CONTINUOUS | depth range`.
For discrete models it also listed the class IDs and names underneath
(`1 Sand · 2 Clay · ...`). This matters for usability: the user can see which
words and classes the LLM will understand.

### 4. Query input and translation

- A text area, for example *"What is the probability that cumulative clay thickness
  exceeds 10 m within 0 to 30 m depth?"*
- A **Query** button calls `ig.query_from_text(...)` behind a spinner, with errors
  shown inline.
- The `interpretation` string is shown right away. It is the LLM restating
  the question, and it is the cheap way to catch a misunderstanding.

### 5. Evaluation

- `P, meta = ig.query(f_post_h5, query_dict)` runs behind a spinner.
- A summary line shows the mean of `P` and `meta['N_data']`.

### 6. Display

- `ig.query_plot(...)` makes matplotlib figures. The GUI recorded
  `plt.get_fignums()` before and after the call and rendered each new figure
  (`st.pyplot`), then called `plt.close('all')`. The same trick works in any GUI:
  embed the new figures in a canvas, or save them to PNG and serve them.
- Collapsible sections showed:
  - **Query JSON** (`json.dumps(query_dict, indent=2)`)
  - **System prompt** (the full prompt sent to the LLM, useful for debugging)

## How the LLM translation works (`query_from_text`)

1. **The system prompt is built from the prior file** (`_build_llm_system_prompt`):
   - Role: "translate natural language into a JSON query dict for INTEGRATE".
   - Two query types:
     - `"probability"` returns a `"constraints"` list (combined with AND).
     - `"percentile"` returns one `"metric"` plus a `"percentiles"` list (default `[5, 50, 95]`).
   - Tables of the constraint and metric fields: `im`, `classes`,
     `value_comparison`/`value_threshold`, `thickness_mode`
     (`cumulative` | `first_occurrence`), `thickness_comparison`/`thickness_threshold`,
     `depth_min`/`depth_max`, `depth_min_im`/`depth_max_im` (depth bounds taken per
     realization from a scalar model, e.g. the water table), and `negate`.
   - **The available prior models, generated from the file:** for each model its
     type (DISCRETE / CONTINUOUS / SCALAR / SCALAR-DISCRETE), depth range, number of
     layers, and class IDs with names, plus rules for each type (for example,
     scalar models get no thickness fields).
   - Eight worked JSON examples.
   - Instructions: return JSON only, always include `interpretation` and
     `query_type`, use only the listed `im` and class IDs, and if the question
     can't be expressed, reply exactly `UNSUPPORTED: <reason>`.
2. **Call:** `litellm.completion(model, messages=[system, user], api_key, **_litellm_extra(model))`.
3. **Clean-up:** strip any ```` ``` ```` fences. Empty content raises an error with a
   hint about thinking models. An `UNSUPPORTED:` reply raises `ValueError`.
4. **Parse:** `json.loads`. If that fails, it retries once, sending the bad answer
   back with "Your response was not valid JSON. Output ONLY the JSON object...".
5. **Normalise:** pop `interpretation` and `query_type`, then return either
   `{'constraints': [...]}` or `{'metric': {...}, 'percentiles': [...]}`.

Example output:
```json
{"constraints": [{"im": 2, "classes": [2], "thickness_mode": "cumulative",
                  "thickness_comparison": ">", "thickness_threshold": 10.0,
                  "depth_min": 0.0, "depth_max": 30.0}]}
```

## Suggestions for a new GUI

- **Keep the GUI thin.** Call only the `ig.*` functions above and don't copy logic into the UI.
- **Two-stage flow** (the later React UI did this): *Translate*, then show the
  interpretation and an **editable** JSON box, then *Evaluate*. The user can fix the
  JSON without paying for another LLM call, and can run saved JSON with no LLM at all.
- **Support percentile queries:** if `'metric' in query_dict`, use
  `ig.query_percentile_plot` instead of `ig.query_plot`. The Streamlit version
  only handled probability queries.
- **Add a "Test LLM" button** using `ig.query_test_llm()`.
- **Make the model a free-text LiteLLM string** with presets (Claude, OpenAI,
  local Ollama), plus an optional API key and an optional Ollama host (set
  `OLLAMA_API_BASE`).
- **Allow the system prompt to be edited:** `query_from_text(..., system_prompt=...)`
  accepts a hand-tuned prompt.
- **Add a "Context" text box or file picker:** `query_from_text(..., context=...)`
  takes a string or a path to a `.md`/`.txt` file with expert, project-specific
  vocabulary (e.g. "clay" means classes 2 and 3 of im=2, "shallow" means 0-20 m). It is
  inserted as a "Use-case context" section after the prior model list.
- **Click to drill down:** clicking a point on the map can call
  `ig.query_plot(P, meta, ip=i, query_dict=..., f_prior_h5=..., f_post_h5=...)`.
- **Run long steps in the background:** `ig.query` can be slow on big posteriors,
  so run it in a worker thread or job queue in GUIs that block.

## Dependencies

`litellm` (LLM calls), `h5py`, `matplotlib`, `numpy`. For local LLMs, a running
Ollama server (`ollama serve`) with the model pulled, e.g. `ollama pull qwen3`.
