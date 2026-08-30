# INTEGRATE Web UI

Browser interface for the INTEGRATE probabilistic inversion toolbox: run
rejection inversions, inspect HDF5 project files, and query posteriors with
natural language — a modern replacement for the Streamlit panes in
`streamlit/`.

---

## Requirements

- The repository Python environment at `.venv/` (has `integrate` and its
  dependencies installed; `litellm` is needed only for the Query tool).
- Node.js + npm (only for building / developing the frontend).

## Running

### First time only — build the frontend

```bash
./ui/dev.sh build
```

This runs `npm install && npm run build` in `ui/frontend/` and produces
`ui/frontend/dist/`.

### Production mode (recommended for normal use)

```bash
./ui/start.sh [DATA_DIR] [--port 8000]
# or, once the package is installed (like `integrate_rejection` etc.):
integrate_ui [--port 8000] [--host 127.0.0.1]
```

Serves the REST API **and** the built frontend on one port. Open
<http://localhost:8000>. `integrate_ui` is a setuptools console script
(`ui.backend.main:main`) — it uses the launch directory as workspace, so run
it from the folder holding your `.h5` files. Wheels ship the frontend from
`ui/frontend/dist/`, so build it (`./ui/dev.sh build`) before packaging.

### Development mode

```bash
./ui/dev.sh [DATA_DIR]
```

Runs the backend on **:8000** (uvicorn `--reload`, picking up Python edits)
plus the Vite dev server on **:5173** with hot-reload for the frontend. Open
<http://localhost:5173>. The Vite server proxies API/WebSocket traffic to the
backend, so both surfaces behave identically.

### Choosing the workspace (where your `.h5` files live)

The UI operates on a single **workspace** directory, shown in the sidebar
footer. Resolution order:

1. `INTEGRATE_WORKSPACE` environment variable, if set.
2. A positional `DATA_DIR` argument (`./ui/start.sh examples`).
3. **The directory you launch the script from** — so
   `cd examples && ../ui/start.sh` works.

The file list shows the top-level `*.h5` files of the workspace only (no
subdirectory browsing); run the server from the folder containing your files.

---

## Using the UI

### Rejection — `/`

Form for `ig.integrate_rejection()`: select PRIOR and DATA files, output
name, `N_use`, `nr`, temperature control (`autoT`/`T_base`), CPUs, numpy/JAX
backend, parallelism, `id_use`/`ip_range` slices and advanced acceptance
parameters. Starting a run spawns a dedicated **child process** per job, so
long inversions never block the API and can be stopped safely. Live progress
(phase, data-point counter) and console logs stream in real time; finished
jobs expose result stats (EV / T / χ² series) and a rendered profile plot
per model (`im`).

### Files — `/files`

Browser for the workspace `.h5` files, each classified
(`PRIOR` / `DATA` / `POSTERIOR` / `UNKNOWN`) with a badge. Filter by class,
upload new `.h5` files into the workspace, and open the **inspector** for a
detailed *summary* (realizations, models, datasets) or the full HDF5 *tree*
with shapes, dtypes and attributes.

### Query — `/query`

Natural-language queries over posterior realizations (the `ig_query` pane):

1. **LLM** — if a model is configured on the server (see below) the panel just
   shows which one and no input is needed. Otherwise pick a **provider**
   (OpenAI, Anthropic, Gemini, Groq, Mistral, DeepSeek, xAI, OpenRouter,
   Ollama, or *Other…* for any other LiteLLM provider), paste its **API key**
   (kept in browser memory, sent per request), and choose a **model** from the
   dropdown — populated by a live call to that provider's model list. If the
   list can't be fetched, type the model id manually.
2. **Posterior file** — filter + select among the workspace's `POSTERIOR`
   files (those containing an `i_use` dataset). The linked prior file
   (`f5_prior` attribute) is shown.
3. **Available prior models** — table of `im`, name, type (CONTINUOUS /
   DISCRETE / SCALAR …) and depth range. Expand **“Model parameters (classes,
   names)”** for the full `ig.prior_describe()` listing including every class
   id and name of discrete models.
4. **Query** — plain English, e.g. *“What is the probability that cumulative
   clay thickness exceeds 10 m within 0 to 30 m depth?”* The LLM translates
   it into a structured query dict; you get the LLM's **interpretation**
   (check it!), the mean probability and number of locations, the probability
   map figure, plus expandable **Query JSON** and **System Prompt** sections.
   Percentile questions (“p5/p50/p95 of …”) are handled too.

### Query Volume — `/query-volume`

Visual probabilistic volume computations over a posterior file (the B8–B9
workflow of `examples/integrate_rawmaterial_daugaard.py` as a UI):

1. **LLM / Posterior File** — same as the Query tool.
2. **A. Probability map** — plain-English probability query (e.g. *“probability
   that cumulative sand-and-gravel thickness exceeds 10 m within the top 30 m”*).
   The result is an interactive map of every sounding coloured by probability
   (grey = edge-affected, dropped; dotted line = survey outline).
3. **B. Interactive region growing** — click a sounding to set the **center**
   (snapped to the nearest kept sounding), adjust `P_MIN` (inclusion cutoff) and
   optionally `MAX_AREA_M2` and the advanced geometry knobs (`hull_ratio`,
   `edge_buffer`, `cell_area_k`, `elong_max`), and grow the coherent area in
   place (`ig.find_coherent_area` machinery). **Add area** appends it to the
   list and starts the next; every area is listed and individually removable.
4. **C. Volume computation** — a second plain-English **percentile**
   question (e.g. *“P5, P50 and P95 of the cumulative thickness of sand and
   gravel between 0 and 50 m depth”*) gives per-area volumes in m³
   (`ig.region_volumes`) as a text table plus a P50-bar / P5–P95-whisker chart.

### LLM configuration (Query tool)

Server-side environment variables take effect without any user input. When any
of them configure a model, the browser LLM picker is hidden and that model is
used for every request.

| Variable | Effect |
|---|---|
| `INTEGRATE_LLM_MODEL` | **General knob.** A full LiteLLM model string, e.g. `openai/gpt-4o`, `openrouter/anthropic/claude-3.5-sonnet`, `gemini/gemini-1.5-pro`, `ollama_chat/qwen3:latest`. The provider's own key var (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, …) must also be set. |
| `ANTHROPIC_API_KEY` | *(Back-compat)* Claude becomes the server default (`anthropic/claude-sonnet-4-6`). |
| `OLLAMA_API_KEY` | *(Back-compat)* Ollama becomes the server default (authenticated remote servers). |
| `OLLAMA_HOST` / `OLLAMA_API_BASE` | Ollama server for the model dropdown **and** the LLM calls, e.g. `http://nsmain:11434`. |
| `INTEGRATE_CLAUDE_MODEL` / `INTEGRATE_OLLAMA_MODEL` | Override the back-compat default model ids. |
| `INTEGRATE_WORKSPACE` | Fixed workspace directory (see above). |

Notes:

- With none of the above set, the user picks provider + key + model in the
  browser; the model list is fetched live from the provider's API. Any
  LiteLLM-supported provider works (via the *Other…* option if not in the
  shortlist).
- litellm itself only honors `OLLAMA_API_BASE`; INTEGRATE additionally maps
  the standard `OLLAMA_HOST` onto it, so either variable works (`api_base`
  kwarg → `OLLAMA_API_BASE` → `OLLAMA_HOST` → `http://localhost:11434`).
- Keys sent from the browser are never stored or logged server-side.

---

## Architecture

```text
┌──────────────────────────┐         ┌────────────────────────────┐
│  Frontend (React SPA)    │  HTTP   │  Backend (FastAPI/uvicorn) │
│  Vite build in           │◄───────►│  ui/backend/               │
│  ui/frontend/dist        │   WS    │   ├─ routers/files.py      │
│                          │◄───────►│   ├─ routers/results.py    │
│  /files /query …         │ /api/ws │   ├─ routers/query.py      │
└──────────────────────────┘         │   └─ routers/jobs.py ──┐   │
                                     │   jobmanager.py        │   │
                                     │      │ spawn           ▼   │
                                     │      └────►  worker.py (child process per inversion) │
                                     └────────────────────────────┘
```

**Backend** — Python, FastAPI + uvicorn.

- `routers/files.py` — workspace file listing/classification, HDF5 tree and
  summary extraction (`h5inspect.py`), `.h5` upload.
- `routers/results.py` — posterior stats as JSON series and profile plots
  rendered with matplotlib's Agg backend to PNG; figure capture diffing under
  a global lock (matplotlib is not thread-safe).
- `routers/query.py` — LLM config discovery, prior-model tables
  (`ig.get_prior_model_info`, `ig.prior_describe`), live per-provider model
  listing, and the combined translate → evaluate → render pipeline
  (`ig.query_from_text` → `ig.query` → `ig.query_plot` /
  `ig.query_percentile_plot`).
- `routers/jobs.py` + `jobmanager.py` + `worker.py` — every inversion runs in
  an isolated child process; progress, logs and completion events flow back
  through a multiprocessing queue into one **WebSocket** (`/api/ws`) consumed
  by the frontend.
- `workspace.py` — workspace root resolution and path-confinement
  (`safe_path` refuses escapes outside the workspace).

**Frontend** — TypeScript SPA: **React 19**, **react-router 7**,
**Tailwind CSS 4** (via `@tailwindcss/vite`), **lucide-react** icons, built
with **Vite 8**. `lib/api.ts` is a typed REST client, `lib/jobs.tsx` a global
WebSocket-backed job context, `components/ui.tsx` shared dark-theme
primitives (Card, Field, Button, badges), and `views/` the three feature
views. The backend serves the built SPA with `index.html` fallback for
client-side routes. Linting: **oxlint**; type-checking: `tsc -b`.

### Backend CLI

```bash
.venv/bin/python -m ui.backend.main [--host 127.0.0.1] [--port 8000] [--reload]
```
