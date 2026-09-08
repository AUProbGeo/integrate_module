# INTEGRATE Workbench

A desktop-style web front end for the `integrate` module. Pure Python
(FastHTML + HTMX), no build step, completely separate from the core library —
see [`FRONTEND.md`](FRONTEND.md) for the full plan and progress
checklist.

Status: all modules are working — **01 Data files**, **02 Prior model**
(Generic + **geoprior1d**), **03 Forward** (GA-AEM + Boreholes),
**04 Inversion**, **05 Results**, **06 Plotting** (Library + Profile line),
and **Query**.

---

## Running it

The UI operates on one **working folder** — the directory holding your
`.h5` (and `.gex` / `.xlsx`) files. It defaults to the directory you launch
from; you can also change it from the strip at the top of every page.

```bash
# one-time: install the two extra deps into the repo environment
uv pip install python-fasthtml openpyxl      # or: pip install python-fasthtml openpyxl

# run it, pointed at a data folder
cd examples          # a folder with .h5 files
python -m frontend   # -> http://127.0.0.1:8051
```

Options:

```bash
python -m frontend --port 8051 --host 127.0.0.1 --workspace /path/to/h5files
```

Workspace resolution order: `--workspace` → `$WB_WORKSPACE` →
`$INTEGRATE_WORKSPACE` → current directory.

### Development (auto-reload)

```bash
WB_WORKSPACE=$PWD/examples uvicorn frontend.app:app --reload --port 8051
```

### Tests

```bash
pytest frontend/tests
```

`test_smoke.py` boots the app and hits every route; `test_integrate_api.py`
and `test_xlsx.py` run against `examples/*.h5` and a temp workbook.

---

## What each screen does

| Screen | Backed by |
|---|---|
| **01 Data files** | Lists workspace `*.h5`, classifies each (DATA / PRIOR / POSTERIOR) and shows a class-specific inspector — DATA: soundings, data types, continuous vs discrete per `/D{i}`; PRIOR: realizations, per-`/M{i}` type + dimension + depth range (`ig.get_prior_model_info`); POSTERIOR: soundings, realizations/sounding, linked files. "Open figure" renders `ig.plot_geometry`. |
| **02 Prior model** | `ig.prior_model_layered` / `prior_model_workbench` / `prior_model_workbench_direct` from a parameter form. Run → child-process job with a live progress panel; on done, `ig.plot_prior_stats`. |
| **geoprior1d** | Load a geological Excel spec, edit every sheet in a grid (in-place, with a "Save to" field so you can write a copy and keep the original), then `geoprior1d(file_xlsx, Nreals, dmax, dz, …)` as a child-process job. |
| **03 Forward — GA-AEM** | `ig.prior_data_gaaem(f_prior_h5, file_gex/stm, im, …)` — synthetic GA-AEM response for every prior realization. Only 3 fields shown; the rest are behind "Advanced options". |
| **03 Forward — Boreholes** | `ig.save_borehole_data` — append boreholes (from a `.json` spec) as extra jointly-invertible data types onto a copy of a prior-data file. **Note:** also writes the gridded borehole observations into the DATA file in place. |
| **04 Inversion** | `ig.integrate_rejection(f_prior_h5, f_data_h5, …)` — localized rejection sampling, one posterior ensemble per sounding. Form left, live run panel right. |
| **05 Results** | KPIs for a POSTERIOR file, a chosen `integrate_plot` figure (`plot_profile`, `plot_T_EV`, …), and a per-sounding T / EV / N_UNIQUE table. |
| **06 Plotting — Library** | Pick DATA / PRIOR / POSTERIOR + a file + a curated `integrate_plot` function + its args, render the figure. |
| **06 Plotting — Profile line** | Click a polyline on the survey scatter map, then plot `ig.plot_profile` along it (`ig.find_points_along_line_segments` picks the soundings). |
| **Query** | Plain-English question → `ig.query_from_text` (LLM) → editable query JSON → `ig.query` → probability / percentile map + stats. Pick a **provider** (OpenAI / Anthropic / Gemini / OpenRouter / Ollama / …), paste its **key** (Ollama needs none), and the model **dropdown** loads live from that provider; or set `INTEGRATE_LLM_MODEL` / `ANTHROPIC_API_KEY` on the server. Keys are sent per request, never stored. |
| Working-folder strip | Shown on every page; change the folder and the whole app reloads against it. |

---

## Implementation

**Stack.** [`python-fasthtml`](https://fastht.ml) serves server-rendered
HTML; [HTMX](https://htmx.org) (vendored at `static/htmx.min.js`) swaps
fragments. No bundler, no npm, no client framework. The visual design system
is `static/modernist.css` — copied verbatim from a Claude design canvas —
plus a little layout chrome in `static/app.css`.

**Layout.**

```
frontend/
  app.py            FastHTML app, static mount, `main()` (the `python -m frontend` entry point)
  components.py     shell / nav / buttons / fields / tables — thin wrappers over modernist.css classes
  config.py         workspace resolution
  pages/            one module per screen; each exposes register(rt)
    _jobs_ui.py     shared run/progress panel + job routes (used by prior + geoprior)
  services/
    integrate_api.py  the ONLY web-process module that imports `integrate` (coerce args → ig.* → plain return)
    worker.py         child-process entry points; imports integrate / geoprior1d, but only inside the spawned process
    jobs.py           Job + start()/get()/cancel(); one spawned process per task, a thread drains its queue
    figures.py        matplotlib (Agg) → PNG under static/figures/, hash-cached, one global lock
    xlsx.py           in-place edits on a live openpyxl Workbook kept in the session
    files.py          structural HDF5 scan/classify (vendored from ui/backend/h5inspect.py)
    workspace.py      workspace root + path confinement
    session.py        cookie-keyed server-side state dict
  static/           modernist.css, app.css, htmx.min.js, fonts/, figures/ (generated, gitignored)
  tests/
```

**Core boundary.** In the web process, only `services/integrate_api.py`
imports `integrate`. Long-running calls (`prior_model_*`, `geoprior1d`, and
later `integrate_rejection`, `forward_gaaem`) run in a **spawned child
process** via `services/jobs.start(...)` — `worker.py` imports the heavy
libraries there, sets `MPLBACKEND=Agg`, `chdir`s to the workspace, names
itself `MainProcess` (some `integrate` functions require it), and streams
progress + stdout back over a `multiprocessing.Queue`. A daemon thread folds
those into an in-memory `Job`; the page polls it over HTMX every second and
stops when the job finishes. Nothing in `integrate/` imports `frontend`.

**State.** Per-visitor scratch state (the loaded workbook, last job ids) lives
in an in-process dict keyed by a signed session cookie; the working folder is
process-wide. Both reset on restart — fine for a single-user desktop tool.
Reusing as much of `integrate` as possible is a hard rule (see `FRONTEND.md`
§2); the only logic that originates here is HTTP/HTML plumbing, form coercion,
job orchestration, figure caching, and the spreadsheet editor.
