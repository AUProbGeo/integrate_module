# INTEGRATE Workbench — Frontend Plan

Living document. **A new session should be able to pick up work from this file
alone.** Update the TODO checklist (bottom) as you go, and revise the spec
sections when decisions change.

---

## 1. Purpose & scope

A desktop-oriented web front end for the `integrate` Python module, living
entirely in `frontend/`, completely decoupled from the core library. Based on
the "INTEGRATE Workbench" design canvas (artifact
`a8fac822-53f7-4198-a2cb-6180b4ba5b76`, a Swiss/Modernist static mockup).

Six functional modules, matching the mockup artboards:

| # | Module        | Core functions driven |
|---|---------------|-----------------------|
| 01 | Data files   | HDF5 scan / classify, `ig.plot_geometry` |
| 02 | Prior model  | `ig.prior_model_layered`, `ig.prior_model_workbench`, `ig.prior_model_workbench_direct` |
| 03 | Forward      | `ig.prior_data_gaaem` / `ig.forward_gaaem` |
| 04 | Inversion    | `ig.integrate_rejection` |
| 05 | Results      | `ig.integrate_posterior_stats`, `ig.plot_T_EV`, `ig.plot_profile` |
| 06 | Plotting     | selected `integrate.integrate_plot` functions |
| +  | geoprior1d   | `geoprior1d(file_xlsx, …)` — its own page with an embedded Excel-spec editor (split out of module 02, which was getting a mismatched "form vs spreadsheet" pair of paths) |
| +  | Query        | `ig.query_from_text` → `ig.query` → `ig.query_plot` (LLM translate → editable JSON → evaluate) |


---

## 2. Decisions locked (do not relitigate)

- **⭐ Reuse `integrate` code, do not reimplement it.** Every computation,
  file read/write, classification, statistic and plot the UI shows MUST come
  from an existing `integrate` function. The frontend is glue: forms in →
  `ig.*` call → result/figure out. If something the UI needs isn't in
  `integrate`, prefer adding/extending it in the core module (or ask the
  user) over writing a private copy in `frontend/`. **Only exception:**
  interactive-plotting helpers that `integrate` genuinely does not provide —
  e.g. picking points on a map to define a profile polyline, click-to-select
  a sounding, region drawing. Those may be implemented in `frontend/`
  (client interaction + light geometry), but the moment real data/model
  work starts, hand the picked coordinates/indices back to an `integrate`
  function (`ig.find_points_along_line_segments`, `ig.plot_profile`, …).
- **Stack: FastHTML** (`python-fasthtml`) + **HTMX** (bundled with FastHTML).
  Server-rendered HTML partials, no bundler, no npm, no client framework.
- **`streamlit/` and `ui/` stay untouched.** The user removes them later once
  `frontend/` is complete. Do **not** import from `ui.*` at runtime — instead
  **vendor (copy)** the few useful modules into `frontend/services/` so
  `ui/` can be deleted with no effect.
- **Spreadsheet editor: tier 1 only** — plain HTML `<table>` of `<input>`
  cells, HTMX save-on-blur. No JS grid library. No live formulas. Small
  workbooks only.
- **Query / Query-Volume tools: deferred.** Leave a `pages/query.py` stub and
  a nav slot, implement later.
- **Design system copied verbatim** from the artifact as
  `frontend/static/modernist.css` (see §5 + Appendix A). Restyle = edit that
  one file.
- **Core stays a pure library.** In the **web process**, only
  `frontend/services/integrate_api.py` imports `integrate`. `services/worker.py`
  also imports `integrate` / `geoprior1d`, but *only inside the spawned child
  process* it defines — never at web-process import time. Nothing in
  `integrate/` imports `frontend`.
- Frontend is an **optional extra**: `pip install integrate_module[frontend]`.

---

## 3. Stack details

- **`python-fasthtml`** — routing, `Div/Table/...` FT components, HTMX helpers,
  Starlette under the hood, uvicorn to serve.
- **HTMX** — `hx-get`/`hx-post`/`hx-target`/`hx-swap`; polling via
  `hx-trigger="every 1s"` for job progress. Ships with FastHTML (`Script(src=...)`
  or `fh.HtmxOn`); pin a local copy under `static/` for offline use.
- **matplotlib Agg** — every `integrate_plot` / `plot_*` call renders to PNG on
  disk under `frontend/static/figures/`, referenced by `<img>`. matplotlib is
  not thread-safe → wrap all figure work in a single `threading.Lock`
  (`services/figures.py`).
- **openpyxl** (+ pandas, already a core dep) — spreadsheet read/write.
- **No database.** Per-session state is an in-process dict keyed by a cookie
  (`services/session.py`). Jobs tracked in an in-process registry.
- **Long-running work** (prior gen, forward, inversion) runs in a **spawned
  child process** (`multiprocessing`, start method `spawn`) so it never blocks
  the event loop and can be killed. Progress/logs stream back on a
  `multiprocessing.Queue`; the web process drains the queue on each poll.

Python: 3.10+ (matches `pyproject.toml`).

---

## 4. Directory layout

```
frontend/
  __init__.py
  __main__.py              # `python -m frontend` -> main()
  app.py                   # FastHTML app, route registration, static mount, main()
  config.py                # workspace resolution, constants, paths
  pages/
    __init__.py            # register_all(rt); IMPLEMENTED set (rest -> stubs)
    files.py               # 01 Data files
    prior.py               # 02 Prior model (3 generic resistivity priors)
    geoprior.py            # geoprior1d — Excel-spec editor + run
    _jobs_ui.py            # shared run/progress panel + job routes (register_job_routes / run_panel)
    forward.py             # 03 Forward
    inversion.py           # 04 Inversion
    results.py             # 05 Results
    plotting.py            # 06 Plotting
    profile.py             # 06 Plotting / Profile line (interactive)
    query.py               # Query (LLM -> ig.query)
    workspace_ui.py        # working-folder strip routes (/workspace, /workspace/edit, /workspace/cancel)
    stubs.py               # placeholder pages for modules not yet built
  components.py            # shell(), page_header(), nav(), workspace_bar(), seg(), field(),
                           # stat_grid(), figure_panel(), btn(), tag()  -> map 1:1
                           # h5_table(), figure_panel(), btn(), tag()  -> map 1:1
                           # onto modernist.css classes
  services/
    __init__.py
    integrate_api.py       # only web-process module that imports `integrate`
    files.py               # h5 scan/classify/summary/flat_datasets  (vendored from ui/backend/h5inspect.py)
    jobs.py                # Job dataclass + start()/get()/cancel() + queue-drain thread + PHASE_LABELS
    worker.py              # child-process entry points (run_prior_job / run_geoprior_job); imports integrate/geoprior1d IN the child only
    figures.py             # locked matplotlib -> PNG helper (hash-cached)
    session.py             # cookie-keyed per-session state dict — state_for(session)
    xlsx.py                # in-place edits on a live openpyxl Workbook held in the session
    workspace.py           # workspace root + safe_path  (vendored from ui/backend/workspace.py)
  static/
    modernist.css          # verbatim from the design canvas (Appendix A)
    app.css                # tiny frontend-only additions (nav, layout chrome)
    htmx.min.js            # pinned local copy
    fonts/                 # Archivo woff2 x3 (from the artifact bundle)
    figures/               # generated PNGs (gitignored)
  README.md
  tests/
    test_smoke.py          # app boots, every route 200s
    test_integrate_api.py  # boundary wrappers against examples/*.h5
    test_xlsx.py
```

`.gitignore`: `frontend/.gitignore` covers `static/figures/` and
`**/__pycache__/` (paths relative to `frontend/`).

---

## 5. Design system reference

From the artifact's `Modernist` system (`INTEGRATE Workbench.dc.html`).

**Tokens** (`:root` in `modernist.css`):
- `--color-bg #f3f2f2`, `--color-surface #eae9e9`, `--color-text #201e1d`
- `--color-accent #ec3013`, `--color-accent-2 #e15b47`
- `--color-divider` = 40% ink; neutral & accent tonal ramps `-100..-900`
- `--font-heading` / `--font-body` = **Archivo** (800 weight headings)
- `--space-1..8` (4/8/12/16/24/32 px); `--radius-* = 0` (flat)
- `--shadow-sm/md/lg`

**Type scale:** h1 42 / h2 32 / h3 25 / h4 20 / h5 16 / h6 13 (h6 is an
uppercase 0.08em-tracked eyebrow). Body 15px / 1.55.

**Component classes** (in `modernist.css`, use as-is via `components.py`):
`.btn` + `.btn-primary` `.btn-secondary` `.btn-ghost` `.btn-icon` `.btn-block`;
`.field` (label + control wrapper); `.input` (text/select);
`.radio` (+ `.dot`); `.seg` + `.seg-opt` (segmented control — also used as
sheet tabs / filter toggles); `.card` + `.card-kicker` `.card-title`
`.card-body`; `.table`; `.hr`; `.text-muted`; `.grayscale`.

**Recurring layout idioms from the artboards:**
- Page header row: `INTEGRATE` wordmark (left) + `NN — Title` counter (right,
  muted, tabular-nums), 2px bottom divider.
- `<h2>` title + one muted intro sentence + `.hr`.
- Config laid out in 2–4 column grids of `.field`s under `<h5>` section
  headers (each `<h5>` has a 2px bottom border).
- Stat blocks: tiny uppercase neutral-600 label + big Archivo-800 tabular
  number (see Results KPI row, Data-files "Soundings/Gates/Data types").
- Figure slot: bordered box, uppercase eyebrow with the call name (e.g.
  `ig.plot_geometry()`), matplotlib PNG or a muted placeholder.
- Tables: `sc-raw-table.table` with right-aligned numeric columns, coloured
  `tag` badges for kind/fit.

**Fonts:** vendor the three Archivo woff2 files from the artifact bundle into
`static/fonts/` and keep the `@font-face` rules that reference them (they are
already in the CSS block). Fallback stack `system-ui, sans-serif`.

---

## 6. Architecture conventions

### 6.0 Working folder
- Every page carries a top strip (`components.workspace_bar`): `WORKING
  FOLDER  <path>  [Change…]`. "Change…" (htmx) swaps in an inline form —
  text field + quick-pick buttons for the parent and each sub-folder (● marks
  folders containing `.h5`). Submitting htmx-POSTs `/workspace`; on success the
  route replies `204 + HX-Redirect: /` so the whole app reloads against the
  new folder; on a bad path it re-renders the strip with an error.
- The folder is **process-wide** (`config.set_workspace`, single-user desktop
  tool) and **not persisted** — a fixed default still comes from
  `--workspace` / `WB_WORKSPACE` / `INTEGRATE_WORKSPACE` / CWD.

### 6.1 App shell & routing
- `app.py` builds `FastHTML(hdrs=[Link(rel=stylesheet, modernist.css), Link(app.css), Script(htmx)])`.
- `pages/__init__.py` exposes `PAGES` (ordered). `app.py` iterates it, calling
  each module's `register(app)` which adds its routes. Nav is generated from
  `PAGES`.
- Every page module exports:
  - `register(app)` — add routes.
  - `render(session, **q) -> FT` — full page body (wrapped by `shell()`).
  - partial handlers for HTMX swaps (prefixed route paths, e.g.
    `/prior/run`, `/files/rescan`).
- URL scheme: `/` = Data files (01), `/prior`, `/forward`, `/inversion`,
  `/results`, `/plotting`, `/sheets`, `/query`.

### 6.2 shell()
`components.shell(title, active_slug, *body)` → `<html>` with `<head>` from
hdrs, a left/top nav built from `PAGES` (active item uses `--color-accent`),
and `<main>` holding `page_header(n, title)` + body. Workspace path shown in a
footer (like the `ui/` sidebar footer).

### 6.3 Sessions & state
- `services/session.py`: `get_session(req) -> dict`, sets a `wb_session`
  cookie if absent. Holds: last-selected files per module, spreadsheet working
  copy, last job ids. No persistence across restarts (fine for a desktop tool).

### 6.4 Jobs (`services/jobs.py` + `services/worker.py`)
- `Job` dataclass: `id, kind, params, status(pending|running|done|error|cancelled),
  created_at, started_at, ended_at, pct, phase, count_label, log: deque(maxlen=1000),
  result: dict, error: str`.
- `start_job(kind, params) -> job_id`: spawn `multiprocessing.Process` running
  `worker.run_<kind>(params, queue)`. A per-job daemon thread drains the queue
  into the `Job` object.
- Queue message types: `{"type":"progress","current","total","info":{phase,status}}`,
  `{"type":"log","line"}`, `{"type":"done", ...result}`, `{"type":"error","traceback"}`,
  `{"type":"exit"}`.
- `worker.py` per kind: set `MPLBACKEND=Agg`, `os.chdir(workspace)`,
  `multiprocessing.current_process().name = "MainProcess"` (integrate_rejection
  refuses to run otherwise), `contextlib.redirect_stdout/stderr` to a
  queue-writer, build a `progress_callback(current,total,info)` that puts
  `progress` messages, call the `integrate` function, put `done`/`error`.
  (Model closely on `ui/backend/worker.py:run_rejection_job` — already
  correct; generalise for `prior` / `forward`.)
- `cancel_job(id)`: `process.terminate()`.
- HTMX progress: page renders a `#run-panel` that self-polls
  `hx-get="/<mod>/progress/<job_id>" hx-trigger="every 1s"
  hx-swap="outerHTML"`; when status is terminal the swapped fragment drops the
  `hx-trigger` and shows results / figure / log.
- `integrate`'s callback convention is `progress_callback(current, total, info_dict)`
  where `info_dict` may carry `phase` and `status` (see
  `streamlit/ig_progress.py` for phase labels:
  initializing/generating/computing/sampling/saving/post_processing/completed).

### 6.5 Figures (`services/figures.py`)
```
render_figure(fn, *args, name_hint, **kwargs) -> "static/figures/<hash>.png"
```
Under a module-global `Lock`: `matplotlib.use("Agg")`, close-all, call `fn`,
`plt.gcf().savefig(path, dpi=110, bbox_inches="tight")`, close-all. Filename =
hash of (fn name, args, mtime of source file) so repeats are cache hits.
Serve from `/static/figures/`. Periodically prune (keep last N).

### 6.6 integrate boundary (`services/integrate_api.py`)
- Import `integrate as ig` lazily inside functions (keeps app importable
  without the heavy deps for tests that stub it).
- One thin, typed wrapper per operation the UI needs. **A wrapper's body is
  argument coercion + one (or a few) `ig.*` call(s) + shaping the return.**
  No numerics, no HDF5 poking, no stats, no plotting logic of its own — if
  you're writing an algorithm here, it belongs in `integrate` instead.
  Wrappers never leak `ig` objects to pages — return plain dicts / paths /
  lists.
- Parse helpers: `parse_int_list("1,2")`, `parse_range("0:500") -> ip_range`.
- All file args resolved through `services/workspace.safe_path`.
- Reading `.h5` for display: use `integrate` readers
  (`ig.get_geometry`, `ig.load_data`, `ig.integrate_posterior_stats`,
  `ig.get_prior_model_info`, `ig.hdf5_scan`, …). The vendored
  `services/files.py` classify/tree is only a lightweight structural
  fallback for the file browser — don't grow it into a data layer.

### 6.6a Interactive picking (the one allowed local implementation)
- Client side (HTMX + minimal inline JS on a single static PNG or an SVG
  overlay): capture clicks, collect pixel→data coords, draw the provisional
  polyline / marker.
- Server side: convert picked points to whatever the `integrate` plot
  function wants (line segments, `ii` indices, `ip`, center sounding) and
  call it. Candidates already in core:
  `ig.find_points_along_line_segments`, `ig.plot_profile`,
  `ig.plot_xy`, `ig.get_geometry`, and (Query, later)
  `ig.find_coherent_area` / `ig.grow_connected_region`.
- Keep the local part to interaction + coordinate math only.

### 6.7 HTMX house style
- Full page load renders everything; HTMX only swaps the panel that changed
  (`hx-target` an id, `hx-swap="innerHTML"` or `"outerHTML"`).
- Forms POST to a partial route that returns the updated panel.
- No inline `<script>` beyond the pinned htmx; the spreadsheet save-on-blur
  uses `hx-post` + `hx-trigger="change"` on each `<input>`, no custom JS.
- Keep handlers pure: `(req) -> FT`. Read session via helper, not globals.

---

## 7. Module specs

Each artboard's fields are enumerated here so a page can be built without
re-opening the mockup. All numeric inputs are text fields coerced server-side.

### 01 — Data files (`pages/files.py`, route `/`)
- Filter segmented control: **All files** / **Hide PRIOR\* / POST\***.
- File count label + **Rescan** button (re-list workspace `*.h5`).
- Left: table `File | Type | Size`. Type badge from
  `services/files.classify()` → DATA / PRIOR / POSTERIOR / UNKNOWN.
- Right, on row select (`hx-get /files/inspect/<name>`) — `api.file_detail()`
  returns `stats` (headline `(label,value)`s), `types` (a detail table) and
  `meta`, all **class-specific**:
  - **DATA** — stats: `Soundings` (`len(ig.get_geometry(...)[0])`, fallback
    `d_obs.shape[0]`), `Data types` (# `/D{i}`), `Continuous`, `Discrete`
    (count by `/D{i}/noise_model`: `gaussian` ⇒ continuous, `multinomial*`
    ⇒ discrete; unknown ⇒ infer from `d_obs.ndim`). types table
    `# | Name | Noise model | Kind | Channels | Used`.
  - **PRIOR** — stats: `Realizations` (`/M1.shape[0]`), `Model types` (# `/M{i}`),
    `Prior-data types` (# `/D{i}`). types table
    `im | Name | Type | Dimension | Depth range (m) | Classes` where **Type** =
    CONTINUOUS / DISCRETE / SCALAR and **Dimension** = `len(/M{i}/x)` (= `Nm`),
    range from `x[0]..x[-1]`, classes from `class_id` — all via
    `ig.get_prior_model_info(path, im)` (structural fallback on error).
  - **POSTERIOR** — stats: `Soundings` (`/i_use.shape[0]`),
    `Realizations / sounding` (`/i_use.shape[1]`), `Model types`,
    `Mean T`, `Mean EV`. types table `im | Statistics` (the `/M{i}/<Stat>`
    datasets present). meta (Run details): `f5_data`, `f5_prior`, `inv_time`,
    `N_use`, `date_start`, `date_end` from root attrs.
  - Always: **Datasets** list (`path` + `shape`, flattened) and a
    `ig.plot_geometry(<file>, pl='LINE')` figure panel rendered on "Open figure".
- Vendored classify rule (from `ui/backend/h5inspect.py`):
  `/i_use` dataset ⇒ POSTERIOR; a `D<n>` **group** ⇒ DATA; `M<n>` **dataset**
  ⇒ PRIOR; else UNKNOWN; `OSError` ⇒ UNREADABLE.

### Shared: `pages/_jobs_ui.py`
- `register_job_routes(rt, base, *, out_key, done_extra)` adds
  `{base}/progress/{id}`, `{base}/cancel/{id}` (POST), `{base}/clear` and
  remembers `out_key` / `done_extra(job)->FT` for that base.
- `run_panel(job, base)` renders into a `#wb-run` container: big
  pct-or-elapsed + phase, progress bar (`.wb-bar`, `.indet` while running with
  no callback), streamed stdout `<pre class="wb-log">` (last 16 lines),
  `Cancel` / `Clear`, and — on done — `out_key` filename +
  `A("Open in Data files →", "/")` + whatever `done_extra` returns. While
  running the panel self-polls `{base}/progress/{id}` every 1s
  (`hx-target="this" hx-swap="outerHTML"`); the terminal panel has no trigger.
- Used by `/prior` and `/geoprior`; Forward/Inversion will reuse it.

### 02 — Prior model (`pages/prior.py`) — IMPLEMENTED
Two pages built by one parametrised `_register(rt, base=…, models=…, …)`:

| Page | base | nav | models |
|---|---|---|---|
| **Generic** | `/prior` | `("prior","Generic")` | `layered` |
| **WB** | `/prior-wb` | `("prior-wb","WB")` | `smooth` (L2) · `blocky` (L1) · `sharp` (MGS) |

- `ig.prior_model_workbench` / `prior_model_workbench_direct` stay in the core
  module for reference but are **no longer in the UI** — the smooth/blocky/
  sharp generators supersede them (and `smooth` with `corr_length<=0` gives an
  uncorrelated i.i.d.-per-layer prior, `RHO_dist='log-uniform'` a log-uniform
  marginal — see the core note below).
- **Model dropdown** (`name="model"`, HTMX `change` → `GET {base}/form` swaps
  `<form>#wb-prior-form` outerHTML); hidden when a page has one model.
- Field specs: `pages/prior.py` `GENERIC_MODELS` / `WB_MODELS` dicts of
  `F(name, label, kind, default)` (`kind` ∈ `int|float|str|("select",[…])`).
  `_cast` coerces on POST.
  - **layered**: `N`, `lay_dist` uniform|chi2, `dz` 1, `z_max` 90,
    `NLAY_min/max/deg` 3/6/6, `RHO_dist` log-uniform|uniform|normal|lognormal,
    `RHO_min/max/mean/std` 0.1/100/100/80.
  - **smooth**: `N`, `z1` 0, `z_max` 100, `dz` 1, `nlayers` 0, `p` 2,
    `corr_length` 15 (**≤0 → i.i.d.**), `sigma_logrho` 0.25, **`RHO_dist`
    lognormal|log-uniform|uniform|normal** (default `lognormal`),
    `RHO_ref` 100, `RHO_min/max` 1/300.
  - **blocky**: as smooth but `blocky_scale` 0.25 instead of
    `corr_length`/`sigma_logrho`/`RHO_dist` (L1 ignores `RHO_dist`).
  - **sharp**: `n_jumps_mean` 3, `RHO_dist` (log-uniform default),
    `RHO_min/max/mean/std`.
- Output name `out_name` → `f_prior_h5` (blank ⇒ auto).
- **Run** (`POST {base}/run`) → `api.start_prior_job(model, kwargs, out)` →
  `jobs.start("prior", worker.run_prior_job, …)` → `run_panel(job, base)`.
  `worker.run_prior_job` dispatches all six `prior_model_*` (workbench* kept in
  the dispatch, just unreachable from the UI).
- `done_extra` = a **model-parameter `<select>`** (`im`, from
  `api.prior_model_ims` = the file's `/M<n>` datasets + their names,
  defaulting to **M1**) + a **Plot prior stats** button → `GET
  /prior/figure/{name}?im=<im>` → `api.prior_stats_figure(name, im)` =
  `ig.plot_prior_stats(path, im=<im>)` into `#wb-fig`. Rendered with
  `figure_panel(..., tall=True)` (natural height + scroll) so the wide 1×3
  panel isn't clipped by the 260px `.wb-figure` box. Without `im`,
  `plot_prior_stats` recurses over every M and only the last is saved — the
  picker fixes that.
- Routes (per page, `base` ∈ `/prior`, `/prior-wb`): `{base}`, `{base}/form`,
  `{base}/run` (POST), `{base}/figure/{name}` (`?im=`) + the shared `_jobs_ui`
  routes at that base. (`/geoprior/figure/{name}` takes the same `?im=` and
  `tall=True`.)

**Core note — `ig.prior_model_smooth` (for the WB page):** signature default
`RHO_dist` is now `None`, resolved per branch (`'lognormal'` for L2 —
unchanged historical output; `'log-uniform'` for MGS — unchanged). Explicit
`RHO_dist='log-uniform'` / `'uniform'` on **L2** applies a Gaussian copula:
the correlated GP is kept, the per-layer marginal becomes (log-)uniform on
`[RHO_min, RHO_max]`. `corr_length <= 0` builds `C = sigma_logrho²·I`
(uncorrelated layers, no NaN). L1 (blocky) ignores `RHO_dist`. Covered by
`tests/test_prior_model_smooth.py` (18 pass; default L2 / MGS behaviour
asserted unchanged).

### geoprior1d (`pages/geoprior.py`, `/geoprior`) — IMPLEMENTED
- Split from module 02: geoprior1d is driven by an `.xlsx` geological spec, so
  the page is a spreadsheet editor + a small run form, not a parameter form.
- **Excel-spec editor** (`services/xlsx.py`, live `openpyxl.Workbook` held in
  `session` via `state_for`):
  - Picker: dropdown of workspace `*.xlsx` + **Load**.
  - `#gp-editor`: a `.~lock.<name>#` warning (if present), the **savebar**
    (`#gp-savebar`: a **SAVE TO** text field `#gp-target` — editable, so you
    can write the edits to a *different* file and leave the original alone —
    plus **Save**, **Reload original**, `#gp-status`, and a
    "(loaded from …)" hint when the target differs), and `#gp-grid`.
  - Session tracks `gp_file` (the on-disk source, re-read by *Reload
    original*) separately from `gp_target` (the write target). `_save_to`
    normalises the name (appends `.xlsx`), `safe_path`-confines it, writes,
    and repoints `gp_target`. A missing extension is added; blank ⇒ error.
  - `#gp-grid`: sheet tabs (`.seg`-style buttons, active = primary), an
    editable `<table class="table wb-sheet">` — each `<td>` an
    `<input name="v" class="cell">` that `POST`s `/geoprior/cell`
    (`hx-vals` = `{sheet,r,c}`, `hx-swap="none"`) → `xlsx.set_cell` (coerces
    against the old cell's type; strings like `"2,3,4,5"` stay strings),
    marks dirty, returns an OOB `#gp-status`. **+ Row** / **+ Col** mutate
    the sheet and re-render `#gp-grid`.
  - **Save to .xlsx** → `xlsx.save_book` to the loaded file (only changed
    cells differ; other sheets/formulas/styles untouched), clears dirty.
- **Run form** (`#gp-runform`): `Nreals` 100000, `dmax` 90, `dz` 1,
  `n_processes` -1, `Output .h5` (blank ⇒ auto), and a "Runs on `<gp_target>`"
  note. **Save & run geoprior1d** (`POST /geoprior/run`) always writes the
  current edits to `gp_target` first, then
  `api.start_geoprior_job({file_xlsx: <abs of gp_target>, …})` →
  `jobs.start("geoprior", worker.run_geoprior_job, …)` → `run_panel(job, "/geoprior")`.
- `done_extra` = geoprior `flags` + **Plot prior stats**
  (`GET /geoprior/figure/{name}`).
- **Live summary-stats preview** (`#gp-preview`, below the grid):
  - A **☐ Auto-update summary stats** checkbox + a **realizations** select
    (50 / 100 / 200 / 500 / 1000, capped at `_PREVIEW_MAX` = 2000) +
    **Refresh preview** button, all in `#gp-preview-form`.
  - Every `/geoprior/cell` `/addrow` `/addcol` response (and the toggle
    itself) carries an **`HX-Trigger`** header from `_changed(st)`:
    always `gp-cond` (drives the instant analytic ρ|lithology panel below),
    plus `gp-changed` when `gp_autopreview` is on (drives this geoprior1d
    preview). Two **persistent** listener `Div`s consume them —
    `#gp-preview-trigger` (`gp-changed from:body delay:800ms` → `POST
    /geoprior/preview`) and `#gp-cond-trigger` (`load, gp-cond from:body
    delay:400ms` → `POST /geoprior/cond`). `delay:` debounces (each event
    resets the timer). Persistent listeners + `HX-Trigger` were chosen over
    an OOB `Div` with `hx-trigger="load"` because htmx 2.0.4 does not
    reliably re-fire `load` on OOB-swapped content.
  - `POST /geoprior/preview` (async): saves the live workbook to a
    **scratch** `frontend/_scratch/gp_<sid>.xlsx` (never `gp_target`), then
    `await asyncio.to_thread` → `api.geoprior_preview_run` =
    `geoprior1d(..., n_processes=1, output_file=<scratch gp_<sid>.h5>)`
    **inline in a worker thread — no child process**. geoprior1d generation
    is <1 s for a few hundred realizations; the child-process path's ~2 s is
    almost all interpreter spawn + imports, so the preview skips it (the web
    process is `MainProcess` and `n_processes=1` starts no pool). A
    per-session `threading.Lock` serialises overlapping refreshes; an
    `#gp-preview-spin` `htmx-indicator` shows "Generating realizations…"
    while the POST is in flight.
  - The route returns the figure grid directly (no polling): two panels from
    `api.geoprior_preview_figure(<abs scratch .h5>, im, nr)` =
    `ig.plot_prior_stats(Mkey='M{im}', panels='reals', title='', hardcopy=False)`
    — just the **right-hand realizations panel** (= the Matlab GUI's
    "Lithostratigraphy" `/M2` and "Resistivity" `/M1` views). `im` 2 then 1;
    missing ones skipped. Laid out side by side in `.gp-preview-grid`
    (`1fr 1fr`, stacks < 760 px), each in a 300 px `object-fit:contain` box.
  - `dmax` / `dz` for the preview come from `#gp-runform` via `hx-include`.
    `POST /geoprior/preview/toggle` persists `gp_autopreview` / `gp_preview_n`
    and, when turned on, returns the trigger to fire an immediate run.
  - `/geoprior/load` + `/reload` call `_reset_preview` (cancel job, unlink
    scratch). Scratch dir = `config.SCRATCH_DIR` (`frontend/_scratch/`,
    gitignored) — **never the workspace**.
- **ρ | lithology conditional-prior panel** (`#gp-cond`, last block in
  `#gp-preview`) — the *assumed* resistivity prior per lithology, live on
  every Resistivity/Geology1 edit **regardless of the Auto-update toggle**
  (it is analytic, ~150 ms, runs in the web process):
  - `POST /geoprior/cond` saves the workbook to the same scratch `.xlsx`
    (under `_preview_lock` so it can't collide with a running preview save),
    then `api.cond_resistivity_figure(xlsx, h5_path=<scratch .h5 if it
    exists>, overlay=st['gp_cond_overlay'])`.
  - `cond_resistivity_figure` parses the spec with
    **`geoprior1d.io.extract_prior_info`** (medians, `res_unc` = log10(unc
    factor)/3, class names, RGB colours) and draws one `scipy.stats.norm`
    log-normal PDF per class on a shared **log-ρ** axis, coloured by the
    class RGB — matching the sampler's `10**(log10(res) + res_unc·N(0,1))`.
    With `overlay` + a scratch `.h5`, adds a `histtype='step'` density hist
    of `/M1` grouped by `/M2` per class. One `figures.render`-cached PNG
    (key salted on both file mtimes + the overlay flag). Returns `None` on
    any parse/render failure → the route shows an inline `error_box`.
  - `☐ overlay sampled` checkbox (`#gp-cond-form`) → `POST
    /geoprior/cond/toggle` stores `gp_cond_overlay`, replies
    `HX-Trigger: gp-cond` to redraw. Overlay on with no `.h5` yet → analytic
    only + a "run a preview" note.
  - `#gp-cond-trigger` also fires once on `load`, so the panel is populated
    as soon as the editor renders (and after every `/geoprior/load`
    `/reload`, which are normal swaps).
- Routes: `/geoprior`, `/geoprior/load` `/reload` `/cell` `/addrow` `/addcol`
  `/save` `/run` (POST), `/geoprior/sheet`, `/geoprior/figure/{name}`,
  `/geoprior/preview` `/geoprior/preview/toggle` `/geoprior/cond`
  `/geoprior/cond/toggle` (POST) + shared `_jobs_ui` routes at base
  `/geoprior`.
- Nav: `02 Prior model` is a **section label** with two children —
  `Generic` (`/prior`) and `geoprior1d` (`/geoprior`). `components.PAGES`
  entries are `(slug, num, label, children)` where `children` is
  `((slug, label), …)`; `leaf_slugs()` flattens them for route registration
  and tests.
- geoprior1d — `geoprior1d(input_data, Nreals, dmax, dz, doPlot=0,
  n_processes=-1, output_file=None) -> (name, flag_vector)`.

### 03 — Forward — TWO types, nav sub-group `(("forward","GA-AEM"),("forward-bh","Boreholes"))`

#### GA-AEM (`pages/forward.py`, `/forward`) — IMPLEMENTED
- **Essential** (always shown): `f_prior_h5` (`("h5",["PRIOR"])`), `file_sys`
  (`"system"` = workspace `*.gex` **+** `*.stm`), `im` (resistivity model
  index) 1.
- **`<details class="wb-details">` Advanced options**: `id` 1, `im_height` 0,
  `N` 0 (=all), `Nhank` 280, `Nfreq` 12, `Ncpu` 0, `showInfo` 0; checkboxes
  `parallel` (on), `doMakePriorCopy` (on), `is_log` (off).
- POST: `.stm` pick → `stmfiles=[<abs>]`, else `file_gex=<abs>`.
- **Run** (`POST /forward/run`) → `api.start_forward_job` →
  `worker.run_forward_job` → `ig.prior_data_gaaem(**kw, progress_callback=…)`
  → `run_panel(job, "/forward", out_key="f_prior_data_h5")`.

#### Boreholes (`pages/borehole.py`, `/forward-bh`) — IMPLEMENTED
- The B4 step of `examples/integrate_rawmaterial_daugaard.py`: append
  boreholes as extra jointly-invertible data types.
- **Essential**: `f_prior_h5` (`"priordata"` kind = `api.list_prior_with_data`,
  PRIOR files with `/D<n>` — also needs a discrete lithology model at
  `im_prior`), `f_data_h5` (DATA, for the XY grid), `borehole_json`
  (`"json"` = workspace `*.json`), `im_prior` 2.
- **Advanced**: `range_xyz` 300, `range_data` (blank), `showInfo` 0. Output
  name (blank ⇒ `<prior stem>_BH.h5`).
- **Run** (`POST /forward-bh/run`) → `api.start_borehole_job` →
  `worker.run_borehole_job`: `ig.copy_hdf5_file(src, out)` →
  `ig.read_borehole(json)` → `ig.save_borehole_data(out, f_data_h5, BH,
  im_prior=…, range_xyz=…, doPlot=False)` → returns `(id_prior_list,
  id_borehole_list)`; `run_panel(job, "/forward-bh",
  out_key="f_prior_data_bh_h5")`, `done_extra` lists the borehole `/D` ids
  (to paste into `id_use` on the Inversion page).
- **⚠ side effect:** `ig.save_borehole_data` step 3 writes the gridded
  borehole observations into **`f_data_h5` itself** (new `/D{n}` groups) — the
  DATA file is mutated in place, by design. Re-running appends more groups.

### 04 — Inversion (`pages/inversion.py`, `/inversion`) — IMPLEMENTED
- 2-column: `.wb-cols` at `1.7fr / 1fr` — form left, `#wb-run` right (the run
  panel fills it once a job starts).
- **Essential** (always shown), each on its own full-width `.wb-row`
  (`max-width:760px`) so long `.h5` names are readable: `f_prior_h5`
  (`api.list_prior_with_data` — **only PRIOR files that carry `/D<n>` forward
  data**; plain model-only priors can't be inverted), `f_data_h5` (DATA), then
  a narrow `N_use`.
  **`N_use` defaults to the selected prior's realization count**
  (`api.prior_n_realizations` = `/M1.shape[0]` from `files.summary`); the
  `f_prior_h5` `<select>` has `hx-get="/inversion/nuse"` on change → swaps the
  `#inv-nuse` field (`_nuse_field`) with the new default. Fallback 100000.
- **`<details class="wb-details">` Advanced options**: `nr` **100**; `f_post_h5`
  text (blank ⇒ auto); `id_use` (`"1,2"` → `api.parse_int_list`), `ip_range`
  (`"0:500"` → `api.parse_range`); `Ncpu` 0, `Nchunks` 0, `use_N_best` 0; a
  **Temperature** block = radios `autoT` `auto`|`fixed` (→ `autoT` 1/0) with the
  **`T_base` 1** field directly beneath them; `backend` numpy|jax; `parallel`
  checkbox (on). All still submit via `hx-include="closest form"`.
- **Run inversion** (`POST /inversion/run`) → `api.start_inversion_job(kw)`
  → `worker.run_rejection_job` calls
  `ig.integrate_rejection(**kw, progress_callback=…)` → `run_panel(job,
  "/inversion", out_key="f_post_h5")`.
- `ig.integrate_rejection(f_prior_h5, f_data_h5, f_post_h5='', N_use, id_use=[],
  ip_range=[], nr=1000, autoT=1, T_base=1, Nchunks=0, Ncpu=0, parallel=True,
  use_N_best=0, progress_callback=None, backend='numpy', **kwargs)` — returns
  the posterior `.h5` path.

### 05 — Results (`pages/results.py`, `/results`) — IMPLEMENTED
- POSTERIOR file dropdown (`hx-get /results/view` on change swaps `#results`).
- `api.posterior_kpis(name)` — from `files.summary` + a small `N_UNIQUE` read:
  KPI row (`stat_grid` cols=6) **Soundings / Realizations per sounding / Mean T
  / Mean EV / Mean N_UNIQUE / Inversion time**; plus a `data: … · prior: …`
  linked-files line.
- `api.posterior_table(name, limit=200)` — `ig.get_geometry` for LINE (falls
  back to the file's own `/LINE`), plus `/T` `/EV` `/N_UNIQUE` read directly →
  `#wb-cols` right column table `ip | LINE | T | EV | N_UNIQUE`.
- Left: figure form (`fn` select over `plot_profile` / `_continuous` /
  `_discrete` / `plot_T_EV` / `plot_post_stats`, + `im` / `i1` / `i2`) →
  `GET /results/figure` → `api.render_plot(fn, file, {...})` → PNG into
  `#results-fig`. `i2` 0 ⇒ dropped (⇒ all).

### 06 — Plotting — nav sub-group `(("plotting","Library"),("profile","Profile line"))`

#### Library (`pages/plotting.py`, `/plotting`) — IMPLEMENTED
- Source **segmented control** DATA / PRIOR / POSTERIOR (`hx-get /plotting/form`
  → swaps the `#plot-form` `<form>` outerHTML) → file dropdown filtered by
  class.
- Curated `PLOTS: {source: [Plot(fn, note, (Arg(name,label,default), …)), …]}`
  in `pages/plotting.py`. Selecting a function (also `hx-get /plotting/form`)
  rebuilds its **arg grid**. Started with: DATA — `plot_geometry`,
  `plot_data_xy`, `plot_data`; PRIOR — `plot_prior_stats`; POSTERIOR —
  `plot_T_EV`, `plot_profile`, `plot_post_stats`. Grow over time.
- **Render figure** → `GET /plotting/figure` reads `request.query_params`
  (source/fn/file + the plot's args, numeric-coerced) → `api.render_plot`.

#### Profile line (`pages/profile.py`, `/profile`) — IMPLEMENTED
- The **interactive-picking** feature (FRONTEND.md §2 ⭐ exception / §6.6a):
  the only client JS the frontend owns (`static/picker.js`, loaded from
  `app._HDRS`).
- `api.geometry_map(file, color)`: reads `ig.get_geometry` once (under
  `workspace.cwd()`), **sizes the figure to the survey aspect** (so
  equal-aspect doesn't leave a whitespace slab), then
  `figures.render_interactive(key, draw, figsize=…)` renders the scatter
  (X/Y by ELEVATION or LINE) with `layout="constrained"` and **no
  `bbox_inches`**, reads `ax.get_position()` + `get_xlim/ylim` and returns
  `{url, ax_l/ax_r/ax_b/ax_t (fig fractions, y from bottom), x0/x1/y0/y1
  (data), n}` — cached as a `<key>.json` beside the PNG.
- `_picker()` renders `.wb-picker` = a **top action bar** (Plot profile /
  Undo / Clear / point count / sounding count) + `.wb-picker-plot` (`<img>` +
  SVG overlay `viewBox="0 0 1000 1000" preserveAspectRatio="none"`), meta on
  `data-*` attrs. `picker.js` maps a click's img-fraction → UTM (inverse of
  the axes transform), appends a waypoint, redraws the SVG polyline/markers,
  writes JSON coords into the hidden `#profile-points` input. Re-inits on
  `htmx:afterSwap` (file / colour change re-renders the map).
- The profile result uses `figure_panel(..., tall=True)` → `.wb-figure-tall`
  (natural height, `max-height:80vh`, scrolls) so multi-panel `plot_profile`
  output isn't clipped by the 260px `.wb-figure` box.
- **`x-axis` select**: `auto` (default) / `x` / `y` / `index` (along the line)
  / `id`. `auto` → the handler compares the waypoints' X vs Y span and picks
  `x` (E–W line) or `y` (N–S line).
- **`tolerance` field** (m, default **10** — the `buffer` value from
  `integrate_rawmaterial_daugaard.py`): the band half-width around the drawn
  line. A wide tolerance grabs a swath of off-line soundings and smears the
  section, so keep it tight. `find_points_along_line_segments` returns indices
  in *file order*, but `plot_profile` `argsort`s `ii` by the chosen axis
  (`x`/`y`/`id`) internally, so the section comes out ordered.
- `POST /profile/plot` → `api.profile_along_line(file, points, im,
  gap_threshold, xaxis, tolerance)` → (under `workspace.cwd()`)
  `ig.get_geometry` + `ig.find_points_along_line_segments(X, Y, Xl, Yl,
  tolerance)` → `igp.plot_profile(file, ii=idx, im=…, xaxis=<resolved>, gap_threshold=…,
  hardcopy=False)` — the pattern from
  `examples/integrate_rawmaterial_daugaard.py` — PNG into `#profile-fig`,
  header shows the resolved `xaxis` + the sounding count.

### Shared: `render_plot`, `workspace.cwd()`, figure CWD
- `api.render_plot(fn, file, params)` = drop empty params, add
  `hardcopy=False`, `getattr(integrate.integrate_plot, fn)(<abs path>, **params)`
  via `figures.render` (hash-cached PNG).
- `figures.render` / `render_interactive` hold the matplotlib lock and
  `os.chdir(workspace)` for the call; `services/workspace.cwd()` is the shared
  context manager for any other `ig.*` call that must resolve `f5_data` /
  `f5_prior` linked files by their bare relative names (e.g. the
  `get_geometry` + `find_points_along_line_segments` in `profile_along_line`).
- `hardcopy=False` stops most `integrate_plot` functions writing their own
  `<name>_M*.png` into the workspace (some still do — see F4.3).

### Spreadsheet — DROPPED
The standalone Spreadsheet nav item is gone; the geoprior1d page already has
the same `services/xlsx` editor. `services/xlsx.py` stays (used by geoprior).

### + Query (`pages/query.py`, `/query`) — IMPLEMENTED
Mirrors `ui/backend/routers/query.py` / `streamlit/ig_query.py`.
- **Posterior picker**: `api.list_posteriors_with_prior()` — POSTERIOR files
  whose `f5_prior` link resolves in the workspace (`api.posterior_prior`).
- **LLM block**: `api.llm_env_config()` (mirror of `ui`'s `_env_llm`:
  `INTEGRATE_LLM_MODEL` › `ANTHROPIC_API_KEY` › `OLLAMA_API_KEY`). If the
  server has one, show it read-only. Else:
  - **Provider** select (OpenAI / Anthropic / Gemini / Groq / Mistral /
    DeepSeek / xAI / OpenRouter / Ollama / Other…) + **API key** (`password`,
    "sent per request, never stored server-side").
  - Changing the provider or key (or **Load models**) HTMX-POSTs
    `/query/provider-models` → `api.list_provider_models(provider, key)`
    (port of `ui`'s `_list_provider_models`: Ollama `/api/tags`, Anthropic
    `/v1/models`, Gemini `/v1beta/models`, else OpenAI-compat `{base}/models`)
    → swaps `#query-modelsel` with a `<select name="model">` whose **values are
    the full LiteLLM strings** (`<prefix><id>`, prefix from `_PROVIDERS`), or a
    text `model` input + the error on failure.
  - Ollama needs no key (host from `OLLAMA_API_BASE` / `OLLAMA_HOST`).
    "Other…" → plain full-id text field.
- **Prior models table** (`api.query_models`): `im | name | type | depth range
  | classes` from `ig.get_prior_model_info`, plus a `<details>` with
  `ig.prior_describe()` output.
- **Run query** (`POST /query/translate`, `async` → `asyncio.to_thread`):
  `api.query_translate` → `ig.query_from_text(text, prior, model, api_key)` →
  shows the **interpretation** + an **editable `query_json` textarea**.
  The button carries `hx-on::before-request` clearing `#query-out` (so the
  screen visibly "starts over") + `hx-indicator="#query-wait"`, which reveals a
  `.wb-wait` line — "● Request sent to the LLM — waiting for a reply…" — while
  the POST is in flight (`.htmx-request .wb-wait { display:block }`, pulsing
  dot via `@keyframes wb-pulse`). The Evaluate button has the same pair
  (`#query-eval-wait`, "Evaluating the query over the posterior…").
- **Ollama base-URL fix**: `ig._litellm_extra` forwards `OLLAMA_API_BASE` /
  `OLLAMA_HOST` to LiteLLM **unnormalised**, but `OLLAMA_HOST` is commonly a
  bare `host:port` (no scheme) → LiteLLM "Request URL is missing an 'http://'
  protocol". `api.query_translate` therefore forces a normalised
  `os.environ["OLLAMA_API_BASE"] = _ollama_host()` (adds `http://` if absent —
  the same value the model dropdown queried) for `model.startswith("ollama")`
  calls, saving the previous value (sentinel for "unset") and restoring it in
  `finally`.
- **Evaluate** (`POST /query/evaluate`): `api.query_evaluate` →
  `ig.query(post, query_dict)` → `ig.query_plot` (probability) /
  `ig.query_percentile_plot` (`"metric" in dict`) via `figures.render_all`
  (multi-figure) → figure(s) + `mean probability` / `percentiles` +
  `n_locations`.
- All `ig.*` calls run under `workspace.cwd()` (linked-file resolution).

---

## 8. `integrate` API quick reference

Verified against `integrate/__init__.py` and source on 2026-09-06.

```python
prior_model_layered(lay_dist='uniform', dz=1, z_max=90, NLAY_min=3, NLAY_max=6,
    NLAY_deg=6, RHO_dist='log-uniform', RHO_min=0.1, RHO_max=5000, RHO_mean=100,
    RHO_std=80, N=100000, save_sparse=True, RHO_threshold=0.001, **kwargs)

prior_model_workbench(N=100000, p=2, z1=0, z_max=100, dz=1, lay_dist='uniform',
    nlayers=0, NLAY_min=3, NLAY_max=6, NLAY_deg=5, RHO_dist='log-uniform',
    RHO_min=1, RHO_max=300, RHO_mean=180, RHO_std=80, chi2_deg=100,
    RHO_threshold=0.001, **kwargs)

prior_model_workbench_direct(N=100000, RHO_dist='log-uniform', z1=0, z_max=100,
    nlayers=0, p=2, NLAY_min=3, NLAY_max=6, RHO_min=1, RHO_max=300, RHO_mean=180,
    RHO_std=80, chi2_deg=100, RHO_threshold=0.001, **kwargs)

integrate_rejection(f_prior_h5='prior.h5', f_data_h5='...', f_post_h5='',
    N_use=..., id_use=[], ip_range=[], nr=1000, autoT=1, T_base=1, Nchunks=0,
    Ncpu=0, parallel=True, use_N_best=0, T_N_above=None, T_P_acc_level=None,
    progress_callback=None, console_progress=None, backend='numpy', **kwargs)
    # -> f_post_h5 path

forward_gaaem(C=np.array(()), ...)          # confirm full sig at integrate.py:968
prior_data_gaaem(...)                       # exported via integrate.integrate

# plotting (integrate.integrate_plot, also re-exported on ig)
plot_geometry, plot_profile, plot_profile_continuous, plot_profile_discrete,
plot_T_EV, plot_data_xy, plot_data, plot_prior_stats, plot_post_stats,
plot_feature_2d, plot_cumulative_probability_profile, plot_boreholes, ...
# plot_prior_stats(f_prior_h5, Mkey, nr=100, panels=('hist','stats','reals'),
#   fontsize=None, ...) — panels= selects a subset of the 1x3 layout;
#   'reals' alone = just the right-hand realizations panel (added for the
#   geoprior1d live preview).  plot_boreholes(..., fontsize=None).

integrate_posterior_stats(...)              # T / EV / N_UNIQUE series
get_geometry(f_h5)                          # UTMX/UTMY/LINE for maps

# query (deferred)
query_from_text, query, query_probability, query_percentile, query_plot,
query_percentile_plot, get_prior_model_info, prior_describe, query_test_llm
```

**Progress callback:** `progress_callback(current, total, info_dict)` where
`info_dict` optionally has `phase` and `status`. Some functions call it with 2
args — tolerate both.

**`MainProcess` guard:** `integrate_rejection` aborts unless
`multiprocessing.current_process().name == "MainProcess"`. The worker child
must set that name.

---

## 9. Vendored from `ui/backend/` (copy, then adapt)

| Source | Dest | Notes |
|--------|------|-------|
| `ui/backend/h5inspect.py` | `frontend/services/files.py` | `classify/tree/summary/_json_safe`. Drop FastAPI-isms; keep pure functions. |
| `ui/backend/workspace.py` | `frontend/services/workspace.py` | `get_workspace()`, `safe_path()` — use `WB_WORKSPACE` **or** `INTEGRATE_WORKSPACE` env, else CWD. |
| `ui/backend/worker.py` | `frontend/services/worker.py` | `_QueueWriter`, `run_rejection_job`. Add `run_prior_job`, `run_forward_job`. |
| `ui/backend/jobmanager.py` | inspiration for `frontend/services/jobs.py` | Its version is asyncio+WebSocket; **simplify** to thread-drains-queue + HTMX polling. |
| `streamlit/ig_progress.py` | phase-label dict in `services/jobs.py` | reuse `_PHASE_LABELS`. |

Do **not** add `ui` back to `[tool.setuptools] packages` dependence; frontend
must stand alone.

---

## 10. Packaging

Done in `pyproject.toml` (2026-09-08). The frontend's runtime deps were added
to the **main `dependencies`** list rather than an optional extra — `integrate`
already hard-depends on `litellm` / `streamlit` / `fastapi` / `uvicorn`, so an
`[frontend]` extra would have been the only optional group while every other
web dep is unconditional; keeping them together is less confusing.

```toml
[project]
dependencies = [ ..., "python-fasthtml", "openpyxl", "httpx" ]
# httpx: used directly by services/integrate_api.list_provider_models
#        (only transitively present before, via litellm)

[project.scripts]
integrate      = "frontend.app:main"            # `integrate` now launches THIS frontend
integrate_www  = "integrate.integrate_www_cli:main"   # old streamlit CLI, renamed

[tool.setuptools]
packages = [ ..., "frontend", "frontend.pages", "frontend.services" ]

[tool.setuptools.package-data]
frontend = ["static/**/*"]   # ship modernist.css / app.css / picker.js / htmx / fonts
```

The `integrate` console script points at `frontend.app:main` (argparse
`--host/--port/--workspace` → `uvicorn.run`); the previous streamlit launcher
is still installed, as `integrate_www`. `python -m frontend` works too
(`frontend/__main__.py`). `numpy` / `h5py` / `matplotlib` are already core
deps; `pytest` is already in `[project.optional-dependencies].dev`.

`frontend.app:main` — argparse `[--host 127.0.0.1] [--port 8051] [--workspace DIR]`,
resolves workspace (arg > `WB_WORKSPACE` > `INTEGRATE_WORKSPACE` > CWD), then
`uvicorn.run(app, ...)`. Run it from the folder holding the `.h5` files.

---

## 11. Dev workflow

```bash
# one-time
.venv/bin/pip install -e ".[frontend]"      # or: pip install python-fasthtml openpyxl

# run (from a data folder)
cd examples && python -m frontend --port 8051
# open http://127.0.0.1:8051

# tests
.venv/bin/pytest frontend/tests -q
```

Live reload during dev: `uvicorn frontend.app:app --reload --port 8051`
(set `WB_WORKSPACE=$PWD/examples`).

Test data: `examples/*.h5` — `DAUGAARD_AVG*.h5` (DATA/PRIOR),
`DAUGAARD_POSTERIOR*.h5` (POSTERIOR). `examples/daugaard_standard.xlsx` for the
sheet editor.

---

## 12. Conventions for future sessions

- **Update the checklist in §13 every session.** Status legend:
  `⬜ not started` · `🟡 impl, not tested` · `✅ impl + tested` · `⛔ blocked`.
  Each task has an **impl** and a **tested** cell.
- "Tested" = has an automated test in `frontend/tests/` **and** was run by the
  user in the browser at least once (note the date).
- **Reuse `integrate`, don't reimplement it** (§2 ⭐). Before writing any
  computation/IO/stats/plot logic in `frontend/`, find the `integrate`
  function that does it. Missing? Extend core or ask the user. The only
  code that legitimately originates in `frontend/` is HTTP/HTML plumbing,
  form-value coercion, job orchestration, figure-file caching, the small
  spreadsheet editor, and interactive-picking interaction (§6.6a).
- Keep `integrate` calls inside `services/integrate_api.py`. If you need a new
  core function, add a wrapper there, and list it in §8.
- Keep `modernist.css` unedited; put frontend-only rules in `app.css`.
- When a module's real behaviour diverges from §7, edit §7 — this file is the
  spec of record.
- Prefer many small HTMX partial routes over client JS.
- Commit granularity: one module (or one phase task) per commit. Commit
  message trailer per repo policy (Co-Authored-By + Claude-Session).

---

## 13. TODO checklist

Legend: **impl** / **tested** each ∈ `⬜ 🟡 ✅ ⛔`

### Phase 0 — Scaffolding
| ID | Task | impl | tested |
|----|------|------|--------|
| F0.1 | Create `frontend/` package tree (§4), empty modules with docstrings | ✅ | ✅ |
| F0.2 | Extract `modernist.css` from artifact into `static/` (Appendix A) + vendor 3 Archivo woff2 | ✅ | ✅ |
| F0.3 | Pin `static/htmx.min.js` (v2.0.4); add `static/app.css` (nav/layout chrome) | ✅ | ✅ |
| F0.4 | `frontend/.gitignore` for `static/figures/`, pycache | ✅ | ✅ |
| F0.5 | `pyproject.toml`: frontend runtime deps (`python-fasthtml` / `openpyxl` / `httpx`) added to main `dependencies`; `frontend*` packages + `static/**/*` package-data (§10). Run via `python -m frontend` (no console script) | ✅ | ✅ (`python -m pytest frontend/tests` green; `import fasthtml, openpyxl, httpx` ok) |

### Phase 1 — App skeleton
| ID | Task | impl | tested |
|----|------|------|--------|
| F1.1 | `config.py` (workspace resolution) + `services/workspace.py` (vendored) | ✅ | ✅ |
| F1.2 | `services/session.py` cookie-keyed dict | ✅ | 🟡 (not yet used by a page) |
| F1.3 | `app.py`: FastHTML app (`default_hdrs=False`, vendored htmx), static mount, `main()` | ✅ | ✅ |
| F1.4 | `components.py`: `shell/page_header/nav/btn/tag/field/text_input/select/seg/stat/stat_grid/eyebrow/figure_panel/error_box` | ✅ | ✅ |
| F1.5 | Nav renders from `components.PAGES`; active state; workspace footer | ✅ | ✅ |
| F1.6 | `tests/test_smoke.py` — boots, all nav routes 200, static assets, list partials | ✅ | ✅ (4 pass) |
| F1.7 | `pages/stubs.py` — placeholder pages for 02–06 + sheets + query | ✅ | ✅ |
| F1.8 | Working-folder strip (`components.workspace_bar` + `pages/workspace_ui.py`): shows current folder at top of every page; "Change…" → inline form with text field + parent/sub-folder quick-picks (● = contains .h5); process-wide, not persisted | ✅ | ✅ (browser: switch out and back on `examples/`) |

### Phase 2 — Services
| ID | Task | impl | tested |
|----|------|------|--------|
| F2.1 | `services/files.py` (vendored classify/tree/summary + `flat_datasets`) + `workspace.list_files` | ✅ | ✅ (via /files) |
| F2.2 | `services/figures.py` locked Agg→PNG + hash cache + prune | ✅ | ✅ (plot_geometry renders ~0.5s) |
| F2.3 | `services/jobs.py` — `Job` dataclass + `start(kind,target,params)` (spawn child) + daemon thread draining the queue + `get`/`cancel` + `PHASE_LABELS` | ✅ | ✅ (prior + geoprior jobs run/cancel on `examples/`) |
| F2.4 | `services/worker.py` — `_QueueWriter`, `_prep` (MPLBACKEND, chdir, `MainProcess`), `run_prior_job` (dispatch to `ig.prior_model_{layered,workbench,workbench_direct}`), `run_geoprior_job` (`geoprior1d`). `run_rejection_job`/`run_forward_job` TODO | ✅ | ✅ (prior + geoprior) |
| F2.5 | `services/integrate_api.py` — parsing helpers + module-01 wrappers + module-02 (`list_xlsx`, `start_prior_job`, `start_geoprior_job`, `prior_stats_figure`); grow per module | 🟡 | ✅ (01 + 02 parts) |
| F2.6 | `tests/test_integrate_api.py` against `examples/*.h5` (parsing + `file_detail` per class + `list_xlsx`) | ✅ | ✅ |

### Phase 3 — Modules
| ID | Task | impl | tested |
|----|------|------|--------|
| F3.1 | **01 Data files** — list, filter seg (by class), rescan, `ig.plot_geometry` figure on demand, datasets list | ✅ | ✅ (browser on `examples/`) |
| F3.1a | Inspect panel — **class-specific** stats + detail table (see §7.01): DATA = Soundings / Data types / Continuous / Discrete + per-`/D{i}` table (kind from `noise_model`); PRIOR = Realizations / Model types / Prior-data types + per-`/M{i}` table (Type + Dimension=`len(x)` + depth range + classes, via `ig.get_prior_model_info`); POSTERIOR = Soundings / Realizations-per-sounding (`/i_use` is [Np,Nr]) / Model types / Mean T / Mean EV + per-`/M{i}` posterior-stats table + Run details (linked files, `inv_time`, dates) | ✅ | ✅ (browser: DATA/PRIOR/POSTERIOR on `examples/`) |
| F3.2 | **02 Prior model** — **two pages** from one parametrised register: **Generic** (`/prior`, `layered`) and **WB** (`/prior-wb`, `smooth`/`blocky`/`sharp`); old `workbench`/`workbench_direct` dropped from the UI (core fns kept). Per-model field grid (HTMX-swapped `<form>`), Run → child job, self-polling run panel, Cancel + Clear, on done: output + "Open in Data files" + **Plot prior stats** (`im` picker default M1; `tall=True`). WB `smooth` exposes `RHO_dist` (L2 copula → log-uniform marginal) + `corr_length≤0` = i.i.d. | ✅ | ✅ (curl: `/prior-wb` page + `smooth`/`blocky`/`sharp` form swaps; `RHO_dist` on smooth not blocky; ran `smooth RHO_dist=log-uniform corr_length=0 N=800` → M1 ∈ [10,1000], lag-1 corr ≈ 0. `tests/test_prior_model_smooth.py` 18 pass) |
| F3.2b | **geoprior1d** (own page `/geoprior`, nav slot) — `.xlsx` picker + **live spreadsheet editor** (`services/xlsx.py`, session-held `openpyxl.Workbook`): sheet tabs, editable cells (`/geoprior/cell` coerces per old type, marks dirty), + Row / + Col, **editable SAVE TO field** (write edits to a different file, leave original alone), Save, **Reload original**, `.~lock` warning. Run form (`Nreals/dmax/dz/n_processes`) → **Save & run** (writes `gp_target` first) → `geoprior1d(...)` child-process job, shared run panel, flags + Plot prior stats | ✅ | ✅ (browser + curl: load daugaard_standard.xlsx, edit cell, **Save as `MY_EDITED_SPEC` → new file gets the edit, original B2 stays 30**, run N=250 → valid PRIOR .h5) |
| F3.2c | **geoprior1d live summary-stats preview** — ☐ Auto-update + realizations select (50–1000, cap 2000) + Refresh; while on, cell/+row/+col/toggle responses send `HX-Trigger: gp-changed` → a persistent `#gp-preview-trigger` (`gp-changed from:body delay:800ms`, debounced) → `POST /geoprior/preview` saves a **scratch** `.xlsx` (never `gp_target`) then `asyncio.to_thread`→`api.geoprior_preview_run` = `geoprior1d(n_processes=1, output_file=<scratch .h5>)` **inline (no child spawn)**, per-session lock, `#gp-preview-spin` indicator; returns two side-by-side panels (`.gp-preview-grid` 1fr 1fr, 300 px boxes) from `api.geoprior_preview_figure` = `plot_prior_stats(Mkey='M{im}', panels='reals')` for M2 then M1. `_reset_preview` unlinks scratch on load/reload. Needs core `plot_prior_stats(panels=…)`. | ✅ | ✅ (browser: toggle on → immediate render; cell edit → htmx log shows gp-changed→/geoprior/preview→both panels refresh; ~1.9 s for 200 reals) |
| F3.2d | **geoprior1d ρ\|lithology conditional-prior panel** (`#gp-cond`) — analytic overlaid log-normal ρ priors, one per lithology, coloured by spec RGB, on a log-ρ axis; `api.cond_resistivity_figure` parses via `geoprior1d.io.extract_prior_info`, draws `scipy.stats.norm` PDFs (σ = log10(unc)/3, matching the sampler). Live on **every** Resistivity/Geology1 edit via a second `HX-Trigger` event `gp-cond` (fires regardless of Auto-update) → persistent `#gp-cond-trigger` (`load, gp-cond from:body delay:400ms`) → `POST /geoprior/cond` (web-process, ~150 ms, `_preview_lock`-guarded save). ☐ `overlay sampled` (`/geoprior/cond/toggle` → `gp_cond_overlay`) adds `/M1`-by-`/M2` step-hists from the scratch preview `.h5` when one exists. Malformed sheet → inline `error_box`. | ✅ | ✅ (unit test `test_cond_resistivity_figure`: analytic + overlay + bad-file→None; browser: renders on load, redraws on a Resistivity median edit with Auto-update **off**; smoke test for `/geoprior/cond*` routes + `gp-cond` header) |
| F2.7 | `pages/_jobs_ui.py` — shared `register_job_routes` + `run_panel` (used by /prior + /geoprior) | ✅ | ✅ |
| F3.8 | `services/xlsx.py` + `tests/test_xlsx.py` (edit round-trip, type coercion, add row/col) | ✅ | ✅ |
| F3.3 | **04 Inversion** — 2-col layout; **3 essential fields** (PRIOR, DATA, N_use — default = prior's realization count, live-updated on prior change via `/inversion/nuse`) + collapsed `<details>` Advanced (nr=100, f_post_h5, id_use/ip_range, autoT/T_base, Ncpu/Nchunks/use_N_best, backend, parallel) → `ig.integrate_rejection` child job → shared run panel | ✅ | ✅ (curl + browser: N_use tracks the prior; id_use=1, ip_range=0:2 → POSTERIOR .h5) |
| F3.4 | **05 Results** — POSTERIOR picker, KPI row (`api.posterior_kpis`), figure form (`plot_profile`/`_continuous`/`_discrete`/`plot_T_EV`/`plot_post_stats` + im/i1/i2 → `api.render_plot`), per-sounding table (`api.posterior_table`) | ✅ | ✅ (browser: DAUGAARD_POSTERIOR.h5 KPIs + plot_profile render + table with LINE) |
| F3.5 | **03 Forward / GA-AEM** — 3 essential fields (prior, `.gex`/`.stm` system file, `im`) + collapsed `<details>` Advanced → `ig.prior_data_gaaem` child job | ✅ | ✅ (curl: N=30 → prior-data .h5) |
| F3.5b | **03 Forward / Boreholes** (`/forward-bh`, own nav sub-item) — prior-data + DATA + `.json` pickers, `im_prior`, advanced (`range_xyz`/`range_data`/`showInfo`) → `copy_hdf5_file` + `read_borehole` + `save_borehole_data` child job; done panel lists the new borehole `/D` ids. **Mutates `f_data_h5` in place** (by design). | ✅ | ✅ (curl: DAUGAARD_PRIOR_MERGED_DATA + daugaard_12boreholes.json → 12 `/D` ids) |
| F2.10 | `_forms` kinds `system` (gex+stm) / `json` / `priordata`; `api.list_system_files` / `list_json` / `list_prior_with_data` (PRIOR files with `/D<n>` forward data — used by Inversion + Borehole prior pickers) | ✅ | ✅ |
| F2.8 | `pages/_forms.py` — declarative `F(name,label,kind,default)` + `field_grid` + `cast` (kinds: int/float/str/bool/select/h5/gex) shared by forward + inversion | ✅ | ✅ |
| F3.6 | **06 Plotting / Library** — DATA/PRIOR/POSTERIOR seg → class-filtered file dropdown, curated `PLOTS` map, per-function arg grid, Render → `api.render_plot` | ✅ | ✅ (browser: plot_T_EV on POSTERIOR, plot_geometry on DATA) |
| F3.7 | **06 Plotting / Profile line** (`/profile`, own nav sub-item) — interactive: scatter map (`api.geometry_map` + `figures.render_interactive` + `static/picker.js` SVG overlay), click waypoints → `api.profile_along_line` (`find_points_along_line_segments` → `plot_profile(ii=…, xaxis='x')`) | ✅ | ✅ (browser: 3 clicks → 749 soundings → profile PNG; curl too) |
| F2.9 | `services/figures.render` / `render_interactive` — `os.chdir(workspace)` under lock; `workspace.cwd()` context manager for other `ig.*` calls (`profile_along_line`) so linked `f5_data`/`f5_prior` resolve | ✅ | ✅ |
| F2.11 | `figures.render_interactive(key, draw)` — fixed-size single-axes render + returns axes-box/data-limit meta (cached as `<key>.json`) for the browser click→data transform | ✅ | ✅ |
| F3.7 | **Spreadsheet** — file pick, sheet tabs, editable table, cell save, add row/col, save-to-xlsx | ⬜ | ⬜ |
| F3.8 | `services/xlsx.py` + `tests/test_xlsx.py` | ⬜ | ⬜ |
| F3.9 | **Query** — posterior+prior picker, LLM provider/key block + live model dropdown (or server-env), prior-models table, translate (`query_from_text`) → editable JSON → evaluate (`query`+`query_plot`); clears + shows a waiting indicator on run | ✅ | ✅ (full flow via Ollama `qwen3.8` → interpretation + JSON → evaluate → mean P=0.551 · 11693 locations + probability map) |

### Phase 4 — Polish
| ID | Task | impl | tested |
|----|------|------|--------|
| F4.1 | Cover/landing page (artboard 00) as `/about` or `/` hero strip | ⬜ | ⬜ |
| F4.2 | Error surface: friendly panel on wrapper/job exceptions + traceback toggle | ⬜ | ⬜ |
| F4.3 | Figure hygiene — hash-cached PNGs + prune done; `hardcopy=False` stops most `integrate_plot` hardcopies; `.~lock` + empty-workspace warnings done. **TODO:** `plot_prior_stats` / `plot_profile_continuous` still drop `<name>_M*.png` into the workspace | 🟡 | 🟡 |
| F4.4 | `frontend/README.md` (run instructions + implementation overview) | ✅ | ✅ |
| F4.5 | Full-run smoke: prior → forward → inversion → results on `examples/` | ⬜ | ⬜ |

### Phase 5 — Query tool (deferred; expand when scheduled)
| ID | Task | impl | tested |
|----|------|------|--------|
| F5.1 | LLM config discovery (`api.llm_env_config`) + browser model/key fields | ✅ | ✅ |
| F5.2 | Posterior select + prior-models table + `prior_describe` (`api.query_models`) | ✅ | ✅ |
| F5.3 | translate → editable JSON → evaluate pipeline (`figures.render_all` for multi-figure query plots) | ✅ | ✅ (end-to-end via Ollama: `ollama_chat/qwen3.8:latest` → interpretation + JSON → evaluate → mean P=0.551 · 11693 locations + probability map) |
| F5.4 | Percentile branch (`"metric" in query_dict` → `query_percentile_plot`) | ✅ | 🟡 |
| F5.5 | Provider picker + API-key field → live model **dropdown** (`api.list_provider_models`: Ollama `/api/tags`, Anthropic `/v1/models`, Gemini `/v1beta/models`, else OpenAI-compat `{base}/models`); dropdown values are full LiteLLM ids | ✅ | ✅ (Ollama: 11 models loaded live) |
| F5.6 | "Run query" clears `#query-out` + shows a pulsing "waiting for a reply" indicator (`hx-on::before-request` + `hx-indicator` / `.wb-wait`); same on Evaluate | ✅ | ✅ |
| F5.7 | Ollama fix: force normalised `OLLAMA_API_BASE` (scheme-prefixed) around `query_from_text` for `ollama*` models — works around bare `OLLAMA_HOST=host:port` in the env | ✅ | ✅ |

---

## Appendix A — extracting `modernist.css`

The design system CSS (≈10 KB, tokens + `@font-face` + component classes) and
the three Archivo `woff2` files are bundled in the artifact. To regenerate:

```python
# WebFetch https://claude.ai/code/artifact/a8fac822-53f7-4198-a2cb-6180b4ba5b76
# -> saves full bundled HTML to a local file (path printed in the tool result).
import json, base64, gzip, re
html = open("<saved-artifact-html>").read()
man = json.loads(re.search(r'<script type="__bundler/manifest">(.*?)</script>', html, re.S).group(1))
tmpl = json.loads(re.search(r'<script type="__bundler/template">(.*?)</script>', html, re.S).group(1))

# CSS: slice tmpl from ':root {' to the first '</style>'
a = tmpl.find(":root {"); css = tmpl[a:tmpl.find("</style>", a)]
open("frontend/static/modernist.css", "w").write(css)

# Fonts: manifest entries with mime 'font/woff2' -> write bytes to static/fonts/.
# The @font-face src url("<uuid>") in the CSS must be rewritten to the saved
# filenames, e.g. url("fonts/archivo-latin.woff2").
for uuid, e in man.items():
    if e["mime"] == "font/woff2":
        data = base64.b64decode(e["data"])
        if e.get("compressed"): data = gzip.decompress(data)
        open(f"frontend/static/fonts/{uuid}.woff2", "wb").write(data)
```

The three woff2 correspond to Google Fonts' `latin`, `latin-ext`,
`vietnamese` subsets of Archivo (weights 400/600/800 share files). Rename them
sensibly and fix the CSS `src` urls. Alternative (online only): replace the
`@font-face` block with
`<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;800&display=swap">`.

Extracted mockup HTML for the six artboards (field lists, layout) is
transcribed in §7 — you do not need the artifact to build the pages.
