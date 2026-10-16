"""
Run time of the EM forward models: fixed vs. variable flight height
====================================================================

Timing-only comparison of the three EM forward backends (``ga-aem``, ``simpeg``,
``anemone``) for the same set of N prior realizations, in two scenarios:

A. **tTEM, fixed flight height** (Daugaard): only the GEX file is needed.
B. **SkyTEM Diamond, variable flight height** (Samsoe): the GEX file and the
   system response (SR2) file are needed. Every prior realization gets its own
   Tx height (M4), as the data do.

For ``anemone`` and the variable-height scenario both modes are timed: the default
(one anemone setup per 1 m height bin) and ``rescale_height=True`` (one setup, kernel
rescaled to each model's height). ``anemone`` is also timed on ``cuda`` when a GPU
is available.

Nothing is compared except computation time. Each run is wall-clock time of
``ig.prior_data_em`` and INCLUDES the one-time setup of the backend (STM files for
ga-aem, anemone ``Forward`` setup per height); a short untimed warm-up run first
removes library start-up cost (imports, torch initialisation).

The result is two tables: one line per run, and a summary with ms per sounding
for every backend in both scenarios, the fastest marked ``*``.

Run time with N=10000 on a CPU-only machine is several minutes (simpeg and anemone with
height bins are the slow ones). Lower ``N`` for a quick test; with small N the one-time
setup dominates.
"""
# %%
import contextlib
import os
import time

import numpy as np

import integrate as ig

N = 10_000            # prior realizations to forward model
N_WARM = 20           # untimed warm-up realizations
HEIGHT_RANGE = (20.0, 50.0)   # [m] Tx height of the variable-height scenario (uniform)
SAVE_TABLE = True     # write forward_timing_table.csv

# %% Input files
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:       # interactive cell execution
    HERE = os.getcwd()


def find_file(name, folders):
    """First existing HERE/<folder>/<name>."""
    for folder in folders:
        f = os.path.join(HERE, folder, name)
        if os.path.exists(f):
            return f
    raise FileNotFoundError('%s not found in %s' % (name, [os.path.join(HERE, x) for x in folders]))


# Daugaard tTEM
F_GEX_TTEM = find_file('TX07_20231016_2x4_RC20-33.gex', ['.', 'gallery/45_forward'])

# Samsoe SkyTEM Diamond: GEX (2-point dummy waveform) + system response
F_GEX_SKY = find_file('20251114_DEN_Diamond_Samsoe_PROTECT_SR.gex', ['SAMSOE', '.'])
F_SR2_SKY = find_file('20251114_DEN_Diamond_Samsoe_PROTECT_SR.sr2', ['SAMSOE', '.'])

OUT = os.path.join(HERE, 'forward_timing_output')
os.makedirs(OUT, exist_ok=True)

# %% Backends to time
# ``variable_only``: only meaningful when the flight height varies
configs = []
configs.append(dict(label='ga-aem (cpu)', method='ga-aem', device='cpu', kw={}, variable_only=False))
configs.append(dict(label='simpeg (cpu)', method='simpeg', device='cpu', kw={}, variable_only=False))
configs.append(dict(label='anemone (cpu)', method='anemone', device='cpu', kw={}, variable_only=False))
configs.append(dict(label='anemone (cpu, rescale_height)', method='anemone', device='cpu',
                    kw=dict(rescale_height=True), variable_only=True))
try:
    import torch
    if torch.cuda.is_available():
        configs.append(dict(label='anemone (cuda)', method='anemone', device='cuda',
                            kw={}, variable_only=False))
        configs.append(dict(label='anemone (cuda, rescale_height)', method='anemone', device='cuda',
                            kw=dict(rescale_height=True), variable_only=True))
except ImportError:
    pass

# %% Prior: the same N layered models for both scenarios
f_prior_h5 = ig.prior_model_layered(N=N, lay_dist='uniform', NLAY_min=1, NLAY_max=5,
                                    RHO_min=1, RHO_max=3000, dz=1, z_max=149,
                                    f_prior_h5=os.path.join(OUT, 'PRIOR_N%d.h5' % N), showInfo=0)

# Tx height of the variable-height scenario in M4 (M2, M3 are used by the layered prior)
rng = np.random.default_rng(1)
tx_height = rng.uniform(HEIGHT_RANGE[0], HEIGHT_RANGE[1], N)
ig.save_prior_model(f_prior_h5, tx_height, im=4, name='Tx_height', force_replace=True, showInfo=0)

scenarios = []
scenarios.append(dict(name='A: tTEM, fixed height', short='fixed', file_gex=F_GEX_TTEM,
                      kw={}, variable=False))
scenarios.append(dict(name='B: SkyTEM, variable height', short='variable', file_gex=F_GEX_SKY,
                      kw=dict(file_sr2=F_SR2_SKY, im_height=4), variable=True))


# %% Timing
def clear_anemone_cache():
    """Make every anemone run pay for its own setup."""
    try:
        from integrate import anemone_forward
        anemone_forward._clear_forward_cache()
    except ImportError:
        pass


def run_once(sc, cfg, n, f_out):
    kw = dict(sc['kw'])
    kw.update(cfg['kw'])
    t0 = time.perf_counter()
    with open(os.devnull, 'w') as devnull:          # the backends print progress
        with contextlib.redirect_stdout(devnull):
            ig.prior_data_em(f_prior_h5, sc['file_gex'], method=cfg['method'],
                             device=cfg['device'], N=n, doMakePriorCopy=True,
                             f_prior_data_h5=f_out, force_replace=True,
                             showInfo=-1, **kw)
    return time.perf_counter() - t0


results = []     # one dict per timed run
for sc in scenarios:
    for cfg in configs:
        if cfg['variable_only'] and not sc['variable']:
            continue
        f_out = os.path.join(OUT, 'TIMING_%s_%s_%s.h5' % (
            sc['short'], cfg['method'], cfg['device']))
        row = dict(scenario=sc['name'], label=cfg['label'], time=np.nan, error='')
        print('%-28s %-32s ...' % (sc['name'], cfg['label']), end=' ', flush=True)
        try:
            run_once(sc, cfg, N_WARM, f_out)       # warm-up, not timed
            clear_anemone_cache()
            row['time'] = run_once(sc, cfg, N, f_out)
            print('%8.1f s' % row['time'])
        except Exception as exc:                   # backend not installed, etc.
            row['error'] = '%s: %s' % (type(exc).__name__, str(exc).split('\n')[0][:80])
            print('FAILED (%s)' % row['error'])
        if os.path.exists(f_out):
            os.remove(f_out)
        results.append(row)

# %% Tables
print('\nForward timing, N = %d prior realizations (setup included)' % N)
print('%-28s %-32s %10s %12s %9s' % ('scenario', 'forward', 'time [s]', 'ms/sounding', 'relative'))
for sc in scenarios:
    rows = []
    for r in results:
        if r['scenario'] == sc['name']:
            rows.append(r)
    t_ok = []
    for r in rows:
        if np.isfinite(r['time']):
            t_ok.append(r['time'])
    t_best = min(t_ok) if len(t_ok) > 0 else np.nan
    for r in rows:
        if np.isfinite(r['time']):
            print('%-28s %-32s %10.1f %12.2f %8.1fx%s' % (
                r['scenario'], r['label'], r['time'], 1000 * r['time'] / N,
                r['time'] / t_best, '  <- fastest' if r['time'] == t_best else ''))
        else:
            print('%-28s %-32s %10s %12s %9s  %s' % (
                r['scenario'], r['label'], '-', '-', '-', r['error']))

# Summary: ms per sounding, one column per scenario, fastest marked with *
labels = []
for cfg in configs:
    labels.append(cfg['label'])
best = {}
for sc in scenarios:
    t_ok = []
    for r in results:
        if r['scenario'] == sc['name'] and np.isfinite(r['time']):
            t_ok.append(r['time'])
    best[sc['name']] = min(t_ok) if len(t_ok) > 0 else np.nan

print('\nSummary, ms per sounding (* = fastest in the scenario, - = not applicable/failed)')
line = '%-34s' % 'forward'
for sc in scenarios:
    line += '%30s' % sc['name']
print(line)
csv_lines = ['forward,' + ','.join([s['short'] + '_s' for s in scenarios])]
for lab in labels:
    line = '%-34s' % lab
    csv = lab
    for sc in scenarios:
        val = np.nan
        for r in results:
            if r['scenario'] == sc['name'] and r['label'] == lab:
                val = r['time']
        if np.isfinite(val):
            mark = '*' if val == best[sc['name']] else ' '
            line += '%29.2f%s' % (1000 * val / N, mark)
            csv += ',%.3f' % val
        else:
            line += '%30s' % '-'
            csv += ','
    print(line)
    csv_lines.append(csv)

print('')
for sc in scenarios:
    winner = ''
    for r in results:
        if r['scenario'] == sc['name'] and r['time'] == best[sc['name']]:
            winner = r['label']
    print('Fastest for %-28s: %s' % (sc['name'], winner))

if SAVE_TABLE:
    f_csv = os.path.join(OUT, 'forward_timing_table.csv')
    with open(f_csv, 'w') as f:
        f.write('\n'.join(csv_lines) + '\n')
    print('\nwrote %s (total seconds for N=%d)' % (f_csv, N))

# %%
