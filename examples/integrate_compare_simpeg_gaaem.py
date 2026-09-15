#!/usr/bin/env python
# %% [markdown]
# # SimPEG vs GA-AEM (vs AarhusInv) forward comparison
#
# Validates the SimPEG backend (`ig.forward_simpeg` / `method='simpeg'`) against
# GA-AEM and, where available, AarhusInv on the Daugaard tTEM case:
#
#   1. Forward the AarhusInv inversion models (`*_inv.xyz`, one flight line)
#      with SimPEG and GA-AEM and compare gate-by-gate, and both against the
#      AarhusInv synthetic response (`*_syn.xyz`).
#   2. Sensitivity of the SimPEG result to its numerical knobs
#      (`gate_integration`, `hankel_filter`, `n_points_per_path`, filters).
#   3. Timing per sounding for 30 / 60 / 90 layers, single core and pool.
#
# Numbers from this script are recorded in `SIMPEG_VS_GAAEM.md`.
# Lower `N_SOUND` for a quicker run (SimPEG is ~0.2-0.5 s per sounding per core).

# %%
import os
import time

import numpy as np
import integrate as ig

try:
    from integrate.simpeg_forward import _require_simpeg
    _require_simpeg()
except ImportError as e:
    raise SystemExit(str(e))

DUMMY = 9999.0
N_SOUND = 40          # soundings from the inversion line used in comparison 1
N_TIME = 4 * (os.cpu_count() or 1)   # soundings per timing run (>= 2*NCPU so the pool is used)
NCPU = os.cpu_count() or 1
hardcopy = True

# %% Data
case = 'DAUGAARD'
files = ig.get_case_data(case=case, showInfo=1)
file_gex = ig.get_gex_file_from_data(files[0])


def _one(x):
    return x[0] if isinstance(x, (list, tuple)) else x


file_xyz_inv = _one(ig.get_case_data(case=case, filelist=['SCI7_40_ml_sharp2_I02_MOD_inv.xyz']))
file_xyz_syn = _one(ig.get_case_data(case=case, filelist=['SCI7_40_ml_sharp2_I02_MOD_syn.xyz']))
print('gex:', file_gex)

# %% [markdown]
# ## 1. AarhusInv inversion models: SimPEG vs GA-AEM vs AarhusInv

# %%
import libaarhusxyz

inv = libaarhusxyz.XYZ(file_xyz_inv)
fl = inv.flightlines
rho = inv.layer_data['rho'].values.astype(float)
dep_top = inv.layer_data['dep_top'].values.astype(float)
z = dep_top[0]
thickness = np.diff(z)
line_inv = fl['line_no'].values.astype(np.int64)
rec_inv = fl['record'].values.astype(np.int64)
main_line = np.bincount(line_inv).argmax()
sel = np.where(line_inv == main_line)[0]
sel = sel[np.linspace(0, sel.size - 1, min(N_SOUND, sel.size)).astype(int)]
M = rho[sel]
print('inv: %d soundings on line %d, %d layers' % (sel.size, main_line, M.shape[1]))

syn = libaarhusxyz.XYZ(file_xyz_syn)
sfl = syn.flightlines
sdata = syn.layer_data['data'].values.astype(float)
sseg = sfl['segments'].values.astype(int)
skey = sfl['line_no'].values.astype(np.int64) * (sfl['record'].values.max() + 1) + sfl['record'].values.astype(np.int64)
gate_t = np.asarray(syn.model_info['gate times (s)'], dtype=float)
lm_row = {k: i for i, k in zip(np.where(sseg == 1)[0], skey[sseg == 1])}
hm_row = {k: i for i, k in zip(np.where(sseg == 2)[0], skey[sseg == 2])}
sel_key = line_inv[sel] * (sfl['record'].values.max() + 1) + rec_inv[sel]

system = ig.gex_to_em_system(file_gex, showInfo=1)
t_lm = system['moments'][0]['gate_centre']
t_hm = system['moments'][1]['gate_centre']
n_lm, n_hm = t_lm.size, t_hm.size
sl_lm, sl_hm = slice(0, n_lm), slice(n_lm, n_lm + n_hm)

t0 = time.time()
D_sp = ig.forward_simpeg(M, thickness, GEX=system, Ncpu=NCPU, showInfo=1)
t_sp = time.time() - t0
t0 = time.time()
D_ga = ig.forward_gaaem(C=1.0 / M, thickness=thickness, file_gex=file_gex, parallel=False)
t_ga = time.time() - t0
print('forward: simpeg %.1fs (%d cores)   ga-aem %.1fs (1 core)' % (t_sp, NCPU, t_ga))


def _syn_gates(row_idx):
    r = sdata[row_idx]
    m = r != DUMMY
    return gate_t[m], r[m]


def _loglog(t_dst, t_src, y_src):
    return 10.0 ** np.interp(np.log10(t_dst), np.log10(t_src), np.log10(np.abs(y_src)))


def stats(name, r):
    r = np.abs(np.asarray(r, dtype=float))
    r = r[np.isfinite(r)]
    print('  %-8s n=%5d   median = %6.2f %%   90th pct = %6.2f %%   max = %6.2f %%'
          % (name, r.size, 100 * np.median(r), 100 * np.percentile(r, 90), 100 * r.max()))
    return np.median(r), np.percentile(r, 90), r.max()


rel_sg = (D_sp - D_ga) / D_ga
print('\nSimPEG vs GA-AEM  |rel diff| on the GEX gate grid:')
stats('LM', rel_sg[:, sl_lm])
stats('HM', rel_sg[:, sl_hm])
print('  per-gate median |rel diff| LM:', np.round(100 * np.median(np.abs(rel_sg[:, sl_lm]), axis=0), 2))
print('  per-gate median |rel diff| HM:', np.round(100 * np.median(np.abs(rel_sg[:, sl_hm]), axis=0), 2))

r_sp_lm, r_sp_hm, r_ga_lm, r_ga_hm = [], [], [], []
for j, key in enumerate(sel_key):
    for row, t_gex, sl, r_sp, r_ga in ((lm_row, t_lm, sl_lm, r_sp_lm, r_ga_lm),
                                       (hm_row, t_hm, sl_hm, r_sp_hm, r_ga_hm)):
        if key not in row:
            continue
        t_s, a_s = _syn_gates(row[key])
        if t_s.size < 2:
            continue
        r_sp.extend((_loglog(t_s, t_gex, D_sp[j, sl]) - a_s) / a_s)
        r_ga.extend((_loglog(t_s, t_gex, D_ga[j, sl]) - a_s) / a_s)
print('\nSimPEG vs AarhusInv |rel diff| (log-log resampled to the _syn gates):')
stats('LM', r_sp_lm)
stats('HM', r_sp_hm)
print('GA-AEM vs AarhusInv |rel diff|:')
stats('LM', r_ga_lm)
stats('HM', r_ga_hm)

# %% overlay figure
try:
    import matplotlib.pyplot as plt
    t_all = np.r_[t_lm, t_hm]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for j in np.linspace(0, sel.size - 1, 5).astype(int):
        ax.loglog(t_all, np.abs(D_ga[j]), '-', color='C0', lw=1)
        ax.loglog(t_all, np.abs(D_sp[j]), '--', color='C3', lw=1)
        key = sel_key[j]
        for row in (lm_row, hm_row):
            if key in row:
                t_s, a_s = _syn_gates(row[key])
                ax.loglog(t_s, np.abs(a_s), 'ok', ms=3)
    ax.plot([], [], '-', color='C0', label='ga-aem')
    ax.plot([], [], '--', color='C3', label='simpeg')
    ax.plot([], [], 'ok', label='AarhusInv')
    ax.set_xlabel('time [s]')
    ax.set_ylabel('|dB/dt| [V/Am$^4$]')
    ax.set_title('%s line %d -- SimPEG vs GA-AEM vs AarhusInv' % (case, main_line))
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    if hardcopy:
        fig.savefig('%s_compare_simpeg_gaaem.png' % case, dpi=140)
        print('wrote %s_compare_simpeg_gaaem.png' % case)
except Exception as e:  # noqa: BLE001
    print('figure skipped:', e)

# %% [markdown]
# ## 2. Sensitivity to SimPEG numerical options (vs the default run)

# %%
M_s = M[:8]
D_ref = ig.forward_simpeg(M_s, thickness, GEX=system, parallel=False)
print('\nSensitivity (|rel diff| vs default bfield / key_101_2009 / n_points_per_path=2):')
for label, kw in (('gl K=9', dict(gate_integration='gl', gate_quad=9)),
                  ('centre', dict(gate_integration='centre')),
                  ('key_51_2012', dict(hankel_filter='key_51_2012')),
                  ('key_201_2012', dict(hankel_filter='key_201_2012')),
                  ('npp=1', dict(n_points_per_path=1)),
                  ('npp=3', dict(n_points_per_path=3)),
                  ('no filters', dict(apply_filters=False))):
    D = ig.forward_simpeg(M_s, thickness, GEX=system, parallel=False, **kw)
    r = np.abs((D - D_ref) / D_ref)
    print('  %-12s LM median %7.4f %%  max %7.3f %% | HM median %7.4f %%  max %7.3f %%'
          % (label, 100 * np.median(r[:, sl_lm]), 100 * r[:, sl_lm].max(),
             100 * np.median(r[:, sl_hm]), 100 * r[:, sl_hm].max()))

# %% [markdown]
# ## 3. Timing

# %%
print('\nTiming (ms per sounding, excluding the one-off ~2 s coefficient setup):')
rng = np.random.default_rng(0)
for nl in (30, 60, 90):
    thk = np.full(nl - 1, 1.0)
    Mt = 10 ** rng.uniform(0.5, 2.5, size=(N_TIME, nl))
    t0 = time.time()
    ig.forward_simpeg(Mt[:1], thk, GEX=system, parallel=False)
    setup = time.time() - t0
    t0 = time.time()
    ig.forward_simpeg(Mt, thk, GEX=system, parallel=False)
    seq = (time.time() - t0 - setup) / N_TIME
    t0 = time.time()
    ig.forward_simpeg(Mt, thk, GEX=system, parallel=True, Ncpu=NCPU)
    par = (time.time() - t0 - setup) / N_TIME
    t0 = time.time()
    ig.forward_gaaem(C=1.0 / Mt, thickness=thk, file_gex=file_gex, parallel=False)
    ga = (time.time() - t0) / N_TIME
    print('  nl=%2d: simpeg 1 core %6.0f ms | simpeg %d cores %6.0f ms | ga-aem 1 core %5.1f ms'
          % (nl, 1000 * seq, NCPU, 1000 * max(par, 0), 1000 * ga))
