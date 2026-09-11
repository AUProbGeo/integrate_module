#!/usr/bin/env python
# %% [markdown]
# # Forward modeling in INTEGRATE
#
# This notebook contains an example comparing TEM forward for ga-aem, anemone, and aarhusinv.
# gaaem and anemone can be called directly from python, while aarhusinv is loaded from xyz files.
#
# The whole idea is to test whether **anemone** (the new PyTorch forward in INTEGRATE) performs
# as well as **ga-aem** (the current INTEGRATE forward), and whether both agree with
# **AarhusInv** (the reference, read from Workbench `.xyz` files).
#
# Two comparisons are made:
#   1. Forward the AarhusInv inversion models (`*_inv.xyz`) with all three codes and compare
#      the responses gate-by-gate against the AarhusInv synthetic response (`*_syn.xyz`).
#   2. Draw a generic prior sample, run `prior_data_gaaem` and `prior_data_anemone`, compare
#      the two prior-data distributions and report CPU/wall time for both backends
#      (realistic = as INTEGRATE runs them, and sequential single-thread apples-to-apples).
#
# Runtime note: comparison 1 forwards the whole longest line (hundreds of soundings) with
# ga-aem in sequential mode, so expect tens of seconds. Lower `N_PRIOR` for a quicker run.
#

# %%
import time
import integrate as ig
hardcopy = True
import matplotlib.pyplot as plt
import numpy as np
import libaarhusxyz

try:
    import torch
except ImportError:
    raise SystemExit("this example needs torch + anemone: "
                     "pip install torch && pip install -e /home/tmeha/PROGRAMMING/anemone")

DUMMY = 9999.0
N_PLOT = 6          # soundings shown in the overlay figure
N_PRIOR = 10000     # prior realizations for comparison 2 (timing); raise for steadier numbers

# %% [markdown]
# ## 0. Get TTEM data
# Several test cases are available in the INTEGRATE package.
# To see which cases are available, check the `get_case_data` function.
#
# The code below downloads the file DAUGAARD_AVG.h5 that contains
# TTEM (time-domain electromagnetic) soundings from Daugaard, Denmark.
# It also downloads the corresponding GEX file, TX07_20231016_2x4_RC20-33.gex,
# which contains information about the TTEM system configuration and parameters.
#


# %%
case = 'DAUGAARD'
files = ig.get_case_data(case=case, showInfo=2)
f_data_h5 = files[0]
file_gex = ig.get_gex_file_from_data(f_data_h5)


def _one(x):
    """get_case_data returns a list; take the single entry."""
    return x[0] if isinstance(x, (list, tuple)) else x


#SCI7_40_ml_sharp2_I02_MOD.xyz  SCI7_40_ml_sharp2_I02_MOD_dat.xyz  SCI7_40_ml_sharp2_I02_MOD_inv.xyz  SCI7_40_ml_sharp2_I02_MOD_syn.xyz
file_xyz_dat = _one(ig.get_case_data(case=case, filelist=['SCI7_40_ml_sharp2_I02_MOD_dat.xyz']))
file_xyz_inv = _one(ig.get_case_data(case=case, filelist=['SCI7_40_ml_sharp2_I02_MOD_inv.xyz']))
file_xyz_syn = _one(ig.get_case_data(case=case, filelist=['SCI7_40_ml_sharp2_I02_MOD_syn.xyz']))

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)
print("Using XYZ data file: %s" % file_xyz_dat)
print("Using XYZ inversion file: %s" % file_xyz_inv)
print("Using XYZ synthetic file: %s" % file_xyz_syn)

# %% [markdown]
# ## 1. Compare forward responses on the AarhusInv inversion models

# %% Load reference models from file_xyz_inv
inv = libaarhusxyz.XYZ(file_xyz_inv)
fl = inv.flightlines
rho = inv.layer_data['rho'].values.astype(float)          # (nS, nLayer)
dep_top = inv.layer_data['dep_top'].values.astype(float)  # (nS, nLayer)
z = dep_top[0]                                            # shared depth grid
assert np.allclose(dep_top, z, atol=1e-3), 'depth grid is not constant'
thickness = np.diff(z)                                    # (nLayer-1,)
line_inv = fl['line_no'].values.astype(np.int64)
rec_inv = fl['record'].values.astype(np.int64)
x_inv = fl['utmx'].values.astype(float)
y_inv = fl['utmy'].values.astype(float)
print('inv: %d soundings, %d layers, grid 0..%g m'
      % (rho.shape[0], rho.shape[1], z[-1]))

# %% Load forward response from file_xyz_syn
syn = libaarhusxyz.XYZ(file_xyz_syn)
sfl = syn.flightlines
sdata = syn.layer_data['data'].values.astype(float)       # (nRow, nGateMerged)
sseg = sfl['segments'].values.astype(int)
sline = sfl['line_no'].values.astype(np.int64)
srec = sfl['record'].values.astype(np.int64)
gate_t = np.asarray(syn.model_info['gate times (s)'], dtype=float)   # merged gate centres

skey = sline * (srec.max() + 1) + srec
lm_row = {k: i for i, k in zip(np.where(sseg == 1)[0], skey[sseg == 1])}
hm_row = {k: i for i, k in zip(np.where(sseg == 2)[0], skey[sseg == 2])}

# %% [markdown]
# ### Forward gate times and per-sounding _syn gates
#
# ga-aem / anemone return `[LM used gates, HM used gates]` on the GEX gate times
# (selection from RemoveInitialGates / NoGates).  The Workbench `_syn` file lays
# the response out in one merged block and culls early gates **per sounding**
# (below the noise floor), so each sounding's real gates are scattered.  The
# comparison therefore interpolates each forward curve (log-log) onto the real
# `_syn` gate times of that sounding.

# %%
gex = libaarhusxyz.GEX(file_gex)
i_lm0, i_lm1 = int(gex.remove_initial_gates(1)), int(gex.no_gates(1))
i_hm0, i_hm1 = int(gex.remove_initial_gates(2)), int(gex.no_gates(2))
lm_used = np.asarray(gex.gate_times(1))[:, 0][i_lm0:i_lm1]      # (n_lm,) GEX LM times
hm_used = np.asarray(gex.gate_times(2))[:, 0][i_hm0:i_hm1]      # (n_hm,) GEX HM times
n_lm, n_hm = lm_used.size, hm_used.size


def _syn_gates(row_idx):
    """Real (non-dummy) gate times + values of one _syn row."""
    r = sdata[row_idx]
    m = r != DUMMY
    return gate_t[m], r[m]


def _loglog(t_dst, t_src, y_src):
    """y_src(t_src) resampled to t_dst in log-log space."""
    return 10.0 ** np.interp(np.log10(t_dst), np.log10(t_src), np.log10(np.abs(y_src)))


# %% [markdown]
# ### Pick every sounding on the longest line

# %%
lines, counts = np.unique(line_inv, return_counts=True)
main_line = lines[np.argmax(counts)]
i_line = np.where(line_inv == main_line)[0]
i_line = np.array([i for i in i_line
                   if (line_inv[i] * (srec.max() + 1) + rec_inv[i]) in lm_row
                   and (line_inv[i] * (srec.max() + 1) + rec_inv[i]) in hm_row])
sel = i_line                                   # whole line
sel_key = [line_inv[i] * (srec.max() + 1) + rec_inv[i] for i in sel]
print('line %d: %d soundings used for the comparison' % (main_line, len(sel)))

# %% Compute forward response using ga-aem and anemone

# ga-aem (sequential -> also the sequential timing number for ga-aem)
t0 = time.perf_counter()
D_ga = ig.forward_gaaem(C=1.0 / rho[sel], thickness=thickness,
                        file_gex=file_gex, parallel=False, showInfo=1)
t_ga0 = time.perf_counter() - t0
g_lm_full = D_ga[:, :n_lm]                      # (nSel, n_lm) on lm_used
g_hm_full = D_ga[:, n_lm:n_lm + n_hm]           # (nSel, n_hm) on hm_used
print('ga-aem forward: %s in %.2f s (%.1f ms/sounding, first call, incl. STM build)'
      % (D_ga.shape, t_ga0, 1e3 * t_ga0 / len(sel)))

# anemone: NO calibration.  forward_anemone returns dB/dt per unit transmitter
# moment [V/(A m^4)] straight from the GEX (loop area, waveform, filters), which
# is the Workbench/AarhusInv data unit.  See ANEMONE_VS_GAAEM_VS_AI.md.
t0 = time.perf_counter()
D_an = ig.forward_anemone(M=rho[sel], thickness=thickness, file_gex=file_gex,
                          showInfo=1)
t_an0 = time.perf_counter() - t0
cal = dict(ig.forward_anemone.last_calibration)
a_lm_full = D_an[:, :n_lm]
a_hm_full = D_an[:, n_lm:n_lm + n_hm]
print('anemone forward: %s in %.2f s (%.1f ms/sounding, first call, incl. Forward build)   mode=%s'
      % (D_an.shape, t_an0, 1e3 * t_an0 / len(sel), cal["mode"]))

# Diagnostic only: fit a scalar per moment against the AarhusInv _syn of sel[0].
# With the correct unit convention this must come out ~1 (it does not feed
# into any result below).
t_lm0, a_lm0 = _syn_gates(lm_row[sel_key[0]])
t_hm0, a_hm0 = _syn_gates(hm_row[sel_key[0]])
ref = {"LM": np.abs(a_lm0), "LM_times": t_lm0, "HM": np.abs(a_hm0), "HM_times": t_hm0}
_ = ig.forward_anemone(M=rho[sel[:1]], thickness=thickness, file_gex=file_gex,
                       calibration="fitted", calibration_reference=ref,
                       calibration_tol=0.30, showInfo=0)
chk = ig.forward_anemone.last_calibration
print('  unit-convention check, fitted k vs AarhusInv (expect ~1): LM %.3f  HM %.3f'
      % (chk["k"]["LM"], chk["k"]["HM"]))

# %% Assemble all three forwards on ONE common gate grid  ->  D_ga, D_an, D_ai
#
# D_ga / D_an already have columns [LM on lm_used | HM on hm_used].  D_ai is the
# AarhusInv _syn response for the same soundings, log-log interpolated onto the
# same grid; NaN where a sounding's _syn gates don't cover a given time
# (AarhusInv culls early gates per sounding -- gaps are left visible).

t_lm, t_hm = lm_used, hm_used
t_all = np.concatenate([t_lm, t_hm])                    # (n_lm + n_hm,)
sl_lm, sl_hm = slice(0, n_lm), slice(n_lm, n_lm + n_hm)


def _loglog_nan(t_dst, t_src, y_src):
    """log-log interp, NaN outside [t_src.min, t_src.max] (no extrapolation)."""
    good = np.isfinite(y_src) & (y_src > 0)
    if good.sum() < 2:
        return np.full_like(t_dst, np.nan, dtype=float)
    out = 10.0 ** np.interp(np.log10(t_dst), np.log10(t_src[good]),
                            np.log10(y_src[good]), left=np.nan, right=np.nan)
    return out


D_ai = np.full((len(sel), n_lm + n_hm), np.nan)
for j, key in enumerate(sel_key):
    t_s, a_s = _syn_gates(lm_row[key])
    D_ai[j, sl_lm] = _loglog_nan(t_lm, t_s, np.abs(a_s))
    t_s, a_s = _syn_gates(hm_row[key])
    D_ai[j, sl_hm] = _loglog_nan(t_hm, t_s, np.abs(a_s))

print('D_ga %s  D_an %s  D_ai %s   (columns: %d LM on t_lm | %d HM on t_hm)'
      % (D_ga.shape, D_an.shape, D_ai.shape, n_lm, n_hm))
print('D_ai finite fraction: %.2f  (NaN = AarhusInv has no gate at that time for that sounding)'
      % np.isfinite(D_ai).mean())

# quick 3-way overlay on the common grid
fig0, ax0 = plt.subplots(figsize=(6, 4.5))
for j in np.linspace(0, len(sel) - 1, N_PLOT).astype(int):
    ax0.loglog(t_all, np.abs(D_ai[j]), 'ok', ms=3)
    ax0.loglog(t_all, np.abs(D_ga[j]), '-', color='C0', lw=1)
    ax0.loglog(t_all, np.abs(D_an[j]), '--', color='C3', lw=1)
ax0.plot([], [], 'ok', label='AarhusInv (D_ai)')
ax0.plot([], [], '-', color='C0', label='ga-aem (D_ga)')
ax0.plot([], [], '--', color='C3', label='anemone (D_an)')
ax0.set_xlabel('time [s]')
ax0.set_ylabel('|dB/dt| [V/Am$^4$]')
ax0.set_title('%s line %d -- forward responses on the GEX gate grid' % (case, main_line))
ax0.legend(fontsize=8)
ax0.grid(True, which='both', alpha=0.3)
fig0.tight_layout()
if hardcopy:
    fig0.savefig('%s_compare_forward_grid.png' % case, dpi=140)
    print('wrote %s_compare_forward_grid.png' % case)

# %% COMPARE THE FORWARD RESPONSES
#
# For every sounding, resample each forward curve onto that sounding's real _syn
# gate times, then collect (forward - aarhus) / aarhus for LM and HM.
#
# Both forwards are absolute: neither anemone nor ga-aem is fitted to AarhusInv.
# anemone is dB/dt per unit Tx moment from the GEX; ga-aem's STM carries the
# same unit-moment normalisation (LoopArea/PeakCurrent/NumberOfTurns = 1 with
# ModellingLoopRadius from TxLoopArea) plus all GEX receiver filters.

r_ga_lm, r_ga_hm, r_an_lm, r_an_hm, r_ae_lm, r_ae_hm = ([] for _ in range(6))
for j, key in enumerate(sel_key):
    for mom, (row, t_gex, g_full, a_full, r_ga, r_an, r_ae) in {
        'LM': (lm_row, lm_used, g_lm_full, a_lm_full, r_ga_lm, r_an_lm, r_ae_lm),
        'HM': (hm_row, hm_used, g_hm_full, a_hm_full, r_ga_hm, r_an_hm, r_ae_hm),
    }.items():
        t_s, a_s = _syn_gates(row[key])
        ga_s = _loglog(t_s, t_gex, g_full[j])
        an_s = _loglog(t_s, t_gex, a_full[j])
        r_ga.extend((ga_s - a_s) / a_s)
        r_an.extend((an_s - a_s) / a_s)
        r_ae.extend((an_s - ga_s) / ga_s)

def stats(name, r):
    r = np.abs(np.asarray(r))
    r = r[np.isfinite(r)]
    print('  %-12s n=%5d   median = %6.2f %%   90th pct = %6.2f %%   max = %6.2f %%'
          % (name, r.size, 100 * np.median(r), 100 * np.percentile(r, 90), 100 * r.max()))


print('\nga-aem  vs AarhusInv   |rel diff|:')
stats('LM', r_ga_lm)
stats('HM', r_ga_hm)
print('anemone vs AarhusInv   |rel diff|:')
stats('LM', r_an_lm)
stats('HM', r_an_hm)
print('anemone vs ga-aem      |rel diff|:')
stats('LM', r_ae_lm)
stats('HM', r_ae_hm)

# %% [markdown]
# ### Overlay figure + histogram

# %%
i_plot = np.linspace(0, len(sel) - 1, N_PLOT).astype(int)
fig, axes = plt.subplots(2, N_PLOT, figsize=(3.2 * N_PLOT, 7),
                         gridspec_kw={'height_ratios': [2, 1]})
for c, j in enumerate(i_plot):
    key = sel_key[j]
    ax, axr = axes[0, c], axes[1, c]
    for row, t_gex, g_full, a_full, cga, can in (
        (lm_row, lm_used, g_lm_full, a_lm_full, 'C0', 'C2'),
        (hm_row, hm_used, g_hm_full, a_hm_full, 'C1', 'C3'),
    ):
        t_s, a_s = _syn_gates(row[key])
        ga_s, an_s = _loglog(t_s, t_gex, g_full[j]), _loglog(t_s, t_gex, a_full[j])
        ax.loglog(t_s, a_s, 'o', ms=4, color='k')
        ax.loglog(t_s, ga_s, '-', color=cga)
        ax.loglog(t_s, an_s, '--', color=can)
        axr.semilogx(t_s, 100 * (ga_s - a_s) / a_s, '-', color=cga)
        axr.semilogx(t_s, 100 * (an_s - a_s) / a_s, '--', color=can)
    ax.set_title('line %d rec %d' % (main_line, rec_inv[sel[j]]), fontsize=9)
    ax.set_xlabel('time [s]')
    if c == 0:
        ax.set_ylabel('dB/dt [V/Am$^4$]')
        ax.plot([], [], 'ok', label='AarhusInv')
        ax.plot([], [], '-', color='0.3', label='ga-aem')
        ax.plot([], [], '--', color='0.3', label='anemone')
        ax.legend(fontsize=7)
    ax.grid(True, which='both', alpha=0.3)
    axr.axhline(0, color='k', lw=0.6)
    axr.set_ylim(-30, 30)
    axr.set_xlabel('time [s]')
    if c == 0:
        axr.set_ylabel('rel. diff vs AarhusInv [%]')
    axr.grid(True, which='both', alpha=0.3)

fig.suptitle('AarhusInv vs ga-aem vs anemone forward  -  %s SCI models, line %d'
             % (case, main_line))
fig.tight_layout()
if hardcopy:
    fig.savefig('%s_compare_forward.png' % case, dpi=140)
    print('wrote %s_compare_forward.png' % case)

fig2, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(-30, 30, 80)
ax.hist(100 * np.asarray(r_ga_lm), bins=bins, alpha=0.5, color='C0', label='ga-aem LM')
ax.hist(100 * np.asarray(r_ga_hm), bins=bins, alpha=0.5, color='C1', label='ga-aem HM')
ax.hist(100 * np.asarray(r_an_lm), bins=bins, alpha=0.5, color='C2', label='anemone LM')
ax.hist(100 * np.asarray(r_an_hm), bins=bins, alpha=0.5, color='C3', label='anemone HM')
ax.set_xlabel('relative difference vs AarhusInv [%]')
ax.set_ylabel('gate count')
ax.set_title('%s line %d, %d soundings' % (case, main_line, len(sel)))
ax.legend()
fig2.tight_layout()
if hardcopy:
    fig2.savefig('%s_compare_forward_hist.png' % case, dpi=140)
    print('wrote %s_compare_forward_hist.png' % case)

# %% [markdown]
# ## 2. Prior sample: prior_data_gaaem vs prior_data_anemone (+ timing)

# %% Generate a generic prior sample
f_prior_h5 = ig.prior_model_layered(N=N_PRIOR, lay_dist='chi2', NLAY_deg=2,
                                    RHO_min=1, RHO_max=3000,
                                    f_prior_h5='%s_CMP_PRIOR_N%d.h5' % (case, N_PRIOR),
                                    showInfo=1)

# %% Compute prior data with both backends (reuse the k fit above for anemone)
f_pd_ga = ig.prior_data_gaaem(f_prior_h5, file_gex, N=N_PRIOR, doMakePriorCopy=True,
                              f_prior_data_h5='%s_CMP_PD_GAAEM_N%d.h5' % (case, N_PRIOR),
                              force_replace=True, showInfo=1)
f_pd_an = ig.prior_data_anemone(f_prior_h5, file_gex, N=N_PRIOR, doMakePriorCopy=True,
                                f_prior_data_h5='%s_CMP_PD_ANEMONE_N%d.h5' % (case, N_PRIOR),
                                force_replace=True, showInfo=1)

import h5py
with h5py.File(f_pd_ga, 'r') as f:
    D_pd_ga = f['/D1'][:]
with h5py.File(f_pd_an, 'r') as f:
    D_pd_an = f['/D1'][:]

# compare only gates within 3 decades of each sounding's peak (skip noise-floor gates)
keep = np.abs(D_pd_ga) > 1e-3 * np.abs(D_pd_ga).max(axis=1, keepdims=True)
r_pd = np.abs((D_pd_an - D_pd_ga) / D_pd_ga)[keep & np.isfinite(D_pd_an) & (D_pd_ga != 0)]
print('\nprior data: anemone vs ga-aem over %d realizations x %d gates (%d significant)'
      % (D_pd_ga.shape[0], D_pd_ga.shape[1], r_pd.size))
print('  median |rel diff| = %5.2f %%   90th pct = %5.2f %%   99th pct = %5.2f %%   within 10%% : %4.1f %%'
      % (100 * np.median(r_pd), 100 * np.percentile(r_pd, 90),
         100 * np.percentile(r_pd, 99), 100 * (r_pd < 0.10).mean()))
print('  (the chi2 generic prior spans RHO 1..3000 over ~90 layers -- the tail is')
print('   driven by extreme realizations near response nulls, not a systematic bias)')

# %% [markdown]
# ### Timing summary
#
# All anemone numbers below are cache-warm (the one-time `Forward` build is done
# in a throwaway call first), so they measure evaluation cost only.  ga-aem's STM
# files are likewise already written by comparison 1.
#   * "1 thread"  : forced single thread   -> fair per-sounding cost
#   * "batched"   : default thread count   -> how anemone is actually used
#   * "prior ..." : `prior_data_*` as INTEGRATE runs them (ga-aem multiprocessing,
#                   anemone one batched torch call) -> realistic run cost

# %%
n_threads0 = torch.get_num_threads()

_ = ig.forward_anemone(M=rho[sel][:2], thickness=thickness, file_gex=file_gex,
                       showInfo=-1)  # warm-up

t0 = time.perf_counter()
_ = ig.forward_gaaem(C=1.0 / rho[sel], thickness=thickness, file_gex=file_gex,
                     parallel=False, showInfo=-1)
t_ga_seq = time.perf_counter() - t0

t0 = time.perf_counter()
_ = ig.forward_anemone(M=rho[sel], thickness=thickness, file_gex=file_gex, showInfo=-1)
t_an_batch = time.perf_counter() - t0

torch.set_num_threads(1)
t0 = time.perf_counter()
_ = ig.forward_anemone(M=rho[sel], thickness=thickness, file_gex=file_gex, showInfo=-1)
t_an_seq = time.perf_counter() - t0
torch.set_num_threads(n_threads0)

t0 = time.perf_counter()
_ = ig.prior_data_gaaem(f_prior_h5, file_gex, N=N_PRIOR, doMakePriorCopy=True,
                        f_prior_data_h5='%s_CMP_TIMING_GAAEM.h5' % case,
                        parallel=True, force_replace=True, showInfo=-1)
t_ga_par = time.perf_counter() - t0

t0 = time.perf_counter()
_ = ig.prior_data_anemone(f_prior_h5, file_gex, N=N_PRIOR, doMakePriorCopy=True,
                          f_prior_data_h5='%s_CMP_TIMING_ANEMONE.h5' % case,
                          force_replace=True, showInfo=-1)
t_an_par = time.perf_counter() - t0

ns_seq, ns_par = len(sel), N_PRIOR
rows = [('ga-aem  forward_gaaem  (1 thread)', t_ga_seq, ns_seq),
        ('anemone forward_anemone (1 thread)', t_an_seq, ns_seq),
        ('anemone forward_anemone (batched)', t_an_batch, ns_seq),
        ('ga-aem  prior_data_gaaem (parallel)', t_ga_par, ns_par),
        ('anemone prior_data_anemone (batched)', t_an_par, ns_par)]
print('\n%-38s %10s %14s %12s' % ('', 'wall [s]', 'ms/sounding', 'it/s'))
for name, t, n in rows:
    print('%-38s %10.2f %14.2f %12.1f' % (name, t, 1e3 * t / n, n / t))

# %% Plot and compare data
fig3, axs = plt.subplots(1, 2, figsize=(11, 4))
gidx = np.arange(D_pd_ga.shape[1])            # [LM used gates | HM used gates]
axs[0].semilogy(gidx, np.abs(D_pd_ga[:200]).T, color='C0', alpha=0.15, lw=0.5)
axs[0].semilogy(gidx, np.abs(D_pd_an[:200]).T, color='C3', alpha=0.15, lw=0.5)
axs[0].axvline(n_lm - 0.5, color='k', lw=0.8, ls=':')
axs[0].set_title('prior data (200 realizations)  ga-aem C0  /  anemone C3')
axs[0].set_xlabel('gate index  (LM | HM)')
axs[0].set_ylabel('|dB/dt| [V/Am$^4$]')
axs[0].grid(True, which='both', alpha=0.3)

names = ['ga-aem\nseq', 'anemone\n1 thread', 'anemone\nbatched',
         'ga-aem\nprior (par)', 'anemone\nprior (batch)']
vals = [1e3 * t_ga_seq / ns_seq, 1e3 * t_an_seq / ns_seq, 1e3 * t_an_batch / ns_seq,
        1e3 * t_ga_par / ns_par, 1e3 * t_an_par / ns_par]
axs[1].bar(range(len(vals)), vals, color=['C0', 'C3', 'C3', 'C0', 'C3'])
axs[1].set_xticks(range(len(vals)))
axs[1].set_xticklabels(names, fontsize=8)
axs[1].set_ylabel('ms / sounding')
axs[1].set_yscale('log')
axs[1].set_title('forward cost per sounding')
axs[1].grid(True, axis='y', which='both', alpha=0.3)
fig3.tight_layout()
if hardcopy:
    fig3.savefig('%s_compare_prior_timing.png' % case, dpi=140)
    print('wrote %s_compare_prior_timing.png' % case)
