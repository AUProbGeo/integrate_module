"""
Compare forward models to HGG Workbench
=======================================

The DAUGAARD case includes an AarhusInv inversion made in HGG Workbench: the
inversion models (``*_inv.xyz``) and Workbench's forward response of those
models (``*_syn.xyz``). This example computes the response of the same
inversion models with each of the EM forward models available in INTEGRATE,

* ``anemone`` on a CUDA GPU (skipped if no GPU is available)
* ``anemone`` on the CPU
* ``ga-aem``
* ``simpeg`` (skipped if SimPEG is not installed)
* ``anemone`` on the CPU with ``rx_coil_filter='damped2'`` and
  ``gate_integration='boxcar'`` (label ``anemone_wb``)

and compares them gate-by-gate to the Workbench response, used here as the
reference.
"""
# %%
import integrate as ig
hardcopy = True
import matplotlib.pyplot as plt
import numpy as np
import time

# %%
# 0. Get TTEM data
# ----------------

# %%
case = 'DAUGAARD'
files = ig.get_case_data(case=case, showInfo=2)
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)

# %%
# 1. Load the Workbench inversion models and forward response
# ------------------------------------------------------------
#
# The forward models return the used gates of the GEX file, [LM | HM], while
# Workbench lays out the response in one merged block and drops early gates
# per sounding. Each forward response is therefore interpolated (log-log)
# onto the gate times Workbench actually has for that sounding.

# %%
import libaarhusxyz

DUMMY = 9999.0  # Workbench no-value marker


def _one(x):
    """get_case_data returns a list; take the single entry."""
    return x[0] if isinstance(x, (list, tuple)) else x


file_xyz_inv = _one(ig.get_case_data(case=case, filelist=['SCI7_40_ml_sharp2_I02_MOD_inv.xyz']))
file_xyz_syn = _one(ig.get_case_data(case=case, filelist=['SCI7_40_ml_sharp2_I02_MOD_syn.xyz']))
print("Using Workbench inversion models: %s" % file_xyz_inv)
print("Using Workbench forward response: %s" % file_xyz_syn)

# Inversion models (all soundings share one depth grid)
inv = libaarhusxyz.XYZ(file_xyz_inv)
rho_inv = inv.layer_data['rho'].values.astype(float)           # (nS, nLayer)
z_inv = inv.layer_data['dep_top'].values.astype(float)[0]
thickness_inv = np.diff(z_inv)
line_inv = inv.flightlines['line_no'].values.astype(np.int64)
rec_inv = inv.flightlines['record'].values.astype(np.int64)

# Workbench forward response: one row per sounding and moment (segment 1 = LM, 2 = HM)
syn = libaarhusxyz.XYZ(file_xyz_syn)
syn_data = syn.layer_data['data'].values.astype(float)
syn_seg = syn.flightlines['segments'].values.astype(int)
syn_rec = syn.flightlines['record'].values.astype(np.int64)
syn_gate_t = np.asarray(syn.model_info['gate times (s)'], dtype=float)
n_rec = max(syn_rec.max(), rec_inv.max()) + 1
syn_key = syn.flightlines['line_no'].values.astype(np.int64) * n_rec + syn_rec
lm_row = {k: i for i, k in zip(np.where(syn_seg == 1)[0], syn_key[syn_seg == 1])}
hm_row = {k: i for i, k in zip(np.where(syn_seg == 2)[0], syn_key[syn_seg == 2])}

# Use every sounding on the longest line that has both an LM and an HM response
main_line = np.bincount(line_inv).argmax()
key_inv = line_inv * n_rec + rec_inv
i_wb = np.array([i for i in np.where(line_inv == main_line)[0]
                 if key_inv[i] in lm_row and key_inv[i] in hm_row])
print('Workbench line %d: %d soundings, %d layers' % (main_line, len(i_wb), rho_inv.shape[1]))

# Gate centre times of the used GEX gates, i.e. the columns of the forward output
system = ig.gex_to_em_system(file_gex)
t_lm = system['moments'][0]['gate_centre']
t_hm = system['moments'][1]['gate_centre']
n_lm, n_hm = t_lm.size, t_hm.size


def _syn_gates(row_idx):
    """Gate times and values that Workbench has for one _syn row."""
    r = syn_data[row_idx]
    m = r != DUMMY
    return syn_gate_t[m], r[m]


def _loglog(t_dst, t_src, y_src):
    """y_src(t_src) resampled to t_dst in log-log space."""
    return 10.0 ** np.interp(np.log10(t_dst), np.log10(t_src), np.log10(np.abs(y_src)))


# %%
# 2. Forward the inversion models
# -------------------------------
#
# Forward the Workbench inversion models with each forward model. The GPU
# and SimPEG runs are skipped if not available.
#
# All forward models read the receiver-coil filter of the GEX,
# ``RxCoilLPFilter1= 0.87 420E+3``, as a second-order low pass with damping
# 0.87 at 420 kHz, modelled as two first-order filters at 420/0.87 kHz.
# ``anemone_wb`` uses the exact second-order filter instead, and averages
# dB/dt over each gate. This reading of the GEX is inferred from comparisons
# with Workbench (see ``ISSUE_rx_coil_filter.md``).
wb_forwards = [('anemone_gpu', dict(method='anemone', device='cuda')),
               ('anemone_cpu', dict(method='anemone', device='cpu')),
               ('gaaem',       dict(method='ga-aem')),
               ('simpeg',      dict(method='simpeg')),
               ('anemone_wb',  dict(method='anemone', device='cpu',
                                    rx_coil_filter='damped2', gate_integration='boxcar'))]
D_wb = {}
for lab_, kw_ in wb_forwards:
    try:
        t0 = time.time()
        D_wb[lab_] = ig.forward_em(rho_inv[i_wb], thickness_inv, file_gex=file_gex, **kw_)
        print('%-12s forward of %d soundings: %5.1f s' % (lab_, len(i_wb), time.time() - t0))
    except Exception as e:
        print('%-12s skipped (%s)' % (lab_, e))

# %%
# 3. Compare to Workbench
# -----------------------
#
# Relative difference to Workbench, (forward - workbench) / workbench, for
# every gate Workbench has.
rel_wb = {lab_: {'LM': [], 'HM': []} for lab_ in D_wb}
t_wb = {'LM': [], 'HM': []}  # gate time of each relative difference
for j, i in enumerate(i_wb):
    for mom, row, t_gex, sl in (('LM', lm_row, t_lm, slice(0, n_lm)),
                                ('HM', hm_row, t_hm, slice(n_lm, n_lm + n_hm))):
        t_s, d_s = _syn_gates(row[key_inv[i]])
        if t_s.size < 2:
            continue
        t_wb[mom].extend(t_s)
        for lab_, D_ in D_wb.items():
            d_f = _loglog(t_s, t_gex, D_[j, sl])
            rel_wb[lab_][mom].extend((d_f - d_s) / d_s)

print('|rel diff| to HGG Workbench:')
for lab_ in D_wb:
    for mom in ('LM', 'HM'):
        r_ = np.abs(np.asarray(rel_wb[lab_][mom]))
        r_ = r_[np.isfinite(r_)]
        print('  %-12s %s  n=%5d   median = %6.2f %%   90th pct = %6.2f %%   max = %6.2f %%'
              % (lab_, mom, r_.size, 100 * np.median(r_), 100 * np.percentile(r_, 90),
                 100 * r_.max()))

# %%
# Left: forward responses of a few soundings, with Workbench as black dots.
# Middle: relative difference to Workbench against gate time, for every gate
# on the line. Right: distribution of the relative difference.
t_all = np.r_[t_lm, t_hm]
ls_wb = ['-', '--', ':', '-.']
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for j in np.linspace(0, len(i_wb) - 1, 5).astype(int):
    for c_, (lab_, D_) in enumerate(D_wb.items()):
        axs[0].loglog(t_all, np.abs(D_[j]), ls_wb[c_ % len(ls_wb)], color='C%d' % c_, lw=1)
    for row in (lm_row, hm_row):
        t_s, d_s = _syn_gates(row[key_inv[i_wb[j]]])
        axs[0].loglog(t_s, np.abs(d_s), 'ok', ms=3)
for c_, lab_ in enumerate(D_wb):
    axs[0].plot([], [], ls_wb[c_ % len(ls_wb)], color='C%d' % c_, label=lab_)
axs[0].plot([], [], 'ok', label='HGG Workbench')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('|dB/dt| [V/Am$^4$]')
axs[0].set_title('Line %d, 5 soundings' % main_line)
axs[0].legend(fontsize=8)
axs[0].grid(True, which='both', alpha=0.3)

t_ = np.r_[t_wb['LM'], t_wb['HM']]
for c_, lab_ in enumerate(D_wb):
    r_ = 100 * np.asarray(rel_wb[lab_]['LM'] + rel_wb[lab_]['HM'])
    axs[1].semilogx(t_ * (1 + 0.03 * (c_ - 2)), r_, '.', ms=2, color='C%d' % c_, label=lab_)
axs[1].axhline(0, color='k', lw=0.6)
axs[1].set_ylim(-8, 8)
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Relative difference to HGG Workbench [%]')
axs[1].set_title('Line %d, %d soundings, LM and HM' % (main_line, len(i_wb)))
axs[1].legend(fontsize=8, markerscale=5)
axs[1].grid(True, which='both', alpha=0.3)

bins = np.linspace(-8, 8, 65)
for c_, lab_ in enumerate(D_wb):
    r_ = 100 * np.asarray(rel_wb[lab_]['LM'] + rel_wb[lab_]['HM'])
    axs[2].hist(r_, bins=bins, histtype='step', lw=1.5, ls=ls_wb[c_ % len(ls_wb)], color='C%d' % c_, label=lab_)
axs[2].axvline(0, color='k', lw=0.6)
axs[2].set_xlabel('Relative difference to HGG Workbench [%]')
axs[2].set_ylabel('Gate count')
axs[2].set_title('Line %d, %d soundings, LM and HM' % (main_line, len(i_wb)))
axs[2].legend(fontsize=8)
axs[2].grid(True, alpha=0.3)
fig.tight_layout()
if hardcopy:
    fig.savefig('%s_compare_forward_workbench.png' % case, dpi=140)
plt.show()
