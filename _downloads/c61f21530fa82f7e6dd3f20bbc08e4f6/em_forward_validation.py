"""
Validate the EM forward models against Aarhus Workbench (tTEM)
==============================================================

This example is for **tTEM data**, i.e. a towed, ground-based system with the
transmitter and receiver at a **fixed height** above the ground, given by the
GEX file (here Tx at 0.36 m, Rx at 0.33 m, 9.15 m behind the Tx). The prior
therefore has no flight-height parameter, and all forward models use the
height of the GEX file. Varying flight height, as for airborne systems such
as SkyTEM, is not tested here.

The DAUGAARD case includes a deterministic (AarhusInv) inversion made in
Aarhus Workbench (here ``aw``):

* ``*_dat.xyz``: the observed data used in the inversion
* ``*_inv.xyz``: the inverted models
* ``*_syn.xyz``: the forward response of the inverted models

This example uses the inverted models and their Workbench forward response
as reference. The inverted models are stored as prior realizations in
``PRIOR_aw.h5``, and the prior data are computed with ``ig.prior_data_em``
for each of the EM forward models available in the current environment,

* ``ga-aem``
* ``simpeg``
* ``anemone`` on the CPU
* ``anemone`` on a CUDA GPU

each stored in its own prior data file. The forward responses are then
compared gate-by-gate to the Workbench response (accuracy), and the run time
of each forward model is measured for an increasing number of soundings
(efficiency). The run time of Workbench is not available.
"""
# %%
import integrate as ig
hardcopy = True
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import libaarhusxyz

# Forward models to test, as (label, method, device). Those not available in
# the current environment are skipped.
forward_models = [('gaaem',       'ga-aem',  'cpu'),
                  ('simpeg',      'simpeg',  'cpu'),
                  ('anemone_cpu', 'anemone', 'cpu'),
                  ('anemone_gpu', 'anemone', 'cuda')]

# Number of soundings in the run-time test. SimPEG takes about 0.3 s per
# sounding per core, so N=10000 takes a few minutes on a few cores.
N_time = [100, 1000, 10000]


def forward_available(method, device):
    """True if the forward model (and device) can be used in this environment."""
    try:
        if method == 'ga-aem':
            from gatdaem1d import TDAEMSystem  # noqa: F401
        elif method == 'simpeg':
            from integrate.simpeg_forward import _require_simpeg
            _require_simpeg()
        elif method == 'anemone':
            from integrate.anemone_forward import _require_anemone
            _, torch = _require_anemone()
            if device == 'cuda' and not torch.cuda.is_available():
                return False
    except (ImportError, OSError):
        return False
    return True


forward_models = [fm for fm in forward_models if forward_available(fm[1], fm[2])]
if len(forward_models) == 0:
    raise SystemExit('No EM forward model available; install ga-aem, simpeg or anemone')
print('Forward models available: %s' % ', '.join(fm[0] for fm in forward_models))

# %%
# 0. Get the DAUGAARD data
# ------------------------
#
# The GEX file, and the Workbench inversion: observed data (dat), inverted
# models (inv) and their forward response (syn).

# %%
case = 'DAUGAARD'
files = ig.get_case_data(case=case, showInfo=2)
f_data_h5 = files[0]
file_gex = ig.get_gex_file_from_data(f_data_h5)

file_xyz_dat = ig.get_case_data(case=case, filelist=['SCI7_40_ml_sharp2_I02_MOD_dat.xyz'])[0]
file_xyz_inv = ig.get_case_data(case=case, filelist=['SCI7_40_ml_sharp2_I02_MOD_inv.xyz'])[0]
file_xyz_syn = ig.get_case_data(case=case, filelist=['SCI7_40_ml_sharp2_I02_MOD_syn.xyz'])[0]

print("Using GEX file: %s" % file_gex)
print("Using Workbench observed data: %s" % file_xyz_dat)
print("Using Workbench inverted models: %s" % file_xyz_inv)
print("Using Workbench forward response: %s" % file_xyz_syn)

# %%
# 1. Observed data and the Workbench forward response
# ---------------------------------------------------
#
# ``ig.xyz_to_h5`` maps the Workbench gates onto the gates of the GEX file,
# i.e. the gates [LM | HM] that the forward models return. The Workbench
# inversion covers several survey days, processed with different GEX files;
# only the soundings whose gate times match this GEX file (2023-11-09) are
# kept. Gates Workbench did not use are NaN.

# %%
f_data_obs_h5 = ig.xyz_to_h5(file_xyz_dat, file_gex, f_data_h5='DATA_aw_obs.h5')
f_data_aw_h5 = ig.xyz_to_h5(file_xyz_syn, file_gex, f_data_h5='DATA_aw.h5')

DATA_obs = ig.load_data(f_data_obs_h5)
D_obs = DATA_obs['d_obs'][0]
D_obs_std = DATA_obs['d_std'][0]
D_aw = ig.load_data(f_data_aw_h5)['d_obs'][0]
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_aw_h5)
nd = D_aw.shape[0]

# Gate centre times of the used GEX gates, i.e. the columns of the data
system = ig.gex_to_em_system(file_gex)
t_lm = system['moments'][0]['gate_centre']
t_hm = system['moments'][1]['gate_centre']
t_gate = np.r_[t_lm, t_hm]
is_hm = np.r_[np.zeros(t_lm.size, dtype=bool), np.ones(t_hm.size, dtype=bool)]

# tTEM: fixed Tx/Rx height from the GEX file, used by all forward models
tx_z = system['tx_z']
rx_z = system['tx_z'] - system['rx_dz']
rx_x = abs(system['rx_xy'][0])
system_label = 'tTEM, fixed height: Tx %4.2f m, Rx %4.2f m, offset %4.2f m' % (tx_z, rx_z, rx_x)
print(system_label)

rel_obs = (D_aw - D_obs) / D_obs
print('%d soundings, %d gates, %d used by Workbench' % (nd, len(t_gate), np.sum(~np.isnan(D_aw))))
print('Workbench response vs observed data: median |rel diff| = %4.2f %%, median rel std = %4.2f %%'
      % (100 * np.nanmedian(np.abs(rel_obs)), 100 * np.nanmedian(D_obs_std / np.abs(D_obs))))

# %%
# 2. Store the inverted models as prior realizations
# --------------------------------------------------
#
# The inverted models are matched to the soundings of the data by line and
# position (in order of occurrence, as a few soundings share a position),
# and saved as ``M1`` in ``PRIOR_aw.h5``, with the depth to the top of each
# layer as ``x``.

# %%
inv = libaarhusxyz.XYZ(file_xyz_inv)
rho_inv = inv.layer_data['rho'].values.astype(float)  # (n_inv, n_layer)
z_inv = inv.layer_data['dep_top'].values.astype(float)[0]


def sounding_key(line, x, y):
    """(line, x, y, occurrence) of each sounding."""
    df = pd.DataFrame({'line': np.round(line).astype(int), 'x': np.round(x, 2), 'y': np.round(y, 2)})
    df['n'] = df.groupby(['line', 'x', 'y']).cumcount()
    return pd.MultiIndex.from_frame(df)


i_inv = pd.Series(np.arange(len(rho_inv)),
                  index=sounding_key(inv.flightlines['line_no'].values,
                                     inv.flightlines['utmx'].values,
                                     inv.flightlines['utmy'].values)
                  ).reindex(sounding_key(LINE, X, Y)).values
if np.any(np.isnan(i_inv)):
    raise ValueError('%d soundings have no inverted model' % np.sum(np.isnan(i_inv)))
M_aw = rho_inv[i_inv.astype(int)]
print('Inverted models: %d soundings, %d layers, depth 0-%g m' % (M_aw.shape[0], M_aw.shape[1], z_inv[-1]))

f_prior_h5 = 'PRIOR_aw.h5'
ig.save_prior_model(f_prior_h5, M_aw, im=1, x=z_inv, name='Resistivity', unit='Ohm.m',
                    delete_if_exist=True)

# %%
# 3. Forward the inverted models
# ------------------------------
#
# ``ig.prior_data_em`` computes the prior data of ``PRIOR_aw.h5`` with each
# forward model, in its own prior data file ``PRIOR_DATA_aw_<label>.h5``.

# %%
D_fwd = {}
for label, method, device in forward_models:
    f_prior_data_h5 = 'PRIOR_DATA_aw_%s.h5' % label
    t0 = time.time()
    ig.prior_data_em(f_prior_h5, file_gex, method=method, device=device,
                     doMakePriorCopy=True, f_prior_data_h5=f_prior_data_h5,
                     force_replace=True, showInfo=0)
    D_fwd[label] = ig.load_prior_data(f_prior_data_h5, id_use=[1], showInfo=0)[0][0]
    print('%-12s %d soundings: %6.1f s -> %s' % (label, nd, time.time() - t0, f_prior_data_h5))

# %%
# 4. Accuracy relative to Workbench
# ---------------------------------
#
# Relative difference to Workbench, (forward - aw) / aw, for every gate
# Workbench used.

# %%
rel = {label: (D_ - D_aw) / D_aw for label, D_ in D_fwd.items()}
use = ~np.isnan(D_aw)
rel_std = np.nanmedian(D_obs_std / np.abs(D_obs))  # typical relative data noise

print('|rel diff| to Workbench (%d soundings):' % nd)
print('%-12s %-3s %6s %9s %9s %9s' % ('model', '', 'n', 'median', '90th pct', 'max'))
for label in rel:
    for mom, m_ in (('LM', ~is_hm), ('HM', is_hm)):
        r_ = np.abs(rel[label][:, m_][use[:, m_]])
        print('%-12s %-3s %6d %7.2f %% %7.2f %% %7.2f %%'
              % (label, mom, r_.size, 100 * np.median(r_), 100 * np.percentile(r_, 90), 100 * r_.max()))

# %%
# Top left: forward responses of a few soundings, with Workbench as black
# dots. Top right: relative difference to Workbench for every gate, with the
# median relative noise of the observed data as grey band. Bottom left:
# median and 90th percentile of the absolute relative difference per gate,
# LM (circles) and HM (squares). Bottom right: distribution of the relative
# difference.
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

ax = axs[0, 0]
for i in np.linspace(0, nd - 1, 4).astype(int):
    for c_, label in enumerate(D_fwd):
        for m_ in (~is_hm, is_hm):
            ax.loglog(t_gate[m_], np.abs(D_fwd[label][i, m_]), '-', color='C%d' % c_, lw=1)
    ax.loglog(t_gate[use[i]], np.abs(D_aw[i, use[i]]), 'ok', ms=3)
for c_, label in enumerate(D_fwd):
    ax.plot([], [], '-', color='C%d' % c_, label=label)
ax.plot([], [], 'ok', label='aw')
ax.set_xlabel('Time [s]')
ax.set_ylabel('|dB/dt| [V/Am$^4$]')
ax.set_title('Forward response, 4 soundings')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

ax = axs[0, 1]
ax.axhspan(-100 * rel_std, 100 * rel_std, color='0.85', label='data noise (%3.1f %%)' % (100 * rel_std))
T = np.broadcast_to(t_gate, D_aw.shape)
for c_, label in enumerate(rel):
    ax.semilogx(T[use] * (1 + 0.02 * (c_ - 1.5)), 100 * rel[label][use], '.', ms=2,
                color='C%d' % c_, label=label)
ax.axhline(0, color='k', lw=0.6)
ax.set_ylim(-1.5 * 100 * rel_std, 1.5 * 100 * rel_std)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Relative difference to aw [%]')
ax.set_title('%d soundings, LM and HM' % nd)
ax.legend(fontsize=8, markerscale=5)
ax.grid(True, which='both', alpha=0.3)

ax = axs[1, 0]
ax.axhline(100 * rel_std, color='0.5', ls='--', label='data noise')
for c_, label in enumerate(rel):
    r_ = np.where(use, np.abs(rel[label]), np.nan)
    for m_, mk in ((~is_hm & use.any(axis=0), 'o'), (is_hm & use.any(axis=0), 's')):
        ax.loglog(t_gate[m_], 100 * np.nanmedian(r_[:, m_], axis=0), '-' + mk, color='C%d' % c_, ms=4)
        ax.loglog(t_gate[m_], 100 * np.nanpercentile(r_[:, m_], 90, axis=0), ':' + mk, color='C%d' % c_,
                  ms=3, mfc='none')
    ax.plot([], [], '-', color='C%d' % c_, label=label)
ax.plot([], [], '-k', label='median')
ax.plot([], [], ':k', label='90th pct')
ax.set_xlabel('Time [s]')
ax.set_ylabel('|Relative difference to aw| [%]')
ax.set_title('Per gate')
ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

ax = axs[1, 1]
bins = np.linspace(-100 * rel_std, 100 * rel_std, 61)
for c_, label in enumerate(rel):
    ax.hist(100 * rel[label][use], bins=bins, histtype='step', lw=1.5, color='C%d' % c_, label=label)
ax.axvline(0, color='k', lw=0.6)
ax.set_xlabel('Relative difference to aw [%]')
ax.set_ylabel('Gate count')
ax.set_title('%d soundings, LM and HM' % nd)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

fig.suptitle('%s: forward models vs Aarhus Workbench\n%s' % (case, system_label))
fig.tight_layout()
if hardcopy:
    fig.savefig('%s_forward_validation_accuracy.png' % case, dpi=140)
plt.show()

# %%
# 5. Computational efficiency
# ---------------------------
#
# The inverted models are repeated to N soundings, stored in
# ``PRIOR_aw_N<N>.h5``, and forwarded with each forward model. The run time
# includes the setup of GA-AEM (worker pool) and SimPEG (filter
# coefficients) on every call, which dominates for small N. anemone keeps its
# setup between calls, so after section 3 its run time is the forward alone.

# %%
import os
n_cpu = os.cpu_count()
print('CPU cores: %d' % n_cpu)
for label, method, device in forward_models:
    if device == 'cuda':
        import torch
        print('GPU: %s' % torch.cuda.get_device_name(0))
        break

t_run = {label: [] for label, _, _ in forward_models}
for N in N_time:
    f_prior_N_h5 = 'PRIOR_aw_N%d.h5' % N
    ig.save_prior_model(f_prior_N_h5, np.resize(M_aw, (N, M_aw.shape[1])), im=1, x=z_inv,
                        name='Resistivity', unit='Ohm.m', delete_if_exist=True)
    for id_, (label, method, device) in enumerate(forward_models, start=1):
        t0 = time.time()
        ig.prior_data_em(f_prior_N_h5, file_gex, method=method, device=device, id=id_,
                         doMakePriorCopy=False, force_replace=True, showInfo=0)
        t_run[label].append(time.time() - t0)
        print('N=%6d %-12s %8.1f s' % (N, label, t_run[label][-1]))

print('\nForward run time')
print('%-12s %-6s' % ('model', 'device') + ''.join('%14s' % ('N=%d' % N) for N in N_time))
for label, method, device in forward_models:
    print('%-12s %-6s' % (label, device) + ''.join('%12.1f s' % t for t in t_run[label]))
print('\nForward time per sounding')
print('%-12s %-6s' % ('model', 'device') + ''.join('%14s' % ('N=%d' % N) for N in N_time))
for label, method, device in forward_models:
    print('%-12s %-6s' % (label, device)
          + ''.join('%11.3f ms' % (1000 * t / N) for t, N in zip(t_run[label], N_time)))

# %%
# Left: run time against the number of soundings. Right: run time per
# sounding; for large N this approaches the cost of the forward itself.
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for c_, label in enumerate(t_run):
    t_ = np.array(t_run[label])
    axs[0].loglog(N_time, t_, '-o', color='C%d' % c_, label=label)
    axs[1].loglog(N_time, 1000 * t_ / np.array(N_time), '-o', color='C%d' % c_, label=label)
axs[0].set_ylabel('Run time [s]')
axs[0].set_title('Run time')
axs[1].set_ylabel('Run time per sounding [ms]')
axs[1].set_title('Run time per sounding')
for ax in axs:
    ax.set_xlabel('Number of soundings, N')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
fig.suptitle('%s: forward run time, %d layers, %d CPU cores\n%s' % (case, M_aw.shape[1], n_cpu, system_label))
fig.tight_layout()
if hardcopy:
    fig.savefig('%s_forward_validation_runtime.png' % case, dpi=140)
plt.show()
