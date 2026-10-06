"""
Effect of the EM forward model on the posterior
===============================================

This example runs the full INTEGRATE workflow (prior, prior data, rejection
sampling, posterior statistics) once for each of the available EM forward
backends, using the same prior realizations, and compares the results:

* ``anemone`` on a CUDA GPU (skipped if no GPU is available)
* ``anemone`` on the CPU
* ``ga-aem``
* ``simpeg`` (skipped if SimPEG is not installed)

``ga-aem`` is used as the reference. The example compares the forward
responses, the run time of each backend, the CHI2, evidence and annealing
temperature of the posteriors, and the resulting resistivity profiles.
"""
# %%
import integrate as ig
hardcopy = True
import matplotlib.pyplot as plt
import numpy as np
import time

#
N=4_000_000
N=100_000

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
# 0b. Compare the forward models to HGG Workbench
# -----------------------------------------------
#
# The DAUGAARD case also holds an AarhusInv inversion made in HGG Workbench:
# the inversion models (``*_inv.xyz``) and Workbench's forward response of
# those models (``*_syn.xyz``). Each forward model below computes the response
# of the same inversion models, which is then compared gate-by-gate to the
# Workbench response, used here as the reference.
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
# Forward the Workbench inversion models with each forward model. The GPU
# and SimPEG runs are skipped if not available.
wb_forwards = [('anemone_gpu', dict(method='anemone', device='cuda')),
               ('anemone_cpu', dict(method='anemone', device='cpu')),
               ('gaaem',       dict(method='ga-aem')),
               ('simpeg',      dict(method='simpeg'))]
D_wb = {}
for lab_, kw_ in wb_forwards:
    try:
        t0 = time.time()
        D_wb[lab_] = ig.forward_em(rho_inv[i_wb], thickness_inv, file_gex=file_gex, **kw_)
        print('%-12s forward of %d soundings: %5.1f s' % (lab_, len(i_wb), time.time() - t0))
    except Exception as e:
        print('%-12s skipped (%s)' % (lab_, e))

# %%
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
        axs[0].loglog(t_all, np.abs(D_[j]), ls_wb[c_], color='C%d' % c_, lw=1)
    for row in (lm_row, hm_row):
        t_s, d_s = _syn_gates(row[key_inv[i_wb[j]]])
        axs[0].loglog(t_s, np.abs(d_s), 'ok', ms=3)
for c_, lab_ in enumerate(D_wb):
    axs[0].plot([], [], ls_wb[c_], color='C%d' % c_, label=lab_)
axs[0].plot([], [], 'ok', label='HGG Workbench')
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('|dB/dt| [V/Am$^4$]')
axs[0].set_title('Line %d, 5 soundings' % main_line)
axs[0].legend(fontsize=8)
axs[0].grid(True, which='both', alpha=0.3)

t_ = np.r_[t_wb['LM'], t_wb['HM']]
for c_, lab_ in enumerate(D_wb):
    r_ = 100 * np.asarray(rel_wb[lab_]['LM'] + rel_wb[lab_]['HM'])
    axs[1].semilogx(t_ * (1 + 0.03 * (c_ - 1.5)), r_, '.', ms=2, color='C%d' % c_, label=lab_)
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
    axs[2].hist(r_, bins=bins, histtype='step', lw=1.5, ls=ls_wb[c_], color='C%d' % c_, label=lab_)
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

# %%
# Select a profile
# ~~~~~~~~~~~~~~~~

# %%
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
# Find points within buffer distance
Xl = np.array([544000, 543550])
Yl = np.array([6174500, 6176500])
Xl = np.array([543000, 544000, 545400])
Yl = np.array([6176400, 6176000, 6174800])

buffer = 10.0
indices, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)
i_line = indices

plt.figure()
fig = ig.plot_geometry(f_data_h5, pl='ELEVATION');
plt.plot(Xl, Yl,'ko', markersize=15)
plt.plot(X[i_line], Y[i_line], 'ko', markersize=5, zorder=3)
plt.title('Profile line')
plt.show()

i_use = np.arange(len(X))

# %%

# The electromagnetic data (d_obs and d_std) can be plotted using ig.plot_data:
#ig.plot_data(f_data_h5, hardcopy=hardcopy)
# Plot data channel 15 in an XY grid
#ig.plot_data_xy(f_data_h5, data_channel=15, cmap='jet');

# %%
# 1. Set up the prior model
# -------------------------

# %%
# Select how many prior model realizations (N) should be generated
f_prior_h5 = ig.prior_model_layered(N=N,
                                    #lay_dist='uniform',
                                    NLAY_min=5,
                                    NLAY_max=5,                 # Minimum 3 layer
                                    ##lay_dist='chi2',
                                    NLAY_deg=6,
                                    RHO_min=1,
                                    RHO_max=1000,
                                    f_prior_h5='PRIOR_N%d.h5' % N,
                                    showInfo=1)
print('%s is used to hold prior realizations' % (f_prior_h5))


# %%

# Plot summary statistics of the prior model for quality control of the prior choice
#ig.plot_prior_stats(f_prior_h5, im=1, panels=['hist'],hardcopy=hardcopy)

# %%
# 1b. Generate corresponding prior data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Next, we generate a corresponding sample of the prior data distribution,
# once with each forward model: ga-aem, anemone (GPU and CPU) and SimPEG.

# %%
# Separate prior-data files, one per forward model
# prior_data_em() only honours N when doMakePriorCopy=True: the copy is then
# truncated to N realizations of /M1 and the forward is run on all of them.
# With doMakePriorCopy=False the forward is applied to EVERY model in the file,
# whatever N is. So to run each forward on a different N, each forward gets its
# own copy of the prior. The data is '/D1' in each file.
N_anemone_gpu = N
N_anemone_cpu = N # int(N/(10*4))
N_gaaem       = N # int(N/(20*4))
N_simpeg      = N_gaaem

f_stem = '%s_%s' % (f_prior_h5[:-3], file_gex[:-4])
f_prior_data_h5_anemone_gpu = '%s_anemone_gpu_N%d.h5' % (f_stem, N_anemone_gpu)
f_prior_data_h5_anemone_cpu = '%s_anemone_cpu_N%d.h5' % (f_stem, N_anemone_cpu)
f_prior_data_h5_gaaem       = '%s_gaaem_N%d.h5'       % (f_stem, N_gaaem)
f_prior_data_h5_simpeg      = '%s_simpeg_N%d.h5'      % (f_stem, N_simpeg)

# %%
# AnEMone GPU
# ^^^^^^^^^^^

# %%
# Release any cached-but-unused CUDA blocks from earlier runs in THIS process and
# report what this process holds. torch.cuda.empty_cache() is safe, but it cannot
# free memory held by live tensors or by OTHER processes (e.g. old Jupyter
# kernels) -- check those with `nvidia-smi` and restart the kernel if needed.
import gc
try:
    import torch
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        print('CUDA memory (this process): allocated=%.0f MB, reserved=%.0f MB, total=%.0f MB'
              % (torch.cuda.memory_allocated()/2**20,
                 torch.cuda.memory_reserved()/2**20,
                 torch.cuda.get_device_properties(0).total_memory/2**20))
except ImportError:
    pass

t0=time.time()
rps_anemone_gpu = None
try:
    f_prior_data_h5_anemone_gpu = ig.prior_data_em(f_prior_h5, file_gex,
                                    doMakePriorCopy=True,
                                    randomize=False, # copy the FIRST N models, so all three files share them
                                    f_prior_data_h5=f_prior_data_h5_anemone_gpu,
                                    id=1, # The id '/D1' of the prior data
                                    im=1, # the id '/M1' the im of the PRIOR resistivity
                                    method='anemone',
                                    N=N_anemone_gpu, # Number of prior realizations copied AND forwarded
                                    device = 'cuda',
                                    # batch_size (default 1000) sets the peak GPU memory of the run
                                    # itself; lower it (e.g. batch_size=250) if the forward OOMs
                                    # batch_size = 1000,
                                    )
    t_anemone_gpu=time.time()-t0
    rps_anemone_gpu = N_anemone_gpu/t_anemone_gpu # realizations per second
except Exception as e:
    print('GPU not available for anemone (%s)' % e)
    f_prior_data_h5_anemone_gpu = None

# %%
# AnEMone CPU
# ^^^^^^^^^^^

# %%
t0=time.time()
f_prior_data_h5_anemone_cpu = ig.prior_data_em(f_prior_h5, file_gex,
                                doMakePriorCopy=True,
                                randomize=False, # copy the FIRST N models, so all three files share them
                                f_prior_data_h5=f_prior_data_h5_anemone_cpu,
                                id=1,
                                im=1,
                                method='anemone',
                                N=N_anemone_cpu,
                                device = 'cpu'
                                )
t_anemone_cpu=time.time()-t0
rps_anemone_cpu = N_anemone_cpu/t_anemone_cpu

# %%
# GA-AEM
# ^^^^^^

# %%
t0=time.time()
f_prior_data_h5_gaaem = ig.prior_data_em(f_prior_h5, file_gex,
                                doMakePriorCopy=True,
                                randomize=False, # copy the FIRST N models, so all three files share them
                                f_prior_data_h5=f_prior_data_h5_gaaem,
                                id=1,
                                im=1,
                                N=N_gaaem,
                                method='gaaem'
                                )
t_gaaem=time.time()-t0
rps_gaaem = N_gaaem/t_gaaem

# %%
# SimPEG
# ^^^^^^

# %%
# SimPEG is an optional dependency; skip it if it is not installed.
t0=time.time()
rps_simpeg = None
try:
    f_prior_data_h5_simpeg = ig.prior_data_em(f_prior_h5, file_gex,
                                    doMakePriorCopy=True,
                                    randomize=False, # copy the FIRST N models, so all files share them
                                    f_prior_data_h5=f_prior_data_h5_simpeg,
                                    id=1,
                                    im=1,
                                    N=N_simpeg,
                                    method='simpeg'
                                    )
    t_simpeg=time.time()-t0
    rps_simpeg = N_simpeg/t_simpeg
except Exception as e:
    print('SimPEG forward not available (%s)' % e)
    f_prior_data_h5_simpeg = None

# %%
# All prior-data files actually produced, with a (file-safe) label for each.
# Everything below is done for ALL of these files.
f_prior_data_h5_arr = []
labels_arr = []
if f_prior_data_h5_anemone_gpu is not None:
    f_prior_data_h5_arr.append(f_prior_data_h5_anemone_gpu); labels_arr.append('anemone_gpu')
f_prior_data_h5_arr.append(f_prior_data_h5_anemone_cpu); labels_arr.append('anemone_cpu')
f_prior_data_h5_arr.append(f_prior_data_h5_gaaem);       labels_arr.append('gaaem')
if f_prior_data_h5_simpeg is not None:
    f_prior_data_h5_arr.append(f_prior_data_h5_simpeg); labels_arr.append('simpeg')

# %%
# Run time of each forward model, in realizations per second.
print('t_gaaem       = %7.1f ite/s' % (rps_gaaem))
print('t_anemone_cpu = %7.1f ite/s' % (rps_anemone_cpu))
if rps_anemone_gpu is not None:
    print('t_anemone_gpu = %7.1f ite/s' % (rps_anemone_gpu))
if rps_simpeg is not None:
    print('t_simpeg      = %7.1f ite/s' % (rps_simpeg))
print('--')
print('t_anemone vs ga-aem speedup     = %6.1f' % (rps_anemone_cpu/rps_gaaem))
if rps_anemone_gpu is not None:
    print('t_anemone_gpu vs ga-aem speedup = %6.1f' % (rps_anemone_gpu/rps_gaaem))
if rps_simpeg is not None:
    print('t_simpeg vs ga-aem speedup      = %6.1f' % (rps_simpeg/rps_gaaem))

# %%
# Read '/D1' from each of the prior-data files. The copies were made with
# randomize=False, so they all hold the same leading realizations of the prior
# and the first N_common rows (the smallest common sample) are directly
# comparable across files.
N_common = min([N_gaaem] + ([N_simpeg] if f_prior_data_h5_simpeg is not None else []))
D_arr = []
for f_ in f_prior_data_h5_arr:
    D_, _ = ig.load_prior_data(f_, id_use=[1], N_use=N_common, showInfo=0)
    D_arr.append(D_[0])

# Compare every forward against gaaem
D_gaaem = D_arr[labels_arr.index('gaaem')]

plt.figure()
plt.semilogy(D_gaaem[0:10].T, 'k-', linewidth=3, label='gaaem')
for D_, lab_ in zip(D_arr, labels_arr):
    if lab_ == 'gaaem':
        continue
    DD_mean = np.mean(np.abs(D_ - D_gaaem), axis=0)
    plt.semilogy(D_[0:10].T, '-', linewidth=1, label=lab_)
    plt.semilogy(DD_mean.T, ':', label='|%s - gaaem| mean' % lab_)
plt.xlabel('Gate id')
plt.ylabel('dB/dT')
plt.show()

# %%
# Prior data
# ~~~~~~~~~~

# %%
for f_ in f_prior_data_h5_arr:
    ig.plot_data_prior(f_, f_data_h5, nr=1000, id=1, id_data=1, hardcopy=hardcopy)

# %%
# 2. Sample the posterior distribution
# ------------------------------------

# %%
N_use = N   # Number of prior samples to use (use all available)
T_base = 1  # Base annealing temperature for rejection sampling
autoT = 1   # Automatically estimate optimal annealing temperature

# One posterior per prior-data file
f_post_h5_arr = []
for f_prior_data_h5_, lab_ in zip(f_prior_data_h5_arr, labels_arr):
    f_post_h5 = ig.integrate_rejection(f_prior_data_h5_,
                                    f_data_h5,
                                    f_post_h5 = 'POST_%s.h5' % (lab_),
                                    ip_range = i_line,
                                    N_use = N_use,
                                    autoT = autoT,
                                    T_base = T_base,
                                    showInfo=0,
                                    id_use=1,
                                    id_prior = 1, # Each prior file holds one data set, '/D1'
                                    updatePostStat = True)
    f_post_h5_arr.append(f_post_h5)

# %%
# 3. Plot statistics from the posterior
# -------------------------------------

# Compare prior and posterior data (disabled)
#for f_post_h5 in f_post_h5_arr:
#    ig.plot_data_prior_post(f_post_h5, i_plot=i_line[0],hardcopy=hardcopy)
#for f_post_h5 in f_post_h5_arr:
#    ig.plot_data_prior_post(f_post_h5, i_plot=i_line[-1],hardcopy=hardcopy)

# %%
# Evidence and annealing temperature
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The evidence quantifies how well the data fits the model,
# while temperature controls the acceptance rate in rejection sampling.
# Each forward model is compared against gaaem.

# %%
import h5py
CHI2 = []
EV = []
T = []
i=-1
for f_post_h5 in f_post_h5_arr:
    i=i+1
    with h5py.File(f_post_h5, 'r') as f:
        CHI2.append(f['CHI2'][:])
        EV.append(f['EV'][:])
        T.append(f['T'][:])
        print('Mean CHI2 for %s: %.3f - %s' % (f_post_h5, np.nanmean(CHI2[-1]),labels_arr[i]))
        print('Mean logEV for %s: %.3f - %s' % (f_post_h5, np.nanmean(EV[-1]),labels_arr[i]))
        print('Mean T for %s: %.3f - %s' % (f_post_h5, np.nanmean(T[-1]),labels_arr[i]))

i_ref = labels_arr.index('gaaem')   # gaaem is the reference on the x-axis
plt.figure(figsize=(15,5))
for i in range(3):
    plt.subplot(1,3,i+1)
    for j in range(len(f_post_h5_arr)):
        if j == i_ref:
            continue
        if i==0:
            plt.plot(CHI2[i_ref].flatten(),CHI2[j].flatten(), '.', markersize=1, label=labels_arr[j])
            plt.title('Comparison of CHI2 values')
        elif i==1:
            plt.plot(EV[i_ref].flatten(),EV[j].flatten(), '.', markersize=1, label=labels_arr[j])
            plt.title('Comparison of EV values')
        elif i==2:
            plt.plot(T[i_ref].flatten(),T[j].flatten(), '.', markersize=1, label=labels_arr[j])
            plt.title('Comparison of T values')
    xlim = plt.xlim();ylim = plt.ylim()
    lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
    plt.plot(lim, lim, 'k--')
    plt.gca().set_aspect('equal')
    plt.xlabel('gaaem')
    plt.ylabel('other forward')
    plt.legend()
    plt.grid()
plt.show()

# %%
# Resistivity profiles
# ~~~~~~~~~~~~~~~~~~~~

# %%
# Plot resistivity profile for model M1, one figure per forward model
for f_post_h5 in f_post_h5_arr:
    ig.plot_profile(f_post_h5, ii=i_line, im=1, 
                    xaxis='x', gap_threshold=50, 
                    title='Resistivity profile for %s' % f_post_h5,
                    key='HarmonicMean', hardcopy=hardcopy)
