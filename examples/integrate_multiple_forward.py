#!/usr/bin/env python
# %% [markdown]
# # Effect of using different forward models INTEGRATE
#

# %%
import integrate as ig
hardcopy = True
import matplotlib.pyplot as plt
import numpy as np
import time

#
N=500_000


# %% [markdown]
# ## 0. Get TTEM data

# %%
case = 'DAUGAARD'
files = ig.get_case_data(case=case, showInfo=2)
f_data_h5 = files[0]
file_gex= ig.get_gex_file_from_data(f_data_h5)

print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)

# %% [markdown]
# Select a profile

# %%
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
# Find points within buffer distance
Xl = np.array([544000, 543550])
Yl = np.array([6174500, 6176500])
#Xl = np.array([544000, 543550, 543000])
#Yl = np.array([6174500, 6176500, 6176400])
buffer = 10.0
indices, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)
i_line = indices

ig.plot_geometry(f_data_h5, pl='ELEVATION')   # opens its own figure
plt.plot(Xl, Yl,'ko', markersize=15)
plt.plot(X[i_line], Y[i_line], 'ko', markersize=5, zorder=3)
plt.title('Profile line')
plt.show()

i_use = np.arange(len(X))

# %%
# The electromagnetic data (d_obs and d_std) can be plotted using ig.plot_data:
#ig.plot_data(f_data_h5, hardcopy=hardcopy)
# Plot data channel 15 in an XY grid
ig.plot_data_xy(f_data_h5, data_channel=15, cmap='jet');

# %% [markdown]
# ## 1. Set up the prior model ($\rho(\mathbf{m},\mathbf{d})$)

# %%
# Select how many prior model realizations (N) should be generated
f_prior_h5 = ig.prior_model_layered(N=N,
                                    lay_dist='uniform',
                                    NLAY_min=5,
                                    NLAY_max=5,                 # Minimum 3 layer
                                    #lay_dist='chi2',
                                    #NLAY_max=6,
                                    RHO_deg=3,
                                    RHO_max=3000,
                                    f_prior_h5='PRIOR_N%d.h5' % N,
                                    showInfo=1)
print('%s is used to hold prior realizations' % (f_prior_h5))


# %%
# Plot summary statistics of the prior model for quality control of the prior choice
ig.plot_prior_stats(f_prior_h5, im=1, panels=['hist'],hardcopy=hardcopy)


# %% [markdown]
# ### 1b. Generate corresponding prior data
# Next, we generate a corresponding sample of $\rho(\mathbf{d})$ (prior data distribution).
# Here we test using both ga-aem and anemone

# %% Three separate prior-data files, one per forward model
# prior_data_em() only honours N when doMakePriorCopy=True: the copy is then
# truncated to N realizations of /M1 and the forward is run on all of them.
# With doMakePriorCopy=False the forward is applied to EVERY model in the file,
# whatever N is. So to run each forward on a different N, each forward gets its
# own copy of the prior. The data is '/D1' in each file.
N_anemone_gpu = N
N_anemone_cpu = int(N/5)
N_gaaem       = int(N/10)

f_stem = '%s_%s' % (f_prior_h5[:-3], file_gex[:-4])
f_prior_data_h5_anemone_gpu = '%s_anemone_gpu_N%d.h5' % (f_stem, N_anemone_gpu)
f_prior_data_h5_anemone_cpu = '%s_anemone_cpu_N%d.h5' % (f_stem, N_anemone_cpu)
f_prior_data_h5_gaaem       = '%s_gaaem_N%d.h5'       % (f_stem, N_gaaem)

# %% AnEMone GPU
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

# %% AnEMone CPU
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

# %% GA-AEM
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
# All prior-data files actually produced, with a (file-safe) label for each.
# Everything below is done for ALL of these files.
f_prior_data_h5_arr = []
labels_arr = []
if f_prior_data_h5_anemone_gpu is not None:
    f_prior_data_h5_arr.append(f_prior_data_h5_anemone_gpu); labels_arr.append('anemone_gpu')
f_prior_data_h5_arr.append(f_prior_data_h5_anemone_cpu); labels_arr.append('anemone_cpu')
f_prior_data_h5_arr.append(f_prior_data_h5_gaaem);       labels_arr.append('gaaem')

#%%
print('t_gaaem = %3.1fs/s' % (rps_gaaem))
print('t_anemone_cpu = %3.1fs/s' % (rps_anemone_cpu))
print('t_anemone vs ga-aem speedup = %3.1f' % (rps_gaaem/rps_anemone_cpu))
if rps_anemone_gpu is not None:
    print('t_anemone_gpu = %3.1fs/s' % (rps_anemone_gpu))
    print('t_anemone_gpu vs ga-aem speedup = %3.1f' % (rps_gaaem/rps_anemone_gpu))
for f_, lab_ in zip(f_prior_data_h5_arr, labels_arr):
    print('%s holds prior model and data realizations for %s' % (f_, lab_))

# %%
# Read '/D1' from each of the prior-data files. The copies were made with
# randomize=False, so they all hold the same leading realizations of the prior
# and the first N_gaaem rows (the smallest common sample) are directly
# comparable across files.
D_arr = []
for f_ in f_prior_data_h5_arr:
    D_, _ = ig.load_prior_data(f_, id_use=[1], N_use=N_gaaem, showInfo=0)
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

# %% [markdown]
# Prior data

# %%
for f_ in f_prior_data_h5_arr:
    ig.plot_data_prior(f_, f_data_h5, nr=1000, id=1, id_data=1, hardcopy=hardcopy)

# %% [markdown]
# ## 2. Sample the posterior distribution $\sigma(\mathbf{m})$

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

# %% [markdown]
# ## 3. Plot statistics from the posterior $\sigma(\mathbf{m})$
#
# ### Compare prior and posterior data
for f_post_h5 in f_post_h5_arr:
    ig.plot_data_prior_post(f_post_h5, i_plot=i_line[0],hardcopy=hardcopy)
for f_post_h5 in f_post_h5_arr:
    ig.plot_data_prior_post(f_post_h5, i_plot=i_line[-1],hardcopy=hardcopy)

# %% [markdown]
# ### Evidence and annealing temperature
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
    plt.ylabel('anemone')
    plt.legend()
    plt.grid()
plt.show()

# %% [markdown]
# ### Resistivity profiles

# %%
# Plot resistivity profile for model M1, one figure per forward model
for f_post_h5 in f_post_h5_arr:
    ig.plot_profile(f_post_h5, ii=i_line, im=1, key='HarmonicMean', hardcopy=hardcopy)
