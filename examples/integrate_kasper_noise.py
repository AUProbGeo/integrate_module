#!/usr/bin/env python
# %% [markdown]
# # DAUGAARD: correlated noise (log space) along a single profile
#
# 1. Load the DAUGAARD tTEM data
# 2. Select a profile and show it on an XY map
# 3. Set up a prior using `geoprior1d` (standard + valley, merged) and compute prior data (log10 space)
# 4. Set up the noise model: correlated Gaussian noise in log10 space
# 5. Invert the data along the profile, plot the profile (resistivity M1, lithology M2)
#    and collect statistics (temperature, evidence, number of unique accepted models, entropy)
#
# The inversion is restricted to the soundings on the profile (`ip_range=id_line`)
# to keep the runtime short.

# %%
try:
    # Check if the code is running in an IPython kernel (which includes Jupyter notebooks)
    get_ipython()
    # If the above line doesn't raise an error, it means we are in a Jupyter environment
    # Execute the magic commands using IPython's run_line_magic function
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    # If get_ipython() raises an error, we are not in a Jupyter environment
    pass

# %%
import os
import integrate as ig
from geoprior1d import geoprior1d

import numpy as np
import matplotlib.pyplot as plt
import h5py

hardcopy = True

# %% [markdown]
# ## Settings

# %%
N = 100_000        # Number of prior model realizations (increase for production runs)
N = 5_000        # Number of prior model realizations (increase for production runs)
nr = 1000          # Number of posterior realizations per sounding


def cd_correlated(std, corr_range, model='exponential'):
    """
    Correlated Gaussian noise covariance matrix for a single sounding.

    Cd = diag(std) @ R @ diag(std), where R is a correlation matrix with
    R[i,j] = exp(-3 h/a)      (exponential)
    R[i,j] = exp(-3 (h/a)^2)  (gaussian)
    with h = |i-j| (distance in data numbers) and a = corr_range.
    The factor 3 gives the 'practical range': correlation ~0.05 at h = a.
    corr_range = 0 gives R = I, i.e. an uncorrelated (diagonal) Cd.

    std : array of shape (nd,) with the noise std of each data point.
          The diagonal of Cd is std**2 regardless of corr_range.
    """
    std = np.asarray(std, dtype=float)
    nd = len(std)
    h = np.abs(np.arange(nd)[:, None] - np.arange(nd)[None, :])
    if corr_range <= 0:
        R = np.eye(nd)
    elif model == 'exponential':
        R = np.exp(-3 * h / corr_range)
    elif model == 'gaussian':
        R = np.exp(-3 * (h / corr_range)**2)
        R = R + 1e-6 * np.eye(nd)  # nugget: keeps the gaussian model numerically invertible
    else:
        raise ValueError("model must be 'exponential' or 'gaussian'")
    return std[:, None] * R * std[None, :]

# %% [markdown]
# ## 1. Load the DAUGAARD data and GEX file

# %%
case = 'DAUGAARD'
files = ig.get_case_data(case=case)
f_data_h5 = files[0]
file_gex = ig.get_gex_file_from_data(f_data_h5)
print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)

# Plot the observed data (d_obs, d_std) and one data channel on an XY map
ig.plot_data(f_data_h5, hardcopy=hardcopy)
ig.plot_data_xy(f_data_h5, data_channel=15, cmap='jet')

# %% [markdown]
# ## 2. Select a profile and show it on an XY map
#
# The profile is all soundings within a buffer distance of a UTM line segment.
# `id_line` is used both to restrict the inversion (`ip_range=`) and to plot
# the profile (`ii=`).

# %%
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

Xl = np.array([544500, 543150])
Yl = np.array([6175000, 6176500])
buffer = 10.0
id_line, distances, segment_ids = ig.find_points_along_line_segments(
    X, Y, Xl, Yl, tolerance=buffer
)
print("Number of soundings on profile: %d" % len(id_line))

ig.plot_geometry(f_data_h5, pl='ELEVATION', cmap='viridis')
plt.plot(X[id_line], Y[id_line], 'r.', markersize=6, label='Profile (id_line)', zorder=2)
plt.plot(Xl, Yl, 'k--', linewidth=1, label='Line segment')
plt.legend()
plt.title('DAUGAARD survey - selected profile in red')
if hardcopy:
    plt.savefig('DAUGAARD_kasper_profile_map.png', dpi=300)
plt.show()

# %% [markdown]
# ## 3. Prior model and prior data
#
# Two `geoprior1d` priors (standard and buried-valley) are simulated and merged
# into one prior file. Prior tTEM data are computed with GA-AEM in log10 space,
# to match the log-space noise model.

# %%
f_xlsx_files = ['daugaard_standard.xlsx', 'daugaard_valley.xlsx']
ig.get_case_data(case=case, filelist=f_xlsx_files)

f_prior_h5_list = []
for file_xlsx in f_xlsx_files:
    fname = file_xlsx.split('.')[0]
    f_prior_h5, flags = geoprior1d(file_xlsx, Nreals=N, dz=1, dmax=90,
                                   output_file='%s_prior_N%d.h5' % (fname, N))
    f_prior_h5_list.append(f_prior_h5)

f_prior_h5 = ig.merge_prior(f_prior_h5_list, f_prior_merged_h5='daugaard_kasper_prior_N%d.h5' % N)
ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)
ig.prior_describe(f_prior_h5)

# %%
# Prior data in log10 space (written to /D1 in f_prior_h5)
f_prior_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, doMakePriorCopy=False, is_log=True)

# %% [markdown]
# ## 4. Noise model: correlated Gaussian noise in log10 space
#
# The observed data are transformed to log10 space. The noise is described by
# two numbers, `corr_std` and `corr_range`, through `cd_correlated()`:
#
#     Cd = diag(std) @ R(corr_range) @ diag(std),   std = corr_std for all gates
#
# The diagonal of Cd is corr_std^2; corr_range controls how strongly
# neighbouring gates are correlated (0 = uncorrelated).
# The std in the data file (d_std) is NOT used. `cd_correlated` also accepts a
# per-gate std vector, if a gate-dependent std is wanted.

# %% 
# Noise model settings (correlated Gaussian noise in log10 space)
corr_std = 0.03           # Std of the noise in log10 units (sqrt of the diagonal of Cd)
corr_range = 4            # Correlation range in units of data numbers (gates); 0 = uncorrelated
corr_model = 'exponential'  # 'exponential' or 'gaussian' covariance model



# %%
DATA = ig.load_data(f_data_h5)
D_obs = DATA['d_obs'][0]
ns, nd = D_obs.shape

# Gates with non-positive d_obs cannot be used (undefined in log space): mark as missing
i_bad = ~(D_obs > 0)
print("Gates marked as missing (d_obs <= 0): %d" % np.sum(i_bad & np.isfinite(D_obs)))
D_obs[i_bad] = np.nan

# Log-space data. NaN gates (not used) stay NaN; the warnings from log10(NaN) are silenced.
with np.errstate(divide='ignore', invalid='ignore'):
    lD_obs = np.log10(D_obs)

# Full covariance per sounding (the same for all soundings here)
Cd_single = cd_correlated(corr_std * np.ones(nd), corr_range, corr_model)
lCd = np.zeros((ns, nd, nd))
for i in range(ns):
    lCd[i] = Cd_single

# Write the data file (geometry is kept by copying the original file)
f_data_noise_h5 = '%s_kasper_log_corr_cs%g_cr%g_%s.h5' % (
    os.path.splitext(f_data_h5)[0], corr_std, corr_range, corr_model)
ig.copy_hdf5_file(f_data_h5, f_data_noise_h5)
ig.save_data_gaussian(lD_obs, Cd=lCd, f_data_h5=f_data_noise_h5, id=1, is_log=1, f_gex=file_gex)

# %%
# Visualize the noise model for one sounding on the profile
i_plot = id_line[len(id_line) // 2]

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].errorbar(np.arange(nd), lD_obs[i_plot], yerr=np.sqrt(np.diag(lCd[i_plot])), fmt='k.', capsize=2)
ax[0].set_title('log10(d_obs) +/- sqrt(diag(Cd)), sounding %d' % i_plot)
ax[0].set_xlabel('Gate')
ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)

im = ax[1].imshow(lCd[i_plot], cmap='viridis')
ax[1].set_title('Cd: std=%g, range=%g, %s' % (corr_std, corr_range, corr_model))
plt.colorbar(im, ax=ax[1])

# Correlation as a function of gate distance
h = np.arange(nd)
ax[2].plot(h, lCd[i_plot][0, :] / corr_std**2, 'k.-')
ax[2].set_xlabel('Gate distance h')
ax[2].set_ylabel('Correlation')
ax[2].set_title('Correlation vs gate distance')
ax[2].grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
if hardcopy:
    plt.savefig('DAUGAARD_kasper_noise_model.png', dpi=300)
plt.show()

# %% [markdown]
# ## 5. Inversion along the profile
#
# Only the soundings on the profile are inverted (`ip_range=id_line`).
# Posterior statistics (Mean, Median, Std, Mode, Entropy, N_UNIQUE, ...)
# are computed by `updatePostStat=True`.

# %%
f_post_h5 = 'post_kasper_log_corr_N%d.h5' % N
f_post_h5 = ig.integrate_rejection(f_prior_h5, f_data_noise_h5, f_post_h5,
                                   ip_range=id_line,
                                   nr=nr,
                                   updatePostStat=True,
                                   autoT=False,
                                   normalize_likelihood=True, # important to get the full Evidence!
                                   showInfo=1)

# %% [markdown]
# ### Profiles: resistivity (M1) and lithology (M2)

# %%
ig.plot_profile(f_post_h5, im=1, ii=id_line, xaxis='x', gap_threshold=50,
                show_n_unique=True, hardcopy=hardcopy)
ig.plot_profile(f_post_h5, im=2, ii=id_line, xaxis='x', gap_threshold=50,
                show_n_unique=True, hardcopy=hardcopy)

# %% [markdown]
# ### Observed data vs prior and posterior data for one sounding

# %%
ig.plot_data_prior_post(f_post_h5, i_plot=i_plot, is_log=True, hardcopy=hardcopy)

# %% [markdown]
# ### Statistics along the profile
#
# * `T` – annealing temperature needed to accept `nr` realizations (1 = no tempering)
# * `EV` – log evidence
# * `N_UNIQUE` – number of unique accepted prior models
# * `M2/Entropy` – normalised entropy of the lithology, averaged over depth
# * `M1/LogStd` – std of log10(resistivity), averaged over depth

# %%
STATS = {}
with h5py.File(f_post_h5, 'r') as f:
    STATS['T'] = f['/T'][:][id_line]
    STATS['EV'] = f['/EV'][:][id_line]
    STATS['N_UNIQUE'] = f['/N_UNIQUE'][:][id_line]
    STATS['ENTROPY'] = np.nanmean(f['/M2/Entropy'][:][id_line, :], axis=1)
    STATS['LOGSTD'] = np.nanmean(f['/M1/LogStd'][:][id_line, :], axis=1)

print('%8s %10s %10s %10s %10s' % ('T', 'EV', 'N_UNIQUE', 'Entropy', 'LogStd'))
print('%8.2f %10.2f %10.1f %10.3f %10.3f' % (
    np.nanmean(STATS['T']), np.nanmean(STATS['EV']), np.nanmean(STATS['N_UNIQUE']),
    np.nanmean(STATS['ENTROPY']), np.nanmean(STATS['LOGSTD'])))

# %%
keys = ['T', 'EV', 'N_UNIQUE', 'ENTROPY', 'LOGSTD']
fig, ax = plt.subplots(len(keys), 1, figsize=(12, 3 * len(keys)), sharex=True)
for k, key in enumerate(keys):
    ax[k].plot(X[id_line], STATS[key], 'k.')
    ax[k].set_ylabel(key)
    ax[k].grid(True, linestyle='--', linewidth=0.5)
    if key in ['T', 'N_UNIQUE']:
        ax[k].set_yscale('log')
ax[-1].set_xlabel('Easting (m)')
plt.tight_layout()
if hardcopy:
    plt.savefig('DAUGAARD_kasper_profile_stats_N%d.png' % N, dpi=300)
plt.show()

# %%
