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
import os
from itertools import product
import integrate as ig
from geoprior1d import geoprior1d

import numpy as np
import matplotlib.pyplot as plt
import h5py

hardcopy = True

# %% [markdown]
# ## Settings

# %%
N = 10_000_000        # Number of prior model realizations (increase for production runs)
N = 200_000        # Number of prior model realizations (increase for production runs)
nr = 10_000          # Number of posterior realizations per sounding


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
ig.plot_data_xy(f_data_h5, data_channel=15, cmap='jet')
ig.plot_data(f_data_h5, hardcopy=hardcopy)

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

ig.plot_geometry(f_data_h5, pl='ELEVATION', cmap='terrain')
fig = plt.gcf()
ax = fig.axes[0]  # plot_geometry adds the colorbar as a second axes
ax.plot(X[id_line], Y[id_line], 'r.', markersize=6, label='Profile (id_line)', zorder=2)
ax.plot(Xl, Yl, 'k--', linewidth=1, label='Line segment')
ax.legend()
ax.set_title('DAUGAARD survey - selected profile in red')
if hardcopy:
    fig.savefig('DAUGAARD_kasper_profile_map.png', dpi=300)
plt.show()

# %% [markdown]
# ## 3. Prior model and prior data
#
# Two `geoprior1d` priors (standard and buried-valley) are simulated and merged
# into one prior file. Prior tTEM data are computed with GA-AEM in log10 space,
# to match the log-space noise model.

# %%
# f_xlsx_files = ['daugaard_standard.xlsx', 'daugaard_valley.xlsx']
# #ig.get_case_data(case=case, filelist=f_xlsx_files)

# # Merge priors once, and reuse the file on later runs.
# f_prior_merged_h5 = 'daugaard_kasper_prior_N%d.h5' % (N * len(f_xlsx_files))
# if not os.path.exists(f_prior_merged_h5):
#     f_prior_h5_list = []
#     for file_xlsx in f_xlsx_files:
#         fname = file_xlsx.split('.')[0]
#         f_prior_h5, flags = geoprior1d(file_xlsx, Nreals=N, dz=1, dmax=90,
#                                        output_file='%s_prior_N%d.h5' % (fname, N))
#         f_prior_h5_list.append(f_prior_h5)

#     f_prior_h5 = ig.merge_prior(
#         f_prior_h5_list,
#         f_prior_merged_h5=f_prior_merged_h5,
#         showInfo=1,
#     )
    # ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)
    # ig.prior_describe(f_prior_h5)
# else:
#     f_prior_h5 = f_prior_merged_h5
#     print('Using existing prior file: %s' % f_prior_h5)

# %% Create prior models (layered) and save to HDF5 file. The prior is used for all noise-model combinations below.
f_prior_h5 = 'generic_PRIOR_N%d.h5' % N
if not os.path.exists(f_prior_h5):
    f_prior_h5 = ig.prior_model_layered(
        N=N,
        lay_dist='chi2',
        NLAY_deg=3,
        RHO_min=1,
        RHO_max=1000,
        f_prior_h5=f_prior_h5,
        showInfo=1,
        save_sparse=False
    )
else:
    print('Using existing prior file: %s' % f_prior_h5)


ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)
ig.prior_describe(f_prior_h5)

# %%
# Forward-model the prior once in log10 space and reuse it on later runs.
# The HDF5 file name is kept predictable so the inversion can skip this step.
file_basename = os.path.splitext(os.path.basename(file_gex))[0]
f_prior_data_h5 = '%s_%s_Nh280_Nf12.h5' % (
    os.path.splitext(f_prior_h5)[0], file_basename)
if not os.path.exists(f_prior_data_h5):
    f_prior_data_h5 = ig.prior_data_gaaem(
    #f_prior_data_h5 = ig.prior_data(
                f_prior_h5,
        file_gex,
        doMakePriorCopy=True,
        is_log=True,
        force_replace=False,
        showInfo=1,
    )
else:
    print('Using existing prior-data file: %s' % f_prior_data_h5)

# %% Optional linear comparison plot only for diagnostics, using the original data file.
# Keep this separate from the inversion data in log-space.
# f_prior_lin_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, doMakePriorCopy=False, is_log=False)
# ig.plot_data_prior(f_prior_lin_h5, f_data_h5, nr=1000, hardcopy=hardcopy)

# %% [markdown]
# ## 4. Noise model: correlated Gaussian noise in log10 space
#
# The observed data are transformed to log10 space. The noise is described by
# three numbers: `uncorr_std`, `corr_std`, and `corr_range`. The correlated
# component is constructed by `cd_correlated()`:
#
#     Cd = diag(std) @ R(corr_range) @ diag(std),   std = corr_std for all gates
#
# The diagonal of the correlated component is corr_std^2; corr_range controls
# how strongly neighbouring gates are correlated (0 = uncorrelated). An
# independent component adds uncorr_std^2 to every diagonal entry.
# The std in the data file (d_std) is NOT used. `cd_correlated` also accepts a
# per-gate std vector, if a gate-dependent std is wanted.

# %%
# Noise model settings (all combinations are run below)
corr_stds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1]       # Correlated noise std values in log10 units
uncorr_stds = [0, 0.01, 0.02, 0.03, 0.05]     # Uncorrelated noise std values in log10 units
corr_ranges = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]            # Correlation ranges in gate numbers
corr_model = 'exponential'      # 'exponential' or 'gaussian'

corr_stds = [0.00, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32]       # Correlated noise std values in log10 units
uncorr_stds = [0.03]     # Uncorrelated noise std values in log10 units
corr_ranges = [0, 1, 4, 8, 16, 32]            # Correlation ranges in gate numbers
corr_model = 'gaussian'      # 'exponential' or 'gaussian'
corr_model = 'exponential'      # 'exponential' or 'gaussian'


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

i_plot = id_line[len(id_line) // 2]


# %% [markdown]
# ## 5. Inversion and statistics for every noise-model combination
#
# Each combination gets separate data, posterior, and plot files.

# %%
summary = []
ev_profiles = []

for uncorr_std, corr_std, corr_range in product(
        uncorr_stds, corr_stds, corr_ranges):
    case_tag = 'us%g_cs%g_cr%g_%s' % (
        uncorr_std, corr_std, corr_range, corr_model)
    print('\nRunning noise case: %s' % case_tag)

    # Add independent noise variance to the diagonal of the correlated model.
    Cd_correlated = cd_correlated(
        corr_std * np.ones(nd), corr_range, corr_model)
    Cd_uncorrelated = (uncorr_std ** 2) * np.eye(nd)
    Cd_single = Cd_correlated + Cd_uncorrelated
    lCd = np.broadcast_to(Cd_single, (ns, nd, nd)).copy()

    f_data_noise_h5 = '%s_kasper_log_corr_%s.h5' % (
        os.path.splitext(f_data_h5)[0], case_tag)
    ig.copy_hdf5_file(f_data_h5, f_data_noise_h5)
    ig.save_data_gaussian(lD_obs, Cd=lCd, f_data_h5=f_data_noise_h5,
                          id=1, is_log=1, f_gex=file_gex)

    # Visualize this covariance model for one sounding on the profile.
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].errorbar(np.arange(nd), lD_obs[i_plot],
                   yerr=np.sqrt(np.diag(lCd[i_plot])), fmt='k.', capsize=2)
    ax[0].set_title('log10(d_obs) +/- sqrt(diag(Cd)), sounding %d' % i_plot)
    ax[0].set_xlabel('Gate')
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    im = ax[1].imshow(lCd[i_plot], cmap='viridis')
    ax[1].set_title('Cd: uncorr std=%g, corr std=%g, range=%g, %s' %
                    (uncorr_std, corr_std, corr_range, corr_model))
    fig.colorbar(im, ax=ax[1])

    h = np.arange(nd)
    marginal_std = np.sqrt(np.diag(lCd[i_plot]))
    correlation = lCd[i_plot][0, :] / (marginal_std[0] * marginal_std)
    ax[2].plot(h, correlation, 'k.-')
    ax[2].set_xlabel('Gate distance h')
    ax[2].set_ylabel('Correlation')
    ax[2].set_title('Correlation vs gate distance')
    ax[2].grid(True, linestyle='--', linewidth=0.5)
    fig.tight_layout()
    if hardcopy:
        fig.savefig('DAUGAARD_kasper_noise_model_%s.png' % case_tag, dpi=300)
    plt.show()

    f_post_h5 = 'post_kasper_log_corr_%s_N%d.h5' % (case_tag, N)
    f_post_h5 = ig.integrate_rejection(
        f_prior_data_h5, f_data_noise_h5, f_post_h5,
        ip_range=id_line,
        nr=nr,
        updatePostStat=True,
        autoT=False,
        normalize_likelihood=True,
        showInfo=1)

    ig.plot_profile(f_post_h5, im=1, ii=id_line, xaxis='x', gap_threshold=50,
                    show_n_unique=True, hardcopy=hardcopy)
    ig.plot_data_prior_post(f_post_h5, i_plot=i_plot, is_log=True,
                            hardcopy=hardcopy)

    STATS = {}
    with h5py.File(f_post_h5, 'r') as f:
        STATS['T'] = f['/T'][:][id_line]
        STATS['EV'] = f['/EV'][:][id_line]
        STATS['N_UNIQUE'] = f['/N_UNIQUE'][:][id_line]
        STATS['CHI2'] = f['/CHI2'][:][id_line]
        STATS['LOGSTD'] = np.nanmean(f['/M1/LogStd'][:][id_line, :], axis=1)

    ev_profiles.append((uncorr_std, corr_std, corr_range, STATS['EV'].copy()))
    

    summary.append([
        uncorr_std, corr_std, corr_range,
        np.nanmean(STATS['T']), np.nanmean(STATS['EV']),
        np.nanmean(STATS['N_UNIQUE']), np.nanmean(STATS['CHI2']),
        np.nanmean(STATS['LOGSTD'])])

    keys = ['T', 'EV', 'N_UNIQUE', 'CHI2', 'LOGSTD']
    fig, ax = plt.subplots(len(keys), 1, figsize=(12, 3 * len(keys)), sharex=True)
    for k, key in enumerate(keys):
        ax[k].plot(X[id_line], STATS[key], 'k.')
        ax[k].set_ylabel(key)
        ax[k].grid(True, linestyle='--', linewidth=0.5)
        if key in ['T', 'N_UNIQUE']:
            ax[k].set_yscale('log')
    ax[-1].set_xlabel('Easting (m)')
    fig.suptitle(case_tag)
    fig.tight_layout()
    if hardcopy:
        fig.savefig('DAUGAARD_kasper_profile_stats_%s_N%d.png' % (case_tag, N), dpi=300)
    plt.show()


# %%
print('\nProfile-wide mean statistics')
print('%8s %8s %10s %10s %10s %10s %10s %10s' %
    ('UStd', 'CStd', 'Range', 'T', 'EV', 'N_UNIQUE', 'CHI2', 'LogStd'))
for row in summary:
    print('%8.3f %8.3f %10.1f %10.2f %10.2f %10.1f %10.3f %10.3f' % tuple(row))
if hardcopy:
    with open('DAUGAARD_kasper_profile_stats_summary_N%d.txt' % N, 'w') as f:
        f.write('%8s %8s %10s %10s %10s %10s %10s %10s\n' %
                ('UStd', 'CStd', 'Range', 'T', 'EV', 'N_UNIQUE', 'CHI2', 'LogStd'))
        for row in summary:
            f.write('%8.3f %8.3f %10.1f %10.2f %10.2f %10.1f %10.3f %10.3f\n' %
                    tuple(row))

# %% Plot the noise-model parameters with the highest EV at each profile index.
ev_values = np.asarray([row[3] for row in ev_profiles])
best_uncorr = np.full(len(id_line), np.nan)
best_corr = np.full(len(id_line), np.nan)
best_range = np.full(len(id_line), np.nan)
best_ev = np.full(len(id_line), np.nan)
has_ev = np.any(np.isfinite(ev_values), axis=0)
best_case_index = np.full(len(id_line), -1, dtype=int)
best_case_index[has_ev] = np.nanargmax(ev_values[:, has_ev], axis=0)

for profile_index in np.flatnonzero(has_ev):
    case_index = best_case_index[profile_index]
    best_uncorr[profile_index] = ev_profiles[case_index][0]
    best_corr[profile_index] = ev_profiles[case_index][1]
    best_range[profile_index] = ev_profiles[case_index][2]
    best_ev[profile_index] = ev_values[case_index, profile_index]

best_parameters = np.column_stack(
    (X[id_line], Y[id_line], best_ev, best_uncorr, best_corr, best_range))
np.savetxt(
    'DAUGAARD_kasper_best_noise_parameters_N%d.txt' % N,
    best_parameters,
    header='Easting Northing EV UncorrStd CorrStd CorrRange',
    fmt='%.8g')

fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
profile_x = X[id_line]
ax[0].plot(profile_x, best_uncorr, 'k.')
ax[0].set_ylabel('Uncorr std')
ax[1].plot(profile_x, best_corr, 'k.')
ax[1].set_ylabel('Corr std')
ax[2].plot(profile_x, best_range, 'k.')
ax[2].set_ylabel('Corr range')
ax[2].set_xlabel('Easting (m)')
for axis in ax:
    axis.grid(True, linestyle='--', linewidth=0.5)
fig.suptitle('Noise-model parameters with highest EV at each profile index')
fig.tight_layout()
if hardcopy:
    fig.savefig('DAUGAARD_kasper_best_noise_parameters_N%d.png' % N, dpi=300)
plt.show()

#%% Plot the soundings of the profile with the colormap tied to best_corr and best_range. This is a diagnostic plot to see how the noise model varies along the profile.
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sc1 = ax[0].scatter(X[id_line], Y[id_line], c=best_corr, cmap='plasma', s=20)
ax[0].set_title('Best correlated std along profile')
ax[0].set_xlabel('Easting (m)')
ax[0].set_ylabel('Northing (m)')
sc2 = ax[1].scatter(X[id_line], Y[id_line], c=best_range, cmap='plasma', s=20)
ax[1].set_title('Best correlation range along profile')
ax[1].set_xlabel('Easting (m)')
ax[1].set_ylabel('Northing (m)')
fig.colorbar(sc1, ax=ax[0], label='Best correlated std')
fig.colorbar(sc2, ax=ax[1], label='Best correlation range')
fig.tight_layout()
if hardcopy:
    fig.savefig('DAUGAARD_kasper_best_noise_parameters_map_N%d.png' % N, dpi=300)
plt.show()



# %% Plot EV grids for each uncorrelated std.
uncorr_values = np.array(sorted({row[0] for row in summary}))
corr_values = np.array(sorted({row[1] for row in summary}))
range_values = np.array(sorted({row[2] for row in summary}))

for uncorr_std in uncorr_values:
    ev_grid = np.full((len(range_values), len(corr_values)), np.nan)
    for row in summary:
        row_uncorr, corr_std, corr_range, _, ev, *_ = row
        if row_uncorr == uncorr_std:
            corr_index = np.where(corr_values == corr_std)[0][0]
            range_index = np.where(range_values == corr_range)[0][0]
            ev_grid[range_index, corr_index] = ev

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(ev_grid, aspect='auto', origin='lower', cmap='viridis')
    ax.set_xticks(np.arange(len(corr_values)),
                  labels=[f'{value:g}' for value in corr_values])
    ax.set_yticks(np.arange(len(range_values)),
                  labels=[f'{value:g}' for value in range_values])
    ax.set_xlabel('Correlated Standard Deviation')
    ax.set_ylabel('Correlation Range')
    ax.set_title('Evidence (EV), Uncorrelated Std=%g' % uncorr_std)
    fig.colorbar(image, ax=ax, label='Evidence (EV)')

    for range_index in range(len(range_values)):
        for corr_index in range(len(corr_values)):
            value = ev_grid[range_index, corr_index]
            if np.isfinite(value):
                ax.text(corr_index, range_index, f'{value:.2f}',
                        ha='center', va='center', color='white')

    fig.tight_layout()
    if hardcopy:
        fig.savefig('DAUGAARD_kasper_profile_EV_grid_us%g_N%d.png' % (uncorr_std, N), dpi=300)
    plt.show()

# %%
