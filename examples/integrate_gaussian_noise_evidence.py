#!/usr/bin/env python
# %% [markdown]
# # Evidence-based comparison of assumed noise levels
#
# Uses the same 3-layer reference model/data as `integrate_gaussian_noise.py`.
# The SAME observed data (one noise realization) is then inverted 4 times,
# each time assuming a different -- but still uncorrelated Gaussian --
# noise level:
#
#     noise_scale = [0.5, 1, 2, 4]  x  the "true" noise std used to
#                                      generate the data
#
# Each of the 4 assumed-noise inversions is repeated with
# `normalize_likelihood=False` (default) and `normalize_likelihood=True`.
# The Bayesian evidence (`EV`) is then plotted along the profile for all
# 8 cases.
#
# The point: without the Gaussian normalization constant, `EV` is only
# comparable *within* a single noise assumption (same std at every prior
# sample of a given data point) -- it is NOT comparable *across* the 4
# noise-scale cases, because the missing normalization constant itself
# depends on the assumed std. `normalize_likelihood=True` fixes this, so
# EV becomes a fair way to ask "which of these assumed noise levels is
# best supported by the data?".
# %%
try:
    # Check if the code is running in an IPython kernel (which includes Jupyter notebooks)
    get_ipython()
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    pass

import integrate as ig

import numpy as np
import time
import matplotlib.pyplot as plt
import h5py
hardcopy = True


# %% [markdown]
# ## Create the reference model and data
# Same 3-layer reference model as in `integrate_gaussian_noise.py`.

# %%
case = '3layer'
z_max = 60
dx = 1
rho = [120, 10, 120]
M_ref, x_ref, z_ref, M_ref_lith, layer_depths = ig.synthetic_case(
    case='3layer', dx=dx, rho1_1=rho[0], rho1_2=rho[1], x_max=1000, x_range=50)

f_data_h5 = '%s_%d' % (case, z_max)
thickness = np.diff(z_ref)
file_gex = ig.get_case_data(case='DAUGAARD', filelist=['TX07_20231016_2x4_RC20-33.gex'])[0]

# Compute the noise-free reference data
D_ref = ig.forward_gaaem(C=1./M_ref, thickness=thickness, file_gex=file_gex)

# %% [markdown]
# ### Plot the reference model and data

# %%
cmap, clim = ig.get_colormap_and_limits(cmap_type='resistivity')

plt.subplot(2, 1, 1)
xx_ref, zz_ref = np.meshgrid(x_ref, z_ref)
plt.pcolormesh(xx_ref, zz_ref, np.log10(M_ref.T), cmap=cmap, vmin=np.log10(clim[0]), vmax=np.log10(clim[1]))
plt.xlim([x_ref.min(), x_ref.max()])
plt.xlabel('Distance (m)')
plt.ylabel('Depth (m)')
plt.gca().invert_yaxis()
plt.colorbar(label='Resistivity (Ohm-m)')

plt.subplot(2, 1, 2)
plt.semilogy(x_ref, D_ref)
plt.xlim([x_ref.min(), x_ref.max()])
plt.xlabel('Distance (m)')
plt.ylabel('Amplitude')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)


# %% [markdown]
# ## Create prior model and data

# %%
N = 2000000  # sample size
NLAY_min = 3
NLAY_max = 3

f_prior_data_h5 = 'PRIOR_UNIFORM_NL_%d-%d_uniform_N%d_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5' % (NLAY_min, NLAY_max, N)

# make prior model realizations
f_prior_h5 = ig.prior_model_layered(N=N,
                                    lay_dist='uniform', z_max=z_max,
                                    NLAY_min=NLAY_min, NLAY_max=NLAY_max,
                                    RHO_dist='uniform', RHO_min=0.5*min(rho), RHO_max=2*max(rho))

# make prior data realizations
f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex)

ig.plot_prior_stats(f_prior_h5)


# %% [markdown]
# ## Observed data: one noise realization, several assumed noise levels
#
# `D_obs` is generated ONCE using a 3% relative noise level (the "true"
# noise). It is then inverted 4 times, each time assuming a *different*
# uncorrelated Gaussian noise level -- scaled by `noise_scale` relative to
# the true noise -- to mimic not knowing the true noise level precisely.

# %%
rng = np.random.default_rng()
d_std_rel = 0.03      # "true" relative noise level
d_std_base = 1e-12    # noise floor
D_std_true = d_std_rel * D_ref + d_std_base

D_noise = rng.normal(0, D_std_true, D_ref.shape)
D_obs = D_ref + D_noise

noise_scale = [0.5, 1, 2, 4]


def scale_str(s):
    return ("%g" % s).replace('.', 'p')


f_data_h5_arr = []
name_arr = []
for scale in noise_scale:
    # Assumed (uncorrelated Gaussian) noise level used for the inversion --
    # note D_obs itself is unchanged, only the assumed std differs.
    D_std_assumed = scale * D_std_true
    f_out = ig.save_data_gaussian(D_obs, D_std=D_std_assumed,
                                  f_data_h5='data_uncorr_scale_%s.h5' % scale_str(scale),
                                  id=1, showInfo=0, delete_if_exist=True)
    f_data_h5_arr.append(f_out)
    name_arr.append('noise_scale=%s' % scale_str(scale).replace('p', '.'))


# %% [markdown]
# ## Invert with each assumed noise level, with and without normalization
#
# Each of the 4 datasets above is inverted twice: once with the default
# (unnormalized) Gaussian likelihood, and once with
# `normalize_likelihood=True`. `f_post_h5` is set explicitly for each of
# the 8 runs -- the auto-generated posterior filename only depends on the
# prior file, not on the data file or normalize_likelihood, so leaving it
# unset would make every run overwrite the same output file.

# %%
normalize_settings = [False, True]

T = {}       # T[(i_scale, normalize_likelihood)]
EV = {}
CHI2 = {}
f_post_h5_dict = {}
t_elapsed = {}

for i_scale, f_data_h5_i in enumerate(f_data_h5_arr):
    for normalize_likelihood in normalize_settings:
        t0 = time.time()
        f_post_h5 = 'POST_gaussian_noise_evidence_scale_%s_norm%d.h5' % (
            scale_str(noise_scale[i_scale]), int(normalize_likelihood))
        f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5_i,
                                           f_post_h5=f_post_h5,
                                           Ncpu=8,
                                           normalize_likelihood=normalize_likelihood,
                                           )
        t_elapsed[(i_scale, normalize_likelihood)] = time.time() - t0

        with h5py.File(f_post_h5, 'r') as f:
            T[(i_scale, normalize_likelihood)] = f['/T'][:]
            EV[(i_scale, normalize_likelihood)] = f['/EV'][:]
            CHI2[(i_scale, normalize_likelihood)] = f['/CHI2'][:]
        f_post_h5_dict[(i_scale, normalize_likelihood)] = f_post_h5

        print('%s, normalize_likelihood=%s: t_elapsed = %.1f s' % (
            name_arr[i_scale], normalize_likelihood, t_elapsed[(i_scale, normalize_likelihood)]))


# %% [markdown]
# ## Plot the log-evidence along the profile
#
# `EV`, as stored by `integrate_rejection`, is the natural-log evidence
# log(p(data | hypothesis)), not the evidence itself.
#
# Top: log-evidence without normalization -- the offset between noise-scale
# cases is partly an artifact of the missing normalization constant, not
# (only) a real difference in fit, so the 4 curves are not directly
# comparable.
#
# Bottom: log-evidence with `normalize_likelihood=True` -- properly
# comparable across noise-scale cases; the case with the highest
# log-evidence at a given position is the noise level best supported by
# the data there.

# %%
colors = plt.cm.viridis(np.linspace(0, 1, len(noise_scale)))

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for i_scale in range(len(noise_scale)):
    axs[0].plot(x_ref, EV[(i_scale, False)], '.', color=colors[i_scale],
               label=name_arr[i_scale], markersize=4)
axs[0].set_ylabel('log-evidence (not normalized)')
axs[0].set_title('normalize_likelihood=False')
axs[0].legend()
axs[0].grid(True, linestyle='--', linewidth=0.5)

for i_scale in range(len(noise_scale)):
    axs[1].plot(x_ref, EV[(i_scale, True)], '.', color=colors[i_scale],
               label=name_arr[i_scale], markersize=4)
axs[1].set_ylabel('log-evidence (normalized)')
axs[1].set_title('normalize_likelihood=True')
axs[1].set_xlabel('Distance (m)')
axs[1].legend()
axs[1].grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout()
if hardcopy:
    plt.savefig('integrate_gaussian_noise_evidence_EV_profile.png')


# %% [markdown]
# ## Reduced CHI2 per assumed noise level (for context)
#
# CHI2 is unaffected by `normalize_likelihood` (it is de-normalized back
# out internally) -- it is shown here only to confirm the expected
# behaviour: an assumed noise level close to the true one (`noise_scale=1`)
# should give CHI2 close to 1, while too-small/too-large assumed noise
# gives CHI2 well above/below 1.

# %%
plt.figure(figsize=(10, 4))
for i_scale in range(len(noise_scale)):
    plt.plot(x_ref, CHI2[(i_scale, True)], '.', color=colors[i_scale],
             label=name_arr[i_scale], markersize=4)
plt.axhline(1.0, color='k', linestyle=':', linewidth=1)
plt.xlabel('Distance (m)')
plt.ylabel('Reduced CHI2')
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
if hardcopy:
    plt.savefig('integrate_gaussian_noise_evidence_CHI2_profile.png')


# %% [markdown]
# ## Probability of each noise hypothesis
#
# The 4 noise-scale cases are treated as 4 competing hypotheses about the
# (uncorrelated Gaussian) noise level. For each hypothesis i, the actual
# evidence is `exp(EV_i)` (EV being the natural-log evidence plotted
# above), so the posterior probability of hypothesis i, assuming equal
# prior probability across hypotheses, is
#
#     P(H_i | data) = exp(EV_i) / sum_j exp(EV_j)
#
# This is computed with `ig.compute_hypothesis_probability`, which uses
# the log-sum-exp trick internally for numerical stability (evidence
# values are far too small to exponentiate directly). It is only computed
# for the `normalize_likelihood=True` case: since the 4 hypotheses use
# different assumed noise levels, the unnormalized log-evidence is not
# comparable across them (see plot above), so `P(H_i | data)` computed
# from unnormalized EV would not be meaningful.
#
# The result is shown as a cumulative (stacked) area plot along the
# profile: at every position the 4 shaded bands sum to 1, and the
# thickest band identifies the noise level best supported by the data
# there.

# %%
f_post_h5_norm = [f_post_h5_dict[(i_scale, True)] for i_scale in range(len(noise_scale))]
P_H, mode_H, entropy_H = ig.compute_hypothesis_probability(f_post_h5_norm)
# P_H has shape (n_data_points, n_hypotheses); stackplot wants one row per hypothesis
P_H = P_H.T

plt.figure(figsize=(10, 4))
plt.stackplot(x_ref, P_H, labels=name_arr, colors=colors)
plt.xlabel('Distance (m)')
plt.ylabel('P(H_i | data)')
plt.ylim([0, 1])
plt.title('Probability of each noise-scale hypothesis (normalize_likelihood=True)')
plt.legend(loc='upper right')
if hardcopy:
    plt.savefig('integrate_gaussian_noise_evidence_hypothesis_probability.png')

# %%
