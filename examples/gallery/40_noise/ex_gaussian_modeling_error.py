"""
Accounting for forward modeling errors with a Gaussian model
============================================================

An accurate forward model (here GA-AEM, the 'reference') can be expensive, while
an approximate one (here SimPEG) can be cheaper or more convenient. The error
introduced by the approximate forward, DD = d_ref - d_approx, can be treated as
extra Gaussian noise, DD ~ N(d_bias, Cd_model), estimated from prior realizations.
The approximate forward is then used together with

    d_obs - d_bias  and  Cd = Cd_noise + Cd_model

IMPORTANT: DD is heavy-tailed and depends on the model (large for high-amplitude
models). (d_bias, Cd_model) must therefore be estimated from models that are
*consistent with the data*, e.g. the posterior samples of a first, noise-only
inversion. Estimating them from the whole (wide) prior gives a Cd_model far
larger than the measurement noise and a d_bias of several sigma, which degrades
the result.

With log_space=True the whole analysis is done on log10(data): prior data and
observations are log-transformed, the noise std is propagated as std/(d ln10), and
DD = log10(d_ref) - log10(d_app) is a relative (multiplicative) error. Non-positive
data become NaN and are not used.

The example:

1. Computes prior data with the reference and the approximate forward
2. Estimates (d_bias, Cd_model) from (a) the whole prior and (b) the posterior
   samples of the noise-only inversion
3. Inverts the DAUGAARD data with the reference and the approximate forward,
   with and without the modeling error model
4. Validates on synthetic data with known truth, where the benefit of the
   modeling error model can be measured directly
"""
# %%

import integrate as ig
import numpy as np
import h5py
import matplotlib.pyplot as plt
hardcopy = True
use_bias = True     # False: d_bias=0, and the bias is absorbed into Cd_model instead
log_space = True    # True: invert log10(data); modeling error is then a *relative* error,
                    # which is far more homogeneous than the absolute error in linear space


def estimate_modeling_error(D_ref, D_app, idx=None, valid=None, robust=True, use_bias=True):
    """Estimate (d_bias, Cd_model) of DD = D_ref - D_app.

    idx      : rows (prior realizations) to use, e.g. posterior samples (repeats
               weight the estimate by posterior probability). Default: all rows.
    valid    : boolean mask of the gates to use. Other gates get bias=0, Cd=0.
    robust   : median / clipped statistics. DD is heavy tailed, so the plain
               mean/covariance is dominated by a few high-amplitude models.
    use_bias : if False, d_bias=0 and Cd_model = E[DD DD^T] (second moment about
               zero), so the bias is treated as extra error instead of removed.
               (Zero bias with a *centred* covariance would be inconsistent.)
    """
    DD = D_ref - D_app if idx is None else D_ref[idx] - D_app[idx]
    nd = DD.shape[1]
    valid = np.ones(nd, dtype=bool) if valid is None else valid
    X = DD[:, valid]
    X = X[np.isfinite(X).all(axis=1)]       # (log space: drop realizations with d<=0)
    if robust:
        med = np.median(X, axis=0)
        s = 1.4826 * np.median(np.abs(X - med), axis=0)
        X = np.clip(X, med - 3 * s, med + 3 * s)
    if use_bias:
        d_bias_v = np.mean(X, axis=0)
        Cd_v = np.cov(X, rowvar=False)      # np.cov centres X (subtracts d_bias) internally
    else:
        d_bias_v = np.zeros(X.shape[1])
        Cd_v = X.T @ X / X.shape[0]
    d_bias = np.zeros(nd)
    d_bias[valid] = d_bias_v
    Cd_model = np.zeros((nd, nd))
    Cd_model[np.ix_(valid, valid)] = Cd_v
    return d_bias, Cd_model


def to_log(d):
    """log10 of the data; non-positive values (sign changes, noise) become NaN (= not used)."""
    d = np.asarray(d, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(d > 0, np.log10(np.where(d > 0, d, 1.0)), np.nan)


def std_to_log(d_std, d_obs):
    """First order propagation of the std to log10 space: std / (|d| ln 10)."""
    d_obs = np.asarray(d_obs, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(d_obs > 0, d_std / (np.abs(d_obs) * np.log(10)), np.nan)


def log_prior_data_copy(f_prior_data_h5):
    """Copy of a prior data file with /D1 replaced by log10(D1)."""
    import shutil
    f_out = f_prior_data_h5[:-3] + '_log.h5'
    shutil.copy(f_prior_data_h5, f_out)
    with h5py.File(f_out, 'r+') as f:
        f['/D1'][...] = to_log(f['/D1'][:]).astype(f['/D1'].dtype)
        f['/D1'].attrs['is_log'] = True
    return f_out


def total_covariance(D_std, Cd_model):
    """Cd_noise + Cd_model for each sounding. NaN std (masked gate) -> NaN diagonal."""
    return np.array([np.diag(s**2) + Cd_model for s in D_std])


def post_samples(f_post_h5):
    with h5py.File(f_post_h5, 'r') as f:
        return f['i_use'][:].ravel()


def post_summary(f_post_h5):
    with h5py.File(f_post_h5, 'r') as f:
        return np.median(f['T'][:]), np.nanmedian(f['CHI2'][:])


# %%
# 0. Get TTEM data and select a profile
# -------------------------------------
case = 'DAUGAARD'
files = ig.get_case_data(case=case, showInfo=0)
f_data_h5 = files[0]
file_gex = ig.get_gex_file_from_data(f_data_h5)

X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
Xl = np.array([544000, 543550])
Yl = np.array([6174500, 6176500])
i_line, _, _ = ig.find_points_along_line_segments(X, Y, Xl, Yl, tolerance=10.0)

D = ig.load_data(f_data_h5, showInfo=0)
D_obs = D['d_obs'][0][i_line]
D_std = D['d_std'][0][i_line]
geom = dict(UTMX=X[i_line], UTMY=Y[i_line], LINE=LINE[i_line], ELEVATION=ELEVATION[i_line])

# gates that carry data for (most of) the soundings
valid = np.mean(~np.isnan(D_obs) & (D_std < 1e3), axis=0) > 0.5

# %%
# 1. Prior model and prior data (reference and approximate forward)
# -----------------------------------------------------------------
N = 50_000      # prior realizations used for the inversions (use 1_000_000 for a final run)
Nr = min(5000, N)   # prior realizations used for the 'whole prior' error estimate

prior_kw = dict(lay_dist='uniform', NLAY_min=1, NLAY_max=5, RHO_min=1, RHO_max=3000, showInfo=0)
f_prior_h5 = ig.prior_model_layered(N=N, f_prior_h5=f'PRIOR_N{N}.h5', **prior_kw)

f_prior_ref = f'{f_prior_h5[:-3]}_full_ref.h5'
ig.prior_data_em(f_prior_h5, file_gex, method='gaaem', doMakePriorCopy=True,
                 randomize=False, N=N, f_prior_data_h5=f_prior_ref, id=1, im=1)

f_prior_app = f'{f_prior_h5[:-3]}_full_app.h5'
ig.prior_data_em(f_prior_h5, file_gex, method='simpeg', doMakePriorCopy=True,
                 randomize=False, N=N, f_prior_data_h5=f_prior_app, id=1, im=1)

D_ref_all = ig.load_prior_data(f_prior_ref, id_use=[1])[0][0]
D_app_all = ig.load_prior_data(f_prior_app, id_use=[1])[0][0]

# Working space of the inversion: linear data, or log10 of the data.
# Everything below (modeling error, data files, prior data files) uses the *_w variables.
if log_space:
    D_ref_all, D_app_all = to_log(D_ref_all), to_log(D_app_all)
    f_prior_ref_w, f_prior_app_w = log_prior_data_copy(f_prior_ref), log_prior_data_copy(f_prior_app)
    D_std_w, D_obs_w = std_to_log(D_std, D_obs), to_log(D_obs)
else:
    f_prior_ref_w, f_prior_app_w = f_prior_ref, f_prior_app
    D_std_w, D_obs_w = D_std, D_obs

# %%
# 2. Estimate the modeling error from the WHOLE prior (naive)
# -----------------------------------------------------------
d_bias_prior, Cd_prior = estimate_modeling_error(D_ref_all[:Nr], D_app_all[:Nr], valid=valid,
                                                 robust=False, use_bias=use_bias)

# %%
# 3. Data files, and the noise-only inversions (reference, approximate)
# ---------------------------------------------------------------------
f_d_ref = ig.save_data_gaussian(D_obs_w, D_std=D_std_w, f_data_h5='data_ref.h5', is_log=int(log_space),
                                delete_if_exist=True, showInfo=-1, **geom)
f_d_app = ig.save_data_gaussian(D_obs_w, D_std=D_std_w, f_data_h5='data_app.h5', is_log=int(log_space),
                                delete_if_exist=True, showInfo=-1, **geom)
f_d_app_me_prior = ig.save_data_gaussian(D_obs_w - d_bias_prior, Cd=total_covariance(D_std_w, Cd_prior),
                                         f_data_h5='data_app_me_prior.h5',
                                         delete_if_exist=True, showInfo=-1, **geom)

f_post = {}
f_post['ref'] = ig.integrate_rejection(f_prior_ref_w, f_d_ref, f_post_h5='POST_ref.h5',
                                       autoT=1, showInfo=0, updatePostStat=True)
f_post['app'] = ig.integrate_rejection(f_prior_app_w, f_d_app, f_post_h5='POST_app.h5',
                                       autoT=1, showInfo=0, updatePostStat=True)

# %%
# 4. Estimate the modeling error from the POSTERIOR samples of the noise-only run
# -------------------------------------------------------------------------------
# The models that fit the data are a small, well-behaved subset of the prior.
idx_post = post_samples(f_post['app'])
d_bias, Cd_model = estimate_modeling_error(D_ref_all, D_app_all, idx=idx_post, valid=valid,
                                           robust=True, use_bias=use_bias)

sd_noise = np.nanmedian(np.where(D_std_w < 1e3, D_std_w, np.nan), axis=0)
print('gate   noise_std   sd_model(prior)   sd_model(post)   |bias|(prior)   |bias|(post)')
for g in np.where(valid)[0][::4]:
    print('%3d   %9.2e   %13.2e   %14.2e   %13.2e   %12.2e' % (
        g, sd_noise[g], np.sqrt(Cd_prior[g, g]), np.sqrt(Cd_model[g, g]),
        abs(d_bias_prior[g]), abs(d_bias[g])))

sd_m = np.sqrt(np.clip(np.diag(Cd_model), 1e-300, None))
Cd_norm = Cd_model / np.outer(sd_m, sd_m)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
im_ = ax[0].imshow(Cd_model, cmap='viridis')
ax[0].set_title('Modeling error covariance $C_d^{model}$ (posterior)')
ax[0].set_xlabel('Gate'); ax[0].set_ylabel('Gate')
plt.colorbar(im_, ax=ax[0])
ax[1].semilogy(sd_noise, 'b-', label='noise std')
ax[1].semilogy(np.sqrt(np.diag(Cd_prior)), 'k--', label=r'$\sqrt{diag}$ $C_d^{model}$, whole prior')
ax[1].semilogy(np.sqrt(np.diag(Cd_model)), 'k-', label=r'$\sqrt{diag}$ $C_d^{model}$, posterior')
ax[1].semilogy(np.abs(d_bias_prior), 'r--', label='|d_bias|, whole prior')
ax[1].semilogy(np.abs(d_bias), 'r-', label='|d_bias|, posterior')
ax[1].set_xlabel('Gate'); ax[1].legend(fontsize=7); ax[1].grid(True)
im_n = ax[2].imshow(Cd_norm, cmap='viridis', vmin=0, vmax=1)
ax[2].set_title('Normalized $C_d^{model}$ (by variance)')
ax[2].set_xlabel('Gate'); ax[2].set_ylabel('Gate')
plt.colorbar(im_n, ax=ax[2])
plt.tight_layout()
if hardcopy:
    plt.savefig('modeling_error_Cd.png', dpi=150)
plt.show()

# %%
# 5. Invert the field data using the approximate forward + modeling error
# -----------------------------------------------------------------------
f_d_app_me = ig.save_data_gaussian(D_obs_w - d_bias, Cd=total_covariance(D_std_w, Cd_model),
                                   f_data_h5='data_app_me.h5',
                                   delete_if_exist=True, showInfo=-1, **geom)
f_post['app_me_prior'] = ig.integrate_rejection(f_prior_app_w, f_d_app_me_prior, f_post_h5='POST_app_me_prior.h5',
                                                autoT=1, showInfo=0, updatePostStat=True)
f_post['app_me'] = ig.integrate_rejection(f_prior_app_w, f_d_app_me, f_post_h5='POST_app_me.h5',
                                          autoT=1, showInfo=0, updatePostStat=True)

labels = {'ref': 'GA-AEM (reference)',
          'app': 'SimPEG, measurement noise only',
          'app_me_prior': 'SimPEG, noise + modeling error (whole prior)',
          'app_me': 'SimPEG, noise + modeling error (posterior samples)'}

# %%
# Profiles of the posterior harmonic mean
for key in f_post:
    ig.plot_profile(f_post[key], im=1, key='HarmonicMean', xaxis='index',
                    panels=['value'], title=labels[key], hardcopy=hardcopy,
                    f_png=f'profile_{key}.png')

# %%
# Difference (log10) relative to the reference.
# NOTE: the reference posterior is itself annealed (T>>1 when the noise is under-
# estimated), so this measures consistency with the reference, not correctness.
HM = {}
for key in f_post:
    with h5py.File(f_post[key], 'r') as f:
        HM[key] = f['/M1/HarmonicMean'][:]        # [n_sounding, n_depth]

keys = ['app', 'app_me_prior', 'app_me']
fig, ax = plt.subplots(len(keys), 1, figsize=(10, 3 * len(keys)), sharex=True)
for a, key in zip(ax, keys):
    diff = np.log10(HM[key]) - np.log10(HM['ref'])
    p = a.imshow(diff.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                 extent=[0, diff.shape[0], len(diff[0]), 0])
    a.set_title(labels[key] + ' - GA-AEM'); a.set_ylabel('Depth index')
    plt.colorbar(p, ax=a, label=r'$\Delta\log_{10}\rho$')
ax[-1].set_xlabel('Sounding index along profile')
plt.tight_layout()
if hardcopy:
    plt.savefig('profile_difference.png', dpi=150)
plt.show()

print('%-13s %6s %6s %s' % ('run', 'T', 'chi2', 'mean |dlog10 rho| vs ref'))
for key in f_post:
    T, chi2 = post_summary(f_post[key])
    d = np.log10(HM[key]) - np.log10(HM['ref'])
    print('%-13s %6.2f %6.2f %.3f' % (key, T, chi2, np.mean(np.abs(d))))

# %%
# 6. Validation on synthetic data with known truth
# ------------------------------------------------
# Models drawn from a fresh prior realization are forward modeled with the
# reference (GA-AEM), noise is added, and the data are inverted with the
# approximate forward. Now the error against the TRUE model can be measured.
Nt = 100
rng = np.random.default_rng(1)
f_prior_t = ig.prior_model_layered(N=Nt, f_prior_h5=f'PRIOR_test_N{Nt}.h5', **prior_kw)
f_prior_t_ref = f'{f_prior_t[:-3]}_ref.h5'
ig.prior_data_em(f_prior_t, file_gex, method='gaaem', doMakePriorCopy=True,
                 randomize=False, N=Nt, f_prior_data_h5=f_prior_t_ref, id=1, im=1)
D_t = ig.load_prior_data(f_prior_t_ref, id_use=[1])[0][0]
with h5py.File(f_prior_t, 'r') as f:
    M_true = f['/M1'][:]

# noise levels and masked gates taken from the field data
std_t = D_std[:Nt].copy()
D_syn = D_t + rng.standard_normal(D_t.shape) * np.where(np.isnan(std_t), 0, std_t)
D_syn[np.isnan(D_obs[:Nt])] = np.nan
std_t[np.isnan(D_obs[:Nt])] = np.nan
if log_space:       # noise is added in linear space (as in reality), then transformed
    std_t, D_syn = std_to_log(std_t, D_syn), to_log(D_syn)
geom_t = dict(UTMX=np.arange(Nt) * 10.0, UTMY=np.zeros(Nt), LINE=np.ones(Nt), ELEVATION=np.zeros(Nt))

f_d_syn = ig.save_data_gaussian(D_syn, D_std=std_t, f_data_h5='data_syn.h5', is_log=int(log_space),
                                delete_if_exist=True, showInfo=-1, **geom_t)
f_post_syn = {}
f_post_syn['app'] = ig.integrate_rejection(f_prior_app_w, f_d_syn, f_post_h5='POST_syn_app.h5',
                                           autoT=1, showInfo=0, updatePostStat=True)

# modeling error estimated from the posterior of this (noise-only) inversion
d_bias_s, Cd_model_s = estimate_modeling_error(D_ref_all, D_app_all,
                                               idx=post_samples(f_post_syn['app']), valid=valid,
                                               robust=True, use_bias=use_bias)
f_d_syn_me = ig.save_data_gaussian(D_syn - d_bias_s, Cd=total_covariance(std_t, Cd_model_s),
                                   f_data_h5='data_syn_me.h5',
                                   delete_if_exist=True, showInfo=-1, **geom_t)
f_post_syn['app_me'] = ig.integrate_rejection(f_prior_app_w, f_d_syn_me, f_post_h5='POST_syn_app_me.h5',
                                              autoT=1, showInfo=0, updatePostStat=True)

# modeling error estimated from the whole prior (naive)
f_d_syn_me_prior = ig.save_data_gaussian(D_syn - d_bias_prior, Cd=total_covariance(std_t, Cd_prior),
                                         f_data_h5='data_syn_me_prior.h5',
                                         delete_if_exist=True, showInfo=-1, **geom_t)
f_post_syn['app_me_prior'] = ig.integrate_rejection(f_prior_app_w, f_d_syn_me_prior,
                                                    f_post_h5='POST_syn_app_me_prior.h5',
                                                    autoT=1, showInfo=0, updatePostStat=True)

print('\nSynthetic test (%d soundings), error against the TRUE model' % Nt)
print('%-13s %6s %6s %s' % ('run', 'T', 'chi2', 'RMSE log10 rho (harmonic mean vs truth)'))
for key in ['app', 'app_me_prior', 'app_me']:
    with h5py.File(f_post_syn[key], 'r') as f:
        HM_t = f['/M1/HarmonicMean'][:]
    T, chi2 = post_summary(f_post_syn[key])
    rmse = np.sqrt(np.mean((np.log10(HM_t) - np.log10(M_true)) ** 2))
    print('%-13s %6.2f %6.2f %.4f' % (key, T, chi2, rmse))
