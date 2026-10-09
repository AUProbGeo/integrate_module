#!/usr/bin/env python
# %% [markdown]
# # Likelihood of continuous and discrete data, and the evidence
#
# This example shows how the likelihood is computed for two types of data:
#
# * **continuous** data (tTEM dB/dt) with a Gaussian noise model, using `ig.likelihood_gaussian_diagonal()`
# * **discrete** data (lithology from a well log) with a multinomial noise model, using `ig.likelihood_multinomial()`
#
# The likelihood is evaluated for every realization of a prior sample. The evidence
# (marginal likelihood) is then estimated as the average likelihood over the prior sample.
#
# The DAUGAARD tTEM data are used, together with the DAUGAARD well logs, and a prior
# generated with geoprior1d that contains both resistivity (continuous) and lithology (discrete).

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
import integrate as ig
from geoprior1d import geoprior1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

hardcopy = True

N = 10_000  # Number of prior realizations

# %% [markdown]
# ## Get the DAUGAARD data, the well logs, and the prior description (xlsx)

# %%
case = 'DAUGAARD'
files = ig.get_case_data(case=case)  # DAUGAARD_AVG.h5, gex file and daugaard_12boreholes.json
f_xlsx = 'daugaard_standard.xlsx'
ig.get_case_data(case=case, filelist=[f_xlsx])

f_data_org_h5 = 'DAUGAARD_AVG.h5'
file_gex = ig.get_gex_file_from_data(f_data_org_h5)

# Work on a copy of the data file, as the well log data are added to it
f_data_h5 = 'DAUGAARD_AVG_discrete_likelihood.h5'
ig.copy_hdf5_file(f_data_org_h5, f_data_h5)

# %% [markdown]
# ## A: Sample the prior with geoprior1d
# The prior contains M1 (resistivity, continuous) and M2 (lithology, discrete)

# %%
f_prior_h5, flags = geoprior1d(f_xlsx, Nreals=N, dmax=90, dz=1,
                               output_file=f'daugaard_standard_N{N}.h5')
ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)

class_id, class_name = ig.get_discrete_classes(f_prior_h5, im=2)
for cid, cname in zip(class_id, class_name):
    print(f"Class {cid}: {cname}")

# %% [markdown]
# ## B: Load the well log data

# %%
BHOLES = ig.read_borehole('daugaard_12boreholes.json')
BH = BHOLES[0]  # Use one well in this example
ig.plot_boreholes(BH, f_prior_h5, hardcopy=hardcopy)

# %% [markdown]
# ## C: Simulate prior data
# * continuous data: tTEM forward response of M1 (resistivity)  -> prior D1
# * discrete data: the mode of the lithology (M2) in each well log interval  -> prior D2
#
# `ig.save_borehole_data` also stores the observed well log data (class probabilities)
# in the data file, as a multinomial data type.

# %%
f_prior_h5 = ig.prior_data_em(f_prior_h5, file_gex, doMakePriorCopy=True)
id_prior_well, id_data_well = ig.save_borehole_data(f_prior_h5, f_data_h5, BH, im_prior=2)

# %% [markdown]
# ## D: Load prior MODEL and DATA, and the observed DATA

# %%
M, idx = ig.load_prior_model(f_prior_h5)
D, idx = ig.load_prior_data(f_prior_h5)
DATA = ig.load_data(f_data_h5)

print(f"Noise models: {DATA['noise_model']}")
for i in range(len(D)):
    print(f"Prior D{i+1}: shape={D[i].shape},  observed d_obs: shape={DATA['d_obs'][i].shape}")

D_tem, D_well = D[0], D[id_prior_well-1]  # prior data: tTEM (continuous) and well log (discrete)

# Use the tTEM sounding closest to the well
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
ip = np.argmin((X-BH['X'])**2 + (Y-BH['Y'])**2)
print(f"Using data point ip={ip}, at distance {np.hypot(X[ip]-BH['X'], Y[ip]-BH['Y']):.1f} m from {BH['name']}")

# %% [markdown]
# ## E: Likelihood of continuous data (Gaussian)
# log L(m) = -0.5 * sum( (g(m) - d_obs)^2 / d_std^2 ) - 0.5 * sum( log(2*pi*d_std^2) )
#
# `normalize=True` includes the normalization constant (the last term), which is
# needed for the evidence to be a proper probability density of the data.

# %%
d_obs = DATA['d_obs'][0][ip]
d_std = DATA['d_std'][0][ip]
logL_tem = ig.likelihood_gaussian_diagonal(D_tem, d_obs, d_std, normalize=True)

# %% [markdown]
# ## F: Likelihood of discrete data (multinomial)
# The observed data is a probability matrix P_obs (n_class, n_intervals): the probability of
# observing each class in each well log interval.
# The prior data is the class (mode) in each interval for each prior realization.
#
# log L(m) = sum_j log( P_obs[class(m)_j, j] )

# %%
P_obs = DATA['d_obs'][id_data_well-1][ip]
print('P_obs (rows: classes, columns: well log intervals)')
print(P_obs)
logL_well = ig.likelihood_multinomial(D_well, P_obs, class_id)

# %% [markdown]
# ## F2: The multinomial pmf in detail
# This section computes the same log-likelihood as above, step by step, without
# `ig.likelihood_multinomial`.
#
# **Model.** Let the well log have J intervals and K classes. In interval j the
# lithology is a categorical random variable Z_j with probabilities
#
#     p_kj = P(Z_j = k) = P_obs[k, j],    sum_k p_kj = 1  for every j.
#
# A multinomial pmf for n trials with class counts n_1, ..., n_K and class
# probabilities p_1, ..., p_K is
#
#     P(n_1, ..., n_K | n, p) = n! / (n_1! ... n_K!) * prod_k p_k^n_k.
#
# Here each interval is a single observation (n = 1 per interval), so each
# count vector has exactly one 1 and the multinomial coefficient equals 1.
# The pmf of one interval therefore reduces to the probability of the observed
# class:
#
#     P(Z_j = c_j) = p_{c_j, j}
#
# where c_j is the class that prior realization m assigns to interval j, i.e.
# c_j = D_well[m, j].
#
# **Likelihood.** The intervals are treated as independent, so the probabilities
# multiply over the intervals:
#
#     L(m) = prod_j p_{c_j(m), j}      =>     log L(m) = sum_j log p_{c_j(m), j}
#
# **Special cases.**
# * A column of P_obs containing NaN means the interval was not observed: it
#   contributes a factor 1 (log = 0) and is skipped.
# * A NaN class in the prior, or a class that has no row in P_obs, has
#   probability 0: log L(m) = -inf.

# %%
# Map each class ID to its row in P_obs
row_of_class = {}
for k in range(len(class_id)):
    row_of_class[int(class_id[k])] = k

n_int = P_obs.shape[1]
observed_int = []
for j in range(n_int):
    if not np.any(np.isnan(P_obs[:, j])):
        observed_int.append(j)

print(f"Observed intervals: {len(observed_int)} of {n_int}")
print('Column sums of P_obs (should be 1):', np.round(np.sum(P_obs[:, observed_int], axis=0), 6))

# Log-likelihood of every prior realization, one interval at a time
N_prior = D_well.shape[0]
logL_manual = np.zeros(N_prior)
for i in range(N_prior):
    logL_i = 0.0
    for j in observed_int:
        c = D_well[i, j]
        if np.isnan(c) or int(c) not in row_of_class:
            p = 0.0
        else:
            p = P_obs[row_of_class[int(c)], j]
        if p == 0.0:
            logL_i = -np.inf
            break
        logL_i += np.log(p)
    logL_manual[i] = logL_i

# Worked example for the best-matching realization (the one with the largest log L)
i_max = np.argmax(logL_manual)
print(f"Realization {i_max}: prior classes per interval = {D_well[i_max].astype(int).tolist()}")
p_used = []
for j in observed_int:
    c = int(D_well[i_max, j])
    p_used.append(P_obs[row_of_class[c], j])
print('p of observed class per interval:', np.round(p_used, 3))
print(f"log L = sum of log p = {np.sum(np.log(p_used)):.4f}")

# Check against the result of ig.likelihood_multinomial above
print('Matches ig.likelihood_multinomial:', np.allclose(logL_manual, logL_well))

# %% [markdown]
# ## G: Joint likelihood
# The two data types are independent, so the log-likelihoods add up

# %%
logL = logL_tem + logL_well

i_best = np.argmax(logL)
print(f"Most likely prior realization: {i_best}")
print(f"  observed well classes : {BH['class_obs']}")
print(f"  prior well classes    : {D_well[i_best].astype(int).tolist()}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(logL_tem, 50)
ax[0].set_title('log L, tTEM (Gaussian)')
ax[1].hist(logL_well, 50)
ax[1].set_title('log L, well log (multinomial)')
ax[2].plot(logL_tem, logL_well, '.', markersize=2)
ax[2].set_xlabel('log L, tTEM')
ax[2].set_ylabel('log L, well log')
for a in ax:
    a.grid()
plt.tight_layout()
if hardcopy:
    plt.savefig('ex_discrete_likelihood_logL.png', dpi=200)
plt.show()

# %% [markdown]
# ## H: Estimate the evidence
# The evidence is the average likelihood over a sample of the prior:
#
# E = (1/N) * sum_i L(m_i)
#
# It is computed in log-space to avoid underflow: log E = logsumexp(log L) - log(N)

# %%
def log_evidence(logL):
    return logsumexp(logL) - np.log(len(logL))

logE_tem = log_evidence(logL_tem)
logE_well = log_evidence(logL_well)
logE = log_evidence(logL)
print(f"log Evidence, tTEM            : {logE_tem:10.2f}")
print(f"log Evidence, well log        : {logE_well:10.2f}")
print(f"log Evidence, tTEM + well log : {logE:10.2f}")

# %% [markdown]
# Convergence of the evidence estimate with the number of prior realizations

# %%
N_arr = np.unique(np.logspace(1, np.log10(N), 30).astype(int))

logE_tem_arr = np.zeros(len(N_arr))
logE_well_arr = np.zeros(len(N_arr))
logE_arr = np.zeros(len(N_arr))
for i in range(len(N_arr)):
    n = N_arr[i]
    logE_tem_arr[i] = log_evidence(logL_tem[:n])
    logE_well_arr[i] = log_evidence(logL_well[:n])
    logE_arr[i] = log_evidence(logL[:n])

plt.figure()
plt.semilogx(N_arr, logE_tem_arr, '.-', label='tTEM')
plt.semilogx(N_arr, logE_well_arr, '.-', label='well log')
plt.semilogx(N_arr, logE_arr, '.-', label='tTEM + well log')
plt.xlabel('Number of prior realizations')
plt.ylabel('log Evidence')
plt.legend()
plt.grid()
if hardcopy:
    plt.savefig('ex_discrete_likelihood_evidence.png', dpi=200)
plt.show()

# %%
