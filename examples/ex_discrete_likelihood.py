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
