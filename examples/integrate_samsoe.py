"""
Samsø SkyTEM data with INTEGRATE
================================

Short workflow for the Samsø SkyTEM survey:

0. Convert the XYZ file to DATA.h5
1. Set up a generic layered prior (PRIOR.h5)
2. Forward-model the prior data with anemone, using GEX + SR2 (PRIOR_DATA_anemone.h5)
3. Run rejection sampling on the whole data set (POST.h5)
4. Plot a WE and a NS resistivity profile
"""

# %%
import os

os.environ["REJECTION_BACKEND"] = "jax"

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

import integrate as ig

hardcopy = True

F_XYZ = '20260501_PROTECT_Samsoe_SkyTEM_2Hz.XYZ'
F_GEX = '20251114_DEN_Diamond_Samsoe_PROTECT_SR.gex'  # 2-point dummy waveform
F_SR2 = '20251114_DEN_Diamond_Samsoe_PROTECT_SR.sr2'  # system response
# Forward method: only 'anemone' reads the GEX + SR2 pair (gaaem and simpeg do not)
METHOD = 'anemone'
#METHOD = 'ga-aem'
#METHOD = 'simpeg'

    # %% [markdown]
# 0\. Load the XYZ data into DATA.h5
# ----------------------------------
#
# The GEX file only sets which gates are used (gates 5-41); it does not
# affect the observed data.

# %%
# flight altitude ('Alt') is stored as D2, with 5% relative std
# UNITS: the Z_dBdt[xx] / STD_Z_dBdt[xx] columns of this XYZ are in pV/(A m^4), not
# V/(A m^4).  Where to find this for your own data:
#   * the survey report, section "Normalisation" (05_Report/*.pdf): the signal is
#     normalised by Rx area, Tx area, turns and current to V/(A m^4), then given the
#     metric prefix "pico" -> pV/(A m^4);  the channel table lists Z_dBdt[xx] as pV/(Am4);
#   * a quick sanity check: the forward models (anemone, GA-AEM, SimPEG) return
#     V/(A m^4), typically ~1e-6 .. 1e-14 for SkyTEM.  If the observed values are of
#     order 1e6 .. 1e0 (as here, ~1e12 larger), the data are in pV/(A m^4).
#     ig.plot_data_prior() then shows prior and observed 12 decades apart.
# The data must be in the same unit as the forward response, so convert pV -> V:
# data_scale = 1e-12 (it scales both d_obs and d_std).  tTEM XYZ files are already in
# V/(A m^4), so they need no scaling.
f_data_h5 = ig.xyz_to_h5(file_xyz=F_XYZ, file_gex=F_GEX, f_data_h5='DATA.h5',
                         altitude='Alt', altitude_std=0.05, data_scale=1e-12, showInfo=1)
ig.data_set(f_data_h5, id=1, name='dBdT', label='dB/dt', unit='V/Am^4')
ig.data_set(f_data_h5, id=2, name='Altitude', label='Flight altitude', unit='m')

# The RelUnc in the XYZ is the measured noise only (median ~0.0015).  Workbench adds a
# 3% uniform STD on top (GEX: UniformDataSTD=0.03), which xyz_to_h5 does not apply.
# Add it here, in quadrature, to d_std of D1.
rel_err = 2*0.03
D = ig.load_data(f_data_h5)
d_obs, d_std = D['d_obs'][0], D['d_std'][0]
d_std = np.where(np.isfinite(d_obs), np.sqrt(d_std**2 + (rel_err * np.abs(d_obs))**2), d_std)
ig.save_data_gaussian(d_obs, D_std=d_std, f_data_h5=f_data_h5, id=1,
                      name='dBdT', label='dB/dt', unit='V/Am^4', f_gex=F_GEX)

# %%
ig.plot_geometry(f_data_h5, pl='ELEVATION')
ig.plot_data(f_data_h5, hardcopy=hardcopy)

# %% [markdown]
# 1\. Set up a generic prior (PRIOR.h5)
# -------------------------------------

# %%
N = 2_000_000
N = 10_000
f_prior_h5 = ig.prior_model_layered(N=N, lay_dist='uniform', NLAY_min=1, NLAY_max=5,
                                    RHO_min=1, RHO_max=3000, dz=1, z_max=149,
                                    f_prior_h5='PRIOR.h5', showInfo=1)


# %% [markdown]
# 1b. Simulate the flight altitude (M4 = Tx height)
# -------------------------------------------------
#
# M2 and M3 are used by the layered prior, so the Tx height goes into M4.
# The Rx height follows from the GEX offset, so only the Tx height is needed.

# %%
D_obs = ig.load_data(f_data_h5)
alt_obs = D_obs['d_obs'][1].flatten()
alt_min, alt_max = np.percentile(alt_obs, [2.5, 97.5])
Tx_sim = np.random.uniform(alt_min, alt_max, N)
ig.save_prior_model(f_prior_h5, Tx_sim, im=4, name='Tx_altitude', showInfo=1, force_replace=True)

ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)

# %% [markdown]
# 2\. Forward-model the prior data (PRIOR_DATA.h5)
# ------------------------------------------------
#
# anemone is the only forward model that reads the GEX + SR2 pair.
# ``im_height=4`` tells anemone to use M4 as the flight height.

# %%
f_prior_data_h5 = f'PRIOR_DATA_{METHOD}_N{N}.h5'

# Prior EM DATA
ForceForward = True
rescale_height = True  # rescale the anemone setup to each model's height instead of one setup per altitude_bin_width bin; much faster for varying flight height, see forward_anemone
#rescale_height = False  #
if (ForceForward) or (not os.path.exists(f_prior_data_h5)):
    f_prior_data_h5 = ig.prior_data_em(f_prior_h5, file_gex=F_GEX, file_sr2=F_SR2,
                                       method=METHOD, 
                                       im_height=4, 
                                       rescale_height=rescale_height, 
                                       doMakePriorCopy=True, f_prior_data_h5=f_prior_data_h5)
else:
    print(f'Using existing prior data file: {f_prior_data_h5}')

# Prior flight altitude (M4) is stored as D2.
# D2 = identity of M4 (simulated flight altitude), matching the altitude block in DATA.h5
ig.prior_data_identity(f_prior_data_h5, id=2, im=4, showInfo=1)

# %%
ig.plot_data_prior(f_prior_data_h5, f_data_h5, nr=1000, hardcopy=hardcopy)

# %% [markdown]
# 3\. Posterior for the whole data set (POST.h5)
# ----------------------------------------------

# %%
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
buffer = 100.0
i_we = np.where(np.abs(Y - np.median(Y)) <= buffer)[0]
i_ns = np.where(np.abs(X - np.median(X)) <= buffer)[0]
ii=np.concatenate((i_we,i_ns))


f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, # type: ignore
                                   f_post_h5='POST.h5',
                                   id_use=[1],
                                   ip_range = ii,
                                   N_use=N, autoT=1, T_base=1, showInfo=0,
                                   backend='jax')  # 'jax' or 'numpy'

# %% [markdown]
# 4\. Profiles: WE and NS
# -----------------------
#
# The WE profile is the soundings within ``buffer`` metres of the median Y,
# and the NS profile those within ``buffer`` metres of the median X.

# %%
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
buffer = 100.0
i_we = np.where(np.abs(Y - np.median(Y)) <= buffer)[0]
i_ns = np.where(np.abs(X - np.median(X)) <= buffer)[0]

ig.plot_profile(f_post_h5, ii=i_we, im=1, xaxis='x', gap_threshold=150, hardcopy=hardcopy, title='WE profile')
#ig.plot_profile(f_post_h5, ii=i_we, im=4, xaxis='x', gap_threshold=150, hardcopy=hardcopy, title='WE profile')
ig.plot_profile(f_post_h5, ii=i_ns, im=1, xaxis='y', gap_threshold=150, hardcopy=hardcopy, title='NS profile')
plt.show()

# %%
ig.plot_data_prior_post(f_post_h5, i_plot=100)
# %% Flight Height (M4) profile
ig.plot_profile(f_post_h5, ii=i_we, im=4, xaxis='id', gap_threshold=150, hardcopy=hardcopy, title='WE profile')

# %% 
ig.plot_T_EV(f_post_h5, hardcopy=hardcopy)

# %% MAP
ig.plot_feature_2d(f_post_h5,im=1,iz=5, key='HarmonicMean', uselog=1, cmap='jet', s=2, hardcopy=hardcopy)
plt.show()

# %% 
try:
    for ele in np.arange(50,-71,-20):
        plt.figure()
        ig.plot_feature_2d(f_post_h5,im=1,elevation=ele, 
                           key='HarmonicMean', 
                           uselog=1, clim=[1,2000],s=2,
                           plotPoints = True, 
                           hardcopy=hardcopy)
        plt.show()
except:
    pass

# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# %% Inversion in natches on subset of the data, based on flight height intervals

# Load flight height from data
D_obs = ig.load_data(f_data_h5)
alt_obs = D_obs['d_obs'][1].flatten()

# %%
N_sub = 4
# fine the threshold that divides the flight heoight data into N_sub subsets.
# use np.percentile to find the thresholds that divide the data into N_sub equal parts
thresholds = np.percentile(alt_obs, np.linspace(0, 100, N_sub + 1))
print(f'Flight height thresholds for {N_sub} subsets: {thresholds}')

ii_subset = []
for i in range(N_sub):
    if i == 0:
        ii = np.where(alt_obs <= thresholds[i + 1])[0]
    elif i == N_sub - 1:
        ii = np.where(alt_obs > thresholds[i])[0]
    else:
        ii = np.where((alt_obs > thresholds[i]) & (alt_obs <= thresholds[i + 1]))[0]
    ii_subset.append(ii)
    print(f'Subset {i + 1}: {len(ii)} soundings, flight height range: {thresholds[i]} - {thresholds[i + 1]} m')

# %% Prior data subsets: which subsets need to be (re)computed
# Each subset gets its own prior data file, PRIOR_DATA_subset_{i+1}.h5
f_prior_data_subset_h5_list = [f'PRIOR_DATA_subset_{i + 1}.h5' for i in range(N_sub)]
i_compute = [i for i in range(N_sub)
             if ForceForward or not os.path.exists(f_prior_data_subset_h5_list[i])]
for i in range(N_sub):
    if i not in i_compute:
        print(f'Using existing prior data file: {f_prior_data_subset_h5_list[i]}')

# %% Copy the prior models, and simulate the flight height (M4) for each subset
# The flight height is uniform within the subset's threshold limits (+- 4 m)
for i in i_compute:
    f_prior_data_subset_h5 = f_prior_data_subset_h5_list[i]
    ig.copy_prior(f_prior_h5, f_prior_data_subset_h5, showInfo=1)
    alt_min = thresholds[i] - 4
    alt_max = thresholds[i + 1] + 4
    Tx_sim = np.random.uniform(alt_min, alt_max, N)
    ig.save_prior_model(f_prior_data_subset_h5, Tx_sim, im=4, name='Tx_altitude', showInfo=1, force_replace=True)

# %% Compute the prior EM data (D1) for each subset
for i in i_compute:
    f_prior_data_subset_h5 = f_prior_data_subset_h5_list[i]
    ig.prior_data_em(f_prior_data_subset_h5, file_gex=F_GEX, file_sr2=F_SR2,
                     method=METHOD,
                     im_height=4,
                     rescale_height=rescale_height,
                     doMakePriorCopy=False, f_prior_data_h5=f_prior_data_subset_h5)

# %% Prior flight height data (D2 = identity of M4) for each subset
for i in i_compute:
    ig.prior_data_identity(f_prior_data_subset_h5_list[i], id=2, im=4, showInfo=1)


# %% Run rejection sampling on each subset
for i in range(N_sub):
    f_prior_data_subset_h5 = f'PRIOR_DATA_subset_{i + 1}.h5'
    f_post_subset_h5 = f'POST_subset_{i + 1}.h5'
    f_post_subset_h5 = ig.integrate_rejection(f_prior_data_subset_h5, f_data_h5,
                                              f_post_h5=f_post_subset_h5,
                                              id_use=[1],
                                              ip_range=ii_subset[i],
                                              N_use=N, autoT=1, T_base=1, showInfo=0,
                                              backend='jax')  # 'jax' or 'numpy'

# %% Merge the N_sub posteriors into a single posterior file
# All subsets use DATA.h5, but each has its own prior (PRIOR_DATA_subset_{i+1}.h5),
# so the subset priors are merged too (PRIOR_DATA_merged.h5).
f_post_merged_h5 = 'POST_merged.h5'
f_post_merged_h5, _ = ig.merge_posterior([f'POST_subset_{i + 1}.h5' for i in range(N_sub)], f_data_h5,
                                         f_post_merged_h5=f_post_merged_h5,
                                         ip_range_list=ii_subset,
                                         f_prior_merged_h5='PRIOR_DATA_merged.h5', showInfo=1)



# %%

ig.plot_profile(f_post_merged_h5, ii=i_we, im=1, xaxis='x', gap_threshold=150, hardcopy=hardcopy, title='WE profile')
#ig.plot_profile(f_post_merged_h5, ii=i_we, im=4, xaxis='x', gap_threshold=150, hardcopy=hardcopy, title='WE profile')
#ig.plot_profile(f_post_merged_h5, ii=i_we, im=4, xaxis='x', gap_threshold=150, hardcopy=hardcopy, title='WE profile')
ig.plot_profile(f_post_merged_h5, ii=i_ns, im=1, xaxis='y', gap_threshold=150, hardcopy=hardcopy, title='NS profile')
plt.show()

# %%
ig.plot_data_prior_post(f_post_merged_h5, i_plot=100)
# %% Flight Height (M4) profile
ig.plot_profile(f_post_merged_h5, ii=i_we, im=4, xaxis='id', gap_threshold=150, hardcopy=hardcopy, title='WE profile')

# MAP
ig.plot_feature_2d(f_post_merged_h5,im=1,iz=5, key='HarmonicMean', uselog=1, cmap='jet', s=2, hardcopy=hardcopy)
plt.show()


# %% 
try:
    for ele in np.arange(50,-71,-20):
        plt.figure()
        ig.plot_feature_2d(f_post_merged_h5,im=1,elevation=ele, 
                           key='HarmonicMean', 
                           uselog=1, clim=[1,2000],s=2,
                           plotPoints = True, 
                           hardcopy=hardcopy)
        plt.show()
except:
    pass


# %%
ig.plot_T_EV(f_post_merged_h5, hardcopy=hardcopy)