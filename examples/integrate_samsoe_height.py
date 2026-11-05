"""
Samsø SkyTEM data with INTEGRATE, in flight-height intervals
============================================================

As ``integrate_samsoe.py``, but the flight-height intervals, the prior data of
each interval, the inversion of each interval and the merging of the
posteriors are done by one call to ``ig.integrate_rejection_em_height``:

0. Convert the XYZ file to a data file: dB/dt as ``D1``, flight height as ``D2``
1. Set up one generic layered prior (``M1``-``M3``)
2. ``ig.integrate_rejection_em_height``: split the soundings into
   flight-height intervals with similar numbers of soundings, compute the
   prior data of each interval (flight height ``M4``, EM data ``D1``,
   flight-height data ``D2``), sample the posterior of each interval, and
   merge the posteriors
3. Plot the results: data fit, WE and NS profiles, and maps

With ``Nbins = 1`` one prior is used for all flight heights.
"""
# %%


try:
    # Check if the code is running in an IPython kernel (which includes Jupyter notebooks)
    get_ipython()
    # If the above line doesn't raise an error, it means we are in a Jupyter environment
    # Execute the magic commands using IPython's run_line_magic function
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    get_ipython().run_line_magic('aimport', '-scipy')
    get_ipython().run_line_magic('aimport', '-jax')
except:
    # If get_ipython() raises an error, we are not in a Jupyter environment
    # # # # # # # # #%load_ext autoreload
    # # # # # # # # #%autoreload 2
    pass


import os

import h5py
import integrate as ig  # type: ignore
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# User settings
# -------------

# %%
hardcopy = True
showInfo = 1  # 0: no, 1: some, 2: more, 3: all info

F_XYZ = '20260501_PROTECT_Samsoe_SkyTEM_2Hz.XYZ'
F_GEX = '20251114_DEN_Diamond_Samsoe_PROTECT_SR.gex'  # 2-point dummy waveform
F_SR2 = '20251114_DEN_Diamond_Samsoe_PROTECT_SR.sr2'  # system response

#F_INV_XYZ = '20260501_PROTECT_Samsoe_SkyTEM_2Hz.XYZ'


# EM forward model: only 'anemone' reads the GEX + SR2 pair (gaaem and simpeg do not)
METHOD = 'anemone'
#METHOD = 'ga-aem'
#METHOD = 'simpeg'
RESCALE_HEIGHT = True  # anemone: one setup rescaled to each model's flight height
# Earlier +/- transmitter pulses (375 Hz; survey report "Modelling Pulse
# Repetition"). Without them the last gates are ~5 % too high; 1 is enough
# (see integrate_samsoe_check_forward.py).
N_PREVIOUS_PULSES = 1

# Number of prior realizations (per flight-height interval)
N = 1_000_000
# N = 2_000_001
N = 100_001

# Flight-height intervals (see ig.em_height_bins): Nbins intervals with an
# equal number of soundings, edges rounded to dh_round m, intervals wider than
# dh_max m split, and joined until each has at least n_min_bin soundings. The
# prior flight height of an interval is uniform within the interval +- dh_pad m.
# Nbins = 1: one prior for all flight heights.
Nbins = 10
#Nbins = 10
dh_max = 20
dh_round = 0.5
n_min_bin = 200
dh_pad = 2.5

# Profiles to plot: the WE profile is the soundings within buffer m of the
# median Y, the NS profile those within buffer m of the median X.
buffer = 100.0

# Data uncertainty: the RelUnc in the XYZ is the measured noise only (median
# ~0.0015). It is inflated by the factor inflateNoise (1 = use the XYZ std).
# Workbench adds a 3% uniform std (GEX: UniformDataSTD=0.03); here rel_err
# is added, in quadrature, to the (inflated) std of the dB/dt data.
inflateNoise = 2
rel_err = 0.03

# Use the measured flight height (D2) as data, with an absolute std [m].
# The prior of each interval still limits the flight height to the
# interval +- dh_pad, also when it is not used as data.
USE_HEIGHT_DATA = True
#USE_HEIGHT_DATA = False
height_std = 1.0

# %% [markdown]
# 0\. Load the data
# -----------------
#
# The dB/dt data as ``D1`` and the measured flight height ('Height', the
# terrain clearance of the frame) as ``D2``, with the absolute std
# ``height_std``. Do NOT use 'Alt': that is the GNSS altitude above sea level
# (= Height + DEM over land, ~13 m higher on average). The GEX file only sets which gates are
# used (gates 5-41); it does not affect the observed data.
#
# UNITS: the Z_dBdt[xx] / STD_Z_dBdt[xx] columns of this XYZ are in
# pV/(A m^4) (see the survey report, section "Normalisation"), while the
# forward models return V/(A m^4), so ``data_scale=1e-12``.
#
# The data file is written only once; its name holds ``inflateNoise``,
# ``rel_err`` and ``height_std``, so a new setting gives a new file.

# %%
f_data_h5 = f'DATA_SAMSOE_H_nf{inflateNoise:g}_err{rel_err:g}_hstd{height_std:g}m.h5'
if not os.path.exists(f_data_h5):
    ig.xyz_to_h5(file_xyz=F_XYZ, file_gex=F_GEX, f_data_h5=f_data_h5,
                 altitude='Height', altitude_std=height_std, data_scale=1e-12, showInfo=showInfo)
    ig.data_set(f_data_h5, id=2, name='Altitude', label='Flight altitude', unit='m')

    # Inflate the std of the dB/dt data, and add rel_err in quadrature
    D = ig.load_data(f_data_h5, showInfo=showInfo)
    d_obs, d_std = D['d_obs'][0], inflateNoise * D['d_std'][0]
    d_std = np.where(np.isfinite(d_obs), np.sqrt(d_std**2 + (rel_err * np.abs(d_obs))**2), d_std)
    ig.save_data_gaussian(d_obs, D_std=d_std, f_data_h5=f_data_h5, id=1,
                          name='dBdT', label='dB/dt', unit='V/Am^4', f_gex=F_GEX, showInfo=showInfo)

DATA = ig.load_data(f_data_h5, showInfo=showInfo)
alt_obs = DATA['d_obs'][1].flatten()
X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
print(f'{len(X)} soundings, flight height {np.nanmin(alt_obs):3.1f}-{np.nanmax(alt_obs):3.1f} m '
      f'(median {np.nanmedian(alt_obs):3.1f} m)')

# %%
ig.plot_geometry(f_data_h5, pl='ELEVATION')
i_plot = np.arange(0, len(X), 20)
ig.plot_data(f_data_h5, i_plot=i_plot, hardcopy=hardcopy)

# %% [markdown]
# 1\. Set up the prior
# --------------------
#
# One generic layered prior: ``M1`` (resistivity, 1 m cells to 149 m),
# ``M2`` (layer resistivities and depths) and ``M3`` (number of layers). The
# flight height is added as ``M4`` by ``ig.integrate_rejection_em_height``,
# for each interval.

# %%
RHO_min = 1
RHO_max = 1000
NLAY_min = 2
NLAY_max = 6
z_max = 120

f_prior_h5 = f'PRIOR_SAMSOE_N{N}.h5'
if not os.path.exists(f_prior_h5):
    f_prior_h5 = ig.prior_model_layered(N=N, lay_dist='uniform', NLAY_min=NLAY_min, NLAY_max=NLAY_max,
                                        RHO_min=RHO_min, RHO_max=RHO_max, dz=1, z_max=149,
                                        f_prior_h5=f_prior_h5, showInfo=showInfo)
ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy, showInfo=showInfo)

# %% [markdown]
# 2\. Prior data and posterior of each flight-height interval
# ----------------------------------------------------------
#
# The EM prior data ``D1`` are computed with the GEX + SR2 pair, with the
# flight height in ``M4``; the flight-height data ``D2`` (identity of ``M4``)
# are computed when ``id_height=2`` is given.
#
# The prior data files of the intervals (``PRIOR_DATA_N{N}_{METHOD}_h{lo}-{hi}.h5``)
# are reused if they exist; delete them after changing the prior or the
# forward settings. The posteriors of the intervals are merged into one POST
# file; the posterior of each interval is kept (attribute ``f5_post_bin``).

# %%

id_use = [1, 2]
autoT = [1,0]
if not USE_HEIGHT_DATA:
    id_use = [1]  # use both the EM data and the flight-height data
    autoT = [1]  # use both the EM data and the flight-height data

txt_h = f'_hstd{height_std:g}' if USE_HEIGHT_DATA else '_noh'
f_post_h5 = ig.integrate_rejection_em_height(f_prior_h5, f_data_h5,
                                             f_post_h5=f'POST_N{N}_{METHOD}_H{USE_HEIGHT_DATA}_npp{N_PREVIOUS_PULSES}_nf{inflateNoise:g}_err{rel_err:g}{txt_h}.h5',
                                             N_use=N,
                                             id_use=id_use,
                                             autoT=autoT,
                                             T_base=1,
                                             backend='jax',
                                             forward_kwargs=dict(file_gex=F_GEX, file_sr2=F_SR2, method=METHOD,
                                                                 rescale_height=RESCALE_HEIGHT,
                                                                 n_previous_pulses=N_PREVIOUS_PULSES),
                                             id_em=1,
                                             id_height=2,
                                             im_height=4,
                                             Nbins=Nbins,
                                             dh_max=dh_max,
                                             dh_round=dh_round,
                                             n_min_bin=n_min_bin,
                                             dh_pad=dh_pad,
                                             f_prior_data_base=f'PRIOR_DATA_N{N}_{METHOD}_npp{N_PREVIOUS_PULSES}',
                                             shared_prior=True,
                                             showInfo=showInfo)

# Flight-height intervals, and the prior data and posterior of each
with h5py.File(f_post_h5, 'r') as f:
    height_edges = f.attrs['height_edges']
    f_prior_data_bin = list(f.attrs['f5_prior_data_bin'])
    f_post_bin = list(f.attrs['f5_post_bin'])


# %%
# Observed data (red) should lie within the prior data, for the interval of
# the median flight height
k_med = 0
ig.plot_data_prior(f_prior_data_bin[k_med], f_data_h5, nr=1000, id=1, hardcopy=hardcopy)
ig.plot_data_prior(f_prior_data_bin[k_med], f_data_h5, nr=1000, id=2, hardcopy=hardcopy)

# %% [markdown]
# 3\. Plot the results
# --------------------
#
# **Data fit.** Prior (beige) and posterior (black) data, with the observed
# data (red), for a low, a typical and a high flight height, from the
# posterior of the interval of each sounding.

# %%
# (np.argsort puts soundings without a flight height last)
n_h = np.sum(np.isfinite(alt_obs))
i_show=1000
for i in range(len(f_prior_data_bin)):
    ig.plot_data_prior_post(f_post_bin[i], i_plot=i_show, id=1, hardcopy=hardcopy, showInfo=showInfo)

# %%
# Annealing temperature, evidence and data fit (CHI2 ~ 1 is a good fit)
ig.plot_T_EV(f_post_h5, pl='T', hardcopy=hardcopy)
ig.plot_T_EV(f_post_h5, pl='EV', hardcopy=hardcopy)
ig.plot_T_EV(f_post_h5, pl='CHI2', hardcopy=hardcopy)

# %% [markdown]
# **Resistivity profiles** (``M1``) and **flight-height profiles** (``M4``)
# along the WE and NS profiles

# %%
i_we = np.where(np.abs(Y - np.median(Y)) <= buffer)[0]
i_ns = np.where(np.abs(X - np.median(X)) <= buffer)[0]
for ii, xaxis, title in ((i_we, 'x', 'WE profile'), (i_ns, 'y', 'NS profile')):
    ig.plot_profile(f_post_h5, ii=ii, im=1, xaxis=xaxis, gap_threshold=150, hardcopy=hardcopy, title=title)
    ig.plot_profile(f_post_h5, ii=ii, im=4, xaxis=xaxis, gap_threshold=150, hardcopy=hardcopy, title=title)
    plt.show()

# %% [markdown]
# **Maps**: resistivity (harmonic mean) at elevations from 50 m to -70 m, and
# the posterior flight height (``M4``, median and std).

# %%
for ele in np.arange(50, -71, -20):
    ig.plot_feature_2d(f_post_h5, im=1, elevation=ele, key='HarmonicMean', uselog=1, clim=[1, 2000],
                       s=2, hardcopy=hardcopy, showInfo=showInfo)
    plt.show()

# %%
ig.plot_feature_2d(f_post_h5, im=4, iz=0, key='Median', uselog=0, s=2, hardcopy=hardcopy, showInfo=showInfo)
plt.show()
ig.plot_feature_2d(f_post_h5, im=4, iz=0, key='Std', uselog=0, s=2, clim=[0, 5], hardcopy=hardcopy, showInfo=showInfo)
plt.show()

# %%
