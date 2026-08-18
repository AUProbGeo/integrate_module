#!/usr/bin/env python
# %% [markdown]
# # Sdr. Felding: probabilistic raw-material assessment with INTEGRATE
#
# GEUS previously assessed raw-material (sand/gravel) potential at a target
# sub-area in Sdr. Felding using a **sequential, deterministic** workflow:
# invert the tTEM data to a single best-fit resistivity model, interpret
# lithology, and hand-estimate overburden and raw-material volumes,
# constrained to what raw-material boreholes could document. Those results
# live in `ReferenceProjects/Sdr Felding og Daugard til Integrate.pptx` and
# the target-area outline in
# `ReferenceProjects/Integrate_Daugaard_Sdr_Felding/Sdr_Felding_delområde.shp`.
#
# This notebook instead runs the **full INTEGRATE probabilistic workflow**
# (Bayesian rejection sampling over a large ensemble of prior 1D models) and
# compares the resulting raw-material volumes -- this time with quantified
# **uncertainty** -- against the old single-number estimate. The workflow:
#
# 1. Load and merge the tTEM data (collected across several survey dates
#    with different system configurations) and the Sdr Felding boreholes.
# 2. *(Skipped for this site.)* Unlike Daugaard, no deterministic
#    (WorkBench LSQ) resistivity inversion is available for Sdr Felding, so
#    there is nothing to sanity-check a generic-prior probabilistic
#    inversion against here -- we go directly to the target-area prior.
# 3. Load the *target-area* lithology prior, already built by GEUS with
#    `geoprior1d` from geological knowledge of the site.
# 4. Run the full probabilistic INTEGRATE inversion (jointly using tTEM and
#    borehole data -- run from scratch here, since (unlike Daugaard) no
#    precomputed posterior is available for this site), and inspect
#    resistivity/lithology profiles and depth-slice maps.
# 5. Compute posterior statistics relevant to raw-material exploitation:
#    overburden thickness (with uncertainty) at each sounding location, and
#    the total area-integrated volume of overburden, raw material
#    (sand+gravel), and coarser material (gravel), with uncertainty.
# 6. Compare those probabilistic volume estimates to the old deterministic
#    number.
#
# As with Daugaard, recall the caveat from GEUS's own notes on the old
# assessment (`ReferenceProjects/EmailFromMette.md`): the deterministic
# raw-material estimate was **capped at the depth documented by raw-material
# boreholes**, even where the tTEM inversion suggested raw material could
# extend deeper. The probabilistic estimate below is not capped this way, so
# a meaningfully larger estimate is expected here.
#
# Run this notebook from within the `examples/` directory (as with the other
# INTEGRATE examples) so that the case data downloads and the
# `integrate_rawmaterial_utils` helper module are found correctly.

# %%
try:
    get_ipython()
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except Exception:
    pass

# %%
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

import integrate as ig
import integrate_rawmaterial_utils as rmu

hardcopy = True

# %% [markdown]
# ## 1. Load the data
#
# ### 1a. tTEM data
#
# The Sdr. Felding survey was flown across several dates with different
# system (GEX) configurations, so the case data ships as multiple
# `(gex, xyz-files)` groups (see `README_SOENDER_FELDING`) that need to be
# converted to HDF5 individually and then merged into one data file.

# %%
case = 'SOENDER_FELDING'
files = ig.get_case_data(case=case, showInfo=1)

gex_xyz_groups = [
    ('TX07_20240802_2x4_RC20-39.gex',
     ['20240819_AVG_export.xyz', '20240820_AVG_export.xyz',
      '20240821_AVG_export.xyz', '20240911_AVG_export.xyz']),
    ('TX07_20240802_2x4_RC20-39_eksternGPS.gex',
     ['20240911_eksterngps_AVG_export.xyz']),
    ('TX07_20240912_2x4_RC20-39_eksterngps.gex',
     ['20240924_AVG_export.xyz', '20240924_test_AVG_export.xyz',
      '20241007_AVG_export.xyz', '20241008_AVG_export.xyz']),
    ('TX07_20241014_2x4_RC20_33_and_57_EksternGPS.gex',
     ['20241029_AVG_export.xyz']),
    ('TX07_20241202_2x4_RC20_57_EksternGPS.gex',
     ['20241210_AVG_export.xyz']),
    ('TX07_20241202_2x4_RC20_57.gex',
     ['20241210_InternGPS_AVG_export.xyz']),
]

f_data_sub = []
for file_gex, file_xyz in gex_xyz_groups:
    fname = file_gex.split('.')[0]
    f_data_sub.append(ig.xyz_to_h5(file_xyz, file_gex, f_data_h5='%s_data.h5' % fname, showInfo=-1))

# %%
# Merge all data subsets into a single data file for inversion.
# ig.merge_data silently skips subsets whose channel/gate configuration is
# incompatible with the reference gex (it prints "Could not merge ..." for
# those) -- this is expected here, since the later survey dates used
# different system configurations; the merged file below still ends up with
# the large majority of the ~34,000 soundings.
f_gex = gex_xyz_groups[0][0]
f_data_h5 = ig.merge_data(f_data_sub, f_gex, f_data_merged_h5='SDR_FEDL_ALL.h5')

X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
ig.plot_data(f_data_h5, hardcopy=hardcopy, showInfo=-1)
ig.plot_data_xy(f_data_h5, data_channel=20, cmap='jet')
ig.plot_geometry(f_data_h5)

# %% [markdown]
# ### 1b. Boreholes
#
# Some boreholes in the JSON file have no recorded elevation (sentinel value
# -9999); fill those in from the nearest tTEM sounding's elevation, purely
# for plotting -- the inversion itself uses depth intervals relative to each
# borehole's own top (depth 0), so it does not depend on elevation.

# %%
BHOLES = ig.read_borehole('SdrFelding_boreholes.json', showInfo=1)
for ibh in range(len(BHOLES)):
    d = np.sqrt((X - BHOLES[ibh]['X']) ** 2 + (Y - BHOLES[ibh]['Y']) ** 2)
    i_closest = np.argmin(d)
    if BHOLES[ibh]['elevation'] == -9999:
        BHOLES[ibh]['elevation'] = ELEVATION[i_closest]

n_plots_per_figure = 8
for i in range(0, len(BHOLES), n_plots_per_figure):
    ig.plot_boreholes(BHOLES[i:i + n_plots_per_figure], hardcopy=hardcopy, i_start=i)

# %% [markdown]
# ### 1c. Target-area polygon (the old approach's raw-material area)
#
# Unlike Daugaard, the Sdr Felding shapefile has a usable `Delområde`
# attribute field, so the polygon name is read directly rather than matched
# by area.

# %%
f_shp_sdrfelding = os.path.join('ReferenceProjects', 'Integrate_Daugaard_Sdr_Felding', 'Sdr_Felding_delområde.shp')
polygons = rmu.load_target_polygons(f_shp_sdrfelding, name_field='Delområde')
# Give the single sub-area a clearer name for the comparison table
polygons = {'Sdr. Felding': list(polygons.values())[0]}

rmu.plot_polygons_over_points(X, Y, polygons, title='Sdr_Felding', hardcopy=hardcopy)

# %% [markdown]
# ## 2. (Skipped) Deterministic resistivity comparison
#
# No WorkBench LSQ (deterministic) resistivity inversion is registered for
# the Sdr Felding case data, unlike Daugaard (see
# `integrate_rawmaterial_daugaard.py`, Section 2). We proceed directly to
# the target-area lithology prior below.

# %% [markdown]
# ## 3. Target-area lithology prior (built by GEUS with `geoprior1d`)
#
# As for Daugaard, the lithology prior used here was already built by GEUS
# using `geoprior1d` (https://github.com/GEUSjesper/geoprior1d) from a
# geological specification of the Sdr Felding site; it is consumed as-is,
# not rebuilt in this notebook. For reference, such a prior is (re)built
# with:
#
# ```python
# from geoprior1d import geoprior1d
# f_prior_h5, flags = geoprior1d('Sdr_Felding_prior_standard.xlsx', Nreals=N,
#                                 dz=1, dmax=90, output_file='Sdr_Felding_prior_N%d.h5' % N)
# ```

# %%
f_prior_h5 = 'Sdr_Felding_prior_210526_N1000000_dmax90_20260521_1616.h5'
N = 1_000_000   # number of realizations in the downloaded prior

ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)
ig.prior_describe(f_prior_h5)

raw_classes, coarse_classes = rmu.resolve_material_classes(f_prior_h5, im=2)

# %% [markdown]
# ## 4. Full probabilistic (INTEGRATE) inversion
#
# No precomputed posterior is available for Sdr Felding, so this must be run
# from scratch. `N_use` below is a demo-scale subset of the 1,000,000-member
# prior for a tractable runtime; increase it for a production-quality run.

# %%
N_use = 200_000   # demo-scale; increase (up to N) for a production-quality run

f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, f_gex, N=N_use, doMakePriorCopy=True)

# %%
im_prior, r_data, r_dis = 2, 1, 1000   # radii tuned for many, sparsely-spaced boreholes
id_borehole_list = []
for BH in BHOLES:
    id_prior, id_out = ig.save_borehole_data(
        f_prior_data_h5, f_data_h5, BH, im_prior=im_prior,
        nan_freq=0.8, r_data=r_data, r_dis=r_dis, doPlot=False, showInfo=0)
    id_borehole_list.append(id_out)

# %%
id_use = [1] + id_borehole_list   # tTEM (id 1) jointly with all borehole logs
f_post_h5 = ig.integrate_rejection(
    f_prior_data_h5, f_data_h5, f_post_h5='POST_SDRFELDING.h5',
    N_use=N_use, id_use=id_use, nr=1000, T_N_above=50, T_P_acc_level=0.2,
    autoT=1, showInfo=1, updatePostStat=True)

ig.plot_T_EV(f_post_h5, pl='CHI2', hardcopy=hardcopy)
ig.plot_data_prior(f_prior_data_h5, f_data_h5, hardcopy=hardcopy, showInfo=-1)

# %% [markdown]
# ### Resistivity/lithology profile through a few boreholes

# %%
ibh_use = [0, 5, 10, 15]
Xl = np.array([BHOLES[i]['X'] for i in ibh_use])
Yl = np.array([BHOLES[i]['Y'] for i in ibh_use])
indices, distances, segment_ids = ig.find_points_along_line_segments(X, Y, Xl, Yl, tolerance=15.0)
id_line = indices

ig.plot_profile(f_post_h5, im=1, ii=id_line, gap_threshold=100, xaxis='x',
                 hardcopy=hardcopy, alpha=0.9, logstd_min=0.3, logstd_max=0.5)
ig.plot_profile(f_post_h5, im=2, ii=id_line, gap_threshold=100, xaxis='x',
                 hardcopy=hardcopy, alpha=0.9, entropy_min=0.5, entropy_max=1.0)

# %% [markdown]
# ### Depth-slice maps: mean resistivity and mode lithology

# %%
for elevation in [40, 20, 0, -20]:
    ig.plot_feature_2d(f_post_h5, key='HarmonicMean', im=1, elevation=elevation,
                        plotPoints=True, hardcopy=hardcopy)
    plt.show()

for elevation in range(40, -21, -10):
    ig.plot_feature_2d(f_post_h5, key='Mode', im=2, s=0.5, elevation=elevation,
                        plotPoints=True, hardcopy=hardcopy)
    plt.show()

# %% [markdown]
# ## 5. Posterior statistics for raw-material exploitation
#
# ### 5a. Per-point overburden thickness, with uncertainty

# %%
query_overburden = {
    "metric": {
        "im": 2,
        "classes": raw_classes,
        "thickness_mode": "first_occurrence",
        "depth_min": 0.0,
    },
    "percentiles": [5, 50, 95]
}
pct_overburden, meta = ig.query(f_post_h5, query_overburden)

ig.query_percentile_plot(pct_overburden, meta,
                          query_text='Overburden thickness (depth to first raw-material layer)',
                          hardcopy='sdrfelding_overburden_pct' if hardcopy else False)

overburden_90 = pct_overburden[:, 2] - pct_overburden[:, 0]
ig.plot_xy(overburden_90, f_data_h5=f_data_h5, f_prior_h5=f_prior_h5,
           cmap='hot', plotPoints=True, uselog=False,
           title='90%% uncertainty range of overburden thickness (m)',
           hardcopy='sdrfelding_overburden_90pct' if hardcopy else False)

# %% [markdown]
# ### 5b. Area-integrated volumes, with uncertainty
#
# Total overburden, raw-material (sand+gravel), and coarser-material
# (gravel) volume for the Sdr Felding target area, with a Monte Carlo
# P5/P50/P95 range (see
# `integrate_rawmaterial_utils.posterior_volume_montecarlo`).

# %%
N_BOOT = 500
prob_results = {}
for name, polygon in polygons.items():
    print("Computing area-integrated volumes for %s ..." % name)
    prob_results[name] = rmu.posterior_volume_montecarlo(
        f_post_h5, polygon, raw_classes, coarse_classes,
        im=2, n_boot=N_BOOT, area_cell_size=10.0, random_state=0)
    r = prob_results[name]
    print("  n_points=%d, area=%.0f m^2" % (r['n_points_in_polygon'], r['area_total_m2']))
    print("  overburden   P5/50/95 = %s m^3" % np.round(r['overburden']).astype(int))
    print("  raw material P5/50/95 = %s m^3" % np.round(r['raw_material']).astype(int))
    print("  coarser      P5/50/95 = %s m^3" % np.round(r['coarser']).astype(int))

# %% [markdown]
# ## 6. Comparison to the old (deterministic/sequential) result
#
# Reference numbers below are transcribed from
# `ReferenceProjects/Sdr Felding og Daugard til Integrate.pptx` (cross-checked
# against the shapefile polygon area).

# %%
SDRFELDING_REFERENCE = {
    'Sdr. Felding': {'overburden': 10_900_000, 'raw_material': 60_700_000, 'coarser': 16_200_000},
}

rmu.compare_to_reference(prob_results, SDRFELDING_REFERENCE, hardcopy=hardcopy,
                          f_name='sdrfelding_rawmaterial_comparison')

# %% [markdown]
# ### Discussion
#
# As with Daugaard, the value of the probabilistic estimate is the explicit
# uncertainty range, not just a (possibly larger) central number. Any gap
# between the probabilistic P50 and the old deterministic total should be
# interpreted alongside the borehole-depth-cap caveat from GEUS's notes
# (`ReferenceProjects/EmailFromMette.md`) -- the deterministic estimate was
# not allowed to extend raw material below documented borehole depth, while
# the probabilistic estimate here is not depth-capped. Whether classes such
# as "Miocene sand" (see Section 3's printed class list) should count as
# producible raw material at Sdr Felding is a geological judgement call that
# should be reviewed before treating this as a strict apples-to-apples
# comparison, exactly as noted for Daugaard.
