#!/usr/bin/env python
# %% [markdown]
# # Daugaard: probabilistic raw-material assessment with INTEGRATE
#
# GEUS previously assessed raw-material (sand/gravel) potential at three
# target sub-areas in Daugaard ("Delområde 1/2/3") using a **sequential,
# deterministic** workflow: invert the tTEM data to a single best-fit
# resistivity model, interpret lithology, and hand-estimate overburden and
# raw-material volumes per sub-area, constrained to what raw-material
# boreholes could document. Those results live in
# `ReferenceProjects/Sdr Felding og Daugard til Integrate.pptx` and the
# target-area outlines in `ReferenceProjects/Fokusområder_Daugaard/`.
#
# This notebook instead runs the **full INTEGRATE probabilistic workflow**
# (Bayesian rejection sampling over a large ensemble of prior 1D models) and
# compares the resulting raw-material volumes — this time with quantified
# **uncertainty** — against those old single-number estimates. The workflow:
#
# 1. Load the tTEM data and Daugaard boreholes.
# 2. Sanity check: invert with a *generic, resistivity-only* prior and
#    compare against an externally computed **deterministic** (WorkBench
#    LSQ) resistivity inversion.
# 3. Load the *target-area* lithology prior, already built by GEUS with
#    `geoprior1d` from geological knowledge of the site.
# 4. Run the full probabilistic INTEGRATE inversion (jointly using tTEM and
#    borehole data), and inspect resistivity/lithology profiles and
#    depth-slice maps.
# 5. Compute posterior statistics relevant to raw-material exploitation:
#    overburden thickness (with uncertainty) at each sounding location, and
#    the total area-integrated volume of overburden, raw material
#    (sand+gravel), and coarser material (gravel), each with uncertainty.
# 6. Compare those probabilistic volume estimates to the old deterministic
#    numbers, sub-area by sub-area.
#
# One important caveat, from GEUS's own notes on the old assessment (see
# `ReferenceProjects/EmailFromMette.md`): the deterministic raw-material
# estimate was **capped at the depth documented by raw-material boreholes**,
# even where the tTEM inversion suggested raw material could extend deeper.
# The probabilistic estimate below is *not* capped this way, so a
# meaningfully larger estimate is expected here — that is a genuine
# difference in what the two methods can say, not an error in either one.
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
# ### 1a. tTEM data, GEX system file, and boreholes

# %%
case = 'DAUGAARD'
files = ig.get_case_data(case=case, showInfo=1)
f_data_h5 = files[0]
file_gex = ig.get_gex_file_from_data(f_data_h5)
print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)

X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

ig.plot_geometry(f_data_h5, pl='LINE')
ig.plot_data(f_data_h5, hardcopy=hardcopy)
ig.plot_data_xy(f_data_h5, data_channel=15, cmap='jet')

# %%
BHOLES = ig.read_borehole('daugaard_12boreholes.json', showInfo=1)
ig.plot_boreholes(BHOLES)

# %% Optionally scale noise
inflateNoise = 2
if inflateNoise != 0:
    gf=inflateNoise
    print("="*60)
    print("Increasing noise level (std) by a factor of %d" % gf)
    print("="*60)
    D = ig.load_data(f_data_h5)
    D_obs = D['d_obs'][0]
    D_std = D['d_std'][0]*gf
    f_data_old_h5 = f_data_h5
    f_data_h5 = 'DAUGAARD_AVG_gf%g.h5' % (gf) 
    ig.copy_hdf5_file(f_data_old_h5, f_data_h5)
    ig.save_data_gaussian(D_obs, D_std=D_std, f_data_h5=f_data_h5, file_gex=file_gex)

ig.plot_data(f_data_h5, useLog = 0, hardcopy= hardcopy)
plt.show()
# %% [markdown]
# ### 1b. Target-area polygons (the old approach's raw-material areas)
#
# The Daugaard focus-area shapefile has an almost-empty attribute table, so
# the 3 Delområder are matched to their names by polygon area, which is
# reported (and matches exactly) in the old-approach PowerPoint.

# %%
DAUGAARD_AREA_ID_MAP = {
    308812.2: 'Delområde 1',   # DAU_02, DAU_06
    106727.7: 'Delområde 2',   # DAU_07
    196598.0: 'Delområde 3',   # DAU_05, DAU_03
}
f_shp_daugaard = os.path.join('ReferenceProjects', 'Fokusområder_Daugaard', 'Fokusområder_polygon.shp')
polygons = rmu.load_target_polygons(f_shp_daugaard, area_id_map=DAUGAARD_AREA_ID_MAP)

# %% [markdown]
# ### Profile through the centers of the three Delområder, west to east
#
# Build a profile line through the centroid of each target polygon, ordered
# from west (smallest X) to east (largest X), and find the sounding indices
# along that line with `ig.find_points_along_line_segments` -- the same
# function used for profile selection in `integrate_workflow.py` and
# `integrate_profiles.py`. These indices (`id_line`) are reused later, once
# the posterior is available, to plot resistivity and lithology sections
# along this single west-to-east line.

# %%
centroids = {name: (geom.centroid.x, geom.centroid.y) for name, geom in polygons.items()}
centroids_sorted = sorted(centroids.items(), key=lambda kv: kv[1][0])  # west (small X) -> east (large X)
for name, (cx, cy) in centroids_sorted:
    print("  %-15s centroid = (%9.1f, %9.1f)" % (name, cx, cy))

Xl = np.array([cx for _, (cx, cy) in centroids_sorted])
Yl = np.array([cy for _, (cx, cy) in centroids_sorted])

buffer = 10.0

# Widen the profile at both ends so it also picks up every sounding
# strictly west of the westernmost Delområde center and strictly east of
# the easternmost one, not just soundings near the 3 centroids themselves:
# extend the line with one extra waypoint beyond each end of the data's
# X-range (at the Y of the nearest centroid), then re-search along the
# now-longer line.
pad = buffer
Xl_wide = np.concatenate(([X.min() - pad], Xl, [X.max() + pad]))
Yl_wide = np.concatenate(([Yl[0]], Yl, [Yl[-1]]))

indices, distances, segment_ids = ig.find_points_along_line_segments(X, Y, Xl_wide, Yl_wide, tolerance=buffer)
id_line = indices
print("Found %d soundings within %.0f m of the widened west-to-east center profile" % (len(id_line), buffer))

# Sanity check: are the tTEM soundings and the target polygons in the same
# coordinate frame? (Both are UTM32N, but this is worth confirming visually
# rather than assuming.) The selected profile line and its soundings are
# overlaid on the same plot.
rmu.plot_polygons_over_points(X, Y, polygons, title='Daugaard', hardcopy=hardcopy,
                               profile_xy=(Xl_wide, Yl_wide), profile_idx=id_line)

# %% [markdown]
# ## 2. Sanity check: generic resistivity-only prior vs. a deterministic inversion
#
# Before bringing in any geological/lithological knowledge, invert the data
# probabilistically with a *generic* resistivity-only prior (no lithology
# classes — just a layered-Earth resistivity model, as in
# `integrate_getting_started.py`), and compare the resulting resistivity
# section to an externally computed **deterministic** inversion: a WorkBench
# least-squares (LSQ) inversion, in both a smooth and a sharp (blocky)
# regularization variant. There is no deterministic inversion built into
# INTEGRATE itself (the package only performs probabilistic/rejection-based
# inversion) -- the LSQ result is an independently computed reference file.

# %%
N_generic = 2_000_000   # demo-scale; increase for a production-quality run
f_prior_generic_h5 = ig.prior_model_layered(
    N=N_generic, lay_dist='chi2', NLAY_deg=3, RHO_min=1, RHO_max=3000,
    f_prior_h5='PRIOR_GENERIC_N%d.h5' % N_generic, showInfo=1)
ig.plot_prior_stats(f_prior_generic_h5, hardcopy=hardcopy)

f_prior_generic_data_h5 = '%s_%s_Nh280_Nf12.h5' % (f_prior_generic_h5[:-3], file_gex[:-4])
if not os.path.exists(f_prior_generic_data_h5):
    f_prior_generic_data_h5 = ig.prior_data_gaaem(
        f_prior_generic_h5, file_gex, doMakePriorCopy=True,
        f_prior_data_h5=f_prior_generic_data_h5)

# %%
f_post_generic_h5 = ig.integrate_rejection(
    f_prior_generic_data_h5, f_data_h5, f_post_h5='POST_GENERIC.h5',
    N_use=N_generic, id_use=[1], autoT=1, T_base=1, showInfo=0, updatePostStat=True,
    backend='jax')

ig.plot_T_EV(f_post_generic_h5, pl='CHI2', hardcopy=hardcopy)

# %% [markdown]
# ### Download the deterministic (WorkBench LSQ) inversion and reproject it
# onto the same depth grid as the probabilistic posterior, following the
# pattern in `integrate_paper_daugaard_supp.py`.

# %%
import libaarhusxyz

ig.get_case_data(case=case, loadType='WB_smooth')
ig.get_case_data(case=case, loadType='WB_sharp')

f_xyz_list = {
    'smooth': 'SCI7_40_ml_Daugaard_I01_MOD_inv.xyz',
    'sharp':  'SCI7_40_ml_sharp2_I02_MOD_inv.xyz',
}

f_lsq_h5_list = {}
for label, f_xyz in f_xyz_list.items():
    f_lsq_h5 = os.path.splitext(f_xyz)[0] + '.h5'
    ig.copy_hdf5_file(f_post_generic_h5, f_lsq_h5)

    Xp, Yp, _, _ = ig.get_geometry(f_lsq_h5)
    with h5py.File(f_lsq_h5, 'r') as f:
        M1_median = f['/M1/Median'][:]
    with h5py.File(f_prior_generic_data_h5, 'r') as f:
        z = f['/M1'].attrs['x'][:].flatten()

    model = libaarhusxyz.XYZ(f_xyz)
    rho = model.layer_data['rho'].values
    rho_std = model.layer_data['rho_std'].values
    dep_top = model.layer_data['dep_top'].values
    utmx = model.flightlines['utmx'].values
    utmy = model.flightlines['utmy'].values

    M1_mean = np.full_like(M1_median, np.nan)
    M1_std = np.full_like(M1_median, np.nan)
    for i in range(len(utmx)):
        idx = np.argmin((Xp - utmx[i]) ** 2 + (Yp - utmy[i]) ** 2)
        j_idx = np.searchsorted(dep_top[i], z, side='right') - 1
        valid = j_idx >= 0
        M1_mean[idx, valid] = rho[i, j_idx[valid]]
        M1_std[idx, valid] = rho_std[i, j_idx[valid]]

    with h5py.File(f_lsq_h5, 'a') as f:
        for key in ['/M1/LogMean', '/M1/Mean', '/M1/Std']:
            if key in f:
                del f[key]
        f['/M1/Mean'] = M1_mean
        f['/M1/LogMean'] = np.log10(M1_mean)
        f['/M1/Std'] = np.log10(M1_std)
    f_lsq_h5_list[label] = f_lsq_h5

# %%
# Reuse the same west-to-east profile (`id_line`) computed in Section 1b --
# one single profile used consistently throughout the notebook.
for label, f_lsq_h5 in f_lsq_h5_list.items():
    ig.plot_profile(f_lsq_h5, ii=id_line, gap_threshold=100, xaxis='x',
                     panels=['mean', 'std'], im=1, hardcopy=hardcopy,
                     txt='WB_%s' % label)
ig.plot_profile(f_post_generic_h5, ii=id_line, gap_threshold=150, xaxis='x',
                 panels=['median', 'std', 'stats'], im=1, hardcopy=hardcopy,
                 txt='probabilistic_generic')

# %% [markdown]
# **Takeaway**: the probabilistic (generic prior) inversion recovers a
# resistivity section broadly consistent with the deterministic WorkBench
# LSQ result, but additionally quantifies the uncertainty (std) at every
# point, rather than delivering only a single smooth/sharp model. This
# resistivity-only comparison does not yet say anything about raw-material
# volumes -- that requires the lithology (target-area) prior in the next
# section.

# %% [markdown]
# ## 3. Target-area lithology prior (built with `geoprior1d`)
#
# The lithology prior used for the rest of this workflow is built with the
# `geoprior1d` tool (https://github.com/GEUSjesper/geoprior1d) from an Excel
# specification of expected layer sequences, thicknesses, and resistivities
# for the Daugaard geology -- this is the step GEUS owns for a new target
# area: encoding their geological knowledge of the site as a geoprior1d
# Excel workbook.
#
# Two geological scenarios are available for Daugaard: `daugaard_standard.xlsx`
# (background/regional geology) and `daugaard_valley.xlsx` (geology inside a
# buried meltwater valley). Since it is not known in advance, at any given
# sounding location, whether that location sits inside or outside a buried
# valley, realizations are generated independently from *both* Excel
# specifications and then combined into a single prior with `ig.merge_prior`,
# 50% from each scenario -- letting the data itself indicate, location by
# location, which geological scenario it is more consistent with (as in
# `integrate_workflow.py`).

# %%
from geoprior1d import geoprior1d

ig.get_case_data(case=case, filelist=['daugaard_standard.xlsx', 'daugaard_valley.xlsx'])

#N_prior_each = 1_000_000   # realizations per geological scenario; merged prior has 2x this
# N_prior_each should be half of N_generic
N_prior_each = N_generic // 2
f_xlsx_files = ['daugaard_standard.xlsx', 'daugaard_valley.xlsx']

f_prior_h5_list = []
for file_xlsx in f_xlsx_files:
    fname = file_xlsx.split('.')[0]
    f_prior_scenario_h5 = '%s_prior_N%d.h5' % (fname, N_prior_each)
    if not os.path.exists(f_prior_scenario_h5):
        f_prior_scenario_h5, flags = geoprior1d(
            file_xlsx, Nreals=N_prior_each, dz=1, dmax=90,
            output_file=f_prior_scenario_h5)
    else:
        print("Using existing prior realizations: %s" % f_prior_scenario_h5)
    f_prior_h5_list.append(f_prior_scenario_h5)

f_prior_merged_h5 = 'daugaard_merged_prior_N%d.h5' % N_prior_each
if not os.path.exists(f_prior_merged_h5):
    f_prior_h5 = ig.merge_prior(f_prior_h5_list, f_prior_merged_h5=f_prior_merged_h5)
else:
    print("Using existing merged prior: %s" % f_prior_merged_h5)
    f_prior_h5 = f_prior_merged_h5

ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)
ig.prior_describe(f_prior_h5)

# Resolve which lithology classes count as "raw material" and "coarser
# material" by name -- see integrate_rawmaterial_utils.resolve_material_classes.
raw_classes, coarse_classes = rmu.resolve_material_classes(f_prior_h5, im=2)

# %% [markdown]
# ## 4. Full probabilistic (INTEGRATE) inversion
#
# No precomputed posterior exists for this newly-built, 50/50 standard+valley
# merged prior, so the rejection sampler is run from scratch. `N_use` below
# subsets the 2,000,000-member prior to a demo-scale run for a tractable
# runtime (forward-modeling the full prior for a single GEX configuration
# alone takes on the order of hours); increase it for a production-quality
# run. Boreholes are included via `ig.save_borehole_data` as an additional,
# jointly inverted data type, exactly as in `integrate_workflow.py`.

# %%
N_use = N_generic   # demo-scale subset of the 2,000,000-member prior; increase for production

f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, N=N_use, doMakePriorCopy=True)

im_prior, r_data, r_dis = 2, 4, 300
id_borehole_list = []
for BH in BHOLES:
    id_prior, id_out = ig.save_borehole_data(
        f_prior_data_h5, f_data_h5, BH, im_prior=im_prior, r_data=r_data,
        r_dis=r_dis, doPlot=False, showInfo=0)
    id_borehole_list.append(id_out)

id_use = [1] + id_borehole_list   # tTEM (id 1) jointly with all borehole logs
f_post_h5 = ig.integrate_rejection(
    f_prior_data_h5, f_data_h5, f_post_h5='POST_DAUGAARD_MERGED.h5',
    N_use=N_use, id_use=id_use, nr=1000, T_N_above=50, T_P_acc_level=0.2,
    autoT=1, showInfo=1, updatePostStat=True)

ig.plot_T_EV(f_post_h5, pl='CHI2', hardcopy=hardcopy)

# %% [markdown]
# ### Resistivity/lithology profile through the Delområde centers
#
# Reuse the west-to-east profile through the three Delområde centroids
# (`id_line`, found in Section 1b -- see the polygon/profile alignment plot
# there) to plot posterior harmonic-mean resistivity (`im=1`) and mode
# lithology (`im=2`) sections along that single line.

# %%
ig.plot_profile(f_post_h5, im=1, ii=id_line, key='HarmonicMean', gap_threshold=100,
                 xaxis='x', hardcopy=hardcopy)
ig.plot_profile(f_post_h5, im=2, ii=id_line, panels=['mode'], gap_threshold=100,
                 xaxis='x', hardcopy=hardcopy)

# %% [markdown]
# ### Depth-slice maps: mean resistivity and mode lithology

# %%
for elevation in [40, 30, 20, 10]:
    ig.plot_feature_2d(f_post_h5, im=1, key='HarmonicMean', elevation=elevation,
                        uselog=1, cmap='jet', s=2, hardcopy=hardcopy)
    plt.show()

for elevation in [40, 30, 20, 10]:
    ig.plot_feature_2d(f_post_h5, im=2, key='Mode', elevation=elevation,
                        cmap='jet', s=2, hardcopy=hardcopy)
    plt.show()

# %% [markdown]
# ## 5. Posterior statistics for raw-material exploitation
#
# ### 5a. Per-point overburden thickness, with uncertainty
#
# The thickness (down to the first occurrence of a raw-material class) is
# queried per sounding location via `ig.query_percentile`; the width of the
# 90% interval (P95-P5) is a per-point uncertainty map, following the same
# pattern used in `integrate_workflow_soenderfelding.py`.

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
                          hardcopy='daugaard_overburden_pct' if hardcopy else False)

overburden_90 = pct_overburden[:, 2] - pct_overburden[:, 0]
ig.plot_xy(overburden_90, f_data_h5=f_data_h5, f_prior_h5=f_prior_h5,
           cmap='hot', plotPoints=True, uselog=False,
           title='90%% uncertainty range of overburden thickness (m)',
           hardcopy='daugaard_overburden_90pct' if hardcopy else False)

# %% [markdown]
# ### 5b. Area-integrated volumes, with uncertainty, per Delområde
#
# For each target sub-area: total overburden volume, total raw-material
# (sand+gravel) volume, and total coarser-material (gravel) volume, each
# with a Monte Carlo P5/P50/P95 range (see
# `integrate_rawmaterial_utils.posterior_volume_montecarlo`).

# %%
N_BOOT = 500
prob_results = {}
for name, polygon in polygons.items():
    print("Computing area-integrated volumes for %s ..." % name)
    prob_results[name] = rmu.posterior_volume_montecarlo(
        f_post_h5, polygon, raw_classes, coarse_classes,
        im=2, n_boot=N_BOOT, area_cell_size=5.0, random_state=0)
    r = prob_results[name]
    print("  n_points=%d, area=%.0f m^2" % (r['n_points_in_polygon'], r['area_total_m2']))
    print("  overburden   P5/50/95 = %s m^3" % np.round(r['overburden']).astype(int))
    print("  raw material P5/50/95 = %s m^3" % np.round(r['raw_material']).astype(int))
    print("  coarser      P5/50/95 = %s m^3" % np.round(r['coarser']).astype(int))

# %% [markdown]
# ## 6. Comparison to the old (deterministic/sequential) results
#
# Reference numbers below are transcribed from
# `ReferenceProjects/Sdr Felding og Daugard til Integrate.pptx` (cross-checked
# against the shapefile polygon areas).

# %%
DAUGAARD_REFERENCE = {
    'Delområde 1': {'overburden': 865_000,   'raw_material': 4_500_000, 'coarser': 1_700_000},
    'Delområde 2': {'overburden': 566_000,   'raw_material': 1_700_000, 'coarser': 1_000_000},
    'Delområde 3': {'overburden': 452_000,   'raw_material': 3_300_000, 'coarser': 2_200_000},
}

rmu.compare_to_reference(prob_results, DAUGAARD_REFERENCE, hardcopy=hardcopy,
                          f_name='daugaard_rawmaterial_comparison')

# %% [markdown]
# ### Discussion
#
# The probabilistic estimates come with an explicit uncertainty range instead
# of a single number, which is the key practical addition of this workflow:
# a decision-maker can see not just a central raw-material volume estimate,
# but how confident that estimate is, sub-area by sub-area.
#
# If the probabilistic P50 raw-material volume is noticeably larger than the
# old deterministic estimate, recall the caveat from GEUS's notes on the old
# assessment (`ReferenceProjects/EmailFromMette.md`): the deterministic
# estimate was capped at the depth documented by raw-material boreholes, even
# where the tTEM inversion suggested raw material could extend deeper. The
# probabilistic model is not capped this way, so part of any gap reflects
# that methodological difference rather than a disagreement about the
# shallow geology. Conversely, if the *lower* end of the uncertainty range
# (P5) is still above the old estimate, or its interval brackets it, that is
# a useful cross-check that the two approaches broadly agree where they
# should. Whether deep/older units matched by `resolve_material_classes`
# (e.g. Miocene sand) should count as producible raw material at all is a
# geological judgement call that should be reviewed before treating this as
# a strict apples-to-apples comparison (see Section 3's printed class list).
