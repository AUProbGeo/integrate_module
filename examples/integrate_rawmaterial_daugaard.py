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
# This notebook is organised in three parts:
#
# **Part A -- Resistivity only: probabilistic vs. deterministic.**
# Invert the tTEM data with a *generic* resistivity-only prior (a plain
# layered-Earth model, no lithology classes) and compare the resulting
# harmonic-mean / std resistivity section to an externally computed
# **deterministic** WorkBench least-squares (LSQ) inversion. This part is
# purely a demonstration of how probabilistic resistivity-only inversion and
# deterministic inversion relate to each other -- it is not used for the
# raw-material estimate.
#
# **Part B -- The full INTEGRATE workflow.**
# Build an *informed* prior from two geological scenarios
# (`daugaard_standard.xlsx` + `daugaard_valley.xlsx`, built with `geoprior1d`,
# merged 50/50), forward-model the prior tTEM data, load and add the borehole
# data, run the joint (tTEM + borehole) rejection inversion, demonstrate the
# `ig.query` tool, automatically grow a data-driven raw-material area from the
# posterior probability map (B8), and turn it into low / median / high
# volume estimates alongside Mette's target polygons (B9). Volumes are built
# from per-sounding thickness percentiles times the Voronoi-cell area around
# each sounding.
#
# **Part C -- Comparison to Mette's original estimate.**
# Put the probabilistic per-polygon volumes next to the single-number
# estimates from the old sequential/deterministic assessment.
#
# One important caveat, from GEUS's own notes on the old assessment (see
# `ReferenceProjects/EmailFromMette.md`): the deterministic raw-material
# estimate was **capped at the depth documented by raw-material boreholes**,
# even where the tTEM inversion suggested raw material could extend deeper.
# The probabilistic estimate here is *not* capped this way, so a meaningfully
# larger estimate is expected -- a genuine methodological difference, not an
# error in either approach.
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
# ## 0. Settings and fixed file names
#
# Every expensive step (prior sampling, forward modelling, rejection) writes
# to a **fixed** path defined below and is skipped with `os.path.exists` when
# that file already exists. This makes it possible to run any one section on
# its own: as long as the file-name variables in this cell are defined, a
# later section can just load `f_post_h5` (etc.) without re-running the
# earlier ones -- e.g. run Part C directly once `f_post_h5` exists.
#
# The names do NOT encode `N` / `inflateNoise`. If you change either, delete
# the affected `DAUGAARD_*.h5` files (or bump `SUFFIX`) so they are
# regenerated.

# %%
# --- run-size settings -------------------------------------------------
N = 2_000_000   # production-scale
N = 42_000      # demo-scale; increase for a production-quality run
# Prior size used everywhere: the generic prior (Part A) and each of the two
# geological-scenario priors merged into the informed prior (Part B, N // 2
# realizations each).

inflateNoise = 2        # multiply the tTEM data std by this (0 = use as-is)

SUFFIX = ''             # optional tag appended to every generated file name

# --- fixed output file names -----------------------------------------
# working data file (noise-inflated copy; == downloaded file if inflateNoise == 0)
f_data_work_h5 = 'DAUGAARD_AVG_gf%g%s.h5' % (inflateNoise, SUFFIX)

# Part A -- resistivity-only, generic prior
f_prior_generic_h5      = 'DAUGAARD_PRIOR_GENERIC%s.h5'       % SUFFIX
f_prior_generic_data_h5 = 'DAUGAARD_PRIOR_GENERIC_DATA%s.h5'  % SUFFIX
f_post_generic_h5       = 'DAUGAARD_POSTERIOR_GENERIC%s.h5'   % SUFFIX

# Part B -- informed prior + boreholes
f_prior_scenario_h5_map = {
    'daugaard_standard': 'DAUGAARD_PRIOR_STANDARD%s.h5' % SUFFIX,
    'daugaard_valley':   'DAUGAARD_PRIOR_VALLEY%s.h5'   % SUFFIX,
}
f_prior_merged_h5  = 'DAUGAARD_PRIOR_MERGED%s.h5'          % SUFFIX
f_prior_data_h5    = 'DAUGAARD_PRIOR_MERGED_DATA%s.h5'     % SUFFIX
f_prior_data_bh_h5 = 'DAUGAARD_PRIOR_MERGED_DATA_BH%s.h5'  % SUFFIX
f_post_h5          = 'DAUGAARD_POSTERIOR%s.h5'             % SUFFIX

# %% [markdown]
# ## 1. Load the data
#
# ### 1a. tTEM data and GEX system file

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

# %% Optionally scale the data noise (fixed name -> built once, reused after)
if inflateNoise != 0:
    print("=" * 60)
    print("Using tTEM data with noise (std) inflated by a factor of %d" % inflateNoise)
    print("=" * 60)
    f_data_src_h5 = f_data_h5
    f_data_h5 = f_data_work_h5
    if not os.path.exists(f_data_h5):
        D = ig.load_data(f_data_src_h5)
        ig.copy_hdf5_file(f_data_src_h5, f_data_h5)
        ig.save_data_gaussian(D['d_obs'][0], D_std=D['d_std'][0] * inflateNoise,
                              f_data_h5=f_data_h5, file_gex=file_gex)

ig.plot_data(f_data_h5, useLog=0, hardcopy=hardcopy)
plt.show()

# %% [markdown]
# ### 1b. Target-area polygons and the west-to-east profile line
#
# The Daugaard focus-area shapefile (the areas Mette selected from the
# deterministic interpretation) has an almost-empty attribute table, so the 3
# Delområder are matched to their names by polygon area, which is reported
# (and matches exactly) in the old-approach PowerPoint. The polygons are
# loaded here because the profile line used for the resistivity sections in
# Parts A and B runs through their centroids; the polygons themselves are
# used for the volume analysis in Part C.

# %%
DAUGAARD_AREA_ID_MAP = {
    308812.2: 'Delområde 1',   # DAU_02, DAU_06
    106727.7: 'Delområde 2',   # DAU_07
    196598.0: 'Delområde 3',   # DAU_05, DAU_03
}
f_shp_daugaard = os.path.join('ReferenceProjects', 'Fokusområder_Daugaard', 'Fokusområder_polygon.shp')
polygons = rmu.load_target_polygons(f_shp_daugaard, area_id_map=DAUGAARD_AREA_ID_MAP)

# %% [markdown]
# Build a profile line through the centroid of each target polygon, ordered
# from west (smallest X) to east (largest X), and find the sounding indices
# along that line with `ig.find_points_along_line_segments` -- the same
# function used for profile selection in `integrate_workflow.py` and
# `integrate_profiles.py`. These indices (`id_line`) are reused in Parts A
# and B to plot resistivity/lithology sections along this single line.

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
# # Part A -- Resistivity only: probabilistic (generic prior) vs. deterministic
#
# Invert the tTEM data with a *generic* resistivity-only prior (a plain
# layered-Earth model, `lay_dist='chi2'`, no lithology classes, as in
# `integrate_getting_started.py`). Plot the posterior harmonic-mean and std
# resistivity along the west-to-east profile, then load the externally
# computed deterministic WorkBench LSQ inversion and plot it the same way.
# The point of this part is only to show how the probabilistic
# resistivity-only result and the deterministic result relate to each other.

# %%
if not os.path.exists(f_prior_generic_h5):
    ig.prior_model_layered(
        N=N, lay_dist='chi2', NLAY_deg=3, RHO_min=1, RHO_max=3000,
        f_prior_h5=f_prior_generic_h5, showInfo=1)
ig.plot_prior_stats(f_prior_generic_h5, hardcopy=hardcopy)

# %%
if not os.path.exists(f_prior_generic_data_h5):
    ig.prior_data_gaaem(f_prior_generic_h5, file_gex, doMakePriorCopy=True,
                        f_prior_data_h5=f_prior_generic_data_h5)

# %%
if not os.path.exists(f_post_generic_h5):
    ig.integrate_rejection(
        f_prior_generic_data_h5, f_data_h5, f_post_h5=f_post_generic_h5,
        N_use=N, id_use=[1], autoT=1, T_base=1, showInfo=0,
        updatePostStat=True, backend='jax')

ig.plot_T_EV(f_post_generic_h5, pl='CHI2', hardcopy=hardcopy)

# %% [markdown]
# ### Probabilistic resistivity-only profile: harmonic mean and std (log space)

# %%
ig.plot_profile(f_post_generic_h5, ii=id_line, im=1, panels=['harmonicmean', 'std'],
                xaxis='x', gap_threshold=150, hardcopy=hardcopy,
                txt='probabilistic_generic')

# %% [markdown]
# ### Deterministic (WorkBench LSQ) inversion
#
# Download the WorkBench LSQ result (smooth and sharp variants) and reproject
# it onto the same depth grid as the probabilistic posterior, following the
# pattern in `integrate_paper_daugaard_supp.py`, then plot it as a normal
# profile (mean + std, log space) along the same `id_line`.

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
    M1_stdf = np.full_like(M1_median, np.nan)   # raw WorkBench rho_std (uncertainty factor, >= 1)
    for i in range(len(utmx)):
        idx = np.argmin((Xp - utmx[i]) ** 2 + (Yp - utmy[i]) ** 2)
        j_idx = np.searchsorted(dep_top[i], z, side='right') - 1
        valid = j_idx >= 0
        M1_mean[idx, valid] = rho[i, j_idx[valid]]
        M1_stdf[idx, valid] = rho_std[i, j_idx[valid]]

    # Aarhus Workbench RHO_STD is the *standard-deviation factor* (STDF): the
    # parameter is estimated in log space, so it is multiplicative --
    #   68% CI      = [rho / RHO_STD, rho * RHO_STD]
    #   std(log10 rho) = log10(RHO_STD)   <- this is what the std panel plots
    #   std(ln    rho) = ln(RHO_STD)
    # RHO_STD == 1 -> perfectly resolved; large -> unresolved (Workbench caps
    # it at 99). It is NOT additive and NOT a percentage, so (RHO_STD - 1) is
    # only valid when RHO_STD is close to 1 (see the smooth vs. sharp header:
    # /MODEL TYPE, /MODEL UNIT, /INVERSION DATA SPACE = Logarithmic).
    M1_logstd = np.log10(M1_stdf)           # std of log10(rho); values > 1 = unresolved
    M1_std    = M1_mean * np.log(M1_stdf)   # approx linear-space std [ohm-m] (rho * sigma_ln)

    # f_lsq_h5 was copied from the probabilistic posterior, so it still holds
    # that run's /M1 statistics. Overwrite EVERY statistic plot_profile might
    # read, or a stale probabilistic value shows through. In particular the
    # std panel plots /M1/LogStd. The LSQ result is a single model, so
    # mean = median = harmonic mean = the model itself.
    with h5py.File(f_lsq_h5, 'a') as f:
        for key in ['/M1/Mean', '/M1/LogMean', '/M1/Median', '/M1/HarmonicMean',
                    '/M1/Std', '/M1/LogStd']:
            if key in f:
                del f[key]
        f['/M1/Mean'] = M1_mean
        f['/M1/Median'] = M1_mean
        f['/M1/HarmonicMean'] = M1_mean
        f['/M1/LogMean'] = np.log10(M1_mean)
        f['/M1/Std'] = M1_std           # linear-space std [ohm-m]
        f['/M1/LogStd'] = M1_logstd     # std of log10(rho) -- what the std panel plots
    f_lsq_h5_list[label] = f_lsq_h5

# %%
# Same west-to-east profile (`id_line`) as the probabilistic plot above.
for label, f_lsq_h5 in f_lsq_h5_list.items():
    ig.plot_profile(f_lsq_h5, ii=id_line, im=1, panels=['mean', 'std'],
                    xaxis='x', gap_threshold=100, hardcopy=hardcopy,
                    txt='WB_%s' % label)
ig.plot_profile(f_post_generic_h5, ii=id_line, im=1, panels=['harmonicmean', 'std'],
                xaxis='x', gap_threshold=150, hardcopy=hardcopy,
                txt='probabilistic_generic')

# %% [markdown]
# **Takeaway**: the probabilistic resistivity-only inversion recovers a
# section broadly consistent with the deterministic WorkBench LSQ result, but
# additionally quantifies the uncertainty (std) at every point rather than
# delivering only a single smooth/sharp model. This resistivity-only
# comparison says nothing about raw-material volumes -- that needs the
# informed lithology prior in Part B.


# %% [markdown]
# # Part B -- The full INTEGRATE workflow (informed prior + boreholes)
#
# ### B1. Boreholes

# %%
BHOLES = ig.read_borehole('daugaard_12boreholes.json', showInfo=1)
ig.plot_boreholes(BHOLES)

# %% [markdown]
# ### B2. Informed prior from two geological scenarios
#
# The prior for the real workflow is built with `geoprior1d`
# (https://github.com/GEUSjesper/geoprior1d) from Excel specifications of the
# expected layer sequences, thicknesses, and resistivities for the Daugaard
# geology -- this is the step GEUS owns for a new target area.
#
# Two scenarios are available: `daugaard_standard.xlsx` (background/regional
# geology) and `daugaard_valley.xlsx` (geology inside a buried meltwater
# valley). Since it is not known in advance whether a given sounding sits
# inside or outside a buried valley, `N // 2` realizations are generated
# independently from *each* Excel specification and combined into a single
# prior with `ig.merge_prior` (50% from each scenario), letting the data
# indicate, location by location, which scenario it is more consistent with
# -- the same construction used in `integrate_workflow.py` and
# `integrate_daugaard_multi_prior.py`.

# %%
from geoprior1d import geoprior1d

ig.get_case_data(case=case, filelist=['daugaard_standard.xlsx', 'daugaard_valley.xlsx'])

# N_prior_each realizations per geological scenario; the merged prior has 2x this.
N_prior_each = N // 2
f_xlsx_files = ['daugaard_standard.xlsx', 'daugaard_valley.xlsx']

f_prior_h5_list = []
for file_xlsx in f_xlsx_files:
    fname = file_xlsx.split('.')[0]
    f_scen_h5 = f_prior_scenario_h5_map[fname]
    if not os.path.exists(f_scen_h5):
        geoprior1d(file_xlsx, Nreals=N_prior_each, dz=1, dmax=90, output_file=f_scen_h5)
    else:
        print("Using existing prior realizations: %s" % f_scen_h5)
    f_prior_h5_list.append(f_scen_h5)

if not os.path.exists(f_prior_merged_h5):
    ig.merge_prior(f_prior_h5_list, f_prior_merged_h5=f_prior_merged_h5)
else:
    print("Using existing merged prior: %s" % f_prior_merged_h5)
f_prior_h5 = f_prior_merged_h5

ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)
ig.prior_describe(f_prior_h5)

# Resolve which lithology classes count as "raw material" (sand+gravel) and
# "coarser material" (gravel) by name -- see
# integrate_rawmaterial_utils.resolve_material_classes.
raw_classes, coarse_classes = rmu.resolve_material_classes(f_prior_h5, im=2)

# %% [markdown]
# ### B3. Prior tTEM data
#
# Forward-model the merged prior once (this is the expensive step; the result
# is reused for the borehole prior data below).

# %%
if not os.path.exists(f_prior_data_h5):
    ig.prior_data_gaaem(f_prior_h5, file_gex, doMakePriorCopy=True,
                        f_prior_data_h5=f_prior_data_h5)

# %% [markdown]
# ### B4. Borehole prior data
#
# Add the boreholes as extra, jointly inverted data types onto a fresh copy
# of the tTEM prior-data file. `ig.save_borehole_data` takes the whole
# `BHOLES` list in one call (no Python loop) and returns one prior/data `/D`
# index per borehole, exactly as in `integrate_workflow.py`. Boreholes are
# appended right after the tTEM data (`/D1`), so on the cached path the data
# IDs are the deterministic `2 .. 1+len(BHOLES)`. Skipped entirely if the
# posterior already exists.

# %%
im_prior = 2
id_borehole_list = list(range(2, 2 + len(BHOLES)))   # deterministic /D indices (cached path)

if not os.path.exists(f_post_h5) and not os.path.exists(f_prior_data_bh_h5):
    ig.copy_hdf5_file(f_prior_data_h5, f_prior_data_bh_h5)
    id_prior_list, id_borehole_list = ig.save_borehole_data(
        f_prior_data_bh_h5, f_data_h5, BHOLES,
        im_prior=im_prior, range_xyz=300,
        doPlot=False, showInfo=0)
else:
    print("Skipping borehole prior-data build (posterior or %s already exists)."
          % f_prior_data_bh_h5)

# %% [markdown]
# ### B5. Joint (tTEM + borehole) rejection inversion

# %%
N_use = N   # subset of the merged prior used in the rejection sampler

if not os.path.exists(f_post_h5):
    id_use = [1] + id_borehole_list   # tTEM (id 1) jointly with all borehole logs
    ig.integrate_rejection(
        f_prior_data_bh_h5, f_data_h5, f_post_h5=f_post_h5,
        N_use=N_use, id_use=id_use, nr=1000, T_N_above=50, T_P_acc_level=0.2,
        autoT=1, showInfo=1, updatePostStat=True)
else:
    print("Using existing posterior: %s" % f_post_h5)

ig.plot_T_EV(f_post_h5, pl='CHI2', hardcopy=hardcopy)

# %% [markdown]
# ### B6. Inspect the posterior
#
# Resistivity (harmonic mean, `im=1`) and mode lithology (`im=2`) along the
# west-to-east profile, then depth-slice maps.

# %%
ig.plot_profile(f_post_h5, im=1, ii=id_line, key='HarmonicMean', gap_threshold=100,
                xaxis='x', hardcopy=hardcopy)
ig.plot_profile(f_post_h5, im=2, ii=id_line, panels=['mode'], gap_threshold=100,
                xaxis='x', hardcopy=hardcopy)

# %%
for elevation in [40, 30, 20, 10]:
    ig.plot_feature_2d(f_post_h5, im=1, key='HarmonicMean', elevation=elevation,
                       uselog=1, s=2, hardcopy=hardcopy)
    plt.show()

for elevation in [40, 30, 20, 10]:
    ig.plot_feature_2d(f_post_h5, im=2, key='Mode', elevation=elevation,
                       s=2, hardcopy=hardcopy)
    plt.show()

# %% [markdown]
# ### B7. The `ig.query` tool
#
# We nee dto find WHERE we potentially have raw material worth producing!

# %%
# %%
#os.environ['ANTHROPIC_API_KEY']='sk-ant-XXXXX'
query_raw, interp_raw, prompt_raw = ig.query_from_text(
    'What is the probbability that the cumulative thickness af raw materials (any sand and gravel) is greater than 10 m within the top 30 m, and where the oberburden (the top layer of non-raw material) is no more than 3 meters thick?', 
    f_prior_h5=f_prior_h5, 
    api_key=os.environ.get('ANTHROPIC_API_KEY'))

# %%
query_raw = {'constraints': [{'im': 2,
   'classes': [2, 5, 6],
   'thickness_mode': 'cumulative',
   'thickness_comparison': '>',
   'thickness_threshold': 10.0,
   'depth_min': 0.0,
   'depth_max': 30.0,
   'negate': False},
  {'im': 2,
   'classes': [1, 3, 4, 7, 8],
   'thickness_mode': 'first_occurrence',
   'thickness_comparison': '<=',
   'thickness_threshold': 3.0,
   'depth_min': 0.0,
   'depth_max': 30.0,
   'negate': False}]}

P_raw, meta_raw = ig.query(f_post_h5, query_raw)

ig.query_plot(P_raw, meta_raw,
              query_text="P(raw material)",
              text_panel=True,
              hardcopy='daugaard_P_raw' if hardcopy else False)


# %% The same probability map as B7, with the target polygons and the
# west-to-east profile line overlaid for context.
fig, ax = plt.subplots(figsize=(9, 8))
sc = ax.scatter(X, Y, c=P_raw, s=6, cmap='hot_r', vmin=0, vmax=1)
for nm, geom in polygons.items():
    xs, ys = geom.exterior.xy
    ax.plot(xs, ys, 'k-', lw=1.5)
    ax.annotate(nm, (geom.centroid.x, geom.centroid.y), ha='center', fontsize=8)
ax.plot(Xl_wide, Yl_wide, 'r--', lw=1.2, label='W-E profile')
ax.plot(X[id_line], Y[id_line], 'r.', ms=3)
ax.set_aspect('equal')
ax.set_xlabel('UTM X (m)')
ax.set_ylabel('UTM Y (m)')
ax.set_title('P(raw material) with target areas + profile')
fig.colorbar(sc, ax=ax, label='P_raw')
ax.legend(fontsize=8)
if hardcopy:
    fig.savefig('daugaard_P_raw_context.png', dpi=200, bbox_inches='tight')
plt.show()


# %% [markdown]
# ### B8. Automatic region search: grow an area of potential raw material
#
# Delineate an area of potential raw material by letting a region grow itself
# outward from the most promising spot: start at the sounding with the highest
# raw-material probability `P_raw`, then repeatedly add the neighbouring
# sounding with the next-highest `P_raw` -- as long as it is still >= `P_MIN`.
# The region simply keeps spreading into adjacent high-probability ground and
# stops on its own when the surrounding soundings are no longer likely enough.
#
# `P_MIN` is the one knob: "include every connected sounding where the
# probability of raw material is at least this". P_MIN = 0.5 -> the region is
# the contiguous patch (around the most likely spot) where raw material is
# more likely than not. Lower it to grow the region, raise it to shrink it.
# `MAX_AREA_M2` is an optional hard cap (None = no cap).
#
# Neighbours come from a Voronoi tessellation of the sounding locations (two
# soundings are neighbours iff their Voronoi cells share an edge); each cell
# is clipped to a *concave* hull of the soundings so its area is meaningful.
# The region area is then the sum of its soundings' cell areas, and its
# raw-material volume is sum(cell_area * thickness_percentile) (see B9).
#
# Before growing, **edge-affected soundings are dropped** (`flag_edge_cells`):
# a sounding on the sparse rim of the survey has a huge, badly constrained
# Voronoi cell that would otherwise let the region balloon into empty ground.
# A sounding is dropped if its cell is unbounded (a convex-hull vertex), or if
# it is within `EDGE_BUFFER` of the survey outline AND its cell is oversized
# (`> CELL_AREA_K x` the interior-median cell) or very elongated (`> ELONG_MAX`).
# Large cells that are *not* on the rim (interior data gaps) are kept.

# %%
import heapq
from scipy.spatial import Voronoi, ConvexHull
from shapely import concave_hull, voronoi_polygons, distance
from shapely import points as sh_points
from shapely import MultiPoint, STRtree
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

# --- region-growing knobs ---
P_MIN       = 0.2      # keep growing while the next sounding has P_raw >= this
MAX_AREA_M2 = None     # optional hard cap on the region area [m^2]; None = no cap
useManual   = False    # True -> seed at (X_start, Y_start) instead of argmax(P_raw)
X_start, Y_start = 543000.0, 6175750.0

# --- edge-affected-cell filter (soundings on the sparse rim of the survey) ---
HULL_RATIO  = 0.10    # concave-hull tightness for the survey outline (0 = tight, 1 = convex)
EDGE_BUFFER = None    # m; a sounding this close to the outline is "near the edge".
                      # None -> auto = 2 * sqrt(interior-median cell area)
CELL_AREA_K = 6.0     # a cell is "oversized" if larger than K x the interior-median cell area
ELONG_MAX   = 4.0     # drop near-edge cells more elongated than this
                      # (perimeter^2 / (4 pi area); 1 = disc); None to skip this test

CELL_SIZE   = 5.0     # m; raster resolution for the polygon-clipped cell areas used in B9


def voronoi_graph(X, Y):
    """(vor, neighbors): scipy Voronoi object + neighbour-index lists (cells sharing an edge).

    Fails on exactly-duplicated coordinates -- deduplicate X, Y first if that happens.
    """
    vor = Voronoi(np.column_stack([X, Y]))
    nb = [set() for _ in range(len(X))]
    for a, b in vor.ridge_points:
        nb[a].add(int(b))
        nb[b].add(int(a))
    return vor, [sorted(s) for s in nb]


def voronoi_cells_ordered(X, Y, boundary):
    """Per-sounding Voronoi cell polygons, in input order, clipped to `boundary`."""
    XY = np.column_stack([X, Y])
    raw = list(voronoi_polygons(MultiPoint(XY), extend_to=boundary).geoms)
    tree = STRtree([Point(xy) for xy in XY])
    cells = [None] * len(XY)
    for cell in raw:
        inside = tree.query(cell, predicate="contains")
        idx = int(inside[0]) if len(inside) else int(tree.nearest(cell.centroid))
        cells[idx] = cell.intersection(boundary)
    return cells


def cells_to_polygon(cells, mask):
    """Union of the masked Voronoi cells -> a single shapely Polygon (largest part)."""
    poly = unary_union([c for i, c in enumerate(cells)
                        if c is not None and mask[i]]).buffer(0)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def flag_edge_cells(X, Y, vor, cells, boundary, edge_buffer=None, k=6.0, elong_max=None):
    """Boolean `good` mask -- False for edge-affected soundings on the sparse survey rim.

    A sounding is dropped if its Voronoi cell is unbounded (a convex-hull
    vertex), or if it lies within `edge_buffer` of the survey outline AND its
    cell is either much larger than the interior-median cell (> k x) or very
    elongated (perimeter^2 / (4 pi area) > elong_max). Large cells that are
    NOT near the rim (interior data gaps) are kept.

    Returns (good, edge_affected, info).
    """
    n = len(X)
    area = np.array([c.area if (c is not None and not c.is_empty) else 0.0 for c in cells])
    peri = np.array([c.length if (c is not None and not c.is_empty) else 0.0 for c in cells])

    # unbounded Voronoi cell -> the sounding is a convex-hull vertex -> definitely edge
    is_open = np.array([
        (len(vor.regions[vor.point_region[i]]) == 0) or (-1 in vor.regions[vor.point_region[i]])
        for i in range(n)])

    if edge_buffer is None:
        base = area[(~is_open) & (area > 0)]
        edge_buffer = 2.0 * np.sqrt(np.median(base)) if base.size else 0.0

    near_boundary = distance(sh_points(np.asarray(X), np.asarray(Y)), boundary.exterior) < edge_buffer

    interior = (~is_open) & (~near_boundary) & (area > 0)
    a_med = float(np.median(area[interior])) if interior.any() else float(np.median(area[area > 0]))
    oversized = area > k * a_med

    if elong_max is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            elong = peri ** 2 / (4.0 * np.pi * np.where(area > 0, area, np.nan))
        stretched = np.nan_to_num(elong, nan=0.0) > elong_max
    else:
        stretched = np.zeros(n, dtype=bool)

    edge_affected = is_open | (near_boundary & (oversized | stretched))
    info = dict(edge_buffer=float(edge_buffer), a_med=a_med,
                n_open=int(is_open.sum()), n_near=int(near_boundary.sum()),
                n_dropped=int(edge_affected.sum()))
    return ~edge_affected, edge_affected, info


def auto_grow_region(P, neighbors, cell_area, p_min=0.5, max_area_m2=None, seed=None):
    """Automatically grow a region outward from the highest-probability sounding.

    Starting at the seed, repeatedly add the adjacent sounding with the highest
    probability P, keeping only those with ``P >= p_min``. The region stops
    growing on its own when no adjacent sounding is likely enough, or when its
    area reaches ``max_area_m2`` (if given).

    Returns (mask, area, order): boolean mask over soundings, the accumulated
    Voronoi-cell area, and the order soundings were added.
    """
    P = np.where(np.isfinite(P), P, -np.inf)
    if seed is None:
        seed = int(np.argmax(P))
    in_region = np.zeros(len(P), dtype=bool)
    in_region[seed] = True
    area = float(cell_area[seed])
    order = [seed]
    seen = np.zeros(len(P), dtype=bool)
    seen[seed] = True
    frontier = []                       # max-heap on P via negative key
    for j in neighbors[seed]:
        heapq.heappush(frontier, (-P[j], j))
        seen[j] = True
    while frontier:
        if max_area_m2 is not None and area >= max_area_m2:
            break
        negp, j = heapq.heappop(frontier)
        if -negp < p_min:               # best remaining neighbour is below cutoff -> done
            break
        if in_region[j]:
            continue
        in_region[j] = True
        area += float(cell_area[j])
        order.append(j)
        for k in neighbors[j]:
            if not seen[k]:
                heapq.heappush(frontier, (-P[k], k))
                seen[k] = True
    return in_region, area, order


# --- Voronoi graph, concave survey outline, per-sounding cells + areas ---
vor, neighbors = voronoi_graph(X, Y)

XY = np.column_stack([X, Y])
survey_poly = concave_hull(MultiPoint(XY), ratio=HULL_RATIO)
if survey_poly.geom_type == "MultiPolygon":
    survey_poly = max(survey_poly.geoms, key=lambda g: g.area)
if survey_poly.is_empty or survey_poly.geom_type != "Polygon":
    survey_poly = Polygon(XY[ConvexHull(XY).vertices])            # fallback to convex hull

voro_cells = voronoi_cells_ordered(X, Y, survey_poly)
cell_area = np.array([c.area if (c is not None and not c.is_empty) else 0.0
                      for c in voro_cells])

# --- drop edge-affected soundings, then grow only on the good ones ---
good, edge_affected, edge_info = flag_edge_cells(
    X, Y, vor, voro_cells, survey_poly,
    edge_buffer=EDGE_BUFFER, k=CELL_AREA_K, elong_max=ELONG_MAX)
print("Edge filter: dropped %d / %d soundings (%d hull-open, edge_buffer=%.0f m, "
      "interior median cell = %.0f m^2)"
      % (edge_info['n_dropped'], len(X), edge_info['n_open'],
         edge_info['edge_buffer'], edge_info['a_med']))

P_raw_eff = np.where(good, P_raw, -np.inf)   # edge-affected soundings are unreachable

if useManual:
    i_start = int(np.argmin((X - X_start) ** 2 + (Y - Y_start) ** 2))
else:
    i_start = int(np.nanargmax(P_raw_eff))
print("Seed sounding %d at (%.0f, %.0f), P_raw=%.2f"
      % (i_start, X[i_start], Y[i_start], P_raw[i_start]))

region_mask, region_area, region_order = auto_grow_region(
    P_raw_eff, neighbors, cell_area, p_min=P_MIN, max_area_m2=MAX_AREA_M2, seed=i_start)
region_mask &= good                          # belt and suspenders
region_area = float(cell_area[region_mask].sum())
print("Grown region (P_raw >= %.2f): %d soundings, %.0f m^2"
      % (P_MIN, region_mask.sum(), region_area))

# %%
# Raw-material volume inside the grown region (P5/P50/P95).
pct_raw_region, _ = ig.query(f_post_h5, {
    "metric": {"im": 2, "classes": raw_classes,
               "thickness_mode": "cumulative", "depth_min": 0.0},
    "percentiles": [5, 50, 95],
})
vol_region = np.nansum(pct_raw_region[region_mask, :] * cell_area[region_mask, None], axis=0)
print("Raw-material volume in region  min/med/max = %s m^3"
      % np.round(vol_region).astype(int))

# %% Turn the auto-grown region into a polygon, in the SAME
# {name: shapely.Polygon} format as `polygons` (Mette's deterministic areas),
# so it can be plotted and fed to the volume helpers exactly the same way.
region_polygon = cells_to_polygon(voro_cells, region_mask)
polygons_prob = {'Probabilistic region': region_polygon}
print("Probabilistic region polygon: area = %.0f m^2  (%d soundings)"
      % (region_polygon.area, region_mask.sum()))

# %%
fig, ax = plt.subplots(figsize=(9, 8))
sc = ax.scatter(X[good], Y[good], c=P_raw[good], s=6, cmap='hot_r', vmin=0, vmax=1)
ax.scatter(X[edge_affected], Y[edge_affected], s=8, marker='x', color='0.6',
           linewidths=0.5, label='edge-affected (dropped, %d)' % edge_affected.sum())
ax.scatter(X[region_mask], Y[region_mask], s=.1, facecolors='none',
           edgecolors='cyan', linewidths=0.7,
           label='grown region (%.0f,000 m$^2$)' % (region_area / 1000))
ax.plot(*survey_poly.exterior.xy, color='0.5', lw=0.8, ls=':', label='survey outline')
for nm, geom in polygons.items():
    xs, ys = geom.exterior.xy
    ax.plot(xs, ys, 'k-', lw=1.5)
for nm, geom in polygons_prob.items():
    xs, ys = geom.exterior.xy
    ax.plot(xs, ys, color='cyan', lw=2, label=nm)
ax.set_aspect('equal')
ax.set_xlabel('UTM X (m)')
ax.set_ylabel('UTM Y (m)')
ax.set_title('Auto-grown raw-material region (from P_raw)')
fig.colorbar(sc, ax=ax, label='P_raw')
ax.legend(fontsize=8)
if hardcopy:
    fig.savefig('daugaard_rawmat_region.png', dpi=200, bbox_inches='tight')
plt.show()


# %% Map of the actual Voronoi cells (clipped to the concave survey outline).
# Good cells are coloured by P_raw; edge-affected cells are hatched grey; the
# auto-grown region's cells are outlined in cyan.
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon


def _cell_patches(cells, idx):
    """matplotlib polygon patches for cells[i] with i in idx (skips empty/multipart bits)."""
    patches, keep = [], []
    for i in idx:
        c = cells[i]
        if c is None or c.is_empty:
            continue
        parts = c.geoms if c.geom_type == "MultiPolygon" else [c]
        for p in parts:
            if p.geom_type == "Polygon" and not p.is_empty:
                patches.append(MplPolygon(np.asarray(p.exterior.coords)))
                keep.append(i)
    return patches, np.asarray(keep)


patches_good, keep_good = _cell_patches(voro_cells, np.where(good)[0])
patches_edge, _ = _cell_patches(voro_cells, np.where(edge_affected)[0])
patches_reg, _ = _cell_patches(voro_cells, np.where(region_mask)[0])

fig, ax = plt.subplots(figsize=(10, 9))
pc_good = PatchCollection(patches_good, cmap='hot_r', edgecolor='0.75', linewidths=0.15)
pc_good.set_array(P_raw[keep_good])
pc_good.set_clim(0, 1)
ax.add_collection(pc_good, autolim=True)

pc_edge = PatchCollection(patches_edge, facecolor='0.85', edgecolor='0.6',
                          linewidths=0.15, hatch='//')
ax.add_collection(pc_edge, autolim=True)

pc_reg = PatchCollection(patches_reg, facecolor='none', edgecolor='cyan', linewidths=0.6)
ax.add_collection(pc_reg, autolim=True)

rx, ry = region_polygon.exterior.xy
ax.plot(rx, ry, color='cyan', lw=2.5, label='probabilistic region')
for geom in polygons.values():
    ax.plot(*geom.exterior.xy, 'k-', lw=1.5)

ax.set_aspect('equal')
ax.autoscale_view()
ax.set_xlabel('UTM X (m)')
ax.set_ylabel('UTM Y (m)')
ax.set_title('Voronoi cells coloured by P_raw  (grey hatch = edge-affected, dropped)')
fig.colorbar(pc_good, ax=ax, label='P_raw')
ax.legend(fontsize=8)
if hardcopy:
    fig.savefig('daugaard_voronoi_cells.png', dpi=200, bbox_inches='tight')
plt.show()




# %% [markdown]
# ### B9. Low / median / high volumes: auto-grown region vs. Mette's polygons
#
# For each area and quantity the volume estimate is
#
#     V(p) = sum_i  thickness_i(p) * voronoi_cell_area_i
#          = (area-weighted mean thickness at percentile p) x area
#
# at p = 5 / 50 / 95 (low / median / high). Quantities:
#   * overburden   -- depth to the first raw-material layer (first_occurrence)
#   * raw material -- cumulative thickness of `raw_classes`
#   * coarser      -- cumulative thickness of `coarse_classes`
#
# Areas: the auto-grown `P_raw` region from B8, and each of Mette's target
# polygons (Section 1b). The per-polygon result, `prob_results`, feeds the
# comparison to Mette's original numbers in Part C.

# %%
PCT = [5, 50, 95]                                    # -> min / median / max


def _thickness_pct(classes, mode):
    """(N_sounding, 3): posterior P5/P50/P95 of a per-sounding thickness metric."""
    pct, _ = ig.query(f_post_h5, {
        "metric": {"im": 2, "classes": classes, "thickness_mode": mode, "depth_min": 0.0},
        "percentiles": PCT})
    return pct


pct_overburden = _thickness_pct(raw_classes, "first_occurrence")
pct_raw        = pct_raw_region                                     # already queried in B8
pct_coarse     = _thickness_pct(coarse_classes, "cumulative") if coarse_classes else None


def area_volumes(area_per_sounding):
    """{quantity: [V_p5, V_p50, V_p95]} + area, for a per-sounding area array (0 outside)."""
    a = np.asarray(area_per_sounding, dtype=float)
    m = a > 0

    def _v(pct_thick):
        if pct_thick is None:
            return np.full(len(PCT), np.nan)
        return np.nansum(pct_thick[m, :] * a[m, None], axis=0)      # [p5, p50, p95] m^3

    return {'overburden':  _v(pct_overburden),
            'raw_material': _v(pct_raw),
            'coarser':      _v(pct_coarse),
            'area_m2':      float(a[m].sum()),
            'n_points':     int(m.sum())}


def _print_vols(label, r):
    print("%-22s area=%9.0f m^2 (n=%d)" % (label, r['area_m2'], r['n_points']))
    for q in ('overburden', 'raw_material', 'coarser'):
        print("    %-12s p5/50/95 = %s m^3" % (q, np.round(r[q]).astype(int)))


# auto-grown probabilistic region (Voronoi-cell areas from B8)
region_results = area_volumes(np.where(region_mask, cell_area, 0.0))
_print_vols("Auto-grown region", region_results)

# Mette's polygons -> prob_results (used by Part C)
prob_results = {}
for name, polygon in polygons.items():
    prob_results[name] = area_volumes(
        rmu.compute_point_footprint_area(X, Y, polygon, cell_size=CELL_SIZE))
    _print_vols("Polygon %s" % name, prob_results[name])

# %%
# Raw-material volume: auto-grown region vs. each polygon (bar = P50, whiskers = P5-P95).
_names = ['Auto-grown\nregion'] + list(prob_results)
_V = np.vstack([region_results['raw_material']]
               + [prob_results[n]['raw_material'] for n in prob_results])
x = np.arange(len(_names))
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x, _V[:, 1], yerr=[_V[:, 1] - _V[:, 0], _V[:, 2] - _V[:, 1]],
       capsize=5, color=['C0'] + ['0.7'] * len(prob_results))
ax.set_xticks(x)
ax.set_xticklabels(_names)
ax.set_ylabel('Raw-material volume (m$^3$)')
ax.set_title('Raw-material volume  (bar = P50, whiskers = P5-P95)')
ax.grid(True, axis='y', ls='--', alpha=0.4)
if hardcopy:
    fig.savefig('daugaard_rawmat_volume_B9.png', dpi=200, bbox_inches='tight')
plt.show()


# %% [markdown]
# # Part C -- Comparison to Mette's original (deterministic) estimate
#
# The headline result: the probabilistic per-polygon volumes (`prob_results`,
# built in B9) next to the single-number estimates from the old
# sequential/deterministic assessment. Reference numbers are transcribed from
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
# The probabilistic estimates come with an explicit min/median/max range
# instead of a single number, which is the key practical addition of this
# workflow: a decision-maker can see not just a central raw-material volume
# estimate, but how confident that estimate is, sub-area by sub-area.
#
# If the probabilistic median raw-material volume is noticeably larger than
# the old deterministic estimate, recall the caveat from GEUS's notes on the
# old assessment (`ReferenceProjects/EmailFromMette.md`): the deterministic
# estimate was capped at the depth documented by raw-material boreholes, even
# where the tTEM inversion suggested raw material could extend deeper. The
# probabilistic model is not capped this way, so part of any gap reflects
# that methodological difference rather than a disagreement about the shallow
# geology. Whether deep/older units matched by `resolve_material_classes`
# (e.g. Miocene sand) should count as producible raw material at all is a
# geological judgement call that should be reviewed before treating this as a
# strict apples-to-apples comparison (see Part B's printed class list).
