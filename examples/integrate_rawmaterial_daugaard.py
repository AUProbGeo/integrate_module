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
# `ig.query` tool, grow a data-driven raw-material region from the posterior
# probability map (B8), and turn it into a low / median / high volume
# estimate (B9), built from per-sounding thickness percentiles times the
# Voronoi-cell area around each sounding. Part B is purely about the
# probabilistic result and ends with a bar chart of the grown region's
# raw-material volume.
#
# **Part C -- Comparison to Mette's original estimate.**
# Everything involving Mette's hand-drawn target polygons lives here: their
# own low/median/high volumes, and the region-vs-polygon comparison next to
# the single-number estimates from the old sequential/deterministic
# assessment.
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
# Every generated file name is prefixed with `N<N>_` (and the working data
# file also encodes `inflateNoise`), so a different `N` writes to a separate
# set of files / figures. If you change `inflateNoise`, delete the affected
# `N*_DAUGAARD_*.h5` files (or bump `SUFFIX`) so they are regenerated.

# %%
# --- run-size settings -------------------------------------------------
N = 1_000_000   # production-scale
N = 100_000      # demo-scale; increase for a production-quality run
#N = 12_000      # demo-scale; increase for a production-quality run
# Prior size used everywhere: the generic prior (Part A) and each of the two
# geological-scenario priors merged into the informed prior (Part B, N // 2
# realizations each).

inflateNoise = 2        # multiply the tTEM data std by this (0 = use as-is)

SUFFIX = '_N%d_iN%d' % (N, inflateNoise)             # optional tag appended to every generated file name

# Prefix prepended to every generated file name (h5 + figures) so that a
# given choice of N writes to its own set of files, e.g. 'N12000_iN2_'.
PREFIX = '' 

# --- fixed output file names -----------------------------------------
# working data file (noise-inflated copy; == downloaded file if inflateNoise == 0)
f_data_work_h5 = '%sDAUGAARD_AVG_gf%g%s.h5' % (PREFIX, inflateNoise, SUFFIX)

# Part A -- resistivity-only, generic prior
f_prior_generic_h5      = '%sDAUGAARD_PRIOR_GENERIC%s.h5'       % (PREFIX, SUFFIX)
f_prior_generic_data_h5 = '%sDAUGAARD_PRIOR_GENERIC_DATA%s.h5'  % (PREFIX, SUFFIX)
f_post_generic_h5       = '%sDAUGAARD_POSTERIOR_GENERIC%s.h5'   % (PREFIX, SUFFIX)

# Part B -- informed prior + boreholes
f_prior_scenario_h5_map = {
    'daugaard_standard': '%sDAUGAARD_PRIOR_STANDARD%s.h5' % (PREFIX, SUFFIX),
    'daugaard_valley':   '%sDAUGAARD_PRIOR_VALLEY%s.h5'   % (PREFIX, SUFFIX),
}
f_prior_merged_h5  = '%sDAUGAARD_PRIOR_MERGED%s.h5'          % (PREFIX, SUFFIX)
f_prior_data_h5    = '%sDAUGAARD_PRIOR_MERGED_DATA%s.h5'     % (PREFIX, SUFFIX)
f_prior_data_bh_h5 = '%sDAUGAARD_PRIOR_MERGED_DATA_BH%s.h5'  % (PREFIX, SUFFIX)
f_post_h5          = '%sDAUGAARD_POSTERIOR%s.h5'             % (PREFIX, SUFFIX)

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

# %%
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
                              profile_xy=(Xl_wide, Yl_wide), profile_idx=id_line,
                              suffix=SUFFIX)


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

# %%
ig.plot_discrete_data_entropy(f_data_h5, id_list=list(range(2, len(BHOLES))))

# %%
# Entropy map over ALL multinomial (borehole) datasets — no id_list needed
fig, ax, sc = ig.plot_discrete_data_entropy(f_data_h5, cmap = 'gray', plotPoints=True)

# Overlay the borehole collar locations from the BHOLES list
bx = [bh['X'] for bh in BHOLES]
by = [bh['Y'] for bh in BHOLES]
ax.plot(bx, by, 'rx', ms=9, mew=1.5, label='boreholes')
for bh in BHOLES:
    ax.annotate(bh['name'], (bh['X'], bh['Y']),                
                xytext=(4, 4), textcoords='offset points', fontsize=7)
ax.legend(loc='best')

fig.savefig('DAUGAARD_entropy_with_boreholes.png', dpi=150, bbox_inches='tight')

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
# We need to find WHERE we potentially have raw material worth producing!

# %%
# %%
useLLM = False
if 'ANTHROPIC_API_KEY' in os.environ:
    useLLM = True
if useLLM:
    #os.environ['ANTHROPIC_API_KEY']='sk-ant-XXXXX'
    query_raw, interp_raw, prompt_raw = ig.query_from_text(
        'What is the probbability that the cumulative thickness af raw materials (any sand and gravel) is greater than 10 m within the top 30 m, and where the oberburden (the top layer of non-raw material) is no more than 3 meters thick?', 
        f_prior_h5=f_prior_h5, 
        api_key=os.environ.get('ANTHROPIC_API_KEY'))
else:
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


# %%
# perform the que
raw_classes = query_raw['constraints'][0]['classes']    # sand + gravel: coarse raw material
fine_classes = query_raw['constraints'][1]['classes']    # everything else: fine, non-raw material (overburden)

P_raw, meta_raw = ig.query(f_post_h5, query_raw)

ig.query_plot(P_raw, meta_raw,
              query_text="P(raw material)",
              text_panel=True,
              hardcopy=PREFIX + 'daugaard_P_raw' + SUFFIX if hardcopy else False)



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
# `P_MIN` is the main knob (the polygon "width"/area): "include every
# connected sounding where the probability of raw material is at least
# this". P_MIN = 0.5 -> the region is the contiguous patch (around the seed)
# where raw material is more likely than not. Lower it to grow the region,
# raise it to shrink it. `MAX_AREA_M2` is an optional hard cap (None = no
# cap).
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
#
# **Organization.** This block first defines the reusable region-analysis
# functions (below), then B9 runs the clean end-to-end pipeline (query ->
# pick point(s) -> grow -> percentile -> volume).
#
# Two entry points:
#   * `find_coherent_area(X, Y, P, p_min, ...)` -- the full pipeline in one
#     call: builds the Voronoi adjacency + per-sounding cells, applies the
#     edge filter, grows a connected region from a seed, and returns the
#     region indices, its boundary polygon, AND the Voronoi scaffold (graph /
#     cells / areas / outline / keep-mask) for reuse. These now live in the
#     `integrate` package (integrate.integrate_query), called as `ig.<name>`.
#   * `grow_connected_region(P, neighbors, cell_area, p_min, ...)` -- the
#     pure graph-grow step on a pre-built scaffold (no Voronoi/shapely).
#     `find_coherent_area` uses it internally.

# %%
# The region-search functions (voronoi_graph, voronoi_cells_ordered,
# cells_to_polygon, flag_edge_cells, grow_connected_region, find_coherent_area)
# are part of the `integrate` package -- call them as ig.<name>(...).
# See the B8 markdown above for what each does.


POS_CENTER = []
POS_CENTER.append((543039.3,6175596.0))   # west end of the profile
POS_CENTER.append((544500.0,6175800.0))   # west end of the profile

AREA_LIST = []

for icenter in range(len(POS_CENTER)):

    # --- central nodel
    X_center = POS_CENTER[icenter][0]
    Y_center = POS_CENTER[icenter][1]

    # --- region-growing knobs (the polygon "width"/area is set by these two) ---
    P_MIN       = 0.2      # keep growing while the next sounding has P_raw >= this
    MAX_AREA_M2 = None     # optional hard cap on the region area [m^2]; None = no cap

    # --- edge-affected-cell filter (soundings on the sparse rim of the survey) ---
    HULL_RATIO  = 0.10    # concave-hull tightness for the survey outline (0 = tight, 1 = convex)
    EDGE_BUFFER = None    # m; a sounding this close to the outline is "near the edge".
                        # None -> auto = 2 * sqrt(interior-median cell area)
    CELL_AREA_K = 6.0     # a cell is "oversized" if larger than K x the interior-median cell area
    ELONG_MAX   = 4.0     # drop near-edge cells more elongated than this
                        # (perimeter^2 / (4 pi area); 1 = disc); None to skip this test

    CELL_SIZE   = 5.0     # m; raster resolution for the polygon-clipped cell areas used in B9


    # Grow the region at the single highest-P_raw sounding. `AREA` (idx / mask /
    # polygon / vor / neighbors / cells / cell_area / boundary / good / ...) is
    # used directly everywhere below, including in B9.
    AREA = ig.find_coherent_area(X, Y, P_raw, P_MIN, max_area_m2=MAX_AREA_M2,
                                X_center = X_center, Y_center = Y_center,
                                hull_ratio=HULL_RATIO, edge_buffer=EDGE_BUFFER,
                                cell_area_k=CELL_AREA_K, elong_max=ELONG_MAX)

    edge_affected = ~AREA['good']
    P_raw_eff = np.where(AREA['good'], P_raw, -np.inf)   # edge-affected soundings are unreachable
    print("Edge filter dropped %d / %d soundings; default region seed %d at (%.0f, %.0f), P_raw=%.2f"
        % (int(edge_affected.sum()), len(X), AREA['seed'],
            X[AREA['seed']], Y[AREA['seed']], P_raw[AREA['seed']]))

    plt.figure(figsize=(8, 8))
    plt.plot(X, Y,'.', markersize=.1, color='lightgray', label='all soundings')
    plt.scatter(X, Y, c=P_raw, s=2, cmap='hot_r', vmin=0, vmax=1, label='kept soundings')
    #plt.plot(X[AREA['idx']], Y[AREA['idx']], 'k.', markersize=1, label='region')
    plt.plot(*AREA['polygon'].exterior.xy, 'k-', lw=2, label='polygon')
    plt.plot(AREA['X_center'], AREA['Y_center'], 'ko', ms=22, label='CENTER')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('P_raw with edge-affected soundings dropped')
    plt.axis('equal')
    plt.show()

    AREA_LIST.append(AREA)  

# %% [markdown]
# ### B9. Volume: query -> percentile -> volume
#
# One `ig.query` percentile query per quantity -- raw material (`raw_classes`:
# sand+gravel) and overburden (`fine_classes`: every other, non-raw class) --
# run once over ALL soundings, exactly like the percentile-query examples in
# `integrate_query.py`. Volume at each percentile = sum(percentile_thickness *
# cell_area) over the grown region's soundings (`AREA['mask']`, from B8), via
# `ig.region_volumes(pct, area)`.

# %%
# ---- A. one percentile query per quantity (run ONCE, shared by every AREA) -
PCT = [5, 50, 95]      # -> low / median / high

query_overburden   = {"im": 2, "classes": fine_classes, "thickness_mode": "cumulative", "depth_min": 0.0}
query_raw_material = {"im": 2, "classes": raw_classes,  "thickness_mode": "cumulative", "depth_min": 0.0}

pct_overburden, _   = ig.query(f_post_h5, {"metric": query_overburden, "percentiles": PCT})
pct_raw_material, _ = ig.query(f_post_h5, {"metric": query_raw_material, "percentiles": PCT})
# each: (N_sounding, len(PCT)) posterior thickness percentiles [m], one row per sounding

# ---- B. volume per grown area (one entry per AREA in AREA_LIST) -----------
AREA_NAMES         = ['Area %d' % i for i in range(len(AREA_LIST))]
V_overburden_list   = []
V_raw_material_list  = []

for iarea, AREA in enumerate(AREA_LIST):

    V_overburden   = ig.region_volumes(pct_overburden, AREA)
    V_raw_material = ig.region_volumes(pct_raw_material, AREA)
    V_overburden_list.append(V_overburden)
    V_raw_material_list.append(V_raw_material)
    print("%-8s overburden   PCT%s = %s m^3" % (AREA_NAMES[iarea], PCT, np.round(V_overburden).astype(int)))
    print("%-8s raw material PCT%s = %s m^3" % (AREA_NAMES[iarea], PCT, np.round(V_raw_material).astype(int)))

    # Voronoi cells coloured by P_raw; edge-affected cells hatched grey; this
    # area's grown region outlined.
    fig, ax, mappable = ig.plot_voronoi_cells(AREA, P=P_raw, vmin=0, vmax=1)
    ax.plot(*AREA['polygon'].exterior.xy, color='k', lw=2.5, label='grown region')
    ax.plot(AREA['X_center'], AREA['Y_center'], 'ko', ms=25, label='center location')
    ax.set_xlabel('UTM X (m)')
    ax.set_ylabel('UTM Y (m)')
    ax.set_title('%s -- Voronoi cells coloured by P_raw  (grey hatch = edge-affected, dropped)' % AREA_NAMES[iarea])
    fig.colorbar(mappable, ax=ax, label='P_raw')
    ax.legend(fontsize=8)
    if hardcopy:
        fig.savefig('%sdaugaard_voronoi_cells_area%d%s.png' % (PREFIX, iarea, SUFFIX), dpi=200, bbox_inches='tight')
    plt.show()

# ---- C. raw-material volume, one bar per grown area ----------------------------
# Purely probabilistic -- comparison to Mette's polygons is in Part C.
Vg = np.vstack(V_raw_material_list)
xg = np.arange(len(AREA_LIST))
fig, ax = plt.subplots(figsize=(3 + 1.5 * len(AREA_LIST), 5))
ax.bar(xg, Vg[:, 1], yerr=[Vg[:, 1] - Vg[:, 0], Vg[:, 2] - Vg[:, 1]], capsize=5, color='C0')
ax.set_xticks(xg)
ax.set_xticklabels(AREA_NAMES)
ax.set_ylabel('Raw-material volume (m$^3$)')
ax.set_title('Raw-material volume per grown area  (bar = P50, whiskers = P5-P95)')
ax.grid(True, axis='y', ls='--', alpha=0.4)
if hardcopy:
    fig.savefig(PREFIX + 'daugaard_rawmat_volume_B' + SUFFIX + '.png', dpi=200, bbox_inches='tight')
plt.show()


# %% [markdown]
# # Part C -- Comparison to Mette's original (deterministic) estimate
#
# Everything that involves Mette's hand-drawn target polygons lives here --
# Part B is about the probabilistic result only. Per-polygon low / median /
# high volumes reuse the same `ig.region_volumes(pct, area)` helper and the
# `pct_*` arrays queried in B9, on an AREA-like dict built from the
# per-sounding area inside each of Mette's polygons (rasterized at
# `CELL_SIZE` resolution via `rmu.compute_point_footprint_area`). Reference
# numbers are transcribed from
# `ReferenceProjects/Sdr Felding og Daugard til Integrate.pptx`
# (cross-checked against the shapefile polygon areas).
#
# The grown region is no longer a single area: every area in `AREA_LIST`
# (grown from its own centre in B8) is carried through the comparison, so
# C2's bar chart shows one bar per grown area alongside one per Mette
# polygon, and C3 overlays both sets on the Voronoi tessellation.

# %%
# C1. Low / median / high volumes for each of Mette's target polygons.
prob_results = {}
for name, polygon in polygons.items():
    footprint_area = rmu.compute_point_footprint_area(X, Y, polygon, cell_size=CELL_SIZE)
    poly_area = {'mask': footprint_area > 0, 'cell_area': footprint_area}
    prob_results[name] = {'overburden':   ig.region_volumes(pct_overburden, poly_area),
                          'raw_material': ig.region_volumes(pct_raw_material, poly_area)}
    print("%-15s overburden   PCT%s = %s m^3" % (name, PCT, np.round(prob_results[name]['overburden']).astype(int)))
    print("%-15s raw material PCT%s = %s m^3" % (name, PCT, np.round(prob_results[name]['raw_material']).astype(int)))

# %%
# C2. Raw-material volume, every grown area vs. Mette's polygons (bar = P50,
# whiskers = P5-P95), then the side-by-side tables from `rmu.compare_to_reference`.
names = AREA_NAMES + list(prob_results)
V = np.vstack(V_raw_material_list + [prob_results[n]['raw_material'] for n in prob_results])
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(5 + 1.2 * len(names), 5))
ax.bar(x, V[:, 1], yerr=[V[:, 1] - V[:, 0], V[:, 2] - V[:, 1]],
       capsize=5, color=['C0'] * len(AREA_NAMES) + ['0.7'] * len(prob_results))
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=30, ha='right')
ax.set_ylabel('Raw-material volume (m$^3$)')
ax.set_title('Raw-material volume  (bar = P50, whiskers = P5-P95)')
ax.grid(True, axis='y', ls='--', alpha=0.4)
if hardcopy:
    fig.savefig(PREFIX + 'daugaard_rawmat_volume_C' + SUFFIX + '.png', dpi=200, bbox_inches='tight')
plt.show()

# %%
# C3. Voronoi overview: cells coloured by P_raw, with Mette's target polygons
# (shades of blue, thin solid lines) and the grown areas (black->light-grey,
# thicker solid lines) drawn on top. The Voronoi scaffold (cells / edge filter
# / survey outline) is identical for every AREA, so AREA_LIST[0] is the backdrop.
fig, ax, mappable = ig.plot_voronoi_cells(AREA_LIST[0], P=P_raw, vmin=0, vmax=1)

def _grey_shades(n, lo=0.0, hi=0.75):
    """n greyscale levels from black (`lo`) to light grey (`hi`)."""
    return [str(lo + (hi - lo) * (i / max(n - 1, 1))) for i in range(n)]

area_greys = _grey_shades(len(AREA_LIST))
for i, AREA in enumerate(AREA_LIST):
    shade = area_greys[i]
    ax.plot(*AREA['polygon'].exterior.xy, color=shade, lw=3.0, zorder=3,
            label=AREA_NAMES[i])
    ax.plot(AREA['X_center'], AREA['Y_center'], marker='o', color=shade,
            ms=12, mec='k', zorder=5)

mette_blues = [plt.cm.Blues(0.45 + 0.5 * (i / max(len(polygons) - 1, 1)))
               for i in range(len(polygons))]
for i, (name, polygon) in enumerate(polygons.items()):
    ax.plot(*polygon.exterior.xy, color=mette_blues[i], lw=1.0, zorder=4,
            label="Mette: %s" % name)

ax.set_xlabel('UTM X (m)')
ax.set_ylabel('UTM Y (m)')
ax.set_title("Voronoi cells (P_raw) -- Mette's polygons vs. the grown areas")
fig.colorbar(mappable, ax=ax, label='P_raw')
ax.legend(fontsize=8, loc='best')
if hardcopy:
    fig.savefig(PREFIX + 'daugaard_voronoi_areas_vs_mette' + SUFFIX + '.png', dpi=200, bbox_inches='tight')
plt.show()

DAUGAARD_REFERENCE = {
    # 'coarser' (gravel-only) is kept for documentation but not compared below --
    # this workflow only distinguishes raw material (sand+gravel) vs. overburden
    # (everything else), not a further gravel-only split.
    'Delområde 1': {'overburden': 865_000,   'raw_material': 4_500_000, 'coarser': 1_700_000},
    'Delområde 2': {'overburden': 566_000,   'raw_material': 1_700_000, 'coarser': 1_000_000},
    'Delområde 3': {'overburden': 452_000,   'raw_material': 3_300_000, 'coarser': 2_200_000},
}

rmu.compare_to_reference(prob_results, DAUGAARD_REFERENCE, quantities=('overburden', 'raw_material'),
                         hardcopy=hardcopy, f_name=PREFIX + 'daugaard_rawmaterial_comparison' + SUFFIX)

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
# geology. Whether deep/older units in `raw_classes` (e.g. Miocene sand)
# should count as producible raw material at all is a
# geological judgement call that should be reviewed before treating this as a
# strict apples-to-apples comparison (see Part B's printed class list).

# %%
