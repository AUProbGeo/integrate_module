"""
Daugaard: probabilistic raw-material assessment with INTEGRATE
==============================================================

GEUS previously assessed raw-material (sand/gravel) potential at three
target sub-areas in Daugaard ("Delområde 1/2/3") using a **sequential,
deterministic** workflow: invert the tTEM data to a single best-fit
resistivity model, interpret lithology, and hand-estimate overburden and
raw-material volumes per sub-area (Mette's results, from
``Sdr Felding og Daugard til Integrate.pptx``).

This example redoes the assessment probabilistically, in a linear order:

1. Load the tTEM data and the boreholes, and show them on a map.
2. Set up two priors: a **generic** resistivity-only prior, and an
   **XLS** prior built with ``geoprior1d`` from ``daugaard_rostof.xlsx``
   (resistivity + lithology).
3. Compute prior data: tTEM data for both priors, and borehole data for
   the XLS prior (the generic prior has no lithology).
4. Check prior data against the observed data before inverting.
5. Invert: generic prior with tTEM, and XLS prior with tTEM only,
   boreholes only, and tTEM + boreholes. This gives ``f_post_h5_list``
   (4 posteriors) from ``f_prior_h5_list`` (2 priors).
6. Plot the posteriors, and compare the generic posterior with the
   deterministic WorkBench inversion.
7. Estimate the producible raw-material volume and compare it with
   Mette's results.

Every expensive step writes to a fixed file name. With ``reuse_h5 = True``
a step is skipped when its file already exists, so the script can be
re-run cheaply; with ``reuse_h5 = False`` all files are deleted and
recomputed.

Run this script from within its own directory, so that the case data and
the ``integrate_rawmaterial_utils`` helper module are found.
"""
# %%
try:
    get_ipython()
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except Exception:
    pass

# %%
# 0. Settings
# -----------
import os

# Trim JAX/XLA compile time on the first run. XLA_FLAGS is read when jax is
# first imported (via `import integrate`), so this must come first.
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=1 --xla_backend_optimization_level=1"

# Choose the forward model (the default is ga-aem)
#os.environ["EM_FORWARD_METHOD"] = "anemone"
#os.environ["EM_FORWARD_DEVICE"] = "cuda"
#os.environ["EM_FORWARD_DEVICE"] = "cpu"
#os.environ["EM_FORWARD_METHOD"] = "ga-aem"

# Rejection-sampling backend used by ig.integrate_rejection: 'jax' or 'numpy'
#os.environ["REJECTION_BACKEND"] = "numpy"
#os.environ["REJECTION_BACKEND"] = "jax"


import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import integrate as ig
import integrate_rawmaterial_utils as rmu

hardcopy = True

N = 2_000_000 +2
#N = 50_001
#N = 3_000_000 + 5        # number of prior realizations (use e.g. 10_000 for a quick demo)
inflateNoise = 2     # multiply the tTEM data std by this (0 = use as-is)
reuse_h5 = False      # True: reuse existing HDF5 files; False: delete them and recompute everything

# Appended to every generated file name, so a different N / inflateNoise
# writes to its own set of files.
SUFFIX = '_N%d_iN%d' % (N, inflateNoise)


def compute(f_h5):
    """Return True if f_h5 must be computed. If reuse_h5 is False, an existing f_h5 is deleted."""
    if os.path.exists(f_h5):
        if reuse_h5:
            print("Using existing %s" % f_h5)
            return False
        print("Deleting existing %s" % f_h5)
        os.remove(f_h5)
    return True

t_start = time.time()

# %%
# 1. Load the data
# ----------------
#
# tTEM data and GEX system file. A working copy of the data file is used,
# with the data std inflated by ``inflateNoise``. The borehole data are
# later added to this copy, so the downloaded file is never modified.

# %%
case = 'DAUGAARD'
f_data_h5 = ig.get_case_data(case=case, showInfo=1)[0]
file_gex = ig.get_gex_file_from_data(f_data_h5)
print("Using data file: %s" % f_data_h5)
print("Using GEX file: %s" % file_gex)

f_data_org_h5 = f_data_h5
f_data_h5 = 'DAUGAARD_AVG_gf%g%s.h5' % (inflateNoise, SUFFIX)
if compute(f_data_h5):
    ig.copy_hdf5_file(f_data_org_h5, f_data_h5)
    if inflateNoise != 0:
        D = ig.load_data(f_data_org_h5)
        ig.save_data_gaussian(D['d_obs'][0], D_std=D['d_std'][0] * inflateNoise,
                              f_data_h5=f_data_h5, file_gex=file_gex)

X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

ig.plot_data(f_data_h5, useLog=0, hardcopy=hardcopy)
ig.plot_data_xy(f_data_h5, data_channel=15, cmap='jet')
plt.show()

# %%
# 2. Load the boreholes
# ---------------------

# %%
f_boreholes = ig.get_case_data(case=case, filelist=['rawmaterial/Daugaard_boringer_rostof.json'])[0]
BHOLES = ig.read_borehole(f_boreholes, showInfo=1)

# Go through all the boreholes, and only retain those that 
# are within min_dis 0m of any tTEM sounding. This is to avoid including boreholes that are too far away from the tTEM survey area.
N_before = len(BHOLES)
min_dis = 50.0
BHOLES = [bh for bh in BHOLES if np.min(np.sqrt((X - bh['X'])**2 + (Y - bh['Y'])**2)) <= min_dis]
N_after = len(BHOLES)
print(f"Filtered boreholes: {N_before} -> {N_after} (kept those within {min_dis} m of any tTEM sounding)")

# %%
# 3. Map of data, boreholes, profile and target areas
# ---------------------------------------------------
#
# A west-to-east profile through a band of boreholes is used for all
# profile plots below (``id_line``). ``pl_data`` are two example soundings
# used for single-sounding plots. Mette's target polygons (Delområde 1/2/3)
# are matched to their names by polygon area.

# %%
BH_PROFILE_NAMES = ['Dau02', 'Dau04', '116. 1602', '116. 1867',
                    'Dau07', '116.  535', 'Dau05', 'Dau03']
bh_profile = sorted([bh for bh in BHOLES if bh['name'] in BH_PROFILE_NAMES],
                    key=lambda bh: bh['X'])   # west -> east
# Extend the line beyond the data at both ends so it covers the full W-E range
buffer = 10.0
Xl = np.concatenate(([X.min() - buffer], [bh['X'] for bh in bh_profile], [X.max() + buffer]))
Yl = np.concatenate(([bh_profile[0]['Y']], [bh['Y'] for bh in bh_profile], [bh_profile[-1]['Y']]))
id_line, _, _ = ig.find_points_along_line_segments(X, Y, Xl, Yl, tolerance=buffer)
print("Found %d soundings within %.0f m of the W-E profile" % (len(id_line), buffer))

pl_data = [id_line[10], id_line[100]]

DAUGAARD_AREA_ID_MAP = {
    308812.2: 'Area 1',
    106727.7: 'Area 2',
    196598.0: 'Area 3',
}
polygons = rmu.load_target_polygons('Fokusområder_polygon_daugaard.shp', area_id_map=DAUGAARD_AREA_ID_MAP)

# %%
fig, ax, sc = ig.plot_xy(ELEVATION, X=X, Y=Y, cmap='jet', colorbar_label='Elevation [m]',
                         title='Daugaard: data, boreholes, profile and target areas', s=3)
# sort X[id_line], Y[id_line] by X to get a continuous profile line
x_sorted_idx = np.argsort(X[id_line])
ax.plot(X[id_line][x_sorted_idx], Y[id_line][x_sorted_idx], 'k-', lw=3, ms=8, label='profile')
# Selected soundings (shown in detail later): large magenta stars drawn on top of everything
for j, i in enumerate(pl_data):
    ax.plot(X[i], Y[i], '*', ms=24, mfc='magenta', mec='k', mew=1.5, zorder=10,        
            label='selected soundings' if j == 0 else None)
    ax.annotate('d%d (%d)' % (j + 1, i), (X[i], Y[i]), xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold', zorder=11, rotation=30,
                rotation_mode='anchor', ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='w', ec='k', alpha=0.8))
ax.plot([bh['X'] for bh in BHOLES], [bh['Y'] for bh in BHOLES], 'ko', ms=12, mew=1.5, label='boreholes')
for bh in BHOLES:
    ax.annotate(bh['name'], (bh['X'], bh['Y']), xytext=(4, 4), textcoords='offset points',
                fontsize=7, color='r')
# Plot the name on TOP of the polygon, not at the centroid, to avoid overlapping with the polygon boundary
for name, polygon in polygons.items():
    ax.plot(*polygon.exterior.xy, 'k-', lw=1.5)
    ax.annotate(name, (polygon.centroid.x, polygon.bounds[3]), xytext=(0, 3),
                textcoords='offset points', color='k', fontsize=10, ha='center', va='bottom')
ax.legend(loc='best')
if hardcopy:
    fig.savefig('daugaard_data_map' + SUFFIX + '.png', dpi=200, bbox_inches='tight')
plt.show()

# %%
# 4. Two priors
# -------------
#
# * **Generic**: a plain layered-Earth resistivity model (no lithology).
# * **XLS**: built with ``geoprior1d`` (https://github.com/GEUSjesper/geoprior1d)
#   from the geological specification ``daugaard_rostof.xlsx``. It holds
#   resistivity (``/M1``) and lithology (``/M2``).

# %%
f_prior_generic_h5 = 'DAUGAARD_PRIOR_GENERIC%s.h5' % SUFFIX
if compute(f_prior_generic_h5):
    ig.prior_model_layered(N=N, lay_dist='uniform', 
                           NLAY_min=2, NLAY_max = 6,
                           RHO_min=5, RHO_max=1000,
                           f_prior_h5=f_prior_generic_h5, showInfo=1)

f_prior_xls_h5 = 'DAUGAARD_PRIOR_ROSTOF%s.h5' % SUFFIX
if compute(f_prior_xls_h5):
    from geoprior1d import geoprior1d
    f_xlsx = ig.get_case_data(case=case, filelist=['rawmaterial/daugaard_rostof.xlsx'])[0]

    t0 = time.time()
    geoprior1d(f_xlsx, Nreals=N,
               dz=1, dmax=90, output_file=f_prior_xls_h5)
    # geoprior1d is not part of INTEGRATE, so store its computation time on
    # the model arrays here (INTEGRATE's own generators do this themselves).
    with h5py.File(f_prior_xls_h5, 'r') as f:
        M_keys = ['/' + k for k in f.keys() if k[0] == 'M']
    ig.save_compute_time(f_prior_xls_h5, M_keys, time.time() - t0, 'geoprior1d', t0)

f_prior_h5_list = [f_prior_generic_h5, f_prior_xls_h5]

for f_prior_h5 in f_prior_h5_list:
    ig.plot_prior_stats(f_prior_h5, hardcopy=hardcopy)
ig.prior_describe(f_prior_xls_h5)

# %%
# 5. Prior data
# -------------
#
# tTEM prior data (``/D1``) are computed for both priors. For the XLS prior
# the boreholes are then added as extra data types (``/D2``, ``/D3``, ...),
# one per borehole, in a copy of the tTEM prior-data file. This also writes
# the borehole observations into ``f_data_h5``.

# %% Prior tTEM data
f_prior_data_generic_h5 = 'DAUGAARD_PRIOR_GENERIC_DATA%s.h5' % SUFFIX
f_prior_data_xls_h5 = 'DAUGAARD_PRIOR_ROSTOF_DATA%s.h5' % SUFFIX
f_prior_data_xls_bh_h5 = 'DAUGAARD_PRIOR_ROSTOF_DATA_BH%s.h5' % SUFFIX

for f_prior_h5, f_prior_data_h5 in [(f_prior_generic_h5, f_prior_data_generic_h5),
                                    (f_prior_xls_h5, f_prior_data_xls_h5)]:
    if compute(f_prior_data_h5):
        ig.prior_data_em(f_prior_h5, file_gex, doMakePriorCopy=True,
                         f_prior_data_h5=f_prior_data_h5)

# %% Prior borehole data
id_borehole_list = list(range(2, 2 + len(BHOLES)))   # /D index of each borehole
if compute(f_prior_data_xls_bh_h5):
    ig.copy_hdf5_file(f_prior_data_xls_h5, f_prior_data_xls_bh_h5)
    _, id_borehole_list = ig.save_borehole_data(
        f_prior_data_xls_bh_h5, f_data_h5, BHOLES,
        im_prior=2, range_xyz=100,
        range_data=1, range_data_i_use=[17, 18, 19, 20],
        doPlot=True, showInfo=0)

f_prior_data_h5_list = [f_prior_data_generic_h5, f_prior_data_xls_bh_h5]

# %%
# 6. Prior data vs. observed data
# -------------------------------
#
# Before inverting, check that the observed tTEM data lie within the spread
# of the prior tTEM data, for both priors. If they do not, the prior is
# inconsistent with the data and should be revised.

# %%
for f_prior_data_h5 in f_prior_data_h5_list:
    ig.plot_data_prior(f_prior_data_h5, f_data_h5, nr=1000, id=1, hardcopy=hardcopy)
    plt.show()

# %%
# The boreholes against the lithology classes of the XLS prior, and a map of
# where the borehole data are informative (low entropy = strong constraint).

# %%
ig.plot_boreholes(BHOLES, f_prior_xls_h5, fontsize=17, n_per_figure=8, hardcopy=hardcopy)
plt.show()

fig, ax, sc = ig.plot_discrete_data_entropy(f_data_h5, cmap='hot',
                                            plotPoints=True, plotPoints_size=1, plotPoints_color='k',
                                            plotPoints_alpha=0.1, hardcopy=False)
ax.plot([bh['X'] for bh in BHOLES], [bh['Y'] for bh in BHOLES], 'rx', ms=9, mew=1.5, label='boreholes')
ax.legend(loc='best')
if hardcopy:
    fig.savefig('daugaard_borehole_entropy' + SUFFIX + '.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# 7. Inversion
# ------------
#
# Four rejection-sampling inversions: the generic prior with tTEM data, and
# the XLS prior with tTEM only, boreholes only, and tTEM + boreholes.

# %%
inversions = [  # (label, prior data file, data types to use)
    ('GENERIC_EM', f_prior_data_generic_h5, [1]),
    ('XLS_EM',     f_prior_data_xls_bh_h5,  [1]),
    ('XLS_BH',     f_prior_data_xls_bh_h5,  id_borehole_list),
    ('XLS_EM_BH',  f_prior_data_xls_bh_h5,  [1] + id_borehole_list),
]


f_post_h5_list = []
t_inversion = {}
for label, f_prior_data_h5, id_use in inversions:
    f_post_h5 = 'DAUGAARD_POST_%s%s.h5' % (label, SUFFIX)
    if compute(f_post_h5):
        t0 = time.time()
        # Anneal the tTEM data (autoT=1) but not the borehole data (autoT=0, T=1),
        # e.g. autoT = [1, 0, 0] for id_use = [1, 2, 3].
        autoT = [1 if i == 1 else 0 for i in id_use]   # data id 1 is tTEM
        ig.integrate_rejection(f_prior_data_h5, f_data_h5, f_post_h5=f_post_h5,
                               N_use=N, id_use=id_use, nr=1000,
                               T_N_above=50,
                               T_P_acc_level=0.2,
                               autoT=autoT, showInfo=1, updatePostStat=True)
        t_inversion[label] = time.time() - t0
    f_post_h5_list.append(f_post_h5)

for label, t in t_inversion.items():
    print('Inversion %s: %.1f s' % (label, t))

f_post = dict(zip([inv[0] for inv in inversions], f_post_h5_list))

for f_post_h5 in f_post_h5_list:
    ig.plot_T_EV(f_post_h5, pl='CHI2', plotPoints=True, plotPoints_color='k', hardcopy=hardcopy)

# %%
# 7b. tTEM data fit
# -----------------
#
# Prior, posterior and observed tTEM data at the example soundings, for all
# four posteriors.

# %%
for i_plot in pl_data:
    for f_post_h5 in f_post_h5_list:
        ig.plot_data_prior_post(f_post_h5, i_plot=i_plot, id=1, hardcopy=hardcopy)

# %%
# 8. Posterior profiles
# ---------------------
#
# Resistivity (harmonic mean and std) along the W-E profile for all four
# posteriors. For the XLS posteriors the lithology (``im=2``) is also
# shown, and the boreholes are overlaid as lithology columns.

# %%
bhole_kw = dict(BHOLES=BHOLES, bhole_im=2, bhole_max_dist=100, bhole_width=10, bhole_label=True)
profile_kw = dict(ii=id_line, xaxis='x', gap_threshold=100, hardcopy=hardcopy)

for label, f_post_h5 in f_post.items():
    is_xls = label.startswith('XLS')
    kw = bhole_kw if is_xls else {}
    ig.plot_profile(f_post_h5, im=1, txt=label, **profile_kw, **kw)
    #
    ig.plot_profile(f_post_h5, im=1, panels=['harmonicmean', 'std'], txt=label, **profile_kw, **kw)
    # same, with the colors faded where the posterior std is large
    ig.plot_profile(f_post_h5, im=1, panels=['harmonicmean', 'std'], alpha=0.95,
                    logstd_min=0.1, logstd_max=1.0, txt=label + '_alpha', **profile_kw, **kw)
    if is_xls:
        ig.plot_profile(f_post_h5, im=2, txt=label, **profile_kw, **bhole_kw)
        # same, with the colors faded where the lithology entropy is large
        ig.plot_profile(f_post_h5, im=2, alpha=0.95, entropy_min=0.1, entropy_max=1.0,
                        txt=label + '_alpha', **profile_kw, **bhole_kw)

# %%
# Details for the joint (tTEM + borehole) posterior: posterior models and
# data at the two example soundings, and depth-slice maps.

# %%
f_post_h5 = f_post['XLS_EM_BH']
for i_plot in pl_data:
    ig.plot_post_stats(f_post_h5, i_plot=i_plot, im=1, hardcopy=hardcopy)
    ig.plot_post_stats(f_post_h5, i_plot=i_plot, im=2, hardcopy=hardcopy)
    ig.plot_data_prior_post(f_post_h5, i_plot=i_plot, hardcopy=hardcopy)

for elevation in [40, 30, 20, 10]:
    ig.plot_feature_2d(f_post_h5, im=1, key='HarmonicMean', elevation=elevation,
                       uselog=1, s=2, hardcopy=hardcopy)
    plt.show()
    ig.plot_feature_2d(f_post_h5, im=2, key='Mode', elevation=elevation,
                       s=2, hardcopy=hardcopy)
    plt.show()

# %%
# 8b. Generic posterior vs. deterministic WorkBench inversion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The WorkBench least-squares results (smooth and sharp) are put onto the
# depth grid of the generic posterior (in a copy of its file) and plotted
# along the same profile, for comparison with the generic posterior above.

# %%
import libaarhusxyz

ig.get_case_data(case=case, loadType='WB_smooth')
ig.get_case_data(case=case, loadType='WB_sharp')

f_xyz_list = {
    'smooth': 'SCI7_40_ml_Daugaard_I01_MOD_inv.xyz',
    'sharp':  'SCI7_40_ml_sharp2_I02_MOD_inv.xyz',
}

with h5py.File(f_prior_data_generic_h5, 'r') as f:
    z = f['/M1'].attrs['x'][:].flatten()

for label, f_xyz in f_xyz_list.items():
    f_lsq_h5 = os.path.splitext(f_xyz)[0] + '.h5'
    ig.copy_hdf5_file(f_post['GENERIC_EM'], f_lsq_h5)

    with h5py.File(f_lsq_h5, 'r') as f:
        M1_median = f['/M1/Median'][:]

    model = libaarhusxyz.XYZ(f_xyz)
    rho = model.layer_data['rho'].values
    rho_std = model.layer_data['rho_std'].values
    dep_top = model.layer_data['dep_top'].values
    utmx = model.flightlines['utmx'].values
    utmy = model.flightlines['utmy'].values

    M1_mean = np.full_like(M1_median, np.nan)
    M1_stdf = np.full_like(M1_median, np.nan)
    for i in range(len(utmx)):
        idx = np.argmin((X - utmx[i]) ** 2 + (Y - utmy[i]) ** 2)
        j_idx = np.searchsorted(dep_top[i], z, side='right') - 1
        valid = j_idx >= 0
        M1_mean[idx, valid] = rho[i, j_idx[valid]]
        M1_stdf[idx, valid] = rho_std[i, j_idx[valid]]

    # WorkBench RHO_STD is a multiplicative std factor (STDF):
    # std(log10 rho) = log10(RHO_STD), which is what the std panel plots.
    M1_logstd = np.log10(M1_stdf)
    M1_std = M1_mean * np.log(M1_stdf)   # approx. linear-space std [ohm-m]

    # Overwrite every statistic plot_profile may read. The LSQ result is a
    # single model, so mean = median = harmonic mean = the model itself.
    with h5py.File(f_lsq_h5, 'a') as f:
        for key in ['/M1/Mean', '/M1/LogMean', '/M1/Median', '/M1/HarmonicMean',
                    '/M1/Std', '/M1/LogStd']:
            if key in f:
                del f[key]
        f['/M1/Mean'] = M1_mean
        f['/M1/Median'] = M1_mean
        f['/M1/HarmonicMean'] = M1_mean
        f['/M1/LogMean'] = np.log10(M1_mean)
        f['/M1/Std'] = M1_std
        f['/M1/LogStd'] = M1_logstd

    ig.plot_profile(f_lsq_h5, im=1, panels=['mean', 'std'], txt='WB_%s' % label, **profile_kw)

# %%
# 9. Raw-material volume
# ----------------------
#
# Where is there raw material worth producing? The query below asks for the
# probability that the cumulative raw-material (sand/gravel) thickness in
# the top 30 m exceeds 10 m, AND the overburden (the top layer of non-raw
# material) is at most 3 m thick. If ``ANTHROPIC_API_KEY`` is set, the same
# query is built from the text with ``ig.query_from_text``.

# %%
query_text = ('What is the probability that the cumulative thickness of raw materials '
              '(any sand and gravel) is greater than 10 m within the top 30 m, and where '
              'the overburden (the top layer of non-raw material) is no more than 3 meters thick?')
if 'ANTHROPIC_API_KEY' in os.environ:
    query_raw, _, _ = ig.query_from_text(query_text, f_prior_h5=f_prior_xls_h5,
                                         api_key=os.environ.get('ANTHROPIC_API_KEY'))
else:
    query_raw = {'constraints': [
        {'im': 2, 'classes': [5], 'thickness_mode': 'cumulative',
         'thickness_comparison': '>', 'thickness_threshold': 10.0,
         'depth_min': 0.0, 'depth_max': 30.0, 'negate': False},
        {'im': 2, 'classes': [1, 2, 3, 4, 6, 7, 8], 'thickness_mode': 'first_occurrence',
         'thickness_comparison': '<=', 'thickness_threshold': 3.0,
         'depth_min': 0.0, 'depth_max': 30.0, 'negate': False}]}

raw_classes = query_raw['constraints'][0]['classes']    # sand + gravel: raw material
fine_classes = query_raw['constraints'][1]['classes']   # everything else: overburden

# %%
# For each XLS posterior: the raw-material probability map ``P_raw``, and the
# P5/P50/P95 overburden and raw-material volume inside each of Mette's
# polygons (sum of thickness percentile x area of each sounding's footprint
# inside the polygon).

# %%
PCT = [5, 50, 95]   # -> low / median / high
CELL_SIZE = 5.0     # m; raster resolution for the per-sounding footprint area
query_overburden = {"metric": {"im": 2, "classes": fine_classes, "thickness_mode": "cumulative",
                               "depth_min": 0.0}, "percentiles": PCT}
query_raw_material = {"metric": {"im": 2, "classes": raw_classes, "thickness_mode": "cumulative",
                                 "depth_min": 0.0}, "percentiles": PCT}

polygon_area = {}
for name, polygon in polygons.items():
    footprint_area = rmu.compute_point_footprint_area(X, Y, polygon, cell_size=CELL_SIZE)
    polygon_area[name] = {'mask': footprint_area > 0, 'cell_area': footprint_area}

xls_labels = ['XLS_EM', 'XLS_BH', 'XLS_EM_BH']
P_raw = {}
pct_overburden = {}
pct_raw_material = {}
prob_results = {}   # prob_results[label][polygon name] = {'overburden': V, 'raw_material': V}
for label in xls_labels:
    P_raw[label], meta_raw = ig.query(f_post[label], query_raw)
    ig.query_plot(P_raw[label], meta_raw, query_text="P(raw material), %s" % label, text_panel=True,
                  hardcopy='daugaard_P_raw_%s%s' % (label, SUFFIX) if hardcopy else False)
    pct_overburden[label], _ = ig.query(f_post[label], query_overburden)
    pct_raw_material[label], _ = ig.query(f_post[label], query_raw_material)

    prob_results[label] = {}
    for name, area in polygon_area.items():
        prob_results[label][name] = {
            'overburden':   ig.region_volumes(pct_overburden[label], area),
            'raw_material': ig.region_volumes(pct_raw_material[label], area)}
        print("%-10s %-12s raw material PCT%s = %s m^3" % (
            label, name, PCT, np.round(prob_results[label][name]['raw_material']).astype(int)))

# %%
# 9b. Data-driven raw-material areas (region growing)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Instead of using Mette's hand-drawn polygons, let an area grow from a seed
# point into neighbouring soundings (Voronoi neighbours) as long as
# ``P_raw >= P_MIN``. This is done separately for each XLS posterior, so each
# case (EM, BH, EM+BH) gets its own areas, from its own ``P_raw`` map. Soundings
# on the sparse rim of the survey (huge, badly constrained Voronoi cells)
# are dropped first. The volume of each grown area is computed as above, from
# the same posterior. If ``P_raw`` at a seed is below ``P_MIN``, the area is
# just the seed's own cell.

# %%
POS_CENTER = [(543039.3, 6175796.0),
              (543930.0, 6175738.0),
              (544810.0, 6175500.0)]
P_MIN = 0.05          # keep growing while the next sounding has P_raw >= this
#P_MIN = 0.1          # keep growing while the next sounding has P_raw >= this
MAX_AREA_M2 = None   # optional hard cap on the area [m^2]
AREA_NAMES = ['Area %d' % i for i in range(len(POS_CENTER))]

AREA_LIST = {}            # AREA_LIST[label] = grown areas, one per seed in POS_CENTER
V_area_raw_material = {}  # V_area_raw_material[label] = P5/P50/P95 volume, one per area
for label in xls_labels:
    AREA_LIST[label] = []
    V_area_raw_material[label] = []
    for name, (X_center, Y_center) in zip(AREA_NAMES, POS_CENTER):
        AREA = ig.find_coherent_area(X, Y, P_raw[label], P_MIN, max_area_m2=MAX_AREA_M2,
                                     X_center=X_center, Y_center=Y_center,
                                     hull_ratio=0.10, edge_buffer=None,
                                     cell_area_k=6.0, elong_max=4.0)
        V_raw = ig.region_volumes(pct_raw_material[label], AREA)
        AREA_LIST[label].append(AREA)
        V_area_raw_material[label].append(V_raw)
        print("%-10s %-8s area = %.0f m^2, raw material PCT%s = %s m^3" % (
            label, name, AREA['area'], PCT, np.round(V_raw).astype(int)))

    # Voronoi cells coloured by P_raw (grey hatch = edge-affected, dropped)
    # with Mette's polygons (blue). Saved twice: BEFORE the grown areas are
    # drawn (P_raw map only; does not depend on P_MIN), and AFTER, with the
    # grown areas (black/dark grey outline, open circle = seed).
    fig, ax, mappable = ig.plot_voronoi_cells(AREA_LIST[label][0], P=P_raw[label], vmin=0, vmax=1)
    for name, polygon in polygons.items():
        ax.plot(*polygon.exterior.xy, 'b-', lw=1, label='Mette: %s' % name)
    ax.set_xlabel('UTM X (m)')
    ax.set_ylabel('UTM Y (m)')
    ax.set_title("P_raw (%s) and Mette's polygons" % label)
    fig.colorbar(mappable, ax=ax, label='P_raw')
    ax.legend(fontsize=8, loc='best')
    if hardcopy:   # BEFORE
        fig.savefig('daugaard_voronoi_P_raw_vs_mette_%s%s.png' % (label, SUFFIX), dpi=200, bbox_inches='tight')

    for i, (name, AREA) in enumerate(zip(AREA_NAMES, AREA_LIST[label])):
        shade = str(0.35 * i / max(len(AREA_LIST[label]) - 1, 1))   # 0 = black ... 0.35 = dark grey
        ax.plot(*AREA['polygon'].exterior.xy, color=shade, lw=3, label=name)
        ax.plot(AREA['X_center'], AREA['Y_center'], 'o', ms=12, mfc='none', mec=shade, mew=2.5)   # open circle
    ax.set_title("P_raw (%s): grown areas (P_MIN = %g) vs. Mette's polygons" % (label, P_MIN))
    ax.legend(fontsize=8, loc='best')
    if hardcopy:   # AFTER
        fig.savefig('daugaard_voronoi_areas_vs_mette_%s%s_Pmin%g.png' % (label, SUFFIX, P_MIN), dpi=200, bbox_inches='tight')
    plt.show()

# %%
# 10. Comparison with Mette's results
# -----------------------------------
#
# Mette's deterministic single-number estimates per target polygon, against
# the probabilistic P5/P50/P95 estimates from the three XLS posteriors.

# %%
DAUGAARD_REFERENCE = {   # keys must match the polygon names in DAUGAARD_AREA_ID_MAP
    # 'coarser' (gravel only) is not compared: this workflow only separates
    # raw material (sand+gravel) from overburden.
    'Area 1': {'overburden': 865_000, 'raw_material': 4_500_000, 'coarser': 1_700_000},
    'Area 2': {'overburden': 566_000, 'raw_material': 1_700_000, 'coarser': 1_000_000},
    'Area 3': {'overburden': 452_000, 'raw_material': 3_300_000, 'coarser': 2_200_000},
}

for label in xls_labels:
    print("=== %s ===" % label)
    rmu.compare_to_reference(prob_results[label], DAUGAARD_REFERENCE,
                             quantities=('overburden', 'raw_material'), hardcopy=hardcopy,
                             f_name='daugaard_rawmaterial_comparison_%s%s' % (label, SUFFIX))

# %%
# Raw-material volume per target polygon: one bar per XLS posterior
# (bar = P50, whiskers = P5-P95) next to Mette's estimate, and on the right
# the grown areas from 9b, one hatched bar per XLS posterior (each posterior's
# own area).

# %%
names = list(polygons)
width = 0.2
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(names))
for k, label in enumerate(xls_labels):
    V = np.array([prob_results[label][n]['raw_material'] for n in names])
    ax.bar(x + (k - 1.5) * width, V[:, 1], width, yerr=[V[:, 1] - V[:, 0], V[:, 2] - V[:, 1]],
           capsize=4, color='C%d' % k, label=label)
ax.bar(x + 1.5 * width, [DAUGAARD_REFERENCE[n]['raw_material'] for n in names], width,
       color='0.5', label='Mette (deterministic)')
xa = len(names) + np.arange(len(AREA_NAMES))
for k, label in enumerate(xls_labels):
    V = np.vstack(V_area_raw_material[label])
    ax.bar(xa + (k - 1) * width, V[:, 1], width, yerr=[V[:, 1] - V[:, 0], V[:, 2] - V[:, 1]],
           capsize=4, color='C%d' % k, hatch='//', label='grown areas (%s)' % label)
ax.set_xticks(np.concatenate((x, xa)))
ax.set_xticklabels(names + AREA_NAMES, rotation=30, ha='right')
ax.set_ylabel('Raw-material volume (m$^3$)')
ax.set_title('Raw-material volume  (bar = P50, whiskers = P5-P95)')
ax.grid(True, axis='y', ls='--', alpha=0.4)
ax.legend(fontsize=8)
if hardcopy:
    fig.savefig('daugaard_rawmat_volume%s_Pmin%g.png' % (SUFFIX, P_MIN), dpi=200, bbox_inches='tight')
plt.show()

# %%
# 11. 1D -> 3D posterior realizations
# -----------------------------------
#
# The posterior realizations in ``/i_use`` are independent from sounding to
# sounding. ``integrate_jointsim`` (joint sequential simulation, in development
# in ~/PROGRAMMING/PROJECT_IDEAS/INTEGRATE_POST_3D/code) picks, for each of
# ``N_SIM`` 3D realizations, one of each sounding's posterior realizations, such
# that neighbouring soundings are horizontally correlated. The indices are
# written to ``/i_use_3d``, shape (Nsoundings, N_SIM), and the posterior
# statistics are then computed from ``/i_use_3d`` instead of ``/i_use``.

# %%
import sys
sys.path.append(os.path.expanduser('~/PROGRAMMING/PROJECT_IDEAS/INTEGRATE_POST_3D/code'))
from jointsim import integrate_jointsim

N_SIM = 50   # number of 3D realizations
i_use_key='i_use_3d'

f_post_h5_org = f_post['XLS_EM_BH']
f_post_h5_3D = f_post_h5_org.replace('.h5', 'N%d_3D.h5' % (N_SIM))
if compute(f_post_h5_3D):
    # 11a. Copy the joint (tTEM + borehole) posterior file
    ig.copy_hdf5_file(f_post_h5_org, f_post_h5_3D)
    # 11b. Simulate N_SIM 3D realizations -> f_post_h5_3D:/i_use_3d
    integrate_jointsim(f_post_h5_3D, n_sim=N_SIM, range_h=200.0, i_use_key=i_use_key)
    # 11c. Posterior statistics from /i_use_3d instead of /i_use
    ig.integrate_posterior_stats(f_post_h5_3D, i_use_key=i_use_key, showInfo=1)

# %%
# 11d. Posterior statistics from the 1D and the 3D realizations, side by side.

# %% Plot profiles of a) PRIOR relaization, b) 1D POSTERIOR realization, c) 3D POSTERIOR realization



# %%
for label, f_post_h5 in [('1D', f_post_h5_org), ('3D', f_post_h5_3D)]:
    ig.plot_profile(f_post_h5, im=1, panels=['harmonicmean', 'std'], txt=label, **profile_kw, **bhole_kw)
    ig.plot_profile(f_post_h5, im=2, txt=label, **profile_kw, **bhole_kw)
    ig.plot_feature_2d(f_post_h5, im=2, key='Mode', elevation=20, s=2, hardcopy=hardcopy)
    plt.show()

# %%
# 11e. Actual realizations of the lithology. ``/i_use_3d`` and ``/i_use`` have
# the same format, (Nsoundings, Nrealizations), and are plotted the same way:
# column ``i_real`` is realization ``i_real``. ``/i_use_3d`` holds the 3D
# realizations, ``/i_use`` the 1D realizations (no horizontal correlation).

# %%
for i_real in range(5):
    for txt, key in [('_3D', i_use_key), ('_1D', 'i_use')]:
        real_kw = dict(i_use_key=key, i_realization=i_real, txt=txt)   # txt is added to the PNG names
        ig.plot_profile(f_post_h5_3D, im=2, panels=['realization'], **real_kw, **profile_kw, **bhole_kw)
        ig.plot_feature_2d(f_post_h5_3D, im=2, key='realization', elevation=40, s=2, hardcopy=hardcopy, **real_kw)
        plt.show()

# %%
# 11f. Resistivity along the profile: a) prior realization, b) 1D posterior
# realization, c) 3D posterior realization, d) 1D posterior harmonic mean,
# e) 3D posterior harmonic mean.

# %% REALS/POST M1
for i_real in range(5):   # realization number is in the PNG name, e.g. _a_prior_real2
    ig.plot_profile(f_post_h5_3D, im=1, panels=['realization'], plot_prior=True, seed=i_real, txt='_a_prior_real%d' % i_real, **profile_kw)
    ig.plot_profile(f_post_h5_3D, im=1, panels=['realization'], i_use_key='i_use', i_realization=i_real, txt='_b_post_real_1D_real%d' % i_real, **profile_kw)
    ig.plot_profile(f_post_h5_3D, im=1, panels=['realization'], i_use_key=i_use_key, i_realization=i_real, txt='_c_post_real_3D_real%d' % i_real, **profile_kw)
    plt.show()
ig.plot_profile(f_post_h5_org, im=1, panels=['harmonicmean'], txt='_d_post_hmean_1D', **profile_kw)
ig.plot_profile(f_post_h5_3D, im=1, panels=['harmonicmean'], txt='_e_post_hmean_3D', **profile_kw)
plt.show()


# %% REALS/POST M2
for i_real in range(5):   # realization number is in the PNG name, e.g. _a_prior_real2
    ig.plot_profile(f_post_h5_3D, im=2, panels=['realization'], plot_prior=True, seed=i_real, txt='_a_prior_real%d' % i_real, **profile_kw)
    ig.plot_profile(f_post_h5_3D, im=2, panels=['realization'], i_use_key='i_use', i_realization=i_real, txt='_b_post_real_1D_real%d' % i_real, **profile_kw)
    ig.plot_profile(f_post_h5_3D, im=2, panels=['realization'], i_use_key=i_use_key, i_realization=i_real, txt='_c_post_real_3D_real%d' % i_real, **profile_kw)
    plt.show()
ig.plot_profile(f_post_h5_org, im=2, panels=['Mode'], txt='_d_post_mode_1D', **profile_kw)
ig.plot_profile(f_post_h5_3D, im=2, panels=['Mode'], txt='_e_post_mode_3D', **profile_kw)
plt.show()

# PROFILE OF PROB(LITHHOLOGY=RAW) ALONG THE PROFILE
# P(class 5 = coarse sand/gravel/stone) along the W-E profile, for the three
# XLS posteriors (panel 'prob' of ig.plot_profile; prob_class=[4, 5] would sum classes).

# %% plot prob(raw) for different types of used data
for label in xls_labels:
    ig.plot_profile(f_post[label], im=2, panels=['prob', 'entropy'], prob_class=5,
                    txt=label, **profile_kw, **bhole_kw)


#%%  Same for the tTEM + borehole posterior from the 1D and the 3D realizations
for label, f_post_h5 in [('1D', f_post_h5_org), ('3D', f_post_h5_3D)]:
    ig.plot_profile(f_post_h5, im=2, panels=['prob', 'entropy'], prob_class=5,
                    txt='_' + label, **profile_kw, **bhole_kw)



# %%
# 12. Raw-material volume from the 3D realizations
# ------------------------------------------------
#
# For each of the ``N_SIM`` 3D realizations and each grown area from 9b
# (XLS_EM_BH): start in the seed sounding at the centre of its thickest
# raw-material run, and grow the connected 3D body of raw-material voxels
# (vertical neighbours, and Voronoi neighbours at the same depth), limited to
# the grown area. Its volume is one sample of the raw-material volume, so the
# N_SIM volumes give the volume distribution directly. Seed without raw
# material -> volume 0.

# %%
label_3d = 'XLS_EM_BH'   # the posterior that f_post_h5_3D is made from
V_3d, bodies_3d = rmu.connected_volume_3d(f_post_h5_3D, AREA_LIST[label_3d], raw_classes=raw_classes,
                                          i_use_key=i_use_key, return_bodies=True)   # V_3d: (n_area, N_SIM) [m^3]
METTE_NAMES = ['Area %d' % (i + 1) for i in range(len(POS_CENTER))]   # seed i lies in Mette's 'Area i+1'

# %%
# Compare per area: Mette (single number), 1D probabilistic in Mette's polygon,
# 1D probabilistic in the grown area, and the 3D realizations in the grown area
# (P5/P50/P95; the 3D volumes are also shown one dot per realization).

# %%
print("\nRaw-material volume [1000 m^3], P5 / P50 / P95 (%s, P_MIN = %g)" % (label_3d, P_MIN))
print("%-8s %10s %22s %22s %22s" % ('area', 'Mette', '1D Mette polygon', '1D grown area', '3D grown area'))
for ia, name in enumerate(METTE_NAMES):
    V_1d_pol = prob_results[label_3d][name]['raw_material'] / 1000
    V_1d_area = V_area_raw_material[label_3d][ia] / 1000
    V_3d_pct = np.percentile(V_3d[ia], PCT) / 1000
    print("%-8s %10.0f %22s %22s %22s" % (name, DAUGAARD_REFERENCE[name]['raw_material'] / 1000,
          '%.0f / %.0f / %.0f' % tuple(V_1d_pol), '%.0f / %.0f / %.0f' % tuple(V_1d_area),
          '%.0f / %.0f / %.0f' % tuple(V_3d_pct)))

fig, axs = plt.subplots(1, len(METTE_NAMES), figsize=(13, 4.5))
methods = ['1D\nMette polygon', '1D\ngrown area', '3D\ngrown area']
for ia, (ax, name) in enumerate(zip(axs, METTE_NAMES)):
    V_all = [prob_results[label_3d][name]['raw_material'],
             V_area_raw_material[label_3d][ia],
             np.percentile(V_3d[ia], PCT)]
    for k, V in enumerate(V_all):
        ax.errorbar(k, V[1], yerr=[[V[1] - V[0]], [V[2] - V[1]]], fmt='o', ms=8, capsize=6,
                    color='C%d' % k, lw=2)
    ax.plot(2 + 0.15 * np.random.default_rng(ia).uniform(-1, 1, N_SIM), V_3d[ia], '.',
            color='C2', alpha=0.4, label='3D realizations')
    ax.axhline(DAUGAARD_REFERENCE[name]['raw_material'], color='k', ls='--', label='Mette (deterministic)')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.set_title(name)
    ax.grid(True, axis='y', ls='--', alpha=0.4)
    ax.legend(fontsize=8)
axs[0].set_ylabel('Raw-material volume (m$^3$)')
fig.suptitle('Raw-material volume (%s): dot = P50, whiskers = P5-P95' % label_3d)
fig.tight_layout()
if hardcopy:
    fig.savefig('daugaard_rawmat_volume_3D%s_Pmin%g.png' % (SUFFIX, P_MIN), dpi=200, bbox_inches='tight')
plt.show()

# %%
# Total raw-material volume over all areas, per method: bars stacked by area
# (P50), whiskers = P5-P95 of the total. For 3D the total is summed per
# realization, so its P5-P95 is the true range of the total. For 1D the P5
# (P95) values are simply added, as if all soundings were at P5 (P95) at once,
# so that range is too wide.

# %%
V_tot = {'Mette': [DAUGAARD_REFERENCE[n]['raw_material'] * np.ones(3) for n in METTE_NAMES],
         '1D\nMette polygon': [prob_results[label_3d][n]['raw_material'] for n in METTE_NAMES],
         '1D\ngrown area': [V_area_raw_material[label_3d][ia] for ia in range(len(METTE_NAMES))],
         '3D\ngrown area': [np.percentile(V_3d[ia], PCT) for ia in range(len(METTE_NAMES))]}
fig, ax = plt.subplots(figsize=(8, 5))
for k, (method, V) in enumerate(V_tot.items()):
    V = np.vstack(V)   # (n_area, 3) = P5/P50/P95 per area
    bottom = 0
    for ia, name in enumerate(METTE_NAMES):
        ax.bar(k, V[ia, 1], bottom=bottom, color='C%d' % ia, edgecolor='k', label=name if k == 0 else None)
        bottom += V[ia, 1]
    lo, hi = V[:, 0].sum(), V[:, 2].sum()
    if method.startswith('3D'):
        lo, hi = np.percentile(V_3d.sum(axis=0), [PCT[0], PCT[2]])   # total per realization
    if method != 'Mette':
        ax.errorbar(k, bottom, yerr=[[bottom - lo], [hi - bottom]], color='k', capsize=8, lw=2)
    ax.text(k, max(hi, bottom), '%.1f M' % (bottom / 1e6), ha='center', va='bottom')
    print("%-20s total raw material P50 = %.0f m^3, P5-P95 = %.0f - %.0f m^3" % (
        method.replace('\n', ' '), bottom, lo, hi))
ax.set_xticks(range(len(V_tot)))
ax.set_xticklabels(list(V_tot))
ax.set_ylabel('Total raw-material volume (m$^3$)')
ax.set_title('Total raw material, Areas 1-3 (%s)' % label_3d)
ax.grid(True, axis='y', ls='--', alpha=0.4)
ax.legend()
if hardcopy:
    fig.savefig('daugaard_rawmat_volume_total%s_Pmin%g.png' % (SUFFIX, P_MIN), dpi=200, bbox_inches='tight')
plt.show()

# %%
# 13. Movie of the 3D raw-material bodies
# ---------------------------------------
#
# One frame per 3D realization: the connected raw-material bodies of the three
# areas (from section 12), with the grown-area outlines drawn at the surface
# elevation. Fixed
# camera, so the frames differ only by the bodies. The frames are written to
# ``movie_3d_bodies/frame_###.png`` and assembled into an MP4 afterwards
# (e.g. in FilmCraft, 0.5 s per frame).

# %%
import pyvista as pv
from matplotlib.colors import ListedColormap, to_hex
from scipy.spatial import cKDTree

# Terrain-following Voronoi prisms (1 m layers) for the soundings of the three
# areas, down to the deepest body voxel; the colour comes from 'area_id' below.
ii_body = np.unique(np.concatenate([b['idx'] for b in bodies_3d]))
z_max = 1 + max(np.where(b['body'].any(axis=(0, 1)))[0].max() for b in bodies_3d)
columns = ig.post_to_voronoi_columns(f_post_h5_3D, ii=ii_body, im=2, key='Mode', z_max=z_max, r_max=25)
columns['values'] = np.zeros_like(columns['values'])   # keep all cells
grid = ig.voronoi_columns_to_grid(columns)

tree = cKDTree(np.column_stack((X, Y)))
colors = [to_hex('C%d' % k) for k in range(len(METTE_NAMES))]   # same colours as the figures above

os.makedirs('movie_3d_bodies', exist_ok=True)
pl = pv.Plotter(off_screen=True, window_size=(1920, 1080))
pl.set_background('white')
pl.enable_depth_peeling()
pl.set_scale(zscale=10)
for k, area in enumerate(AREA_LIST[label_3d]):   # grown-area outlines, on the surface
    xy = np.array(area['polygon'].exterior.coords)
    pl.add_mesh(pv.lines_from_points(np.column_stack((xy, ELEVATION[tree.query(xy)[1]] + 1.0))),
                color=colors[k], line_width=4)
pl.add_legend([(n, c) for n, c in zip(METTE_NAMES, colors)], bcolor='white', size=(0.1, 0.12), loc='upper right')
pl.add_text('Daugaard: connected raw-material bodies (class 5), %s, 3D realizations (vertical exaggeration 10x)'
            % label_3d, position='upper_left', font_size=14, color='k')
pl.view_vector((0.15, -1, 0.9), viewup=(0, 0, 1))   # from the south, obliquely down
pl.reset_camera(bounds=grid.bounds)   # frame the three areas (same in all frames)
pl.camera.zoom(1.3)
cam = pl.camera_position

for r in range(N_SIM):
    # area_id per cell: 1/2/3 inside the body of Area 1/2/3 in realization r, else 0
    nz_grid = grid.cell_data['iz'].max() + 1
    lookup = np.zeros((len(X), nz_grid), dtype=np.int8)
    for k, b in enumerate(bodies_3d):
        L = lookup[b['idx']]
        L[b['body'][r][:, :nz_grid]] = k + 1
        lookup[b['idx']] = L
    grid.cell_data['area_id'] = lookup[grid.cell_data['sounding'], grid.cell_data['iz']]
    body = grid.extract_cells(grid.cell_data['area_id'] > 0).extract_surface(algorithm='dataset_surface')
    pl.add_mesh(body, scalars='area_id', cmap=ListedColormap(colors), clim=[0.5, len(colors) + 0.5],
                show_scalar_bar=False, name='bodies')
    pl.add_text('Realization %2d / %d\n' % (r + 1, N_SIM) + '\n'.join(
        '%s: %.2f million m³' % (n, V_3d[k, r] / 1e6) for k, n in enumerate(METTE_NAMES)),
        position='lower_left', font_size=16, color='k', name='info')
    pl.camera_position = cam
    pl.screenshot('movie_3d_bodies/frame_%03d.png' % r)
pl.close()
print("Wrote %d frames to movie_3d_bodies/" % N_SIM)

# %%
# Discussion
# ^^^^^^^^^^
#
# The probabilistic estimates come with a P5-P95 range instead of a single
# number, so a decision-maker can see how confident the raw-material volume
# estimate is, sub-area by sub-area, and how much the boreholes and the tTEM
# data each contribute (BH vs. EM vs. EM+BH).
#
# Mette's deterministic estimate was capped at the depth documented by the
# raw-material boreholes, even where the tTEM inversion suggested raw
# material could extend deeper. The probabilistic estimate is not capped
# this way, so a larger estimate is expected -- a methodological difference,
# not an error in either approach. Whether deep units in ``raw_classes``
# (e.g. Miocene sand) count as producible raw material is a geological
# judgement that should be reviewed (see the class list from
# ``ig.prior_describe`` in step 4).

# %%
# Computation times: prior models and prior data (stored as attributes on each
# /M and /D array), and the inversions (stored as ``inv_time`` in each posterior).
for f_prior_data_h5 in f_prior_data_h5_list:
    ig.get_prior_timing(f_prior_data_h5)
for label, f_post_h5 in f_post.items():
    with h5py.File(f_post_h5, 'r') as f:
        print("Inversion %-10s: %.1f s" % (label, f.attrs.get('inv_time', np.nan)))
print("Inversion times [s] (this run): %s" % {k: round(v, 1) for k, v in t_inversion.items()})
print("Total running time: %.1f s" % (time.time() - t_start))
