"""
Shared helpers for the raw-material volume/uncertainty analysis in
``integrate_rawmaterial_daugaard.py`` and ``integrate_rawmaterial_sdrfelding.py``.

These two example scripts each run the full INTEGRATE probabilistic workflow
for one GEUS raw-material exploration site, and compare the resulting
area-integrated raw-material volumes (with uncertainty) to the single-number
estimates from the old, deterministic/sequential workflow (see
``examples/ReferenceProjects/``). The non-trivial, site-agnostic pieces of
that comparison — reading the target-area shapefiles, resolving lithology
class IDs to "raw material" / "coarser material" by name, turning point-wise
tTEM sounding statistics into an area-integrated volume with uncertainty, and
building the final comparison table/plot — live here so they are not
duplicated between the two scripts.

Requires geopandas and shapely (``pip install geopandas shapely``, or
``pip install .[examples]`` from the repository root), in addition to the
core INTEGRATE dependencies.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import contains_xy
from scipy.spatial import cKDTree

import integrate as ig


# ----------------------------------------------------------------------
# 1. Target-area polygons
# ----------------------------------------------------------------------

def load_target_polygons(shp_path, name_field=None, area_id_map=None):
    """
    Read a target-area shapefile and return {name: shapely polygon}.

    Two ways of naming the sub-areas are supported, because the two
    reference shapefiles are set up differently:

    - ``name_field``: read the sub-area name directly from an attribute
      column (used for the Sdr Felding shapefile, which has a populated
      ``'Delområde'`` field).
    - ``area_id_map``: a ``{area_m2: name}`` dict; each polygon is matched
      to the reference name whose area is closest (used for the Daugaard
      shapefile, whose attribute table is almost entirely empty — the only
      reliable way to tell the 3 Delområder apart is by polygon area, which
      matches the areas already reported in the old-approach PowerPoint).

    Parameters
    ----------
    shp_path : str
    name_field : str or None
    area_id_map : dict or None

    Returns
    -------
    polygons : dict {name: shapely.Polygon}
    """
    gdf = gpd.read_file(shp_path)
    print("Loaded %s : CRS=%s, %d feature(s)" % (shp_path, gdf.crs, len(gdf)))

    polygons = {}
    if name_field is not None:
        for _, row in gdf.iterrows():
            polygons[str(row[name_field])] = row.geometry
    elif area_id_map is not None:
        ref_areas = np.array(list(area_id_map.keys()), dtype=float)
        for geom in gdf.geometry.values:
            a = geom.area
            j = int(np.argmin(np.abs(ref_areas - a)))
            name = area_id_map[ref_areas[j]]
            polygons[name] = geom
    else:
        raise ValueError("Provide either name_field or area_id_map")

    for name, geom in polygons.items():
        print("  %-20s area = %10.1f m^2" % (name, geom.area))
    return polygons


def plot_polygons_over_points(X, Y, polygons, title='', hardcopy=False,
                               profile_xy=None, profile_idx=None):
    """
    Sanity-check plot: survey sounding locations plus target-area outlines,
    on the same axes. Used right after loading the data and the shapefile to
    visually confirm that both are in the same coordinate frame (both the
    tTEM data and the reference shapefiles are UTM32N, but this is worth
    checking rather than assuming).

    Optionally overlays a selected profile line: the line itself (through
    `profile_xy`, e.g. polygon centroids) and every sounding location that
    was matched onto it (`profile_idx`, e.g. from
    `ig.find_points_along_line_segments`), so the profile actually used for
    later resistivity/lithology sections can be checked visually against
    the target-area polygons in the same figure.

    Parameters
    ----------
    X, Y : ndarray (N,)
        Survey point coordinates.
    polygons : dict {name: shapely.Polygon}
    title : str
    hardcopy : bool
    profile_xy : tuple (Xl, Yl) or None
        Waypoint coordinates the profile line passes through (e.g. the
        sorted polygon centroids), drawn as a connected line.
    profile_idx : array-like of int or None
        Indices into `X`/`Y` of the sounding locations found along the
        profile (e.g. the `indices` returned by
        `ig.find_points_along_line_segments`), highlighted as dots.
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    # Plotted with a tiny markersize (there can be tens of thousands of
    # soundings), but given its own larger legend handle below so it stays
    # legible -- a blanket legend markerscale would also inflate the
    # profile/line markers added further down.
    ax.plot(X, Y, '.', markersize=1, color='0.6')
    soundings_handle = plt.Line2D([], [], marker='.', linestyle='', markersize=8,
                                   color='0.6', label='tTEM soundings')
    handles = [soundings_handle]

    colors = plt.cm.tab10.colors
    for i, (name, geom) in enumerate(polygons.items()):
        xs, ys = geom.exterior.xy
        line, = ax.plot(xs, ys, '-', color=colors[i % len(colors)], linewidth=2, label=name)
        handles.append(line)
    if profile_idx is not None:
        h, = ax.plot(np.asarray(X)[profile_idx], np.asarray(Y)[profile_idx], '.',
                      markersize=6, color='red', zorder=3, label='Profile soundings')
        handles.append(h)
    if profile_xy is not None:
        Xl, Yl = profile_xy
        h, = ax.plot(Xl, Yl, 'k*--', markersize=10, linewidth=1.5, zorder=4, label='Profile line')
        handles.append(h)
    ax.set_xlabel('UTM X (m)')
    ax.set_ylabel('UTM Y (m)')
    ax.set_aspect('equal')
    ax.set_title(title or 'Survey points vs. target-area polygons')
    ax.legend(handles=handles, fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    if hardcopy:
        fig.savefig('%s_polygon_alignment.png' % (title.replace(' ', '_') or 'check'), dpi=200)
    plt.show()


# ----------------------------------------------------------------------
# 2. Lithology class resolution (by name, not hard-coded ID)
# ----------------------------------------------------------------------

def resolve_material_classes(f_prior_h5, im=2,
                              raw_keywords=('sand', 'grus', 'gravel'),
                              coarse_keywords=('grus', 'gravel')):
    """
    Resolve which lithology class IDs count as "raw material" (sand+gravel,
    Danish "råstof") and "coarser material" (gravel only, Danish "grovere
    aflejringer" / stable gravel) for the discrete lithology model `im` in a
    prior HDF5 file, by matching class *names* (case-insensitive, whole-word
    match against the space-separated tokens of the class name), not
    hard-coded class IDs.

    This matters because different sites' priors are not guaranteed to use
    the same class ordering, IDs, or even the same number of classes (e.g.
    the real Daugaard "detailed" prior has 9 classes including "Sandy Till"
    and "Miocene sand" alongside "Meltwater sand"/"Meltwater gravel", not
    just the 8 simple classes in the example geoprior1d Excel sheets).
    Whole-word matching (rather than plain substring matching) is used
    deliberately: a substring match on "sand" would incorrectly tag "Sandy
    Till" (a glacial diamicton, not an exploitable resource) as raw
    material, since "sand" is a substring of "sandy".

    The printed classification is a mechanical first pass, not a geological
    judgement call — always read the printed class list and confirm it
    matches the intended definition of "raw material" before trusting any
    downstream volume estimate (in particular, whether older/deeper units
    like Miocene sand should count as producible raw material at all is a
    genuine geological question, not something this keyword match can
    decide).

    Returns
    -------
    raw_ids : list of int
    coarse_ids : list of int
    """
    info = ig.get_prior_model_info(f_prior_h5, im=im)
    class_id = np.asarray(info['class_id']).astype(int).ravel()
    class_name = np.asarray(info['class_name']).ravel()

    raw_ids, coarse_ids = [], []
    print("Lithology classes in %s (M%d):" % (f_prior_h5, im))
    for cid, name in zip(class_id, class_name):
        tokens = str(name).lower().replace('-', ' ').split()
        is_raw = any(k in tokens for k in raw_keywords)
        is_coarse = any(k in tokens for k in coarse_keywords)
        if is_raw:
            raw_ids.append(int(cid))
        if is_coarse:
            coarse_ids.append(int(cid))
        tag = ' '.join(t for t, flag in (('RAW', is_raw), ('COARSE', is_coarse)) if flag)
        print("  class %2d: %-20s %s" % (cid, name, tag))
    print("  -> verify this classification matches the intended definition "
          "of 'raw material' before trusting downstream volume estimates.")

    if len(raw_ids) == 0:
        raise ValueError("No lithology class name matched %s in %s" % (raw_keywords, f_prior_h5))
    return raw_ids, coarse_ids


# ----------------------------------------------------------------------
# 3. Point footprint area within a polygon
# ----------------------------------------------------------------------

def compute_point_footprint_area(X, Y, polygon, cell_size=5.0):
    """
    Approximate the ground area (m^2) represented by each data point,
    restricted to `polygon`.

    tTEM soundings are dense along flight lines but much more widely spaced
    across lines, so a fixed per-point area is not appropriate. Instead the
    polygon's bounding box is rasterized into `cell_size` x `cell_size`
    cells; cells whose centre falls inside the polygon are each assigned to
    their nearest data point (nearest-neighbour tessellation); a point's
    footprint area is then (number of assigned cells) * cell_size**2. Points
    outside the polygon (or too far from it to be nearest to any interior
    cell) get area 0 and drop out of any area-integrated sum automatically.

    Parameters
    ----------
    X, Y : ndarray (N,)
        Survey point coordinates (all survey points, not pre-filtered).
    polygon : shapely.Polygon
    cell_size : float
        Grid resolution [m] for the area rasterization.

    Returns
    -------
    area : ndarray (N,)
        Footprint area [m^2] per point; 0 for points that own no cell.
    """
    minx, miny, maxx, maxy = polygon.bounds
    xs = np.arange(minx, maxx + cell_size, cell_size) + cell_size / 2
    ys = np.arange(miny, maxy + cell_size, cell_size) + cell_size / 2
    gx, gy = np.meshgrid(xs, ys)
    gx, gy = gx.ravel(), gy.ravel()

    inside = contains_xy(polygon, gx, gy)
    gx, gy = gx[inside], gy[inside]
    if len(gx) == 0:
        return np.zeros(len(X))

    tree = cKDTree(np.column_stack([X, Y]))
    _, nearest_idx = tree.query(np.column_stack([gx, gy]))

    counts = np.bincount(nearest_idx, minlength=len(X))
    return counts.astype(float) * cell_size ** 2


# ----------------------------------------------------------------------
# 4. Area-integrated volume with uncertainty (Monte Carlo over the posterior)
# ----------------------------------------------------------------------

def posterior_volume_montecarlo(f_post_h5, polygon, raw_classes, coarse_classes,
                                 im=2, n_boot=500, area_cell_size=5.0,
                                 percentiles=(5, 50, 95), random_state=None):
    """
    Monte Carlo estimate, with uncertainty, of the area-integrated volumes
    of overburden, raw material (sand+gravel), and coarser material
    (gravel) within `polygon`.

    There is no built-in area-integrated volume function in INTEGRATE (its
    query engine — ``ig.query`` / ``ig.query_percentile`` — only returns
    per-sounding-location thickness statistics; this function turns those
    into a single area-integrated total with uncertainty).

    Method
    ------
    For each of `n_boot` bootstrap draws:
      1. Independently sample one posterior realization index per sounding
         location from that location's own accepted ensemble
         (``i_use[i, :]``, `N_post` values per location).
      2. From the prior lithology model, compute for that realization:
         - overburden thickness: depth from the surface to the first
           occurrence of a raw-material class (``thickness_mode =
           'first_occurrence'``); if no raw-material class occurs anywhere
           in the modelled depth range, the whole profile counts as
           overburden.
         - raw-material thickness: cumulative thickness of `raw_classes`
           over the full profile.
         - coarser-material thickness: cumulative thickness of
           `coarse_classes` over the full profile.
      3. Multiply each per-point thickness by that point's footprint area
         (see `compute_point_footprint_area`) and sum over all points whose
         footprint lies inside `polygon`.
    Repeating this `n_boot` times gives a distribution of total volume
    [m^3] for each of the three quantities; the requested percentiles are
    returned.

    Caveat: this treats per-point uncertainty as *independent* across
    sounding locations (each bootstrap draw samples every point separately).
    It does not model spatial correlation of the inversion uncertainty
    between neighbouring soundings, which would narrow the true
    area-integrated uncertainty somewhat. This is the same level of
    approximation as INTEGRATE's own per-point ``ig.query_percentile``
    output, just aggregated over an area.

    Parameters
    ----------
    f_post_h5 : str
        Path to the posterior HDF5 file (must have `/i_use`, `/UTMX`,
        `/UTMY`, and attrs `f5_prior`).
    polygon : shapely.Polygon
        Target area to integrate over.
    raw_classes, coarse_classes : list of int
        Lithology class IDs, e.g. from `resolve_material_classes`.
    im : int
        Lithology (discrete) model index, default 2.
    n_boot : int
        Number of Monte Carlo draws.
    area_cell_size : float
        Passed to `compute_point_footprint_area`.
    percentiles : sequence of float
    random_state : int or None

    Returns
    -------
    result : dict
        'overburden', 'raw_material', 'coarser' : ndarray (len(percentiles),)
            Requested percentiles of total volume [m^3].
        'samples' : dict of ndarray (n_boot,)
            Raw bootstrap totals, for further analysis/plotting.
        'percentiles' : ndarray
        'n_points_in_polygon' : int
        'area_total_m2' : float
    """
    rng = np.random.default_rng(random_state)

    with h5py.File(f_post_h5, 'r') as f:
        i_use = f['i_use'][:]                       # (N_data, N_post)
        f_prior_h5 = str(f.attrs['f5_prior'])
        f_data_h5 = str(f.attrs.get('f5_data', ''))
        X = f['UTMX'][:] if 'UTMX' in f else None
        Y = f['UTMY'][:] if 'UTMY' in f else None

    if X is None or Y is None:
        # Coordinates are not always duplicated into the posterior file --
        # fall back to the data file referenced by its 'f5_data' attribute
        # (same fallback ig.query()/ig.query_percentile() use internally).
        X, Y, _, _ = ig.get_geometry(f_data_h5)

    with h5py.File(f_prior_h5, 'r') as f:
        M = f['M%d' % im][:]                         # (N_prior, N_depth)
        z = f['M%d' % im].attrs['x'][:].astype(float)

    N_data, N_post = i_use.shape
    n_layers = len(z) - 1
    top, bot = z[:n_layers], z[1:]
    t = bot - top                                    # layer thickness (m)

    area = compute_point_footprint_area(X, Y, polygon, cell_size=area_cell_size)
    in_poly = area > 0
    if not np.any(in_poly):
        raise ValueError("No survey points found inside the target polygon "
                          "-- check that the data and the polygon share a CRS.")
    idx_pts = np.where(in_poly)[0]
    w = area[idx_pts]

    overburden_tot = np.zeros(n_boot)
    raw_tot = np.zeros(n_boot)
    coarse_tot = np.zeros(n_boot)

    for b in range(n_boot):
        draw = rng.integers(0, N_post, size=len(idx_pts))
        real_idx = i_use[idx_pts, draw]              # one realization index per point
        classes = np.round(M[real_idx, :n_layers]).astype(int)   # (n_pts, n_layers)

        is_raw = np.isin(classes, raw_classes)
        is_coarse = np.isin(classes, coarse_classes) if len(coarse_classes) else np.zeros_like(is_raw)

        has_raw = is_raw.any(axis=1)
        first_idx = np.argmax(is_raw, axis=1)
        overburden_thick = np.where(has_raw, top[first_idx], z[-1])

        raw_thick = (is_raw * t).sum(axis=1)
        coarse_thick = (is_coarse * t).sum(axis=1)

        overburden_tot[b] = np.sum(overburden_thick * w)
        raw_tot[b] = np.sum(raw_thick * w)
        coarse_tot[b] = np.sum(coarse_thick * w)

    pct = np.asarray(percentiles)
    return {
        'overburden': np.percentile(overburden_tot, pct),
        'raw_material': np.percentile(raw_tot, pct),
        'coarser': np.percentile(coarse_tot, pct),
        'samples': {'overburden': overburden_tot, 'raw_material': raw_tot, 'coarser': coarse_tot},
        'percentiles': pct,
        'n_points_in_polygon': int(len(idx_pts)),
        'area_total_m2': float(w.sum()),
    }


# ----------------------------------------------------------------------
# 5. Comparison to the old (deterministic/sequential) results
# ----------------------------------------------------------------------

def compare_to_reference(prob_results, reference, quantities=('overburden', 'raw_material', 'coarser'),
                          quantity_labels=None, hardcopy=False, f_name='comparison'):
    """
    Print a comparison table and plot an error-bar chart of the old
    deterministic single-number estimates against the new probabilistic
    P5/P50/P95 estimates, for a set of target sub-areas.

    Parameters
    ----------
    prob_results : dict {area_name: result}
        `result` as returned by `posterior_volume_montecarlo` (must have
        keys matching `quantities`, each an array of percentiles matching
        `reference`'s single 50th-percentile-like point estimate).
    reference : dict {area_name: {quantity: value_m3}}
        Old deterministic estimates, e.g. from the PPTX-derived numbers.
    quantities : tuple of str
        Keys to compare, must exist in both `prob_results[name]` and
        `reference[name]`.
    quantity_labels : dict or None
        Optional {quantity: display label} for plot titles.
    hardcopy : bool
    f_name : str
        Base filename for the saved figure (if hardcopy).

    Returns
    -------
    None (prints the table; shows/saves the plot).
    """
    if quantity_labels is None:
        quantity_labels = {
            'overburden': 'Overburden volume (m$^3$)',
            'raw_material': 'Raw material (sand+gravel) volume (m$^3$)',
            'coarser': 'Coarser material (gravel) volume (m$^3$)',
        }

    names = [n for n in prob_results if n in reference]

    print("\n" + "=" * 100)
    print("Comparison: deterministic (old, sequential) vs. probabilistic (INTEGRATE)")
    print("=" * 100)
    for q in quantities:
        print("\n-- %s --" % quantity_labels.get(q, q))
        print("%-20s %18s %28s" % ("Area", "Old (deterministic)", "New: P5 / P50 / P95 (probabilistic)"))
        for name in names:
            old_val = reference[name].get(q, np.nan)
            p5, p50, p95 = prob_results[name][q]
            print("%-20s %18s m3   %10s / %10s / %10s m3" % (
                name, "{:,.0f}".format(old_val),
                "{:,.0f}".format(p5), "{:,.0f}".format(p50), "{:,.0f}".format(p95)))

    fig, axes = plt.subplots(1, len(quantities), figsize=(6 * len(quantities), 5), squeeze=False)
    axes = axes[0]
    x = np.arange(len(names))
    for ax, q in zip(axes, quantities):
        old_vals = np.array([reference[n].get(q, np.nan) for n in names])
        p5 = np.array([prob_results[n][q][0] for n in names])
        p50 = np.array([prob_results[n][q][1] for n in names])
        p95 = np.array([prob_results[n][q][2] for n in names])

        width = 0.35
        ax.bar(x - width / 2, old_vals, width, color='0.7', label='Old (deterministic)')
        ax.bar(x + width / 2, p50, width, color='C0', label='New (probabilistic, P50)',
               yerr=[p50 - p5, p95 - p50], capsize=4, ecolor='k')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right')
        ax.set_ylabel(quantity_labels.get(q, q))
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    axes[0].legend(fontsize=8)
    fig.suptitle('Deterministic vs. probabilistic raw-material volume estimates')
    fig.tight_layout()
    if hardcopy:
        fig.savefig('%s.png' % f_name, dpi=200, bbox_inches='tight')
    plt.show()
