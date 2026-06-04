"""
Query posterior realizations based on geophysical constraints.

This module provides tools to compute probabilities that posterior realizations
from Bayesian inversion satisfy user-defined constraints (e.g., thickness of
lithology classes, resistivity thresholds).
"""

import json
import os
import numpy as np
import h5py
from tqdm import tqdm


def _get_clipped_thickness(z, depth_min, depth_max):
    """
    Compute per-layer thickness clipped to a depth interval.

    Parameters
    ----------
    z : ndarray (N_depth,)
        Depth values [m] at layer interfaces.
    depth_min : float, None, or ndarray (N_samples,)
        Lower depth bound [m]. Array triggers per-sample mode.
    depth_max : float, None, or ndarray (N_samples,)
        Upper depth bound [m]. Array triggers per-sample mode.

    Returns
    -------
    t : ndarray (N_depth - 1,) or (N_samples, N_depth - 1)
        Clipped thickness per layer. 2-D when bounds are per-sample.
    mask : ndarray (N_depth - 1,) bool or None
        True where clipped thickness > 0 (scalar bounds). None when 2-D.
    """
    n_layers = len(z) - 1
    top = z[:n_layers]
    bot = z[1:]
    lo = depth_min if depth_min is not None else -np.inf
    hi = depth_max if depth_max is not None else np.inf

    if np.ndim(lo) > 0 or np.ndim(hi) > 0:
        # Per-sample bounds: broadcast (N_samples,1) against (N_layers,)
        lo2 = np.asarray(lo)[:, np.newaxis] if np.ndim(lo) > 0 else lo
        hi2 = np.asarray(hi)[:, np.newaxis] if np.ndim(hi) > 0 else hi
        t = np.maximum(np.minimum(bot, hi2) - np.maximum(top, lo2), 0.0)
        return t, None      # 2-D t; mask handled downstream
    else:
        t = np.maximum(np.minimum(bot, hi) - np.maximum(top, lo), 0.0)
        return t, t > 0


def _first_occurrence_thickness(condition, t):
    """
    Thickness of the first contiguous True run per sample row.

    Parameters
    ----------
    condition : ndarray (N_samples, N_layers) bool
    t : ndarray (N_layers,) float

    Returns
    -------
    metric : ndarray (N_samples,) float
    """
    metric = np.zeros(condition.shape[0])
    has_true = condition.any(axis=1)
    first_idx = np.argmax(condition, axis=1)
    t_2d = t.ndim == 2
    for i in np.where(has_true)[0]:
        j = first_idx[i]
        while j < condition.shape[1] and condition[i, j]:
            metric[i] += t[i, j] if t_2d else t[j]
            j += 1
    return metric


def _compute_metric(M_samples, z, metric_def, scalar_model_values=None):
    """
    Compute the raw per-realization metric value for a model.

    Parameters
    ----------
    M_samples : ndarray (N_samples, N_depth)
        Model values for the sampled realizations.
    z : ndarray (N_depth,)
        Depth values [m].
    metric_def : dict
        Metric definition — same fields as a constraint minus comparison fields.
        Supports: im, classes, value_comparison, value_threshold, thickness_mode,
        depth_min, depth_max, depth_max_im, depth_min_im.
    scalar_model_values : dict {im: ndarray (N_samples,)}, optional
        Per-sample scalar model values for dynamic depth bounds.

    Returns
    -------
    metric : ndarray (N_samples,)
        For scalar models: raw model value per realization.
        For depth models: cumulative or first-occurrence thickness [m] of
        layers satisfying the class/value condition.
    """
    depth_min = metric_def.get('depth_min', None)
    depth_max = metric_def.get('depth_max', None)

    if scalar_model_values:
        if 'depth_max_im' in metric_def:
            depth_max = scalar_model_values.get(metric_def['depth_max_im'], depth_max)
        if 'depth_min_im' in metric_def:
            depth_min = scalar_model_values.get(metric_def['depth_min_im'], depth_min)

    t, layer_mask = _get_clipped_thickness(z, depth_min, depth_max)
    n_layers = len(z) - 1

    # Scalar model: return the single value directly (no thickness concept)
    if (z[-1] - z[0]) == 0 or n_layers == 0:
        return M_samples[:, 0].copy()

    if layer_mask is None:
        M_sel = M_samples[:, :n_layers]
        t_sel = t
    else:
        M_sel = M_samples[:, :n_layers][:, layer_mask]
        t_sel = t[layer_mask]

    if 'classes' in metric_def:
        condition = np.isin(np.round(M_sel).astype(int), metric_def['classes'])
    else:
        v_cmp = metric_def.get('value_comparison', '<')
        v_thr = metric_def.get('value_threshold', 0.0)
        condition = M_sel < v_thr if v_cmp == '<' else M_sel > v_thr

    mode = metric_def.get('thickness_mode', 'cumulative')
    if mode == 'cumulative':
        return (condition * t_sel).sum(axis=1)
    else:
        return _first_occurrence_thickness(condition, t_sel)


def _evaluate_constraint(M_samples, z, constraint, scalar_model_values=None):
    """
    Evaluate one constraint for a batch of realizations.

    Parameters
    ----------
    M_samples : ndarray (N_samples, N_depth)
        Model values for the sampled realizations.
    z : ndarray (N_depth,)
        Depth values [m].
    constraint : dict
        Constraint definition.
    scalar_model_values : dict {im: ndarray (N_samples,)}, optional
        Per-sample values of scalar models. Used to resolve `depth_max_im`
        and `depth_min_im` constraint fields into per-sample depth bounds.

    Returns
    -------
    valid : ndarray (N_samples,) bool
        True for each realization that satisfies the constraint.
    """
    n_layers = len(z) - 1
    is_scalar = (z[-1] - z[0]) == 0 or n_layers == 0

    raw = _compute_metric(M_samples, z, constraint, scalar_model_values)

    if is_scalar:
        if 'classes' in constraint:
            valid = np.isin(np.round(raw).astype(int), constraint['classes'])
        else:
            v_cmp = constraint.get('value_comparison', '<')
            v_thr = constraint.get('value_threshold', 0.0)
            valid = raw < v_thr if v_cmp == '<' else raw > v_thr
    else:
        t_cmp = constraint.get('thickness_comparison', '>')
        t_thr = constraint.get('thickness_threshold', 0.0)
        _ops = {
            '>':  lambda m: m > t_thr,
            '<':  lambda m: m < t_thr,
            '>=': lambda m: m >= t_thr,
            '<=': lambda m: m <= t_thr,
        }
        valid = _ops.get(t_cmp, _ops['>'])(raw)

    return ~valid if constraint.get('negate', False) else valid


def _collect_needed_ims(items):
    """Return the set of model indices (im) referenced in a constraint list or metric dict."""
    needed = set()
    if isinstance(items, dict):
        items = [items]
    for c in items:
        needed.add(c['im'])
        for field in ('depth_max_im', 'depth_min_im'):
            if field in c:
                needed.add(c[field])
    return needed


def _load_query_inputs(f_post_h5, needed_ims):
    """
    Load posterior indices, coordinates, and prior model arrays.

    Parameters
    ----------
    f_post_h5 : str
        Path to the posterior HDF5 file.
    needed_ims : set of int
        Model indices to pre-load from the prior file.

    Returns
    -------
    i_use : ndarray (N_data, N_post)
    X, Y : ndarray or None
    prior_models : dict {im: (M, z, is_discrete)}
    N_data, N_post : int
    """
    with h5py.File(f_post_h5, 'r') as f:
        i_use = f['i_use'][:]
        f_prior_h5 = str(f.attrs.get('f5_prior', ''))
        f_data_h5 = str(f.attrs.get('f5_data', ''))
        X = f['UTMX'][:] if 'UTMX' in f else None
        Y = f['UTMY'][:] if 'UTMY' in f else None

    if not f_prior_h5:
        raise ValueError("Posterior file missing 'f5_prior' attribute.")
    if not os.path.isfile(f_prior_h5):
        raise FileNotFoundError(f"Prior file not found: {f_prior_h5}")

    if (X is None or Y is None) and f_data_h5 and os.path.isfile(f_data_h5):
        with h5py.File(f_data_h5, 'r') as f:
            if X is None and 'UTMX' in f:
                X = f['UTMX'][:]
            if Y is None and 'UTMY' in f:
                Y = f['UTMY'][:]

    N_data, N_post = i_use.shape

    prior_models = {}
    for im in needed_ims:
        key = f'M{im}'
        with h5py.File(f_prior_h5, 'r') as f:
            M = f[key][:]
            z = f[key].attrs['x'].astype(float)
            is_discrete = bool(f[key].attrs.get('is_discrete', 0))
        prior_models[im] = (M, z, is_discrete)

    return i_use, X, Y, prior_models, N_data, N_post


def _build_scalar_vals(prior_models, idx):
    """Return per-sample values for all scalar models in prior_models."""
    scalar_vals = {}
    for im_s, (M_s, z_s, _) in prior_models.items():
        if (z_s[-1] - z_s[0]) == 0 or len(z_s) - 1 == 0:
            scalar_vals[im_s] = M_s[idx, 0]
    return scalar_vals


def query_probability(f_post_h5, query_dict):
    """
    Compute per-data-point probability that posterior realizations satisfy a query.

    Parameters
    ----------
    f_post_h5 : str
        Path to the posterior HDF5 file.
    query_dict : str or dict
        Path to a JSON file, or a dict with a ``"constraints"`` key.

    Returns
    -------
    P : ndarray (N_data,)
        Probability [0, 1] for each data location.
    meta : dict
        Keys: 'X', 'Y', 'N_data', 'N_post', 'i_use', 'i_use_query'.

    Examples
    --------
    >>> query_def = {
    ...     "constraints": [{
    ...         "im": 2, "classes": [2],
    ...         "thickness_mode": "cumulative",
    ...         "thickness_comparison": ">",
    ...         "thickness_threshold": 10.0,
    ...         "depth_min": 0.0, "depth_max": 30.0
    ...     }]
    ... }
    >>> P, meta = query_probability('f_post.h5', query_def)
    """
    if isinstance(query_dict, str):
        with open(query_dict, 'r') as fh:
            query_dict = json.load(fh)

    constraints = query_dict['constraints']
    needed_ims = _collect_needed_ims(constraints)
    i_use, X, Y, prior_models, N_data, N_post = _load_query_inputs(f_post_h5, needed_ims)

    P = np.zeros(N_data)
    i_use_query = []
    for i in tqdm(range(N_data), desc='Evaluating probability query', unit='location'):
        idx = i_use[i]
        valid = np.ones(N_post, dtype=bool)
        scalar_vals = _build_scalar_vals(prior_models, idx)
        for c in constraints:
            M, z, _ = prior_models[c['im']]
            valid &= _evaluate_constraint(M[idx, :], z, c, scalar_model_values=scalar_vals)
        P[i] = valid.mean()
        i_use_query.append(idx[valid])

    meta = {
        'X': X, 'Y': Y,
        'N_data': N_data, 'N_post': N_post,
        'i_use': i_use, 'i_use_query': i_use_query,
    }
    return P, meta


def query_percentile(f_post_h5, query_dict):
    """
    Compute per-data-point percentiles of a metric over posterior realizations.

    Rather than asking "what fraction of realizations satisfy condition X?", this
    asks "what is the p5/p50/p95 of metric X across realizations?".  The metric
    is defined by the same fields as a probability constraint, minus the comparison
    fields (``thickness_comparison``, ``thickness_threshold``, ``negate``).

    Parameters
    ----------
    f_post_h5 : str
        Path to the posterior HDF5 file.
    query_dict : str or dict
        Path to a JSON file, or a dict with a ``"metric"`` key and an optional
        ``"percentiles"`` key (default ``[5, 50, 95]``).

    Returns
    -------
    percentile_values : ndarray (N_data, n_percentiles)
        Requested percentile values for each data location.
    meta : dict
        Keys: 'X', 'Y', 'N_data', 'N_post', 'i_use', 'percentiles'.

    Examples
    --------
    >>> query_def = {
    ...     "metric": {
    ...         "im": 2, "classes": [1, 2],
    ...         "thickness_mode": "cumulative",
    ...         "depth_max": 30.0
    ...     },
    ...     "percentiles": [5, 50, 95]
    ... }
    >>> pct_values, meta = query_percentile('f_post.h5', query_def)
    >>> # pct_values shape: (N_data, 3) — p5, p50, p95 per location
    """
    if isinstance(query_dict, str):
        with open(query_dict, 'r') as fh:
            query_dict = json.load(fh)

    metric_def = query_dict['metric']
    percentiles = query_dict.get('percentiles', [5, 50, 95])
    needed_ims = _collect_needed_ims(metric_def)
    i_use, X, Y, prior_models, N_data, N_post = _load_query_inputs(f_post_h5, needed_ims)

    M_main, z_main, _ = prior_models[metric_def['im']]
    n_pct = len(percentiles)
    result = np.zeros((N_data, n_pct))

    for i in tqdm(range(N_data), desc='Evaluating percentile query', unit='location'):
        idx = i_use[i]
        scalar_vals = _build_scalar_vals(prior_models, idx)
        values = _compute_metric(M_main[idx, :], z_main, metric_def,
                                 scalar_model_values=scalar_vals)
        result[i, :] = np.percentile(values, percentiles)

    meta = {
        'X': X, 'Y': Y,
        'N_data': N_data, 'N_post': N_post,
        'i_use': i_use,
        'percentiles': percentiles,
    }
    return result, meta


def query(f_post_h5, query_dict):
    """
    Dispatcher: route to query_probability() or query_percentile() based on query_dict.

    If ``query_dict`` contains a ``"metric"`` key, calls :func:`query_percentile`.
    Otherwise calls :func:`query_probability` (backward compatible with all existing
    ``"constraints"``-based dicts).

    Parameters
    ----------
    f_post_h5 : str
        Path to the posterior HDF5 file.
    query_dict : str or dict
        Query definition.  See :func:`query_probability` and
        :func:`query_percentile` for the respective schemas.

    Returns
    -------
    result, meta
        See the delegated function for details.
    """
    if not os.path.isfile(str(f_post_h5)):
        print(f"[ig.query] Posterior file not found: {f_post_h5}")
        return None, {}
    if isinstance(query_dict, str):
        with open(query_dict, 'r') as fh:
            query_dict = json.load(fh)
    if 'metric' in query_dict:
        return query_percentile(f_post_h5, query_dict)
    return query_probability(f_post_h5, query_dict)


def query_plot(P, meta, ip=None, query_dict=None, f_prior_h5=None, f_post_h5=None, title=None,
               query_text=None, interpretation=None, text_panel=False, hardcopy=False):
    """
    Plot query results and optionally detailed model visualization for a data point.

    If ip is None, displays the XY probability map showing P(x, y).
    If ip is provided (together with query_dict and f_prior_h5/f_post_h5), skips the
    probability map and shows only the detailed single-point visualization of all
    posterior realizations and the query-matching subset.

    Parameters
    ----------
    P : ndarray (N_data,)
        Probability array from query().
    meta : dict
        Metadata dict from query() containing 'X', 'Y', 'i_use', 'i_use_query'.
    ip : int, optional
        Data point index to visualize in detail. If None, only shows probability map.
    query_dict : dict, optional
        Query dict used in query(). Required for detailed visualization.
    f_prior_h5 : str, optional
        Path to prior HDF5 file. If not provided, will be extracted from f_post_h5.
    f_post_h5 : str, optional
        Path to posterior HDF5 file. Used to automatically extract prior file path
        if f_prior_h5 is not provided.
    title : str, optional
        Custom title for the probability map. If None, a title is built from
        query_text and interpretation (if provided), or 'Query Probability Map'.
    query_text : str, optional
        The original natural-language query string. Shown in the figure title,
        or in the text panel if text_panel=True.
    interpretation : str, optional
        The LLM interpretation string returned by query_from_text(). Shown as a
        second line in the figure title, or in the text panel if text_panel=True.
    text_panel : bool, optional
        If True and query_text or interpretation is provided, adds a narrow text
        column to the right of the probability map. The query text appears at the
        top and the interpretation below. Default False.
    hardcopy : bool or str, optional
        Save the probability map figure. If True, saves as 'query_plot.png'.
        If a string, uses that as the filename (a '.png' extension is appended
        if the string has no extension). Default False.

    Examples
    --------
    >>> P, meta = query(f_post_h5, query_def)
    >>> query_plot(P, meta)  # Just probability map
    >>> query_plot(P, meta, title='Custom Query Title')  # Custom title
    >>> query_plot(P, meta, ip=1000, query_dict=query_def, f_post_h5='posterior.h5')
    >>> query_plot(P, meta, ip=1000, query_dict=query_def, f_prior_h5='prior.h5')
    >>> # With LLM query text and interpretation:
    >>> query_dict, interp = ig.query_from_text(text, f_prior_h5)
    >>> P, meta = ig.query(f_post_h5, query_dict)
    >>> ig.query_plot(P, meta, query_text=text, interpretation=interp)
    """
    import matplotlib.pyplot as plt

    # Auto-extract prior file path from posterior file if needed
    if f_prior_h5 is None and f_post_h5 is not None:
        with h5py.File(f_post_h5, 'r') as f:
            f_prior_h5 = str(f.attrs.get('f5_prior', ''))
            if not f_prior_h5:
                print("Warning: Could not extract f5_prior attribute from posterior file")

    X = meta['X']
    Y = meta['Y']

    # Plot XY probability map only when no specific point is requested
    if ip is None:
        has_text = text_panel and (query_text is not None or interpretation is not None)
        if has_text:
            fig = plt.figure(figsize=(11, 6))
            gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
            ax = fig.add_subplot(gs[0])
            ax_text = fig.add_subplot(gs[1])
        else:
            fig, ax = plt.subplots(figsize=(8, 6))

        # Determine title
        if has_text:
            _title = 'Query Probability Map'
        elif title is not None:
            _title = title
        elif query_text is not None or interpretation is not None:
            parts = []
            if query_text is not None:
                parts.append(f"Query: {query_text}")
            if interpretation is not None:
                parts.append(f"Interpreted as: {interpretation}")
            _title = '\n'.join(parts)
        else:
            _title = 'Query Probability Map'

        import textwrap
        _title = '\n'.join(
            '\n'.join(textwrap.wrap(line, width=60)) if len(line) > 60 else line
            for line in _title.splitlines()
        )

        # Background dots so P=0 (white) areas are visible, then probability scatter
        from integrate.integrate_plot import plot_xy
        ax.scatter(X, Y, c='black', s=2, alpha=0.5)
        _, ax, sc = plot_xy(P, X=X, Y=Y,
                            cmap='hot_r', clim=[0, 1],
                            title=_title, colorbar=True, colorbar_label='Probability',
                            ax=ax, s=1)
        ax.set_xlabel('UTMX [m]')
        ax.set_ylabel('UTMY [m]')

        if has_text:
            import textwrap
            ax_text.set_axis_off()
            CHARS = 36       # characters per wrapped line
            LH = 0.062       # axes-fraction height per text line (fontsize 8)
            LABEL_GAP = 0.03 # gap between bold label and text box
            SECTION_GAP = 0.07  # gap between sections

            y = 0.97
            if query_text is not None:
                ax_text.text(0.02, y, "Query:", transform=ax_text.transAxes,
                             fontsize=8, fontweight='bold', va='top')
                y -= LH + LABEL_GAP
                wrapped_q = textwrap.fill(query_text, CHARS)
                n_q = wrapped_q.count('\n') + 1
                ax_text.text(0.02, y, wrapped_q, transform=ax_text.transAxes,
                             fontsize=7.5, va='top',
                             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='none'))
                y -= n_q * LH + SECTION_GAP
            if interpretation is not None:
                ax_text.text(0.02, y, "Interpretation:", transform=ax_text.transAxes,
                             fontsize=8, fontweight='bold', va='top')
                y -= LH + LABEL_GAP
                wrapped_i = textwrap.fill(interpretation, CHARS)
                ax_text.text(0.02, y, wrapped_i, transform=ax_text.transAxes,
                             fontsize=7.5, va='top',
                             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f4e8', edgecolor='none'))

        plt.tight_layout()

    # If ip provided and we have necessary data, plot detailed model view
    if ip is not None and query_dict is not None and f_prior_h5 is not None:
        # Load prior model for the first constraint
        im = query_dict['constraints'][0]['im']
        with h5py.File(f_prior_h5, 'r') as f:
            M = f[f'M{im}'][:]
            # Read model attributes
            is_discrete = bool(f[f'M{im}'].attrs.get('is_discrete', 0))
            class_id = None
            class_name = None
            prior_cmap = None

            # Always try to read colormap from prior file if available
            if 'cmap' in f[f'M{im}'].attrs.keys():
                try:
                    cmap_array = f[f'M{im}'].attrs['cmap'][:]
                    from matplotlib.colors import ListedColormap
                    # Format is [3, nlev] or [4, nlev] - transpose to get [nlev, 3] or [nlev, 4]
                    prior_cmap = ListedColormap(cmap_array.T)
                except Exception:
                    prior_cmap = None

            # Read class information for discrete models
            if is_discrete:
                if 'class_id' in f[f'M{im}'].attrs.keys():
                    class_id = f[f'M{im}'].attrs['class_id'][:].flatten()
                if 'class_name' in f[f'M{im}'].attrs.keys():
                    class_name = f[f'M{im}'].attrs['class_name'][:].flatten()

        # Get posterior and query-matching indices
        i_use = meta['i_use'][ip, :]
        i_use_query = meta['i_use_query'][ip]

        # Calculate statistics
        n_total = len(i_use)
        n_accepted = len(i_use_query)
        probability = P[ip]

        # Get all posterior realizations
        M_use_all = M[i_use]

        # Create filtered version with NaN for non-matching realizations
        # Convert to float to allow NaN values
        M_use_filtered = M_use_all.astype(float)
        # Create mask: True where i_use is in i_use_query
        matching_mask = np.isin(i_use, i_use_query)
        # Set non-matching realizations to NaN
        M_use_filtered[~matching_mask, :] = np.nan

        # Create detailed model plot
        plt.figure(figsize=(12, 8))

        # Determine color limits for discrete models
        # Use the full range of class IDs from the prior file
        if is_discrete and class_id is not None:
            vmin_plot = np.min(class_id) - 0.5
            vmax_plot = np.max(class_id) + 0.5
        else:
            vmin_plot = None
            vmax_plot = None

        # Subplot 1: All posterior realizations
        plt.subplot(2, 1, 1)
        # Use colormap from prior file if available
        if prior_cmap is not None and vmin_plot is not None:
            im1 = plt.imshow(M_use_all.T, aspect='auto', cmap=prior_cmap, interpolation='nearest',
                           vmin=vmin_plot, vmax=vmax_plot)
        elif prior_cmap is not None:
            im1 = plt.imshow(M_use_all.T, aspect='auto', cmap=prior_cmap, interpolation='nearest')
        else:
            im1 = plt.imshow(M_use_all.T, aspect='auto', cmap='jet', interpolation='nearest')

        plt.title(f'All Posterior Realizations (Point {ip})\n'
                  f'Total Realizations: {n_total} | Accepted: {n_accepted} | Probability: {probability:.3f}')
        plt.xlabel('Realization index')
        plt.ylabel('Layer index')

        # Create colorbar with class names if discrete
        if is_discrete and class_id is not None and class_name is not None:
            cbar1 = plt.colorbar(im1)
            cbar1.set_ticks(class_id)
            # Create tick labels with format "ClassName (ID)"
            tick_labels = [f'{name} ({int(cid)})' for name, cid in zip(class_name, class_id)]
            cbar1.set_ticklabels(tick_labels)
            cbar1.ax.invert_yaxis()
        else:
            plt.colorbar(im1, label='Model value')

        # Subplot 2: Query-matching realizations only (others set to NaN)
        plt.subplot(2, 1, 2)
        # Use colormap from prior file if available
        if prior_cmap is not None and vmin_plot is not None:
            im2 = plt.imshow(M_use_filtered.T, aspect='auto', cmap=prior_cmap, interpolation='nearest',
                           vmin=vmin_plot, vmax=vmax_plot)
        elif prior_cmap is not None:
            im2 = plt.imshow(M_use_filtered.T, aspect='auto', cmap=prior_cmap, interpolation='nearest')
        else:
            im2 = plt.imshow(M_use_filtered.T, aspect='auto', cmap='jet', interpolation='nearest')

        plt.title('Query-Matching Realizations Only (non-matching set to NaN)')
        plt.xlabel('Realization index')
        plt.ylabel('Layer index')

        # Create colorbar with class names if discrete
        if is_discrete and class_id is not None and class_name is not None:
            cbar2 = plt.colorbar(im2)
            cbar2.set_ticks(class_id)
            tick_labels = [f'{name} ({int(cid)})' for name, cid in zip(class_name, class_id)]
            cbar2.set_ticklabels(tick_labels)
            cbar2.ax.invert_yaxis()
        else:
            plt.colorbar(im2, label='Model value')

        plt.tight_layout()

    _VALID_EXTS = {'.png', '.jpg', '.jpeg', '.pdf', '.svg', '.eps', '.tif', '.tiff', '.webp'}
    if hardcopy is not False and hardcopy is not None:
        if isinstance(hardcopy, str):
            safe = hardcopy.replace(':', '_').replace('/', '_')
            f_png = safe if os.path.splitext(safe)[1].lower() in _VALID_EXTS else safe + '.png'
        else:
            f_png = 'query_plot.png'
        plt.savefig(f_png)
        print(f"Figure saved to {f_png}")

    plt.show()


def query_percentile_plot(percentile_values, meta, query_text=None, interpretation=None,
                          text_panel=False, hardcopy=False):
    """
    Plot one probability map per requested percentile as side-by-side subplots.

    Parameters
    ----------
    percentile_values : ndarray (N_data, n_percentiles)
        Output of query_percentile().
    meta : dict
        Metadata dict from query_percentile() containing 'X', 'Y', 'percentiles'.
    query_text : str, optional
        Original query string — shown as figure suptitle.
    interpretation : str, optional
        LLM interpretation string — shown below query_text if provided.
    text_panel : bool, optional
        If True, add a narrow text column to the right of the maps.
    hardcopy : bool or str, optional
        Save figure to disk.  True → 'query_percentile_plot.png'; a string is
        used as the filename (.png appended if no extension).

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    percentiles = meta.get('percentiles', [5, 50, 95])
    n_pct = len(percentiles)
    X = meta.get('X')
    Y = meta.get('Y')

    ncols = n_pct + (1 if text_panel else 0)
    width_ratios = [4] * n_pct + ([1.5] if text_panel else [])
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * n_pct + (1.5 if text_panel else 0), 5),
                             gridspec_kw={'width_ratios': width_ratios} if text_panel else {},
                             squeeze=False)

    map_axes = axes[0, :n_pct]
    vmin = percentile_values.min()
    vmax = percentile_values.max()

    from integrate.integrate_plot import plot_xy
    for k, (pct, ax) in enumerate(zip(percentiles, map_axes)):
        vals = percentile_values[:, k]
        if X is not None and Y is not None:
            _, ax, _ = plot_xy(vals, X=X, Y=Y,
                               cmap='viridis', clim=[vmin, vmax],
                               colorbar=True, colorbar_label='[m]',
                               ax=ax, s=5)
            ax.set_xlabel('UTMX')
            if k == 0:
                ax.set_ylabel('UTMY')
        else:
            ax.plot(vals)
            ax.set_xlabel('Location index')
        ax.set_title(f'P{pct}  (median={np.median(vals):.1f})')

    if text_panel:
        tax = axes[0, n_pct]
        tax.axis('off')
        txt = ''
        if query_text:
            txt += f'Query:\n{query_text}\n\n'
        if interpretation:
            txt += f'Interpretation:\n{interpretation}'
        if txt:
            tax.text(0.05, 0.95, txt, transform=tax.transAxes,
                     fontsize=7, va='top', wrap=True)

    suptitle_parts = [t for t in [query_text, interpretation] if t]
    if suptitle_parts and not text_panel:
        fig.suptitle('\n'.join(suptitle_parts), fontsize=8, y=1.01)

    plt.tight_layout()

    if hardcopy:
        fname = hardcopy if isinstance(hardcopy, str) else 'query_percentile_plot'
        if '.' not in os.path.basename(fname):
            fname += '.png'
        plt.savefig(fname, bbox_inches='tight', dpi=150)
        print(f"Figure saved to {fname}")

    plt.show()
    return fig


def save_query(query, path):
    """
    Save a query dict to a JSON file.

    Parameters
    ----------
    query : dict
        Query definition dictionary.
    path : str
        Output JSON file path.
    """
    with open(path, 'w') as f:
        json.dump(query, f, indent=2)
    print(f"Query saved to {path}")


def load_query(path):
    """
    Load a query dict from a JSON file.

    Parameters
    ----------
    path : str
        Input JSON file path.

    Returns
    -------
    query : dict
        Query definition dictionary.
    """
    with open(path, 'r') as f:
        return json.load(f)


def get_prior_model_info(f_prior_h5, im):
    """
    Return metadata for prior model im.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the prior HDF5 file.
    im : int
        Model index.

    Returns
    -------
    info : dict
        Keys: 'name', 'is_discrete', 'z', 'class_id', 'class_name'.
    """
    key = f'M{im}'
    with h5py.File(f_prior_h5, 'r') as f:
        ds = f[key]
        info = {
            'name':        ds.attrs.get('name', key),
            'is_discrete': bool(ds.attrs.get('is_discrete', 0)),
            'z':           ds.attrs['x'].astype(float),
            'class_id':    ds.attrs.get('class_id', None),
            'class_name':  ds.attrs.get('class_name', None),
        }
    return info


def _build_llm_system_prompt(f_prior_h5):
    """
    Build a system prompt for the LLM that includes the query schema and prior model context.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the prior HDF5 file.

    Returns
    -------
    prompt : str
        System prompt string.
    """
    # Collect all model keys from the prior file
    with h5py.File(f_prior_h5, 'r') as f:
        model_keys = sorted([k for k in f.keys() if k.startswith('M') and k[1:].isdigit()])

    model_sections = []
    for key in model_keys:
        im = int(key[1:])
        info = get_prior_model_info(f_prior_h5, im)
        z = info['z']
        n_layers = len(z) - 1
        depth_min = float(z[0])
        depth_max = float(z[-1])
        name = info['name'] if info['name'] != key else key

        is_scalar = (depth_max - depth_min) == 0 or n_layers == 0

        if is_scalar:
            kind = 'SCALAR-DISCRETE' if info['is_discrete'] else 'SCALAR'
            lines = [f"  Model im={im}: {name} ({kind}) — single value per realization, no depth profile"]
            lines.append("    Use only 'value_comparison' and 'value_threshold'. Do NOT include any thickness fields.")
            if info['is_discrete'] and info['class_id'] is not None and info['class_name'] is not None:
                lines.append("    Classes (use these integer IDs in the 'classes' field):")
                ids = info['class_id'].flatten()
                names = info['class_name'].flatten()
                for cid, cname in zip(ids, names):
                    lines.append(f"      {int(cid)} = {cname}")
        elif info['is_discrete']:
            lines = [f"  Model im={im}: {name} (DISCRETE), depth {depth_min:.1f}–{depth_max:.1f} m, {n_layers} layers"]
            if info['class_id'] is not None and info['class_name'] is not None:
                lines.append("    Classes (use these integer IDs in the 'classes' field):")
                ids = info['class_id'].flatten()
                names = info['class_name'].flatten()
                for cid, cname in zip(ids, names):
                    lines.append(f"      {int(cid)} = {cname}")
            else:
                lines.append("    (Class IDs not available in prior file)")
        else:
            lines = [f"  Model im={im}: {name} (CONTINUOUS), depth {depth_min:.1f}–{depth_max:.1f} m, {n_layers} layers"]
            lines.append("    Use 'value_comparison' ('<' or '>') and 'value_threshold' for this model.")

        model_sections.append('\n'.join(lines))

    models_text = '\n'.join(model_sections)

    prompt = f"""You are a geophysics query assistant for the INTEGRATE probabilistic inversion module.
Your task is to translate a natural-language query about geological or geophysical properties
into a valid JSON query dict that can be executed by the query() function.

## Query types

Choose the query type based on the user's intent:

**"probability"** — the user asks for a probability, likelihood, or yes/no fraction.
  Example: "What is the probability that clay thickness exceeds 10 m?"
  Response includes "query_type": "probability" and a "constraints" list.

**"percentile"** — the user asks for a distribution, typical value, or p5/p50/p95.
  Example: "What are the p5, p50, p95 of the cumulative thickness of sand above 10 m depth?"
  Response includes "query_type": "percentile", a single "metric" object, and a "percentiles" list.

## Response format

You must always respond with a single JSON object. Required top-level keys:
- "interpretation": 1–2 sentence plain-English confirmation of what you understood.
- "query_type": either "probability" or "percentile".
- For probability: "constraints" (list of constraint objects — see below).
- For percentile: "metric" (a single metric object — see below) and "percentiles" (list of ints,
  default [5, 50, 95] if not specified by the user).

Probability response structure:
```json
{{
  "interpretation": "...",
  "query_type": "probability",
  "constraints": [ {{ ... }} ]
}}
```

Percentile response structure:
```json
{{
  "interpretation": "...",
  "query_type": "percentile",
  "metric": {{ ... }},
  "percentiles": [5, 50, 95]
}}
```

## Constraint fields

A constraint list contains one or more constraint objects combined with logical AND
(every constraint must be satisfied).

## Constraint Fields

| Field                | Type        | Required          | Valid values                        | Description                                      |
|----------------------|-------------|-------------------|-------------------------------------|--------------------------------------------------|
| im                   | int         | always            | 1, 2, 3, ...                        | Prior model index (see Available Models below)   |
| classes              | list[int]   | discrete only     | class IDs from the model            | Match any of these class IDs (discrete models)   |
| value_comparison     | str         | continuous only   | "<" or ">"                          | Compare model value against threshold            |
| value_threshold      | float       | continuous only   | any float                           | Threshold for continuous value comparison        |
| thickness_mode       | str         | depth models only | "cumulative" or "first_occurrence"  | How to aggregate thickness of matching layers    |
| thickness_comparison | str         | depth models only | ">", "<", ">=", "<="                | Operator applied to the computed thickness       |
| thickness_threshold  | float       | depth models only | any float (meters)                  | Thickness threshold in meters                    |
| depth_min            | float       | optional          | any float                           | Upper boundary of depth interval [m]             |
| depth_max            | float       | optional          | any float                           | Lower boundary of depth interval [m]             |
| depth_max_im         | int         | optional          | SCALAR model im                     | Per-realization depth_max from a scalar model    |
| depth_min_im         | int         | optional          | SCALAR model im                     | Per-realization depth_min from a scalar model    |
| negate               | bool        | optional          | true or false (default: false)      | If true, invert the constraint result            |

> **Cross-model depth bounds:** `depth_max_im` / `depth_min_im` take the `im` index of a
> SCALAR model and use its per-realization value as the depth boundary. This enables
> queries like "Sand above the water table" where the cutoff depth varies per realization.
> Use `depth_max_im` to cut at the scalar model's value from above; use `depth_min_im`
> to cut from below. These may be combined with fixed `depth_min` / `depth_max`.

### thickness_mode explained
- "cumulative": sum the thickness of ALL matching layers within the depth interval
- "first_occurrence": thickness of the FIRST contiguous block of matching layers

### Scalar models (marked SCALAR or SCALAR-DISCRETE above)
These store a single value per realization, not a depth profile. For scalar models:
- Omit ALL thickness fields (`thickness_mode`, `thickness_comparison`, `thickness_threshold`, `depth_min`, `depth_max`).
- Use only `im`, `value_comparison`, `value_threshold`, and optionally `negate`.

## Available Prior Models

{models_text}

## Examples

## Metric fields (for percentile queries)

A metric object defines WHAT to measure per realization (no comparison or threshold).
It uses the same fields as a constraint, minus: thickness_comparison, thickness_threshold, negate.

| Field            | Type      | Required          | Valid values                        | Description                                   |
|------------------|-----------|-------------------|-------------------------------------|-----------------------------------------------|
| im               | int       | always            | 1, 2, 3, ...                        | Prior model index                             |
| classes          | list[int] | DISCRETE only     | class IDs from the model            | Thickness of these classes is measured        |
| value_comparison | str       | CONTINUOUS only   | "<" or ">"                          | Condition on value before measuring thickness |
| value_threshold  | float     | CONTINUOUS only   | any float                           | Threshold for the value condition             |
| thickness_mode   | str       | depth models only | "cumulative" or "first_occurrence"  | How to aggregate thickness                    |
| depth_min        | float     | optional          | any float                           | Upper depth boundary [m]                      |
| depth_max        | float     | optional          | any float                           | Lower depth boundary [m]                      |
| depth_max_im     | int       | optional          | SCALAR model im                     | Per-realization depth_max from scalar model   |
| depth_min_im     | int       | optional          | SCALAR model im                     | Per-realization depth_min from scalar model   |

For SCALAR models used as a metric: returns the raw scalar value; no thickness fields needed.

### Example 1: Discrete cumulative constraint
Query: "Probability that cumulative clay thickness exceeds 10 m within 0–30 m depth"
```json
{{
  "interpretation": "Probability that the cumulative thickness of clay (class 2) exceeds 10 m within 0–30 m depth.",
  "query_type": "probability",
  "constraints": [
    {{
      "im": 2,
      "classes": [2],
      "thickness_mode": "cumulative",
      "thickness_comparison": ">",
      "thickness_threshold": 10.0,
      "depth_min": 0.0,
      "depth_max": 30.0,
      "negate": false
    }}
  ]
}}
```

### Example 2: Continuous cumulative constraint
Query: "Probability that resistivity is below 100 ohm-m for at least 25 m within 0–50 m"
```json
{{
  "interpretation": "Probability that resistivity (im=1) is below 100 ohm-m for a cumulative thickness of at least 25 m within 0–50 m depth.",
  "query_type": "probability",
  "constraints": [
    {{
      "im": 1,
      "value_comparison": "<",
      "value_threshold": 100.0,
      "thickness_mode": "cumulative",
      "thickness_comparison": ">",
      "thickness_threshold": 25.0,
      "depth_min": 0.0,
      "depth_max": 50.0,
      "negate": false
    }}
  ]
}}
```

### Example 3: Multi-constraint AND
Query: "Probability that clay > 5 m within 0–20 m AND resistivity > 500 ohm-m for >= 1 m within 20–60 m"
```json
{{
  "interpretation": "Probability that cumulative clay (class 2) thickness exceeds 5 m within 0–20 m AND resistivity (im=1) exceeds 500 ohm-m for at least 1 m within 20–60 m. Both constraints must hold simultaneously.",
  "query_type": "probability",
  "constraints": [
    {{
      "im": 2,
      "classes": [2],
      "thickness_mode": "cumulative",
      "thickness_comparison": ">",
      "thickness_threshold": 5.0,
      "depth_min": 0.0,
      "depth_max": 20.0,
      "negate": false
    }},
    {{
      "im": 1,
      "value_comparison": ">",
      "value_threshold": 500.0,
      "thickness_mode": "cumulative",
      "thickness_comparison": ">",
      "thickness_threshold": 1.0,
      "depth_min": 20.0,
      "depth_max": 60.0,
      "negate": false
    }}
  ]
}}
```

### Example 4: First-occurrence with negation
Query: "Probability that the first occurrence of clay at the surface is less than 3 m thick"
```json
{{
  "interpretation": "Probability that the first contiguous block of clay (class 2) starting from the surface is less than 3 m thick, within 0–30 m depth.",
  "query_type": "probability",
  "constraints": [
    {{
      "im": 2,
      "classes": [2],
      "thickness_mode": "first_occurrence",
      "thickness_comparison": "<",
      "thickness_threshold": 3.0,
      "depth_min": 0.0,
      "depth_max": 30.0,
      "negate": false
    }}
  ]
}}
```

### Example 5: Scalar model query (no thickness fields)
Query: "Probability that the water table is shallower than 5 m"
```json
{{
  "interpretation": "Probability that the water table depth (im=3, SCALAR) is less than 5 m.",
  "query_type": "probability",
  "constraints": [
    {{
      "im": 3,
      "value_comparison": "<",
      "value_threshold": 5.0,
      "negate": false
    }}
  ]
}}
```

### Example 6: Cross-model depth constraint (dynamic depth bound from scalar model)
Query: "Probability that Sand and Grus have a cumulative thickness above the water table exceeding 5 m"
```json
{{
  "interpretation": "Probability that Sand (class 1) and Grus (class 2) have a cumulative thickness exceeding 5 m within the zone above the water table (im=3), starting from the surface.",
  "query_type": "probability",
  "constraints": [
    {{
      "im": 2,
      "classes": [1, 2],
      "thickness_mode": "cumulative",
      "thickness_comparison": ">",
      "thickness_threshold": 5.0,
      "depth_min": 0.0,
      "depth_max_im": 3,
      "negate": false
    }}
  ]
}}
```

### Example 7: Percentile query — thickness distribution
Query: "What are the p5, p50, and p95 of the cumulative thickness of Sand and Grus within 0 to 30 m depth?"
```json
{{
  "interpretation": "P5/P50/P95 of the cumulative thickness of Sand (class 1) and Grus (class 2) within 0–30 m depth.",
  "query_type": "percentile",
  "metric": {{
    "im": 2,
    "classes": [1, 2],
    "thickness_mode": "cumulative",
    "depth_min": 0.0,
    "depth_max": 30.0
  }},
  "percentiles": [5, 50, 95]
}}
```

### Example 8: Percentile query — cross-model depth bound
Query: "What is the typical (median) thickness of Sand and Grus above the water table?"
```json
{{
  "interpretation": "P5/P50/P95 of the cumulative thickness of Sand (class 1) and Grus (class 2) above the water table (depth bounded per realization by im=3).",
  "query_type": "percentile",
  "metric": {{
    "im": 2,
    "classes": [1, 2],
    "thickness_mode": "cumulative",
    "depth_min": 0.0,
    "depth_max_im": 3
  }},
  "percentiles": [5, 50, 95]
}}
```

## Instructions

- Respond with ONLY a valid JSON object. No markdown fences, no extra commentary.
- Always include "interpretation" (1–2 sentences) and "query_type" ("probability" or "percentile").
- Use only the model indices (im) and class IDs listed under Available Prior Models above.
- If the query cannot be expressed with the available schema and models, respond with exactly:
  UNSUPPORTED: <brief reason>
- Do not invent class IDs or model indices that are not listed above.
"""
    return prompt


def _litellm_extra(model):
    """Return extra_body kwargs for litellm.completion to disable thinking on Ollama models."""
    if model.startswith('ollama'):
        return {'extra_body': {'think': False}}
    return {}


def query_from_text(text, f_prior_h5, model='anthropic/claude-sonnet-4-6', api_key=None, max_tokens=4096, verbose=False):
    """
    Translate a natural-language query into a query dict using an LLM.

    Uses LiteLLM to interpret the user's text query in the context of the
    available prior models and the integrate query schema, returning a query
    dict and a plain-English interpretation of what the LLM understood.

    Parameters
    ----------
    text : str
        Natural language description of the query, e.g.
        "What is the probability that cumulative clay thickness exceeds 10 m?".
    f_prior_h5 : str
        Path to the prior HDF5 file. Model metadata (class names, depth ranges,
        discrete/continuous type) is read automatically and included in the
        LLM prompt so the model knows what constraints are valid.
    model : str, optional
        LiteLLM model string (default: 'anthropic/claude-sonnet-4-6'). Any
        LiteLLM-supported model works, e.g. 'openai/gpt-4o'.
    api_key : str, optional
        Provider API key. If None, the relevant environment variable
        (e.g. ANTHROPIC_API_KEY) is used.
    verbose : bool, optional
        If True, print the system prompt and LLM response for inspection.

    Returns
    -------
    query_dict : dict
        Query dict ready to pass to ig.query(f_post_h5, query_dict).
    interpretation : str
        Plain English confirmation of what the LLM understood the query to mean.
        Check this before running ig.query() to catch misunderstandings cheaply.
    system_prompt : str
        The full system prompt sent to the LLM. Useful for inspection and debugging.

    Raises
    ------
    ImportError
        If the litellm package is not installed.
    ValueError
        If the LLM reports the query is unsupported, or if the response
        cannot be parsed as valid JSON.

    Notes
    -----
    Requires either the api_key parameter or the relevant provider environment
    variable to be set. Install the dependency with: pip install litellm

    Examples
    --------
    >>> import integrate as ig
    >>> query_dict, interpretation, system_prompt = ig.query_from_text(
    ...     "Probability that cumulative clay thickness > 10 m within 0-30 m",
    ...     f_prior_h5='prior.h5',
    ...     api_key='sk-ant-...',
    ... )
    >>> print(interpretation)
    >>> P, meta = ig.query('posterior.h5', query_dict)
    >>> ig.query_plot(P, meta)
    """
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "The 'litellm' package is required for query_from_text(). "
            "Install it with: pip install litellm"
        )

    system_prompt = _build_llm_system_prompt(f_prior_h5)

    if verbose:
        print("=== SYSTEM PROMPT ===")
        print(system_prompt)
        print("=== USER TEXT ===")
        print(text)

    def _strip_fences(s):
        if s.startswith("```"):
            s = s.split("\n", 1)[-1]
            if s.endswith("```"):
                s = s.rsplit("```", 1)[0].strip()
        return s

    response_obj = litellm.completion(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        api_key=api_key,
        **_litellm_extra(model),
    )

    msg = response_obj.choices[0].message
    response = _strip_fences((msg.content or '').strip())

    if not response:
        raise ValueError(
            f"Model '{model}' returned empty content. "
            "If this is a thinking model (e.g. Qwen3, DeepSeek-R1), ensure /no_think "
            "is in the prompt or increase max_tokens."
        )

    if verbose:
        print("=== LLM RESPONSE ===")
        print(response)

    if response.startswith("UNSUPPORTED:"):
        reason = response[len("UNSUPPORTED:"):].strip()
        raise ValueError(f"Query cannot be expressed with the current schema: {reason}")

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        if verbose:
            print("=== JSON PARSE FAILED — RETRYING ===")
        retry_obj = litellm.completion(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
                {"role": "assistant", "content": response},
                {"role": "user", "content": "Your response was not valid JSON. Output ONLY the JSON object with no extra text, no markdown fences, no explanation."},
            ],
            api_key=api_key,
            **_litellm_extra(model),
        )
        response = _strip_fences(retry_obj.choices[0].message.content.strip())
        if verbose:
            print("=== RETRY RESPONSE ===")
            print(response)
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as e2:
            raise ValueError(
                f"LLM response could not be parsed as JSON after retry: {e2}\nRaw response:\n{response}"
            )

    interpretation = parsed.pop('interpretation', '')
    query_type = parsed.pop('query_type', 'probability')
    print(f"Interpretation: {interpretation}")

    # Build the canonical query dict based on query_type
    if query_type == 'percentile':
        query_dict = {
            'metric': parsed.get('metric', {}),
            'percentiles': parsed.get('percentiles', [5, 50, 95]),
        }
    else:
        # probability (default, backward compatible)
        query_dict = {'constraints': parsed.get('constraints', [])}

    return query_dict, interpretation, system_prompt


def title_from_json(file_json, f_prior_h5=None, model='anthropic/claude-sonnet-4-6',
                    api_key=None, showInfo=1):
    """
    Return a plain-language description of what a query JSON dict will do.

    Uses an LLM to produce a short human-readable summary suitable for a figure
    title or log message. If the LLM is unavailable (missing package, no API key,
    network error), returns an empty string.

    Parameters
    ----------
    file_json : str or dict
        Path to a query JSON file, or a query dict directly (e.g. from
        ``ig.load_query()``).
    f_prior_h5 : str, optional
        Path to the prior HDF5 file. When provided, real model names, depth
        ranges, and class labels are included in the prompt so the description
        uses geological names instead of numeric model/class IDs.
    model : str, optional
        LiteLLM model string (default: 'anthropic/claude-sonnet-4-6').
    api_key : str, optional
        Provider API key. If None, the relevant environment variable is used.
    showInfo : int, optional
        0 = silent; 1 = print a message when the LLM cannot be reached (default);
        2 = also print the exception detail.

    Returns
    -------
    description : str
        One-sentence plain-English summary of the query, or an empty string if
        the LLM could not be reached.

    Examples
    --------
    >>> description = ig.title_from_json('my_query.json')
    >>> description = ig.title_from_json('my_query.json', f_prior_h5='prior.h5')
    >>> query = ig.load_query('query_ex1.json')
    >>> title = ig.title_from_json(query, f_prior_h5='prior.h5')
    >>> title = ig.title_from_json(query, showInfo=0)  # silent on failure
    """
    try:
        import litellm
    except ImportError:
        if showInfo >= 1:
            print("[ig.title_from_json] LLM unavailable: 'litellm' package not installed "
                  "(pip install litellm). Returning empty description.")
        return ''

    if isinstance(file_json, str):
        try:
            with open(file_json, 'r') as fh:
                query_dict = json.load(fh)
        except Exception as e:
            if showInfo >= 1:
                print(f"[ig.title_from_json] Could not read query file: {e}")
            return ''
    else:
        query_dict = dict(file_json)

    # Collect the im indices referenced by this query
    if 'constraints' in query_dict:
        items = query_dict['constraints']
    elif 'metric' in query_dict:
        items = [query_dict['metric']]
    else:
        items = []
    needed_ims = _collect_needed_ims(items) if items else set()

    # Build an optional model-context block from the prior file
    model_context = ''
    if f_prior_h5 and needed_ims:
        try:
            lines = ['Available models referenced in this query:']
            for im in sorted(needed_ims):
                info = get_prior_model_info(f_prior_h5, im)
                z = info['z']
                depth_min, depth_max = float(z[0]), float(z[-1])
                is_scalar = (depth_max - depth_min) == 0 or len(z) - 1 == 0
                name = info['name']
                if is_scalar:
                    kind = 'scalar-discrete' if info['is_discrete'] else 'scalar'
                    lines.append(f"  im={im}: '{name}' ({kind})")
                elif info['is_discrete']:
                    lines.append(f"  im={im}: '{name}' (discrete), depth {depth_min:.1f}–{depth_max:.1f} m")
                    if info['class_id'] is not None and info['class_name'] is not None:
                        ids = info['class_id'].flatten()
                        names = info['class_name'].flatten()
                        for cid, cname in zip(ids, names):
                            lines.append(f"    class {int(cid)} = {cname}")
                else:
                    lines.append(f"  im={im}: '{name}' (continuous), depth {depth_min:.1f}–{depth_max:.1f} m")
            model_context = '\n'.join(lines)
        except Exception:
            model_context = ''

    system_prompt = (
        "You are a geophysics assistant for the INTEGRATE probabilistic inversion module. "
        "The user will provide a query dict in JSON format. "
        "Suggest a short figure title (max ~10 words) for the result this query produces. "
        "Use geological language and real model/class names where available. "
        "Write the title in title case. Do not start with 'Computes', 'Shows', or 'Displays'. "
        "Do not include JSON syntax. Reply with only the title — no preamble, no full stop."
    )
    if model_context:
        system_prompt += f"\n\n{model_context}"

    try:
        response_obj = litellm.completion(
            model=model,
            max_tokens=128,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(query_dict)},
            ],
            api_key=api_key,
            **_litellm_extra(model),
        )
        description = (response_obj.choices[0].message.content or '').strip()
        return description
    except Exception as e:
        if showInfo >= 1:
            print(f"[ig.title_from_json] LLM call failed — returning empty description. "
                  f"Use ig.query_test_llm() to diagnose. (model='{model}')")
        if showInfo >= 2:
            print(f"  Detail: {e}")
        return ''


def query_test_llm(model='anthropic/claude-sonnet-4-6', api_key=None, verbose=1):
    """
    Test whether a given LLM model and API key are working correctly.

    Sends a minimal JSON-generation prompt and checks that the response is
    valid JSON. Prints a summary and returns a status dict.

    Parameters
    ----------
    model : str, optional
        LiteLLM model string (default: 'anthropic/claude-sonnet-4-6').
    api_key : str, optional
        Provider API key. If None, the relevant environment variable is used.
    verbose : int, optional
        0 = silent, 1 = summary only (default), 2 = full response included.

    Returns
    -------
    result : dict
        Keys: 'ok' (bool), 'model', 'response' (str or None), 'error' (str or None).
    """
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "The 'litellm' package is required. Install it with: pip install litellm"
        )

    test_prompt = 'Reply with exactly this JSON and nothing else: {"status": "ok"}'
    result = {'ok': False, 'model': model, 'response': None, 'error': None}

    try:
        response_obj = litellm.completion(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": test_prompt}],
            api_key=api_key,
            **_litellm_extra(model),
        )
        msg = response_obj.choices[0].message
        raw = (msg.content or '').strip()
        # strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0].strip()
        result['response'] = raw

        if not raw:
            reasoning = getattr(msg, 'reasoning_content', None)
            hint = " (model returned empty content — it may be a thinking model with all tokens used for reasoning)" if not reasoning else f" (content was empty; reasoning_content present, length {len(reasoning)})"
            raise ValueError(f"Empty response from model{hint}")

        json.loads(raw)  # validate JSON
        result['ok'] = True
        if verbose >= 1:
            print(f"[query_test_llm] OK — model '{model}' responded with valid JSON.")
        if verbose >= 2:
            print(f"  Response: {raw}")
    except Exception as e:
        result['error'] = str(e)
        if verbose >= 1:
            print(f"[query_test_llm] FAILED — model '{model}'")
            print(f"  Error: {e}")
        if verbose >= 2 and result['response']:
            print(f"  Raw response: {result['response']}")

    return result
