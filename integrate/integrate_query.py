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
    depth_min : float or None
        Lower depth bound [m].
    depth_max : float or None
        Upper depth bound [m].

    Returns
    -------
    t : ndarray (N_depth - 1,)
        Clipped thickness for each layer [m].
    mask : ndarray (N_depth - 1,) bool
        True where clipped thickness > 0.
    """
    n_layers = len(z) - 1
    top = z[:n_layers]
    bot = z[1:]
    lo = depth_min if depth_min is not None else -np.inf
    hi = depth_max if depth_max is not None else np.inf
    t = np.minimum(bot, hi) - np.maximum(top, lo)
    t = np.maximum(t, 0.0)
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
    for i in np.where(has_true)[0]:
        j = first_idx[i]
        while j < condition.shape[1] and condition[i, j]:
            metric[i] += t[j]
            j += 1
    return metric


def _evaluate_constraint(M_samples, z, constraint):
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

    Returns
    -------
    valid : ndarray (N_samples,) bool
        True for each realization that satisfies the constraint.
    """
    depth_min = constraint.get('depth_min', None)
    depth_max = constraint.get('depth_max', None)
    t, layer_mask = _get_clipped_thickness(z, depth_min, depth_max)

    # Property values per layer (column i = layer top z[i] to z[i+1])
    n_layers = len(z) - 1
    M_sel = M_samples[:, :n_layers][:, layer_mask]   # (N_samples, N_sel)
    t_sel = t[layer_mask]                             # (N_sel,)

    # Per-layer boolean condition
    if 'classes' in constraint:
        # Discrete: match any of the specified class IDs
        condition = np.isin(np.round(M_sel).astype(int), constraint['classes'])
    else:
        # Continuous: compare value against threshold
        v_cmp = constraint.get('value_comparison', '<')
        v_thr = constraint.get('value_threshold', 0.0)
        condition = M_sel < v_thr if v_cmp == '<' else M_sel > v_thr

    # Compute thickness metric
    mode = constraint.get('thickness_mode', 'cumulative')
    if mode == 'cumulative':
        metric = (condition * t_sel).sum(axis=1)      # (N_samples,)
    else:  # first_occurrence
        metric = _first_occurrence_thickness(condition, t_sel)

    # Apply thickness comparison
    t_cmp = constraint.get('thickness_comparison', '>')
    t_thr = constraint.get('thickness_threshold', 0.0)
    _ops = {
        '>':  lambda m: m > t_thr,
        '<':  lambda m: m < t_thr,
        '>=': lambda m: m >= t_thr,
        '<=': lambda m: m <= t_thr,
    }
    valid = _ops.get(t_cmp, _ops['>'])(metric)

    return ~valid if constraint.get('negate', False) else valid


def query(f_post_h5, query_dict):
    """
    Compute per-data-point probability that posterior realizations satisfy a query.

    Parameters
    ----------
    f_post_h5 : str
        Path to the posterior HDF5 file (output of integrate_rejection).
    query_dict : str or dict
        Path to a JSON file, or a dict defining the query.

    Returns
    -------
    P : ndarray (N_data,)
        Probability [0, 1] for each data location.
    meta : dict
        Metadata with keys:

        - 'X', 'Y' : coordinate arrays (or None)
        - 'N_data', 'N_post' : data location count and samples per location
        - 'i_use' : ndarray (N_data, N_post), all posterior indices from f_post_h5
        - 'i_use_query' : list of N_data arrays, indices that match the query
          for each data location

    Notes
    -----
    Constraints are applied sequentially (logical AND). A realization is
    counted as satisfying the query only if it passes every constraint.
    The probability is the fraction of accepted realizations.

    The prior file path is read from the 'f5_prior' attribute of the
    posterior file. Coordinates (UTMX, UTMY) are read from the posterior
    file if present, otherwise from the data file ('f5_data' attribute).

    Examples
    --------
    >>> query_def = {
    ...     "constraints": [{
    ...         "im": 2, "classes": [2],
    ...         "thickness_mode": "cumulative",
    ...         "thickness_comparison": ">",
    ...         "thickness_threshold": 10.0,
    ...         "depth_min": 0.0, "depth_max": 30.0, "negate": False
    ...     }]
    ... }
    >>> P, meta = query('f_post.h5', query_def)
    """
    if isinstance(query_dict, str):
        with open(query_dict, 'r') as fh:
            query_dict = json.load(fh)

    constraints = query_dict['constraints']

    # Load posterior metadata and acceptance indices
    with h5py.File(f_post_h5, 'r') as f:
        i_use = f['i_use'][:]                          # (N_data, N_post)
        f_prior_h5 = str(f.attrs.get('f5_prior', ''))
        f_data_h5 = str(f.attrs.get('f5_data', ''))
        X = f['UTMX'][:] if 'UTMX' in f else None
        Y = f['UTMY'][:] if 'UTMY' in f else None

    if not f_prior_h5:
        raise ValueError("Posterior file missing 'f5_prior' attribute.")
    if not os.path.isfile(f_prior_h5):
        raise FileNotFoundError(f"Prior file not found: {f_prior_h5}")

    # Fall back to data file for coordinates if needed
    if (X is None or Y is None) and f_data_h5 and os.path.isfile(f_data_h5):
        with h5py.File(f_data_h5, 'r') as f:
            if X is None and 'UTMX' in f:
                X = f['UTMX'][:]
            if Y is None and 'UTMY' in f:
                Y = f['UTMY'][:]

    N_data, N_post = i_use.shape

    # Pre-load all required prior model arrays (once per model index)
    prior_models = {}
    for c in constraints:
        im = c['im']
        if im not in prior_models:
            key = f'M{im}'
            with h5py.File(f_prior_h5, 'r') as f:
                M = f[key][:]
                z = f[key].attrs['x'].astype(float)
                is_discrete = bool(f[key].attrs.get('is_discrete', 0))
            prior_models[im] = (M, z, is_discrete)

    # Evaluate constraints for each data location
    P = np.zeros(N_data)
    i_use_query = []
    for i in tqdm(range(N_data), desc='Evaluating query', unit='location'):
        idx = i_use[i]                             # (N_post,) indices into prior
        valid = np.ones(N_post, dtype=bool)
        for c in constraints:
            M, z, _ = prior_models[c['im']]
            valid &= _evaluate_constraint(M[idx, :], z, c)
        P[i] = valid.mean()
        i_use_query.append(idx[valid])             # Store matching indices

    meta = {
        'X': X,
        'Y': Y,
        'N_data': N_data,
        'N_post': N_post,
        'i_use': i_use,
        'i_use_query': i_use_query
    }
    return P, meta


def query_plot(P, meta, ip=None, query_dict=None, f_prior_h5=None, f_post_h5=None, title=None):
    """
    Plot query results and optionally detailed model visualization for a data point.

    Always displays a probability map showing P(x, y). If ip is provided along with
    query_dict and (f_prior_h5 or f_post_h5), also displays a detailed visualization of
    the posterior models and query-matching models for that specific data location.

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
        Custom title for the probability map. If None, uses 'Query Probability Map'.

    Examples
    --------
    >>> P, meta = query(f_post_h5, query_def)
    >>> query_plot(P, meta)  # Just probability map
    >>> query_plot(P, meta, title='Custom Query Title')  # Custom title
    >>> query_plot(P, meta, ip=1000, query_dict=query_def, f_post_h5='posterior.h5')
    >>> query_plot(P, meta, ip=1000, query_dict=query_def, f_prior_h5='prior.h5')
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

    # Always plot probability map
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    # Plot black dots underneath to make P=0 (white) visible
    ax.scatter(X, Y, c='black', s=2, alpha=0.5)
    sc = ax.scatter(X, Y, c=P, cmap='hot_r', vmin=0, vmax=1, s=1)
    if ip is not None and X is not None and Y is not None:
        ax.plot(X[ip], Y[ip], 'kx', markersize=12, markeredgewidth=2, label=f'Point {ip}')
        ax.legend()
    plt.colorbar(sc, label='Probability')
    ax.set_xlabel('UTMX [m]')
    ax.set_ylabel('UTMY [m]')
    ax.set_title(title if title is not None else 'Query Probability Map')
    ax.set_aspect('equal')
    # add grid lines
    ax.grid(True, linestyle='--', alpha=0.5)
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

    plt.show()


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
