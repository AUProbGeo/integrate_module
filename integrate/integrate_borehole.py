
def compute_P_obs_discrete(depth_top=None, depth_bottom=None, lithology_obs=None, z=None, class_id=None, lithology_prob=0.8, P_prior=None, W=None):
    """
    Compute discrete observation probability matrix from depth intervals and lithology observations.

    This function creates a probability matrix where each depth point is assigned
    probabilities based on observed lithology classes within specified depth intervals.

    Parameters
    ----------
    depth_top : array-like, optional
        Array of top depths for each observation interval. Required if W is not provided.
    depth_bottom : array-like, optional
        Array of bottom depths for each observation interval. Required if W is not provided.
    lithology_obs : array-like, optional
        Array of observed lithology class IDs for each interval. Required if W is not provided.
    z : array-like, optional
        Array of depth/position values where probabilities are computed. Required if W is not provided.
    class_id : array-like, optional
        Array of unique class identifiers (e.g., [0, 1, 2] for 3 lithology types). Required if W is not provided.
    lithology_prob : float or array-like, optional
        Probability assigned to the observed class. Can be:
        - float: Same probability for all intervals (default is 0.8)
        - array: Array of probabilities, one per interval (must match length of lithology_obs)
    P_prior : ndarray, optional
        Prior probability matrix of shape (nclass, nm). If None, uses uniform distribution
        for depths not covered by observations. Default is None.
    W : dict, optional
        Well/borehole dictionary containing observation data. If provided, overrides
        the individual parameters. Expected keys:
        - 'depth_top': Array of top depths
        - 'depth_bottom': Array of bottom depths
        - 'class_obs': Array of observed class IDs (e.g., lithology, soil type)
        - 'class_prob': Probability or array of probabilities (optional, defaults to 0.8)
        - 'X': X coordinate of well location (optional, not used in this function)
        - 'Y': Y coordinate of well location (optional, not used in this function)
        Default is None.

    Returns
    -------
    P_obs : ndarray
        Probability matrix of shape (nclass, nm) where nclass is the number of classes
        and nm is the number of depth points. For each depth point covered by observations,
        the observed class gets probability lithology_prob and other classes share (1-lithology_prob).
        Depths not covered by any observation contain NaN or prior probabilities if provided.

    Examples
    --------
    >>> # Traditional usage with individual parameters
    >>> depth_top = [0, 10, 20]
    >>> depth_bottom = [10, 20, 30]
    >>> lithology_obs = [1, 2, 1]  # clay, sand, clay
    >>> z = np.arange(30)
    >>> class_id = [0, 1, 2]  # gravel, clay, sand
    >>> P_obs = compute_P_obs_discrete(depth_top, depth_bottom, lithology_obs, z, class_id)
    >>> print(P_obs.shape)  # (3, 30)

    >>> # With different probabilities per interval
    >>> lithology_prob = [0.9, 0.7, 0.85]  # Higher confidence in first interval
    >>> P_obs = compute_P_obs_discrete(depth_top, depth_bottom, lithology_obs, z, class_id, lithology_prob=lithology_prob)

    >>> # Using well dictionary (cleaner interface)
    >>> W = {'depth_top': [0, 10, 20], 'depth_bottom': [10, 20, 30],
    ...      'class_obs': [1, 2, 1], 'class_prob': [0.9, 0.7, 0.85],
    ...      'X': 543000.0, 'Y': 6175800.0}
    >>> P_obs = compute_P_obs_discrete(z=z, class_id=class_id, W=W)
    """
    import numpy as np

    # Override parameters with W dictionary if provided
    if W is not None:
        if 'depth_top' in W:
            depth_top = W['depth_top']
        if 'depth_bottom' in W:
            depth_bottom = W['depth_bottom']
        if 'class_obs' in W:
            lithology_obs = W['class_obs']
        if 'class_prob' in W:
            lithology_prob = W['class_prob']
        # Note: X and Y coordinates stored for reference but not used in this function
        # X_well = W.get('X', None)
        # Y_well = W.get('Y', None)

    # Validate required parameters
    if depth_top is None or depth_bottom is None or lithology_obs is None:
        raise ValueError("depth_top, depth_bottom, and lithology_obs must be provided either as arguments or in W dictionary")
    if z is None or class_id is None:
        raise ValueError("z and class_id are required parameters")

    nm = len(z)
    nclass = len(class_id)

    # Convert lithology_prob to array if it's a scalar
    lithology_prob_array = np.atleast_1d(lithology_prob)
    if len(lithology_prob_array) == 1:
        # Scalar case: broadcast to all intervals
        lithology_prob_array = np.full(len(lithology_obs), lithology_prob_array[0])
    elif len(lithology_prob_array) != len(lithology_obs):
        raise ValueError(f"lithology_prob array length ({len(lithology_prob_array)}) must match lithology_obs length ({len(lithology_obs)})")

    # Initialize with NaN or prior
    if P_prior is not None:
        P_obs = P_prior.copy()
    else:
        P_obs = np.zeros((nclass, nm)) * np.nan

    # Loop through each depth point
    for im in range(nm):
        # Loop through each observation interval
        for i in range(len(depth_top)):
            # Check if current depth is within this interval
            if z[im] >= depth_top[i] and z[im] < depth_bottom[i]:
                # Get the probability for this specific interval
                lithology_prob_i = lithology_prob_array[i]
                # Compute probability for non-hit classes
                P_nohit = (1 - lithology_prob_i) / (nclass - 1)

                # Assign probabilities for all classes
                for ic in range(nclass):
                    if class_id[ic] == lithology_obs[i]:
                        P_obs[ic, im] = lithology_prob_i
                    else:
                        P_obs[ic, im] = P_nohit

    return P_obs

def _compute_mode_sequential(M_lithology, z, depth_top, depth_bottom, nl, showInfo=1):
    """
    Sequential computation of mode lithology for each realization and depth interval.

    This is the original implementation extracted for maintainability.
    """
    import numpy as np
    from tqdm import tqdm
    import time

    nreal = len(M_lithology)
    lithology_mode = np.zeros((nreal, nl), dtype=int)

    # Show info based on showInfo level
    if showInfo == 1:
        print(f'compute_P_obs_sparse: Processing {nreal} realizations, {nl} intervals')

    # Start timing
    t_start = time.time()

    # Extract mode lithology for each realization and depth interval
    iterator = np.arange(nreal)
    if showInfo > 1:
        iterator = tqdm(iterator, desc='compute_P_obs_sparse')

    for im in iterator:
        M_test = M_lithology[im]
        for i in range(len(depth_top)):
            z_top = depth_top[i]
            z_bottom = depth_bottom[i]
            id_top = np.argmin(np.abs(z - z_top))
            id_bottom = np.argmin(np.abs(z - z_bottom))

            if id_top == id_bottom:
                lithology_layer = M_test[id_top]
                lithology_mode_layer = lithology_layer
            else:
                lithology_layer = M_test[id_top:id_bottom]
                # Find the most frequent lithology in this layer
                values, counts = np.unique(lithology_layer, return_counts=True)
                lithology_mode_layer = values[np.argmax(counts)]

            lithology_mode[im, i] = lithology_mode_layer

    # Show timing information if showInfo > 1
    if showInfo > 1:
        t_elapsed = time.time() - t_start
        time_per_model = t_elapsed / nreal
        print(f'compute_P_obs_sparse: Total runtime: {t_elapsed:.2f} seconds')
        print(f'compute_P_obs_sparse: Time per model: {time_per_model*1000:.2f} ms')

    return lithology_mode

def _compute_mode_worker(args):
    """
    Worker function for parallel processing of lithology mode computation.

    Processes a chunk of realizations assigned to this worker.
    """
    import numpy as np
    import integrate as ig

    # Unpack arguments
    chunk_indices, shared_memory_refs, z, depth_top, depth_bottom, nl = args

    # Reconstruct shared array
    [M_lithology], worker_shm_objects = ig.reconstruct_shared_arrays(shared_memory_refs)

    try:
        # Initialize output for this chunk
        lithology_mode_chunk = np.zeros((len(chunk_indices), nl), dtype=int)

        # Process assigned realizations
        for local_idx, im in enumerate(chunk_indices):
            M_test = M_lithology[im]
            for i in range(nl):
                z_top = depth_top[i]
                z_bottom = depth_bottom[i]
                id_top = np.argmin(np.abs(z - z_top))
                id_bottom = np.argmin(np.abs(z - z_bottom))

                if id_top == id_bottom:
                    lithology_mode_chunk[local_idx, i] = M_test[id_top]
                else:
                    lithology_layer = M_test[id_top:id_bottom]
                    values, counts = np.unique(lithology_layer, return_counts=True)
                    lithology_mode_chunk[local_idx, i] = values[np.argmax(counts)]

        return lithology_mode_chunk

    finally:
        # Close shared memory in worker
        for shm in worker_shm_objects:
            shm.close()

def _compute_mode_parallel(M_lithology, z, depth_top, depth_bottom, nl, Ncpu, showInfo=1):
    """
    Parallel computation of mode lithology using multiprocessing.

    Splits realizations across worker processes and uses shared memory for efficiency.
    """
    import numpy as np
    import multiprocessing
    from multiprocessing import Pool
    import integrate as ig
    import time
    from tqdm import tqdm

    # Setup
    if Ncpu < 1:
        Ncpu = multiprocessing.cpu_count()

    nreal = len(M_lithology)

    # Show info based on showInfo level
    if showInfo == 1:
        print(f'compute_P_obs_sparse: Processing {nreal} realizations, {nl} intervals (parallel, {Ncpu} CPUs)')
    elif showInfo > 1:
        print(f'compute_P_obs_sparse: Processing {nreal} realizations, {nl} intervals')
        print(f'compute_P_obs_sparse: Using {Ncpu} CPU cores in parallel mode')

    # Start timing
    t_start = time.time()

    # Create shared memory for M_lithology
    shared_memory_refs, shm_objects = ig.create_shared_memory([M_lithology])

    try:
        # Split realizations into many small chunks for better progress tracking
        # More chunks = more frequent progress updates and better load balancing
        min_chunk_size = 100   # Minimum realizations per chunk
        max_chunks = 500       # Cap to avoid excessive overhead
        n_chunks = min(max(nreal // min_chunk_size, Ncpu), max_chunks)

        if showInfo > 1:
            print(f'compute_P_obs_sparse: Splitting into {n_chunks} chunks for {Ncpu} workers')

        realization_chunks = np.array_split(np.arange(nreal), n_chunks)

        # Create worker arguments (include small arrays directly)
        worker_args = [
            (chunk_indices, shared_memory_refs, z, depth_top, depth_bottom, nl)
            for chunk_indices in realization_chunks
        ]

        # Execute in parallel
        with Pool(processes=Ncpu) as p:
            if showInfo > 1:
                # Use imap to get results as they complete and show progress
                results = list(tqdm(
                    p.imap(_compute_mode_worker, worker_args),
                    total=len(worker_args),
                    desc='compute_P_obs_sparse (parallel chunks)',
                    unit='chunk'
                ))
            else:
                # Use regular map without progress tracking
                results = p.map(_compute_mode_worker, worker_args)

        # Concatenate results
        lithology_mode = np.concatenate(results, axis=0)

        # Show timing information if showInfo > 1
        if showInfo > 1:
            t_elapsed = time.time() - t_start
            time_per_model = t_elapsed / nreal
            print(f'compute_P_obs_sparse: Total runtime: {t_elapsed:.2f} seconds')
            print(f'compute_P_obs_sparse: Time per model: {time_per_model*1000:.2f} ms')
            print(f'compute_P_obs_sparse: Speedup with {Ncpu} cores vs sequential: ~{Ncpu*0.7:.1f}x (estimated)')

        return lithology_mode

    finally:
        # Cleanup shared memory
        ig.cleanup_shared_memory(shm_objects)

def welllog_compute_P_obs_class_mode(M_lithology=None, depth_top=None, depth_bottom=None, lithology_obs=None, z=None, class_id=None, lithology_prob=0.8, W=None, parallel=False, Ncpu=-1, showInfo=1):
    """
    Compute observation probability matrix from well log class observations by extracting mode class from prior models.

    This function processes discrete class models (e.g., lithology) from a prior ensemble to create
    well log observations. For each depth interval, it finds the most frequent (mode) class within
    that interval from each prior model, then creates a probability matrix based on how well these
    modes match the observed classes.

    **Simplified Usage**: When called with only `W=W`, the function returns just the probability
    matrix `P_obs` without computing class mode, and class_mode is returned as None.

    Parameters
    ----------
    M_lithology : ndarray, optional
        Array of lithology models from prior ensemble, shape (nreal, nz) where nreal is the
        number of realizations and nz is the number of depth points. If None, only P_obs
        is computed and lithology_mode is returned as None. Default is None.
    depth_top : array-like, optional
        Array of top depths for each observation interval. Required if W is not provided.
    depth_bottom : array-like, optional
        Array of bottom depths for each observation interval. Required if W is not provided.
    lithology_obs : array-like, optional
        Array of observed lithology class IDs for each interval. Required if W is not provided.
    z : array-like, optional
        Array of depth/position values corresponding to M_lithology depth discretization.
        Required if M_lithology is provided and W does not contain depth information.
    class_id : array-like, optional
        Array of unique class identifiers (e.g., [0, 1, 2] for 3 lithology types).
        Required if W is not provided.
    lithology_prob : float or array-like, optional
        Probability assigned to the observed class. Can be:
        - float: Same probability for all intervals (default is 0.8)
        - array: Array of probabilities, one per interval (must match length of lithology_obs)
    W : dict, optional
        Well/borehole dictionary containing observation data. If provided, overrides
        the individual parameters. Expected keys:
        - 'depth_top': Array of top depths
        - 'depth_bottom': Array of bottom depths
        - 'class_obs': Array of observed class IDs (e.g., lithology, soil type)
        - 'class_prob': Probability or array of probabilities (optional, defaults to 0.8)
        - 'X': X coordinate of well location (optional, not used in this function)
        - 'Y': Y coordinate of well location (optional, not used in this function)
        Default is None.
    parallel : bool, optional
        Enable parallel processing for large ensembles. Default is False.
        When True and parallel processing is available, distributes realization
        processing across multiple CPU cores for significant speedup.
        Recommended for N > 10,000 realizations. Only used when M_lithology is provided.
    Ncpu : int, optional
        Number of CPU cores to use for parallel processing. Default is -1 (auto-detect).
        Only used when parallel=True and M_lithology is provided.
    showInfo : int, optional
        Control information output level. Default is 1.
        - 0: No information printed
        - 1: Single line info (number of realizations, intervals)
        - >1: Progress bar with tqdm, runtime statistics, and time per model
        Only applies when M_lithology is provided.

    Returns
    -------
    P_obs : ndarray
        Probability matrix of shape (nclass, n_obs) where nclass is the number of classes
        and n_obs is the number of observation intervals. Each column represents the
        probability distribution for one depth interval.
    class_mode : ndarray or None
        If M_lithology is provided: Array of mode class values extracted from prior models,
        shape (nreal, n_obs). For each realization and observation interval, contains the most
        frequent class ID within that depth range.
        If M_lithology is None: Returns None.

    Examples
    --------
    >>> # Full usage: Load prior lithology models and compute mode
    >>> M_lithology = f_prior['M2'][:]  # Shape: (100000, 50)
    >>> z = np.linspace(0, 100, 50)
    >>> class_id = [0, 1, 2]  # sand, clay, gravel
    >>>
    >>> # Define observations
    >>> depth_top = [0, 20, 40]
    >>> depth_bottom = [20, 40, 60]
    >>> lithology_obs = [1, 0, 1]  # clay, sand, clay
    >>>
    >>> # Compute well log observations with class mode
    >>> P_obs, class_mode = welllog_compute_P_obs_class_mode(M_lithology, depth_top, depth_bottom,
    ...                                                       lithology_obs, z, class_id)
    >>> print(P_obs.shape)  # (3, 3) - 3 classes, 3 observations
    >>> print(class_mode.shape)  # (100000, 3) - mode for each realization and interval

    >>> # Simplified usage: Only compute P_obs using well dictionary
    >>> W = {'depth_top': [0, 20, 40], 'depth_bottom': [20, 40, 60],
    ...      'class_obs': [1, 0, 1], 'class_prob': [0.9, 0.8, 0.85],
    ...      'X': 543000.0, 'Y': 6175800.0}
    >>> P_obs, class_mode = welllog_compute_P_obs_class_mode(W=W, class_id=class_id)
    >>> print(P_obs.shape)  # (3, 3) - 3 classes, 3 observations
    >>> print(class_mode)  # None (no prior models provided)
    >>>
    >>> # Using parallel processing for large ensembles
    >>> P_obs, class_mode = welllog_compute_P_obs_class_mode(M_lithology, depth_top, depth_bottom,
    ...                                                       lithology_obs, z, class_id,
    ...                                                       parallel=True, Ncpu=8)
    >>>
    >>> # Control output verbosity
    >>> P_obs, class_mode = welllog_compute_P_obs_class_mode(M_lithology, ..., showInfo=0)  # Silent
    >>> P_obs, class_mode = welllog_compute_P_obs_class_mode(M_lithology, ..., showInfo=1)  # Single line info
    >>> P_obs, class_mode = welllog_compute_P_obs_class_mode(M_lithology, ..., showInfo=2)  # Progress bar + timing

    Notes
    -----
    The function extracts class mode for each depth interval by:
    1. Finding depth indices corresponding to interval boundaries
    2. Extracting class values within the interval
    3. Computing the most frequent (mode) class
    4. Assigning probabilities based on match with observed class

    This approach is suitable for well log observations where each interval represents
    the dominant class within that depth range, rather than the full depth profile.

    **Simplified Mode** (M_lithology=None):
    When M_lithology is not provided, the function only computes the P_obs probability matrix
    from the observed class data. This is useful when you only need the probability
    representation of observations without extracting mode class from prior models.

    Parallel processing uses shared memory for M_lithology array to minimize memory overhead.
    Expected speedup: 4-8x on 8-core machines for large ensembles (N > 100,000 realizations).
    The parallel implementation distributes realizations across worker processes while
    maintaining identical results to the sequential version.

    Both sequential and parallel modes support timing information via showInfo parameter:
    - showInfo=0: Silent mode
    - showInfo=1: Single line summary (default)
    - showInfo>1: Detailed timing (progress bar for sequential, runtime stats for both)
    """
    import numpy as np

    # Override parameters with W dictionary if provided
    if W is not None:
        if 'depth_top' in W:
            depth_top = W['depth_top']
        if 'depth_bottom' in W:
            depth_bottom = W['depth_bottom']
        if 'class_obs' in W:
            lithology_obs = W['class_obs']
        if 'class_prob' in W:
            lithology_prob = W['class_prob']
        # Note: X and Y coordinates stored for reference but not used in this function
        # X_well = W.get('X', None)
        # Y_well = W.get('Y', None)

    # Validate required parameters
    if depth_top is None or depth_bottom is None or lithology_obs is None:
        raise ValueError("depth_top, depth_bottom, and lithology_obs must be provided either as arguments or in W dictionary")
    if class_id is None:
        raise ValueError("class_id is required parameter")

    # Validate z is provided when M_lithology is provided
    if M_lithology is not None and z is None:
        raise ValueError("z is required when M_lithology is provided")

    # Get dimensions
    nclass = len(class_id)
    nl = len(lithology_obs)
    n_obs = nl

    # Convert lithology_prob to array if it's a scalar
    lithology_prob_array = np.atleast_1d(lithology_prob)
    if len(lithology_prob_array) == 1:
        # Scalar case: broadcast to all intervals
        lithology_prob_array = np.full(n_obs, lithology_prob_array[0])
    elif len(lithology_prob_array) != n_obs:
        raise ValueError(f"lithology_prob array length ({len(lithology_prob_array)}) must match lithology_obs length ({n_obs})")

    # Compute mode lithology only if M_lithology is provided
    lithology_mode = None
    if M_lithology is not None:
        nreal = len(M_lithology)
        # Compute mode lithology using sequential or parallel method
        import integrate as ig
        if parallel and ig.use_parallel():
            # Parallel execution path
            lithology_mode = _compute_mode_parallel(M_lithology, z, depth_top, depth_bottom, nl, Ncpu, showInfo)
        else:
            # Sequential execution path (original implementation)
            lithology_mode = _compute_mode_sequential(M_lithology, z, depth_top, depth_bottom, nl, showInfo)

    # Convert observed lithologies to P_obs probabilities
    P_obs = np.zeros((nclass, n_obs)) * np.nan
    for i in range(n_obs):
        # Get the probability for this specific interval
        lithology_prob_i = lithology_prob_array[i]

        for j in range(nclass):
            if class_id[j] == lithology_obs[i]:
                P_obs[j, i] = lithology_prob_i
            else:
                P_obs[j, i] = (1 - lithology_prob_i) / (nclass - 1)

    return P_obs, lithology_mode

def rescale_P_obs_temperature(P_obs, T=1.0):
    """
    Rescale discrete observation probabilities by temperature and renormalize.

    This function applies temperature annealing to probability distributions by raising
    each probability to the power (1/T), then renormalizing each column (depth point)
    so that probabilities sum to 1. Higher temperatures (T > 1) flatten the distribution,
    while lower temperatures (T < 1) sharpen it.

    Parameters
    ----------
    P_obs : ndarray
        Probability matrix of shape (nclass, nm) where nclass is the number of classes
        and nm is the number of model parameters (e.g., depth points).
        Each column should represent a probability distribution over classes.
    T : float, optional
        Temperature parameter for annealing. Default is 1.0 (no scaling).
        - T = 1.0: No change (original probabilities)
        - T > 1.0: Flattens distribution (less certain)
        - T < 1.0: Sharpens distribution (more certain)
        - T → ∞: Approaches uniform distribution
        - T → 0: Approaches one-hot distribution

    Returns
    -------
    P_obs_scaled : ndarray
        Temperature-scaled and renormalized probability matrix of shape (nclass, nm).
        Each column sums to 1.0. NaN values in input are preserved in output.

    Examples
    --------
    >>> P_obs = np.array([[0.8, 0.6, 0.5],
    ...                   [0.1, 0.2, 0.3],
    ...                   [0.1, 0.2, 0.2]])
    >>> P_scaled = rescale_P_obs_temperature(P_obs, T=2.0)
    >>> print(P_scaled)  # More uniform distribution
    >>> P_scaled = rescale_P_obs_temperature(P_obs, T=0.5)
    >>> print(P_scaled)  # Sharper distribution

    Notes
    -----
    The temperature scaling follows the Boltzmann distribution:
        P_new(c) ∝ P_old(c)^(1/T)

    After scaling, each column (depth point) is renormalized:
        P_new(c) = P_new(c) / sum_c(P_new(c))

    This is commonly used in simulated annealing and rejection sampling to control
    the strength of discrete observations during Bayesian inference.
    """
    import numpy as np

    # Copy to avoid modifying the original
    P_obs_scaled = P_obs.copy()

    # Get shape
    nclass, nm = P_obs.shape

    # Apply temperature scaling: p^(1/T)
    # Handle special case where T=1 (no scaling needed)
    if T != 1.0:
        P_obs_scaled = np.power(P_obs_scaled, 1.0 / T)

    # Renormalize each column (each depth point) to sum to 1
    for im in range(nm):
        col_sum = np.nansum(P_obs_scaled[:, im])

        # Only renormalize if the sum is non-zero and not NaN
        if col_sum > 0 and not np.isnan(col_sum):
            P_obs_scaled[:, im] = P_obs_scaled[:, im] / col_sum

    return P_obs_scaled

def Pobs_to_datagrid(P_obs, X, Y, f_data_h5, r_data=10, r_dis=100, doPlot=False):
    """
    Convert point-based discrete probability observations to gridded data with distance-based weighting.

    This function distributes discrete probability observations (e.g., from a borehole) across
    a spatial grid using distance-based weighting. Observations at location (X, Y) are applied
    to nearby grid points with decreasing influence based on distance. Temperature annealing
    is used to reduce the strength of observations far from the source point.

    Parameters
    ----------
    P_obs : ndarray
        Probability matrix of shape (nclass, nm) where nclass is the number of classes
        and nm is the number of model parameters (e.g., depth points).
        Each column represents a probability distribution over discrete classes.
    X : float
        X coordinate (e.g., UTM Easting) of the observation point.
    Y : float
        Y coordinate (e.g., UTM Northing) of the observation point.
    f_data_h5 : str
        Path to HDF5 data file containing survey geometry (X, Y coordinates).
    r_data : float, optional
        Inner radius in meters within which observations have full strength.
        Default is 10 meters.
    r_dis : float, optional
        Outer radius in meters for distance-based weighting. Beyond this distance,
        observations are fully attenuated (temperature → ∞). Default is 100 meters.
    doPlot : bool, optional
        If True, creates diagnostic plots showing weight distributions.
        Default is False.

    Returns
    -------
    d_obs : ndarray
        Gridded observation data of shape (nd, nclass, nm) where nd is the number
        of spatial locations in the survey. Each location gets temperature-scaled
        probabilities based on distance from (X, Y).
    i_use : ndarray
        Binary mask of shape (nd, 1) indicating which grid points should be used
        (1) or ignored (0) in the inversion. Points with temperature < 100 are used.
    T_all : ndarray
        Temperature values of shape (nd, 1) applied to each grid point based on distance
        from the observation point.

    Notes
    -----
    The function uses distance-based temperature annealing:
    1. Computes distance-based weights using `get_weight_from_position()`
    2. Converts distance weight to temperature: T = 1 / w_dis
    3. Caps maximum temperature at 100 (very weak influence)
    4. For each grid point:

       - If T < 100: include point (i_use=1) and apply temperature scaling
       - If T ≥ 100: exclude point (i_use=0) and set observations to NaN

    Temperature scaling reduces probability certainty with distance:

    - T = 1 (close to observation): Original probabilities preserved
    - T > 1 (far from observation): Probabilities become more uniform
    - T ≥ 100 (very far): Observations effectively ignored

    Examples
    --------
    >>> # Borehole observation at specific location
    >>> P_obs = compute_P_obs_discrete(depth_top, depth_bottom, lithology, z, class_id)
    >>> X_well, Y_well = 543000.0, 6175800.0
    >>> d_obs, i_use, T_all = Pobs_to_datagrid(P_obs, X_well, Y_well, 'survey_data.h5',
    ...                                  r_data=10, r_dis=100)
    >>> # Write to data file
    >>> ig.save_data_multinomial(d_obs, i_use=i_use, id=2, f_data_h5='survey_data.h5')

    See Also
    --------
    rescale_P_obs_temperature : Temperature scaling function
    compute_P_obs_discrete : Create P_obs from depth intervals
    get_weight_from_position : Distance-based weighting function
    """
    import numpy as np
    import integrate as ig

    # Get grid dimensions from data file
    X_grid, Y_grid, _, _ = ig.get_geometry(f_data_h5)
    nd = len(X_grid)
    nclass, nm = P_obs.shape

    # Initialize output arrays
    i_use = np.zeros((nd, 1))
    d_obs = np.zeros((nd, nclass, nm)) * np.nan

    # Compute distance-based weights for all grid points
    w_combined, w_dis, w_data, i_use_from_func = ig.get_weight_from_position(
        f_data_h5, X, Y, r_data=r_data, r_dis=r_dis, doPlot=doPlot
    )

    # Convert distance weight to temperature
    # w_dis is 1 at observation point, decreases with distance
    # T = 1/w_dis means T increases with distance (weaker influence)
    #T_all = 1 / w_dis
    #T_all = 1 / w_data
    T_all = 1 / w_combined

    # Cap maximum temperature at 100 (beyond this, observation has negligible effect)
    T_all[T_all > 100] = 100

    # Apply temperature scaling to each grid point
    for ip in np.arange(nd):
        T = T_all[ip]

        # Only use points where temperature is reasonable (< 100)
        if T < 100:
            i_use[ip] = 1
            # Scale probabilities based on distance (higher T = more uniform distribution)
            P_obs_local = rescale_P_obs_temperature(P_obs, T=T)
            d_obs[ip, :, :] = P_obs_local
        # else: i_use[ip] = 0 and d_obs[ip] stays NaN

    return d_obs, i_use, T_all



def get_weight_from_position(f_data_h5,x_well=0,y_well=0, i_ref=-1, r_dis = 400, r_data=2, useLog=True, doPlot=False, plFile=None, showInfo=0):
    """Calculate weights based on distance and data similarity to a reference point.

    This function computes three sets of weights:
    1. Combined weights based on both spatial distance and data similarity
    2. Distance-based weights
    3. Data similarity weights

    Parameters
    ----------
    f_data_h5 : str
        Path to HDF5 file containing geometry and observed data
    x_well : float, optional
        X coordinate of reference point (well), by default 0
    y_well : float, optional 
        Y coordinate of reference point (well), by default 0
    i_ref : int, optional
        Index of reference point, by default -1 (auto-calculated as closest to x_well,y_well)
    r_dis : float, optional
        Distance range parameter for spatial weighting, by default 400
    r_data : float, optional
        Data similarity range parameter for data weighting, by default 2

    Returns
    -------
    tuple
        (w_combined, w_dis, w_data) where:
        - w_combined: Combined weights from distance and data similarity
        - w_dis: Distance-based weights
        - w_data: Data similarity-based weights

    Notes
    -----
    The weights are calculated using Gaussian functions:
    - Distance weights use exp(-dis²/r_dis²)
    - Data weights use exp(-sum_dd²/r_data²)
    where dis is spatial distance and sum_dd is cumulative data difference
    """
    import integrate as ig
    import numpy as np
    import matplotlib.pyplot as plt
    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
    DATA = ig.load_data(f_data_h5, showInfo=showInfo)
    id=0
    d_obs = DATA['d_obs'][id]
    d_std = DATA['d_std'][id]
    # index if position in X and Y with smallets distance to well
    if i_ref == -1:
        i_ref = np.argmin((X-x_well)**2 + (Y-y_well)**2)

    # select gates to use 
    # find the number of data points for each gate that has non-nan values
    n_not_nan = np.sum(~np.isnan(d_obs), axis=0)
    n_not_nan_freq = n_not_nan/d_obs.shape[0]
    # use the data for which n_not_nan_freq>0.7
    # 0.7 should be an option to select!
    i_use = np.where(n_not_nan_freq>0.8)[0]
    # only use i_use values that are not nan
    i_use = i_use[~np.isnan(d_obs[i_ref,i_use])]
    # select gates to use, manually
    if useLog:
        d_ref = np.log10(d_obs[i_ref,i_use])
        d_test = np.log10(d_obs[:,i_use])
    else:
        d_ref = d_obs[i_ref,i_use]
        d_test =d_obs[:,i_use]
    dd = np.abs(d_test - d_ref)
    sum_dd = np.sum(dd, axis=1)
    w_data = np.exp(-1*sum_dd**2/r_data**2)
    

    # Compute the distance from each data point to the actual borehole location
    dis = np.sqrt((X-x_well)**2 + (Y-y_well)**2)
    w_dis = np.exp(-1*dis**2/r_dis**2)

    w_combined = w_data * w_dis

    cmap = 'hot_r'
    #cmap = 'jet'

    if doPlot:
        plt.figure(figsize=(15,5))
        for i in range(3):
            plt.subplot(1,3,i+1)
            plt.plot(X,Y,'.', markersize=1.02, color='lightgray') 
            #plt.scatter(X[i_use], Y[i_use], c=w[i_use], cmap='jet', s=1, zorder=3, vmin=0, vmax=1, marker='.')
                 
            if i==0:
                i_use = np.where(w_combined>0.001)[0]
                plt.scatter(X[i_use],Y[i_use],c=w_combined[i_use], s=1, cmap=cmap, vmin=0, vmax=1, marker='.', zorder=3)  
                plt.title('Combined weights')          
            elif i==1:
                i_use = np.where(w_dis>0.001)[0]                
                plt.scatter(X[i_use],Y[i_use],c=w_dis[i_use], s=1, cmap=cmap, vmin=0, vmax=1, marker='.', zorder=3)
                plt.title('XY distance weights')
            elif i==2:
                i_use = np.where(w_data>0.001)[0]                                
                plt.scatter(X[i_use],Y[i_use],c=w_data[i_use], s=0.2, cmap=cmap, vmin=0, vmax=1, marker='.', zorder=3)
                plt.title('Data distance weights')
            plt.axis('equal')
            plt.colorbar()
            plt.grid()
            plt.plot(x_well,y_well,'wo', zorder=6, markersize=2)
            plt.plot(x_well,y_well,'ko', zorder=5, markersize=4)
            plt.plot(x_well,y_well,'wo', zorder=4, markersize=6)
    

        plt.suptitle('Weights')
        plt.xlabel('X')
        plt.ylabel('Y')
        if plFile is None:
            plFile = 'weights_%d_%d_%d_rdis%d_rdata%d.png' % (x_well,y_well,i_ref,r_dis,r_data)
        plt.savefig(plFile, dpi=300)

    return w_combined, w_dis, w_data, i_ref


def prior_data_borehole_class_mode(f_prior_h5, im_prior, BH, parallel=False, **kwargs):
    """
    Compute mode-class prior data for a borehole and save to prior HDF5 file.

    Reads M{im_prior} from f_prior_h5, extracts the most frequent class in
    each observed depth interval for every prior realization (via
    welllog_compute_P_obs_class_mode), then stores the result as a new dataset
    in f_prior_h5.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the prior HDF5 file.
    im_prior : int
        Index of the discrete model parameter (e.g. 2 for /M2 lithology).
    BH : dict
        Borehole dictionary with keys depth_top, depth_bottom, class_obs,
        class_prob, X, Y, name, method.
    parallel : bool, optional
        Enable parallel mode computation. Default is False.
    **kwargs
        showInfo : int, optional
            Verbosity level. Default is 1.

    Returns
    -------
    P_obs : ndarray
        Probability matrix, shape (nclass, n_intervals).
    id_prior : int
        Dataset index of the new /D{id_prior} entry added to f_prior_h5.
    """
    import h5py
    import integrate as ig

    showInfo = kwargs.get('showInfo', 1)

    with h5py.File(f_prior_h5, 'r') as f:
        z           = f['M%d' % im_prior].attrs['x']
        class_id    = f['M%d' % im_prior].attrs['class_id']
        M_lithology = f['M%d' % im_prior][:]

    P_obs, class_mode = welllog_compute_P_obs_class_mode(
        M_lithology, z=z, class_id=class_id, W=BH,
        parallel=parallel, showInfo=showInfo)

    id_prior = ig.save_prior_data(f_prior_h5, class_mode, showInfo=showInfo)

    return P_obs, id_prior


def prior_data_borehole_class_layer(f_prior_h5, im_prior, BH, **kwargs):
    """
    Compute layer-probability prior data for a borehole using identity mapping.

    Uses compute_P_obs_discrete (no prior ensemble needed) and stores an
    identity prior data reference in f_prior_h5 via prior_data_identity.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the prior HDF5 file.
    im_prior : int
        Index of the discrete model parameter (e.g. 2 for /M2 lithology).
    BH : dict
        Borehole dictionary with keys depth_top, depth_bottom, class_obs,
        class_prob, X, Y, name, method.
    **kwargs
        showInfo : int, optional
            Verbosity level. Default is 1.

    Returns
    -------
    P_obs : ndarray
        Probability matrix, shape (nclass, n_intervals).
    id_prior : int
        Dataset index of the identity /D{id_prior} entry in f_prior_h5.
    """
    import h5py
    import integrate as ig

    showInfo = kwargs.get('showInfo', 1)

    if BH is None:
        print('prior_data_borehole_class_layer: BH is None, returning None')
        return None, None

    with h5py.File(f_prior_h5, 'r') as f:
        z        = f['M%d' % im_prior].attrs['x']
        class_id = f['M%d' % im_prior].attrs['class_id']

    P_obs = compute_P_obs_discrete(z=z, class_id=class_id, W=BH)

    f_prior_h5, id_prior = ig.prior_data_identity(
        f_prior_h5, im=im_prior, doMakePriorCopy=False, showInfo=showInfo)

    return P_obs, id_prior


def prior_data_borehole(f_prior_h5, im_prior, BH, parallel=False, **kwargs):
    """
    Compute prior data for a single borehole and save to the prior HDF5 file.

    Dispatches to the appropriate implementation based on BH['method']:

    - ``'mode_probability'``  → :func:`prior_data_borehole_class_mode`
      (recommended — fast and robust)
    - ``'layer_probability'`` → :func:`prior_data_borehole_class_layer`
    - ``'class_exact'`` / ``'layer_probability_independent'`` → NotImplementedError

    Parameters
    ----------
    f_prior_h5 : str
        Path to the prior HDF5 file.
    im_prior : int
        Index of the discrete model parameter (e.g. 2 for /M2 lithology).
    BH : dict
        Borehole dictionary. Key ``'method'`` selects the integration approach
        and defaults to ``'mode_probability'`` if absent.
    parallel : bool, optional
        Enable parallel mode computation for ``'mode_probability'``. Default False.
    **kwargs
        showInfo : int, optional
            Verbosity level. Default is 1.

    Returns
    -------
    P_obs : ndarray or None
        Probability matrix, shape (nclass, n_intervals).
    id_prior : int or None
        Dataset index of the new /D entry added to f_prior_h5.

    Examples
    --------
    >>> P_obs, id_prior = ig.prior_data_borehole(f_prior_h5, im_prior=2, BH=BH, parallel=True)
    >>> # Then extrapolate to survey grid:
    >>> d_obs, i_use, T_use = ig.Pobs_to_datagrid(P_obs, BH['X'], BH['Y'], f_data_h5)
    >>> id_out, _ = ig.save_data_multinomial(d_obs, i_use=i_use, id_prior=id_prior, f_data_h5=f_data_h5)
    """
    showInfo = kwargs.get('showInfo', 1)

    if BH is None:
        return None, None

    method = BH.get('method', 'mode_probability')

    if showInfo > 0:
        print('prior_data_borehole: borehole=%s  method=%s' % (BH.get('name', '?'), method))

    if method == 'mode_probability':
        return prior_data_borehole_class_mode(
            f_prior_h5, im_prior, BH, parallel=parallel, **kwargs)
    elif method == 'layer_probability':
        return prior_data_borehole_class_layer(
            f_prior_h5, im_prior, BH, **kwargs)
    elif method in ('class_exact', 'layer_probability_independent'):
        raise NotImplementedError("Method '%s' not implemented yet" % method)
    else:
        raise ValueError("Unknown method: %s" % method)


def save_borehole_data(f_prior_h5, f_data_h5, BH, **kwargs):
    """
    Compute and save prior and observed data for a single borehole in one call.

    Combines the three-step borehole ingestion workflow into a single function:

    1. Compute mode-class prior data and save to f_prior_h5
       (via :func:`prior_data_borehole`)
    2. Extrapolate point observations to the survey grid with
       distance-based weighting (via :func:`Pobs_to_datagrid`)
    3. Save the gridded observations to f_data_h5
       (via ``ig.save_data_multinomial``)

    Parameters
    ----------
    f_prior_h5 : str
        Path to the prior HDF5 file.
    f_data_h5 : str
        Path to the observed-data HDF5 file.
    BH : dict
        Borehole dictionary with keys depth_top, depth_bottom, class_obs,
        class_prob, X, Y, name, method. Two optional keys control the
        distance-weighting radii when r_data / r_dis are not passed as
        explicit kwargs:

        * ``range_data`` (float, optional) — data-space similarity radius.
          Survey points whose EM data response is similar to the borehole
          location receive higher weight; points that are more dissimilar are
          down-weighted. Default: 1,000,000 (effectively no cutoff).
        * ``range_dis`` (float, optional) — geographic XY distance [m] beyond
          which the borehole exerts no influence on nearby survey points.
          Default: 300 m.
    **kwargs
        im_prior : int, optional
            Index of the discrete model parameter in f_prior_h5 (e.g. 2 for /M2).
            Default is 2.
        parallel : bool, optional
            Enable parallel mode computation. Default is False.
        r_data : float, optional
            Data-space similarity radius. Overrides ``BH['range_data']`` when
            provided. Resolution order: explicit kwarg > BH['range_data'] >
            1,000,000 (no cutoff).
        r_dis : float, optional
            Geographic XY fade-out distance [m]. Overrides ``BH['range_dis']``
            when provided. Resolution order: explicit kwarg > BH['range_dis'] >
            300 m.
        doPlot : bool, optional
            Plot distance-weight maps. Default is False.
        showInfo : int, optional
            Verbosity level (0 = silent, 1 = one summary line per borehole).
            Default is 1.

    Returns
    -------
    id_prior : int
        Dataset index of the new /D entry added to f_prior_h5.
    id_out : int
        Dataset index of the new /D entry added to f_data_h5.

    Examples
    --------
    >>> # Single borehole
    >>> id_prior, id_data = ig.save_borehole_data(f_prior_h5, f_data_h5, BH)

    >>> # All boreholes — collect data IDs for joint inversion
    >>> id_borehole_list = []
    >>> for BH in BHOLES:
    ...     _, id_out = ig.save_borehole_data(f_prior_h5, f_data_h5, BH,
    ...                                       r_data=2, r_dis=300, parallel=True)
    ...     id_borehole_list.append(id_out)
    >>> f_post_h5 = ig.integrate_rejection(f_prior_h5, f_data_h5,
    ...                                    id_use=[1] + id_borehole_list)
    """
    import integrate as ig

    im_prior = kwargs.get('im_prior', 2)
    parallel = kwargs.get('parallel', False)
    r_data   = kwargs.get('r_data', BH.get('range_data', 1_000_000))
    r_dis    = kwargs.get('r_dis',  BH.get('range_dis',  300))
    doPlot   = kwargs.get('doPlot', False)
    showInfo = kwargs.get('showInfo', 1)

    # Step 1: compute mode-class prior data and save to f_prior_h5
    P_obs, id_prior = prior_data_borehole(
        f_prior_h5, im_prior, BH,
        parallel=parallel, showInfo=showInfo)

    # Step 2: extrapolate point observations to the survey grid
    d_obs, i_use, T_use = Pobs_to_datagrid(
        P_obs, BH['X'], BH['Y'], f_data_h5,
        r_data=r_data, r_dis=r_dis, doPlot=doPlot)

    # Step 3: save gridded observations to f_data_h5
    id_out, _ = ig.save_data_multinomial(
        D_obs=d_obs,
        i_use=i_use,
        id_prior=id_prior,
        f_data_h5=f_data_h5,
        showInfo=showInfo)

    if showInfo > 0:
        print('save_borehole_data: %s  →  prior D%d, data D%d'
              % (BH.get('name', '?'), id_prior, id_out))

    return id_prior, id_out
