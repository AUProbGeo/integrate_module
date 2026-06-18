"""
INTEGRATE Core Module - Probabilistic Geophysical Data Integration

This module implements rejection sampling algorithms for Bayesian inversion and
probabilistic data integration in geophysics, with particular focus on electromagnetic
(EM) data analysis. The module provides comprehensive tools for prior model generation,
forward modeling, likelihood computation, and posterior sampling.

Key Features:
    - Rejection sampling for Bayesian inversion
    - Parallel processing with shared memory optimization
    - Temperature annealing for improved sampling efficiency
    - Support for multiple data types (TDEM, multinomial, etc.)
    - Integration with GA-AEM electromagnetic forward modeling
    - Automatic temperature estimation and adaptive sampling

Main Functions:
    - integrate_rejection(): Main rejection sampling workflow (now in integrate_rejection module)
    - prior_data(): Integration of forward modeling with prior structure
    - forward_gaaem(): Electromagnetic forward modeling interface
    - likelihood_*(): Various likelihood calculation functions (now in integrate_rejection module)
    - posterior_*(): Posterior analysis and statistics

Author: Thomas Mejer Hansen
Email: tmeha@geo.au.dk
"""

import h5py
import numpy as np
import os.path
import subprocess
from sys import exit
import sys
import types
import multiprocessing
from multiprocessing import Pool
from multiprocessing import shared_memory
from multiprocessing import get_context
from functools import partial
import time

# %% Set up logging.. USed to test creation and use of shared memory
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING) # For production
#logger.setLevel(logging.DEBUG)  # For debugging    
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def is_notebook():
    """
    Check if the code is running in a Jupyter notebook or IPython shell.

    Returns
    -------
    bool
        True if running in a Jupyter notebook or IPython shell, False otherwise.
    """
    try:
        # Get the shell type from IPython
        shell = get_ipython().__class__.__name__
        
        if shell == 'ZMQInteractiveShell':
            # Additional check for VS Code
            import sys
            if 'vscode' in sys.modules:
                return False
            return True
        else:
            return False
            
    except NameError:  # If get_ipython is not defined (standard Python)
        return False


def use_parallel(**kwargs):
    """
    Determine if parallel processing can be used based on the environment.

    Parallel processing is supported on all platforms. The module handles
    platform differences internally: Linux uses fork for performance, while
    Windows and macOS use spawn for correctness. No `if __name__ == "__main__":`
    guard is required in user scripts on any platform.

    Parameters
    ----------
    showInfo : int, optional
        If greater than 0, prints information about the environment and
        parallel processing status. Default is 0.

    Returns
    -------
    bool
        True if parallel processing is safe, False otherwise.
    """
    import os
    showInfo = kwargs.get('showInfo', 0)

    if is_notebook():
        if showInfo>0:
            print('Notebook detected. Parallel processing is OK.')
    else:
        if os.name == 'posix':
            if os.uname().sysname == 'Darwin':
                if showInfo>0:
                    print('macOS detected. Parallel processing is OK (using spawn).')
            else:
                if showInfo>0:
                    print('Linux/POSIX detected. Parallel processing is OK (using fork).')
        else:
            if showInfo>0:
                print('Windows detected. Parallel processing is OK (using spawn).')

    return True

    


def logl_T_est(logL, N_above=10, P_acc_lev=0.2):
    """
    Estimate a temperature (T_est) based on a given logarithmic likelihood (logL), 
    a number (N_above), and an acceptance level (P_acc_lev).

    Parameters
    ----------
    logL : numpy.ndarray
        An array of logarithmic likelihoods.
    N_above : int, optional
        The number of elements above which to consider in the sorted logL array. 
        Default is 10.
    P_acc_lev : float, optional
        The acceptance level for the calculation. Default is 0.2.
    
    Returns
    -------
    float
        The estimated temperature. It's either a positive number or infinity.

    Notes
    -----
    The function sorts the logL array in ascending order after normalizing the data 
    by subtracting the maximum value from each element. It then removes any NaN values 
    from the sorted array. If the sorted array is not empty, it calculates T_est based 
    on the N_above+1th last element in the sorted array and the natural logarithm of 
    P_acc_lev. If the sorted array is empty, it sets T_est to infinity.
    """
    sorted_logL = np.sort(logL - np.nanmax(logL))
    sorted_logL = sorted_logL[~np.isnan(sorted_logL)]
    
    if sorted_logL.size > 0:
        logL_lev = sorted_logL[-N_above-1]
        T_est = logL_lev / np.log(P_acc_lev)
        T_est = np.nanmax([1, T_est])
    else:
        T_est = np.inf

    return T_est


def lu_post_sample_logl(logL, ns=1, T=1):
    """
    Perform LU post-sampling log-likelihood calculation.

    Parameters
    ----------
    logL : array-like
        Array of log-likelihood values.
    ns : int, optional
        Number of samples to generate. Defaults to 1.
    T : float, optional
        Temperature parameter. Defaults to 1.

    Returns
    -------
    tuple
        A tuple containing the generated samples and the acceptance probabilities.
        
        i_use_all : numpy.ndarray
            Array of indices of the selected samples.
        P_acc : numpy.ndarray
            Array of acceptance probabilities.
    """

    N = len(logL)
    P_acc = np.exp((1/T) * (logL - np.nanmax(logL)))
    P_acc[np.isnan(P_acc)] = 0

    Cum_P = np.cumsum(P_acc)
    Cum_P = Cum_P / np.nanmax(Cum_P)
    dp = 1 / N
    p = np.array([i * dp for i in range(1, N+1)])

    i_use_all = np.zeros(ns, dtype=int)
    for is_ in range(ns):
        r = np.random.rand()
        i_use = np.where(Cum_P > r)[0][0]
        i_use_all[is_] = i_use
    
    return i_use_all, P_acc

def integrate_update_prior_attributes(f_prior_h5, **kwargs):
    """
    Update the 'is_discrete' attribute of datasets in an HDF5 file.

    This function iterates over all datasets in the provided HDF5 file. 
    If a dataset's name starts with 'M', the function checks if the dataset 
    has an 'is_discrete' attribute. If not, it checks if the dataset appears 
    to represent discrete data by sampling the first 1000 elements and checking 
    how many unique values there are. If there are fewer than 20 unique values, 
    it sets 'is_discrete' to 1; otherwise, it sets 'is_discrete' to 0. 
    The 'is_discrete' attribute is then added to the dataset.

    Parameters
    ----------
    f_prior_h5 : str
        The path to the HDF5 file to process.
    showInfo : int, optional
        Level of verbosity for output (default is 0).
    """

    showInfo = kwargs.get('showInfo', 0)

    # Check that hdf5 files exists
    if not os.path.isfile(f_prior_h5):
        if showInfo>=2:
            print('integrate_update_prior_attributes: File %s does not exist' % f_prior_h5)
        exit()  

    with h5py.File(f_prior_h5, 'a') as f:  # open file in append mode
        for name, dataset in f.items():
            if showInfo>0:
                print("integrate_update_prior_attributes: Checking %s" % (name))
            if name.upper().startswith('M'):
                # Check if the attribute 'is_discrete' exists
                if 'x' in dataset.attrs:
                    pass
                else:
                    if 'z' in dataset.attrs:
                        dataset.attrs['x'] = dataset.attrs['z']
                    else:
                        x = np.arange(dataset.shape[1])
                        dataset.attrs['x'] = x
                        print(dataset.attrs)
                        #if 'M1' in f.keys():
                        #    if 'x' in f['/M1'].attrs.keys():
                        #        f[name].attrs['x'] = f['/M1'].attrs['x']
                        #        print('Setting %s/x = /M1/x ' % name)
                        #    else:
                        #        print('No x attribute found in %s' % name)    
                
                if 'is_discrete' in dataset.attrs:
                    if (showInfo>0):
                        print('%s: %s.is_discrete=%d' % (f_prior_h5,name,dataset.attrs['is_discrete']))
                else:
                    # Check if M is discrete
                    M_sample = dataset[:1000]  # get the first 1000 elements
                    class_id = np.unique(M_sample)
                    print(class_id)
                    if len(class_id) < 20:
                        is_discrete = 1
                        dataset.attrs['class_id'] = class_id
                        ## convert class_id to an array of strings and save it as an attribute if the attribute does not 
                        ## already exist
                        if 'class_name' not in dataset.attrs:
                            dataset.attrs['class_name'] = np.array([str(x) for x in class_id])
                        
                    else:
                        is_discrete = 0

                    if (showInfo>0):
                        print(f'Setting is_discrete={is_discrete}, for {name}')
                    dataset.attrs['is_discrete'] = is_discrete

                if dataset.attrs['is_discrete']==1:
                    if not ('class_id' in dataset.attrs):
                        M_sample = dataset[:1000]  # get the first 1000 elements
                        class_id = np.unique(M_sample)
                        dataset.attrs['class_id'] = class_id
                    if not ('class_name' in dataset.attrs):
                        # Convert class_id to an array of strings and save it as an attribute if the attribute does not
                        class_id = dataset.attrs['class_id']
                        dataset.attrs['class_name'] = [str(x) for x in class_id]



def integrate_posterior_stats(f_post_h5='POST.h5', ip_range=None, **kwargs):
    """
    Compute posterior statistics for all model parameters in a POST HDF5 file.

    Reads posterior sample indices (i_use) and the corresponding prior model
    realizations, then computes per-location statistics and writes them back
    into the same POST.h5 file.

    Parameters
    ----------
    f_post_h5 : str, optional
        Path to the POST HDF5 file. Default is 'POST.h5'.
    ip_range : array-like or None, optional
        Indices of data locations to process. If None, all locations are processed.
        Locations not in ip_range receive NaN values. Default is None.
    **kwargs : dict
        showInfo : int, optional
            Verbosity level. 0 = silent, 1 = progress bars. Default is 0.
        usePrior : bool, optional
            If True, use randomly drawn prior samples instead of i_use indices
            (useful for computing prior statistics as a baseline). Default is False.
        updateGeometryFromData : bool, optional
            Copy UTMX, UTMY, LINE, ELEVATION from the DATA.h5 file into POST.h5
            if not already present. Default is True.
        computeKL : bool, optional
            Shorthand to enable KL divergence computation for all parameter types.
            If True, both ``computeKL_continuous`` and ``computeKL_discrete`` are
            set to True. Default is False.
        computeKL_continuous : bool, optional
            Compute KL divergence D_KL(posterior || prior) for continuous model
            parameters using log10-space histograms (50 bins). Result is in bits
            (log base 2). Default is False.
        computeKL_discrete : bool, optional
            Compute KL divergence D_KL(posterior || prior) for discrete model
            parameters. Normalised to [0, 1] using log_base = number of classes
            (0 = posterior equals prior, 1 = completely certain). Default is False.

    Writes to POST.h5
    -----------------
    Always written:

    - ``/N_UNIQUE``  [Np]  Number of unique prior realizations used per location.
    - ``/UTMX``, ``/UTMY``, ``/LINE``, ``/ELEVATION``  [Np]
      Geometry copied from DATA.h5 (if ``updateGeometryFromData=True``).

    For each **continuous** model parameter ``/Mx``:

    - ``/Mx/Mean``          [Np, Nm]  Arithmetic mean of posterior realizations.
    - ``/Mx/LogMean``       [Np, Nm]  Geometric mean (exp of mean of log values).
    - ``/Mx/Median``        [Np, Nm]  Median of posterior realizations.
    - ``/Mx/Std``           [Np, Nm]  Standard deviation of posterior realizations.
    - ``/Mx/LogStd``        [Np, Nm]  Standard deviation of log10(posterior realizations).
    - ``/Mx/HarmonicMean``  [Np, Nm]  Trimmed harmonic mean: conductivity samples
      are trimmed 10% each tail, averaged, then inverted back to resistivity.
    - ``/Mx/KL``            [Np, Nm]  KL divergence in bits. Only written when
      ``computeKL_continuous=True``.

    For each **discrete** model parameter ``/Mx``:

    - ``/Mx/Mode``     [Np, Nm]  Most probable class at each location/depth.
    - ``/Mx/Entropy``  [Np, Nm]  Shannon entropy normalised by log(n_classes).
    - ``/Mx/P``        [Np, Nclass, Nm]  Posterior probability of each class.
    - ``/Mx/KL``       [Np, Nm]  KL divergence normalised to [0, 1]. Only written
      when ``computeKL_discrete=True``.

    Returns
    -------
    None
    """
    import h5py
    import numpy as np
    import integrate
    import scipy as sp
    from tqdm import tqdm

    showInfo = kwargs.get('showInfo', 0)
    if showInfo<0:
        disableTqdm=True
    else:
        disableTqdm=False
    usePrior = kwargs.get('usePrior', False)
    updateGeometryFromData = kwargs.get('updateGeometryFromData', True)
    computeKL = kwargs.get('computeKL', False)
    computeKL_continuous = kwargs.get('computeKL_continuous', False) or computeKL
    computeKL_discrete = kwargs.get('computeKL_discrete', False) or computeKL

    # Check if f_prior_h5 attribute exists in the HDF5 file
    with h5py.File(f_post_h5, 'r') as f:
        if 'f5_prior' in f.attrs:
            f_prior_h5 = f.attrs['f5_prior']
        else:
            f_prior_h5 = None
            if showInfo>=1:
                raise ValueError(f"'f5_prior' attribute does not exist in {f_post_h5}")

    # Check if f5_data attribute exists in the HDF5 file
    with h5py.File(f_post_h5, 'r') as f:
        if 'f5_data' in f.attrs:
            f_data_h5 = f.attrs['f5_data']
        else:
            f_data_h5 = None
            if showInfo>=1:
                raise ValueError(f"'f5_data' attribute does not exist in {f_post_h5}")

    # update Geometry from f_data_h5
    if (updateGeometryFromData)&(f_data_h5 is not None):
        with h5py.File(f_data_h5, 'r') as f_data, h5py.File(f_post_h5, 'a') as f_post:
            if '/UTMX' in f_data:
                if '/UTMX' not in f_post:
                    f_data.copy('/UTMX', f_post)
            if '/UTMY' in f_data:
                if '/UTMY' not in f_post:
                    f_data.copy('/UTMY', f_post)
            if '/LINE' in f_data:
                if '/LINE' not in f_post:
                    f_data.copy('/LINE', f_post)
            if '/ELEVATION' in f_data:
                if '/ELEVATION' not in f_post:
                    f_data.copy('/ELEVATION', f_post)
    
    # Load 'i_use' data from the HDF5 file
    try:
        with h5py.File(f_post_h5, 'r') as f:
            i_use = f['i_use'][:]
    except KeyError:
        print(f"Could not read 'i_use' from {f_post_h5}")
        #return

    if usePrior:
        with h5py.File(f_prior_h5, 'r') as f_prior:
            N = f_prior['/M1'].shape[0]
            nr=i_use.shape[1]
            nd=i_use.shape[0]
            # compute i_use of  (nd,nr), with random integer numbers between 0 and N-1
            i_use = np.random.randint(0, N, (nd,nr))

    # Handle ip_range parameter
    nsounding = i_use.shape[0]
    if ip_range is None or len(ip_range) == 0:
        ip_range = np.arange(nsounding)
        if showInfo > 0:
            print(f'Computing posterior statistics for all {nsounding} data points')
    else:
        ip_range = np.asarray(ip_range)
        if showInfo > 0:
            print(f'Computing posterior statistics for {len(ip_range)} of {nsounding} data points')
        # Validate ip_range
        if np.any(ip_range < 0) or np.any(ip_range >= nsounding):
            raise ValueError(f"ip_range contains indices outside valid range [0, {nsounding-1}]")

    # Compute number of unique realizations for each data location
    if showInfo > 0:
        print('Computing number of unique realizations (N_UNIQUE)')

    N_UNIQUE = np.full(nsounding, np.nan)
    for iid in tqdm(ip_range, mininterval=1, disable=disableTqdm, desc='N_UNIQUE', leave=False):
        N_UNIQUE[iid] = len(np.unique(i_use[iid, :]))

    # Save N_UNIQUE to the HDF5 file
    with h5py.File(f_post_h5, 'a') as f_post:
        if '/N_UNIQUE' not in f_post:
            if showInfo > 0:
                print('Creating /N_UNIQUE in %s' % f_post_h5)
            f_post.create_dataset('/N_UNIQUE', data=N_UNIQUE)
        else:
            if showInfo > 0:
                print('Updating /N_UNIQUE in %s' % f_post_h5)
            f_post['/N_UNIQUE'][:] = N_UNIQUE

    # Process each dataset in f_prior_h5
    with h5py.File(f_prior_h5, 'r') as f_prior, h5py.File(f_post_h5, 'a') as f_post:
        for name, dataset in f_prior.items():
            
            if name.upper().startswith('M') and 'is_discrete' in dataset.attrs and dataset.attrs['is_discrete'] == 0:
                if showInfo>2:
                    print('%s: CONTINUOUS' % name)

                nm = dataset.shape[1]
                nsounding, nr = i_use.shape
                m_post = np.zeros((nm, nr))

                # Initialize with NaN for all data points
                M_logmean = np.full((nsounding, nm), np.nan)
                M_mean = np.full((nsounding, nm), np.nan)
                M_std = np.full((nsounding, nm), np.nan)
                M_logstd = np.full((nsounding, nm), np.nan)
                M_median = np.full((nsounding, nm), np.nan)
                M_harmonicmean = np.full((nsounding, nm), np.nan)
                if computeKL_continuous:
                    M_KL = np.full((nsounding, nm), np.nan)

                # Load all prior data into memory
                M_all = dataset[:]

                useSequential = True
                if useSequential:

                    # Precompute log10 prior histograms once — prior is fixed across all soundings.
                    # Using histograms (O(n)) instead of KDE (O(n²)) for speed.
                    # Log10 bins are appropriate for log-normally distributed continuous parameters.
                    if computeKL_continuous:
                        n_bins_kl = 50
                        idx_kl = np.random.choice(M_all.shape[0], nr, replace=False)
                        M_prior_kl = np.log10(np.maximum(M_all[idx_kl, :], 1e-10))
                        kl_bins = []
                        kl_prior_hist = []
                        for _i in range(nm):
                            col = M_prior_kl[:, _i]
                            bins = np.linspace(col.min(), col.max(), n_bins_kl + 1)
                            h, _ = np.histogram(col, bins=bins)
                            h = (h + 1e-10) / (h.sum() + n_bins_kl * 1e-10)
                            kl_bins.append(bins)
                            kl_prior_hist.append(h)

                    # Sequential processing - simple, fast, memory-efficient
                    for iid in tqdm(ip_range, mininterval=1, disable=disableTqdm, desc='%s-continuous' % name, leave=False):
                        ir = np.int64(i_use[iid,:])
                        m_post = M_all[ir,:]

                        M_logmean[iid,:] = np.exp(np.mean(np.log(m_post), axis=0))
                        M_mean[iid,:] = np.mean(m_post, axis=0)
                        M_median[iid,:] = np.median(m_post, axis=0)
                        with np.errstate(invalid='ignore', divide='ignore'):
                            M_logstd[iid,:] = np.std(np.log10(np.maximum(m_post, 1e-10)), axis=0)
                        M_std[iid,:] = np.std(m_post, axis=0)
                        _c = 1.0 / np.maximum(m_post, 1e-10)
                        _k = int(np.floor(0.10 * _c.shape[0]))
                        _cs = np.sort(_c, axis=0)
                        M_harmonicmean[iid, :] = 1.0 / np.mean(_cs[_k:_c.shape[0]-_k, :], axis=0)
                        if computeKL_continuous:
                            m_post_log = np.log10(np.maximum(m_post, 1e-10))
                            for _i in range(nm):
                                h_q, _ = np.histogram(m_post_log[:, _i], bins=kl_bins[_i])
                                h_q = (h_q + 1e-10) / (h_q.sum() + n_bins_kl * 1e-10)
                                M_KL[iid, _i] = np.sum(h_q * np.log2(h_q / kl_prior_hist[_i]))
                elif a==1:

                    # NEW Experimental METHOD
                    # 3. Optimization Constants
                    BATCH_SIZE = 100  # Process 1000 soundings at a time
                    INV_LOG_10 = 1.0 / np.log(10.0) # Pre-calculate constant

                    # 4. Batched Processing
                    # Instead of 40,000 iterations, we do 40.
                    for start_idx in tqdm(range(0, len(ip_range), BATCH_SIZE), 
                                        disable=disableTqdm, 
                                        desc=f'{name}-optimized', 
                                        leave=False):

                        # A. Define Batch Range
                        end_idx = min(start_idx + BATCH_SIZE, len(ip_range))
                        current_iids = ip_range[start_idx:end_idx]
                        
                        # B. Vectorized Indexing
                        # Gather all indices for this batch at once.
                        # Shape: (Batch_Size, K) where K is number of priors used per sounding
                        batch_indices = np.int64(i_use[current_iids, :])
                        
                        # C. Create 3D Data Cube
                        # Fetch data for 1000 soundings simultaneously.
                        # Shape: (Batch_Size, K, nm) -> e.g., (1000, 50, 100)
                        # This is the biggest speedup: one large memory read instead of 1000 tiny ones.
                        m_cube = M_all[batch_indices, :]

                        # D. Compute Statistics (Collapsing Axis 1)
                        
                        # -- Arithmetic Mean & Median --
                        M_mean[current_iids, :] = np.mean(m_cube, axis=1)
                        M_median[current_iids, :] = np.median(m_cube, axis=1)
                        
                        # -- Logarithmic Stats (Optimized) --
                        # Calculate Log ONCE. 
                        # Use maximum to prevent log(0) errors (NaNs)
                        # Shape: (Batch_Size, K, nm)
                        log_cube = np.log(np.maximum(m_cube, 1e-10))
                        
                        # Geometric Mean: exp(mean(log(x)))
                        M_logmean[current_iids, :] = np.exp(np.mean(log_cube, axis=1))
                        
                        # LogStd: std(log10(x)) = std(ln(x)) * (1/ln(10)); reuse log_cube for speed
                        M_logstd[current_iids, :] = np.std(log_cube, axis=1) * INV_LOG_10
                        M_std[current_iids, :] = np.std(m_cube, axis=1)

                        # Harmonic mean (trimmed 10% each tail in conductivity space)
                        _c = 1.0 / np.maximum(m_cube, 1e-10)
                        _nr = _c.shape[1]
                        _k = int(np.floor(0.10 * _nr))
                        _cs = np.sort(_c, axis=1)
                        M_harmonicmean[current_iids, :] = 1.0 / np.mean(_cs[:, _k:_nr-_k, :], axis=1)




                # Create datasets
                for stat in ['Mean', 'Median', 'Std', 'LogStd', 'LogMean', 'HarmonicMean']:
                    if stat not in f_post:
                        dset = '/%s/%s' % (name,stat)
                        if dset not in f_post:
                            if (showInfo>0):
                                print('Creating %s in %s' % (dset,f_post_h5 ))
                            f_post.create_dataset(dset, (nsounding,nm))

                f_post['/%s/%s' % (name,'LogMean')][:] = M_logmean
                f_post['/%s/%s' % (name,'Mean')][:] = M_mean
                f_post['/%s/%s' % (name,'Median')][:] = M_median
                f_post['/%s/%s' % (name,'Std')][:] = M_std
                f_post['/%s/%s' % (name,'LogStd')][:] = M_logstd
                f_post['/%s/%s' % (name,'HarmonicMean')][:] = M_harmonicmean
                if computeKL_continuous:
                    dset = '/%s/KL' % name
                    if dset not in f_post:
                        f_post.create_dataset(dset, (nsounding, nm))
                    f_post[dset][:] = M_KL

            elif name.upper().startswith('M') and 'is_discrete' in dataset.attrs and dataset.attrs['is_discrete'] == 1:
                if showInfo>2:
                    print('%s: DISCRETE' % name)

                nm = dataset.shape[1]
                nsounding, nr = i_use.shape
                # Get number of classes for name
                class_id = f_prior[name].attrs['class_id']
                n_classes = len(class_id)

                # Determine log_base for KL: use n_classes from class_name if available
                if 'class_name' in f_prior[name].attrs:
                    log_base_kl = len(f_prior[name].attrs['class_name'])
                else:
                    log_base_kl = n_classes if n_classes > 1 else 2

                if showInfo>1:
                    print('%s: DISCRETE, N_classes =%d' % (name,n_classes))

                # Initialize with NaN for all data points
                M_mode = np.full((nsounding, nm), np.nan)
                M_entropy = np.full((nsounding, nm), np.nan)
                M_P = np.full((nsounding, n_classes, nm), np.nan)
                if computeKL_discrete:
                    M_KL = np.full((nsounding, nm), np.nan)

                # Create datasets in h5 file
                for stat in ['Mode', 'Entropy']:
                    if stat not in f_post:
                        dset = '/%s/%s' % (name,stat)
                        if dset not in f_post:
                            if (showInfo>0):
                                print('Creating %s in %s' % (dset,f_post_h5 ))
                            f_post.create_dataset(dset, (nsounding,nm))
                for stat in ['P']:
                    if stat not in f_post:
                        dset = '/%s/%s' % (name,stat)
                        if dset not in f_post:
                            if (showInfo>0):
                                print('Creating %s' % dset)
                            f_post.create_dataset(dset, (nsounding,n_classes,nm))

                # Load all prior data into memory
                M_all = dataset[:]

                # Subsample prior once for KL (prior is fixed across soundings)
                if computeKL_discrete:
                    idx_kl = np.random.choice(M_all.shape[0], nr, replace=False)
                    M_prior_kl = M_all[idx_kl, :]

                # Sequential processing - simple, fast, memory-efficient
                for iid in tqdm(ip_range, mininterval=1, disable=disableTqdm, desc='%s-discrete' % name, leave=False):
                    ir = np.int64(i_use[iid,:])
                    m_post = M_all[ir,:]

                    # Compute class probabilities
                    n_count = np.zeros((n_classes,nm))
                    for ic in range(n_classes):
                        n_count[ic,:] = np.sum(class_id[ic]==m_post, axis=0)/nr
                    M_P[iid,:,:] = n_count

                    # Compute mode
                    M_mode[iid,:] = class_id[np.argmax(n_count, axis=0)]

                    # Compute entropy
                    M_entropy[iid,:] = sp.stats.entropy(n_count, base=n_classes)
                    if computeKL_discrete:
                        M_KL[iid, :] = kl_divergence(m_post, M_prior_kl, is_discrete=True, log_base=log_base_kl)

                f_post['/%s/%s' % (name,'Mode')][:] = M_mode
                f_post['/%s/%s' % (name,'Entropy')][:] = M_entropy
                f_post['/%s/%s' % (name,'P')][:] = M_P
                if computeKL_discrete:
                    dset = '/%s/KL' % name
                    if dset not in f_post:
                        f_post.create_dataset(dset, (nsounding, nm))
                    f_post[dset][:] = M_KL


            else: 
                if (showInfo>1):
                    print('%s: NOT RECOGNIZED' % name.upper())
                
            
                

    return None


def sample_from_posterior(is_, d_sim, f_data_h5='tTEM-Djursland.h5', N_use=1000000, autoT=1, ns=400):
    """
    Sample from the posterior distribution.

    Parameters
    ----------
    is_ : int
        Index of data f_data_h5.
    d_sim : ndarray
        Simulated data.
    f_data_h5 : str, optional
        Filepath of the data file. Default is 'tTEM-Djursland.h5'.
    N_use : int, optional
        Number of samples to use. Default is 1000000.
    autoT : int, optional
        Flag indicating whether to estimate temperature. Default is 1.
    ns : int, optional
        Number of samples to draw from the posterior. Default is 400.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        
        i_use : ndarray
            Indices of the samples used.
        T : float
            Temperature.
        EV : float
            Expected value.
        is_index : int
            Index of the posterior sample.
    """
    with h5py.File(f_data_h5, 'r') as f:
        d_obs = f['/D1/d_obs'][is_,:]
        d_std = f['/D1/d_std'][is_,:]
    
    i_use = np.where(~np.isnan(d_obs) & (np.abs(d_obs) > 0))[0]
    d_obs = d_obs[i_use]
    d_var = d_std[i_use]**2

    dd = (d_sim[:, i_use] - d_obs)**2
    #logL = -.5*np.sum(dd/d_var, axis=1)
    logL = np.sum(-0.5 * dd / d_var, axis=1)

    # Compute the annealing temperature
    if autoT == 1:
        T = logl_T_est(logL)
    else:
        T = 1
    maxlogL = np.nanmax(logL)
    
    # Find ns realizations of the posterior, using the log-likelihood values logL, and the annealing tempetrature T 
    i_use, P_acc = lu_post_sample_logl(logL, ns, T)
    
    # Compute the evidence
    exp_logL = np.exp(logL - maxlogL)
    EV = maxlogL + np.log(np.nansum(exp_logL)/len(logL))
    return i_use, T, EV, is_




#def sample_from_posterior_chunk(is_,d_sim,f_data_h5, N_use,autoT,ns):
#    return sample_from_posterior(is_,d_sim,f_data_h5, N_use,autoT,ns) 

#%% integrate_prior_data: updates PRIOR strutcure with DATA
def prior_data(f_prior_in_h5, f_forward_h5, id=1, im=1, doMakePriorCopy=0, parallel=True):
    """
    Update prior structure with forward modeled data.
    
    This function integrates forward modeling results into the prior data structure,
    supporting different data types including TDEM (time-domain electromagnetic) data
    with GA-AEM forward modeling and identity transforms.
    
    Parameters
    ----------
    f_prior_in_h5 : str
        Path to input prior HDF5 file containing prior models.
    f_forward_h5 : str
        Path to forward modeling results HDF5 file.
    id : int, optional
        Data identifier for the prior structure. Default is 1.
    im : int, optional
        Model identifier for the prior structure. Default is 1.
    doMakePriorCopy : int, optional
        Flag to create a copy of the prior file (0=no copy, 1=copy). Default is 0.
    parallel : bool, optional
        Enable parallel processing for forward modeling. Default is True.
    
    Returns
    -------
    str
        Path to the updated prior HDF5 file containing integrated data.
    
    Notes
    -----
    The function automatically detects the data type from the forward modeling file
    and calls appropriate integration methods (GA-AEM for TDEM, identity for direct data).
    Prints error messages for unsupported data types or methods.
    """
    # Check if at least two inputs are provided
    if f_prior_in_h5 is None or f_forward_h5 is None:
        print(f'{__name__}: Use at least two inputs to')
        help(__name__)
        return ''

    # Open HDF5 files
    with h5py.File(f_forward_h5, 'r') as f:
        # Check type=='TDEM'
        if 'type' in f.attrs:
            data_type = f.attrs['type']
        else:
            data_type = 'TDEM'

    f_prior_h5 = ''
    if data_type.lower() == 'tdem':
        # TDEM
        with h5py.File(f_forward_h5, 'r') as f:
            if 'method' in f.attrs:
                method = f.attrs['method']
            else:
                print(f'{__name__}: "TDEM/{method}" not supported')
                return

        if method.lower() == 'ga-aem':
            f_prior_h5, id, im = integrate_prior_data_gaaem(f_prior_in_h5, f_forward_h5, id, im, doMakePriorCopy)
        else:
            print(f'{__name__}: "TDEM/{method}" not supported')
            return
    elif data_type.lower() == 'identity':
        f_prior_h5, id, im = integrate_prior_data_identity(f_prior_in_h5, f_forward_h5, id, im, doMakePriorCopy)
    else:
        print(f'{__name__}: "{data_type}" not supported')
        return

    # update prior data with an attribute defining the prior
    with h5py.File(f_prior_h5, 'a') as f:
        f.attrs[f'/D{id}'] = 'f5_forward'


    integrate_update_prior_attributes(f_prior_h5)

    return f_prior_h5


'''
Forward simulation
'''

def forward_gaaem(C=np.array(()), 
                    thickness=np.array(()), 
                    stmfiles=None, 
                    tx_height=np.array(()), 
                    txrx_dx = -13, 
                    txrx_dy = 0,
                    txrx_dz     = .1,
                    GEX={}, 
                    file_gex=None, 
                    showtime=False, 
                    **kwargs):
    """
    Perform forward modeling using the GA-AEM method.

    Parameters
    ----------
    C : numpy.ndarray, optional
        Conductivity array. Default is np.array(()).
    thickness : numpy.ndarray, optional
        Thickness array. Default is np.array(()).
    stmfiles : list, optional
        List of STM files. Default is None.
    tx_height : numpy.ndarray, optional
        Transmitter height array. Default is np.array(()).
    txrx_dx : float, optional
        X-distance between transmitter and receiver. Default is -13.
    txrx_dy : float, optional
        Y-distance between transmitter and receiver. Default is 0.
    txrx_dz : float, optional
        Z-distance between transmitter and receiver. Default is 0.1.
    GEX : dict, optional
        GEX dictionary. Default is {}.
    file_gex : str, optional
        Path to GEX file. Default is None.
    showtime : bool, optional
        Flag to display execution time. Default is False.
    showInfo : int, optional
        Level of verbosity for output.
    doCompress : bool, optional
        Flag to enable layer compression. Default is True.

    Returns
    -------
    numpy.ndarray
        Forward modeled data array.
    """
    from gatdaem1d import Earth;
    from gatdaem1d import Geometry;
    # Next should probably only be loaded if the DLL is not allready loaded!!!
    from gatdaem1d import TDAEMSystem; # loads the DLL!!
    import integrate as ig
    import time 
    from tqdm import tqdm

    showInfo = kwargs.get('showInfo', 0)
    progress_callback = kwargs.get('progress_callback', None)
    if (showInfo<0):
        disableTqdm=True
    else:
        disableTqdm=False

    doCompress = kwargs.get('doCompress', True)

    # Handle None defaults
    if stmfiles is None:
        stmfiles = []
    if file_gex is None:
        file_gex = ''

    #print(stmfiles)
    #print(file_gex)

    if (len(stmfiles)>0) and (file_gex != '') and (len(GEX)==0):
        # GEX FILE and STM FILES
        if (showInfo)>1:
            print('Using submitted GEX file (%s)' % (file_gex))
        # Try legacy read_gex first, fallback to read_gex_workbench if needed
        try:
            GEX = ig.read_gex(file_gex)
        except (ValueError, KeyError) as e:
            if showInfo > 0:
                print(f"Legacy read_gex() failed ({type(e).__name__}), trying read_gex_workbench()...")
            GEX = ig.read_gex_workbench(file_gex, showInfo=showInfo)
    elif (len(stmfiles)>0):
        # USING STM FILES
        if (showInfo)>1:
            print('Using submitted STM files (%s)' % (stmfiles))

    elif (len(stmfiles)==0) and (file_gex != '') and (len(GEX)==0):
        # ONLY GEX FILE
        stmfiles, GEX = ig.gex_to_stm(file_gex, **kwargs)
    elif (len(stmfiles)>0) and (file_gex == '') and (len(GEX)>0):
        # Using GEX dict and STM FILES
        a = 1
    elif (len(GEX)>0) and (len(stmfiles)>1):
        # using the GEX file in stmfiles
        print('Using submitted GEX and STM files')
    elif (len(GEX)>0) and (len(stmfiles)==0):
        # using GEX file and writing STM files
        print('Using submitted GEX and writing STM files')
        stmfiles = ig.write_stm_files(GEX, **kwargs)
    elif (len(GEX)==0) and (len(stmfiles)>1):
        if (file_gex == ''):
            if (showInfo>-1):
                print('Using STM files without GEX file')
            #return -1
        else:
            print('Converting STM files to GEX')
            # Try legacy read_gex first, fallback to read_gex_workbench if needed
            try:
                GEX = ig.read_gex(file_gex)
            except (ValueError, KeyError) as e:
                if showInfo > 0:
                    print(f"Legacy read_gex() failed ({type(e).__name__}), trying read_gex_workbench()...")
                GEX = ig.read_gex_workbench(file_gex, showInfo=showInfo)
    elif (len(GEX)>0) and (len(stmfiles)==0):
        stmfiles, GEX = ig.gex_to_stm(file_gex, **kwargs)
    elif (file_gex != ''):
        a=1
        #stmfiles, GEX = ig.gex_to_stm(file_gex, **kwargs)
    else:   
        print('Error: No GEX or STM files provided')
        return -1

    if (showInfo>0):
        print('Using STM files : ')
        print(stmfiles)

    if (showInfo>1):        
        if 'filename' in GEX:
            print('Using GEX file: ', GEX['filename'])

    nstm=len(stmfiles)
    if (showInfo>0):
        for i in range(len(stmfiles)):
            print('Using MOMENT:', stmfiles[i])

    if C.ndim==1:
        nd=1
        nl=C.shape[0]
    else:
        nd,nl=C.shape

    nt = thickness.shape[0]
    if nt != (nl-1):
        raise ValueError('Error: thickness array (nt=%d) does not match the number of layers minus 1(nl=%d)' % (nt,nl))

    if (showInfo>0):
        print('nd=%s, nl=%d,  nstm=%d' %(nd,nl,nstm))

    # SETTING UP t1=time.time()
    t1=time.time()
    
    S_LM = TDAEMSystem(stmfiles[0])
    if nstm>1:
        S_HM = TDAEMSystem(stmfiles[1])
        S=[S_LM, S_HM]
    else:
        S=[S_LM]
    t2=time.time()
    t_system = 1000*(t2-t1)
    if showtime:
        print("Time, Setting up systems = %4.1fms" % t_system)

    # Setting up geometry
    if len(GEX)>0:
        # Try legacy read_gex first, fallback to read_gex_workbench if needed
        try:
            GEX = ig.read_gex(file_gex)
        except (ValueError, KeyError) as e:
            if showInfo > 0:
                print(f"Legacy read_gex() failed ({type(e).__name__}), trying read_gex_workbench()...")
            GEX = ig.read_gex_workbench(file_gex, showInfo=showInfo)
        if 'TxCoilPosition1' in GEX['General']:
            # Typical for tTEM system
            txrx_dx = float(GEX['General']['RxCoilPosition1'][0])-float(GEX['General']['TxCoilPosition1'][0])
            txrx_dy = float(GEX['General']['RxCoilPosition1'][1])-float(GEX['General']['TxCoilPosition1'][1])
            txrx_dz = float(GEX['General']['RxCoilPosition1'][2])-float(GEX['General']['TxCoilPosition1'][2])
            if len(tx_height)==0:
                tx_height = -float(GEX['General']['TxCoilPosition1'][2])
                tx_height=np.array([tx_height])

        else:
            # Typical for SkyTEM system
            txrx_dx = float(GEX['General']['RxCoilPosition1'][0])
            txrx_dy = float(GEX['General']['RxCoilPosition1'][1])
            txrx_dz = float(GEX['General']['RxCoilPosition1'][2])
            if len(tx_height)==0:
                tx_height=np.array([40])
    

        # Set geometry once, if tx_height has one value
        if len(tx_height)==1:
            if (showInfo>1):
                print('Using tx_height=%f' % tx_height[0])
            G = Geometry(tx_height=float(tx_height[0]), txrx_dx = txrx_dx, txrx_dy = txrx_dy, txrx_dz = txrx_dz)
        if (showInfo>1):
            print('tx_height=%f, txrx_dx=%f, txrx_dy=%f, txrx_dz=%f' % (tx_height[0], txrx_dx, txrx_dy, txrx_dz))
        
        # Handle both scalar and array values for NumPy 2.x compatibility
        no_gates_ch1 = np.atleast_1d(GEX['Channel1']['NoGates'])[0]
        remove_gates_ch1 = np.atleast_1d(GEX['Channel1']['RemoveInitialGates'])[0]
        ng0 = no_gates_ch1 - remove_gates_ch1
        if nstm>1:
            no_gates_ch2 = np.atleast_1d(GEX['Channel2']['NoGates'])[0]
            remove_gates_ch2 = np.atleast_1d(GEX['Channel2']['RemoveInitialGates'])[0]
            ng1 = no_gates_ch2 - remove_gates_ch2
        else:
            ng1 = 0
        ng = int(ng0+ng1)
    
    else:
        if len(tx_height)==0:
            tx_height=np.array([0])
        G = Geometry(tx_height=float(tx_height[0]), txrx_dx = txrx_dx, txrx_dy = txrx_dy, txrx_dz = txrx_dz)
        # Here we should read the number of gates from the lines in STMFILES that conatin 'NumberOfWindows = 41'
        ng = 41

    # pinrt txrx_dx, txrx_dy, txrx_dz
    if (showInfo>0):
        print('txrx_dx=%f, txrx_dy=%f, txrx_dz=%f' % (txrx_dx, txrx_dy, txrx_dz))
        print('ng=%d' % ng)
        

    D = np.zeros((nd,ng))

    # Compute forward data
    t1=time.time()
    # Throttle callback to ~100 updates so it does not dominate runtime
    progress_step = max(1, nd // 100)

    for i in tqdm(range(nd), mininterval=1, disable=disableTqdm, desc='gatdaem1d', leave=False):
        if progress_callback and ((i + 1) % progress_step == 0 or i + 1 == nd):
            _report_progress(progress_callback, i + 1, nd,
                             'computing', 'Forward modeling (%d/%d soundings)' % (i + 1, nd))
        if C.ndim==1:
            # Only one model
            conductivity = C
        else:
            conductivity = C[i]

        # Update geometry, tx_height is changing!
        if len(tx_height)>1:
            if (showInfo>1):
                print('Using tx_height=%f' % tx_height[i])
            G = Geometry(tx_height=float(tx_height[i]), txrx_dx = txrx_dx, txrx_dy = txrx_dy, txrx_dz = txrx_dz)
    
        #doCompress=True
        if doCompress:
            i_change=np.where(np.diff(conductivity) != 0 )[0]+1
            n_change = len(i_change)
            conductivity_compress = np.zeros(n_change+1)+conductivity[0]
            thickness_compress = np.zeros(n_change)
            for il in range(n_change):
                conductivity_compress[il+1] = conductivity[i_change[il]]
                if il==0:
                    thickness_compress[il]=np.sum(thickness[0:i_change[il]])
                else:   
                    i1=i_change[il-1]
                    i2=i_change[il]
                    #print("i1: %d, i2: %d" % (i1, i2))
                    thickness_compress[il]=np.sum(thickness[i1:i2]) 
            E = Earth(conductivity_compress,thickness_compress)
        else:   
            E = Earth(conductivity,thickness)

        fm0 = S[0].forwardmodel(G,E)
        d = -fm0.SZ
        if nstm>1:
            fm1 = S[1].forwardmodel(G,E)
            d1 = -fm1.SZ
            d = np.concatenate((d,d1))    

        D[i] = d    

        '''
        fm_lm = S_LM.forwardmodel(G,E)
        fm_hm = S_HM.forwardmodel(G,E)
        # combine -fm_lm.SZ and -fm_hm.SZ
        d = np.concatenate((-fm_lm.SZ,-fm_hm.SZ))
        d_ref = D[i]
        '''
        
    t2=time.time()
    if showtime:
        print("Time = %4.1fms per model and %d model tests" % (1000*(t2-t1)/nd, nd))

    return D

def forward_gaaem_chunk(C_chunk, tx_height_chunk, thickness, stmfiles, file_gex, Nhank, Nfreq, **kwargs):
    """
    Perform forward modeling using the GA-AEM method on a chunk of data.

    Parameters
    ----------
    C_chunk : numpy.ndarray
        The chunk of data to be processed.
    tx_height_chunk : numpy.ndarray
        The transmitter heights for this chunk.
    thickness : float
        The thickness of the model.
    stmfiles : list
        A list of STM files.
    file_gex : str
        The path to the GEX file.
    Nhank : int
        The number of Hankel functions.
    Nfreq : int
        The number of frequencies.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    numpy.ndarray
        The result of the forward modeling.
    """
    return forward_gaaem(C=C_chunk, 
                        thickness=thickness, 
                        tx_height=tx_height_chunk,
                        stmfiles=stmfiles, 
                        file_gex=file_gex, 
                        Nhank=Nhank, 
                        Nfreq=Nfreq, 
                        parallel=False, 
                        **kwargs)

# %% PRIOR DATA GENERATORS

# Add this function to check current handle count (Windows only)
def get_process_handle_count():
    """
    Return the number of handles used by the current process (Windows only).
    
    Returns
    -------
    int
        The number of handles used by the current process.
    """
    import psutil
    import os
    return psutil.Process(os.getpid()).num_handles()

def prior_data_gaaem(f_prior_h5, file_gex=None, stmfiles=None, N=0, doMakePriorCopy=True, im=1, id=1, im_height=0, Nhank=280, Nfreq=12, is_log=False, parallel=True, force_replace=False, **kwargs):
    """
    Generate prior data for the GA-AEM method.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the prior data file in HDF5 format.
    file_gex : str, optional
        Path to the file containing geophysical exploration data (.gex format).
    stmfiles : list of str, optional
        List of STM files for system configuration. If not provided, will be
        generated from file_gex.
    N : int, optional
        Number of soundings to consider. Default is 0 (use all).
    doMakePriorCopy : bool, optional
        Flag indicating whether to make a copy of the prior file. Default is True.
    im : int, optional
        Index of the model. Default is 1.
    id : int, optional
        Index of the data. Default is 1.
    im_height : int, optional
        Index of the model for height. Default is 0.
    Nhank : int, optional
        Number of Hankel transform quadrature points. Default is 280.
    Nfreq : int, optional
        Number of frequencies. Default is 12.
    is_log : bool, optional
        Flag to apply logarithmic scaling to data. Default is False.
    parallel : bool, optional
        Flag indicating whether multiprocessing is used. Default is True.
        When True, forward modeling is parallelized across available CPUs.
    **kwargs : dict
        Additional keyword arguments:

        Ncpu : int, optional
            Number of CPUs to use for parallel processing. Default is 0, which
            uses all available CPUs. Only used when parallel=True.
        force_replace : bool, optional
            If True, delete an existing /D{id} dataset before writing.
            If False (default), print a warning and return early if the
            dataset already exists.
        showInfo : int, optional
            Level of verbosity for output (0=silent, 1=normal, 2=verbose).

    Returns
    -------
    str
        Filename of the HDF5 file containing the updated prior data.

    Notes
    -----
    This function computes forward-modeled electromagnetic responses for prior
    model realizations using the GA-AEM forward modeling code. The forward
    modeling can be parallelized for faster computation on multi-core systems.

    Examples
    --------
    >>> # Basic usage with all CPUs
    >>> f_prior_data = prior_data_gaaem(f_prior_h5, file_gex)

    >>> # Use specific number of CPUs
    >>> f_prior_data = prior_data_gaaem(f_prior_h5, file_gex, Ncpu=4)

    >>> # Sequential processing (no parallelization)
    >>> f_prior_data = prior_data_gaaem(f_prior_h5, file_gex, parallel=False)
    """
    import integrate as ig
    import os
    # Safety guard: if somehow called from a worker process, do nothing.
    if multiprocessing.current_process().name != 'MainProcess':
        return None

    type = 'TDEM'
    method = 'ga-aem'
    showInfo = kwargs.get('showInfo', 0)
    Ncpu = kwargs.get('Ncpu', 0)
    # of 'Nproc' is set in kwargs use it
    Ncpu = kwargs.get('Nproc', Ncpu)
    # Pop (not get): the callback must never be pickled to worker processes
    progress_callback = kwargs.pop('progress_callback', None)

    if showInfo>0:
        print('prior_data_gaaem: %s/%s -- starting' % (type, method))

    # Force open/close of hdf5 file
    if showInfo>0:
        print('Forcing open and close of %s' % (f_prior_h5))
    with h5py.File(f_prior_h5, 'r') as f:
        # open and close
        pass

    with h5py.File(f_prior_h5, 'r') as f:
        N_in = f['M1'].shape[0]
    if N==0: 
        N = N_in     
    if N>N_in:
        N=N_in

    # if is not None file_gex
    if (file_gex is not None):
        if not os.path.isfile(file_gex):
            print("ERRROR: file_gex=%s does not exist in the current folder." % file_gex)

    if (stmfiles is not None):
        for i in range(len(stmfiles)):
            if not os.path.isfile(stmfiles[i]):
                print("ERRROR: stmfiles[%d]=%s does not exist in the current folder." % (i,stmfiles[i]))
 

    if doMakePriorCopy:

        # If file_gex is not None, then use it to get the file_base_name
        if (file_gex is not None) and os.path.isfile(file_gex): 
            file_basename = os.path.splitext(os.path.basename(file_gex))[0]
        elif (stmfiles is not None) and (len(stmfiles)>0):
            file_basename = os.path.splitext(os.path.basename(stmfiles[0]))[0]
        else:
            file_basename = 'GAAEM'
        
        print('Using file_basename=%s' % file_basename)

        if N < N_in:
            f_prior_data_h5 = '%s_%s_N%d_Nh%d_Nf%d.h5' % (os.path.splitext(f_prior_h5)[0], os.path.splitext(file_basename)[0], N, Nhank, Nfreq)
        else:
            f_prior_data_h5 = '%s_%s_Nh%d_Nf%d.h5' % (os.path.splitext(f_prior_h5)[0], os.path.splitext(file_basename)[0], Nhank, Nfreq)
        

        if (showInfo>0):
            print("Creating a copy of %s" % (f_prior_h5))
            print("                as %s" % (f_prior_data_h5))
        if (showInfo>1):
                print('  using N=%d of N_in=%d data' % (N,N_in))
        
        # make a copy of the prior file
        ig.copy_hdf5_file(f_prior_h5, f_prior_data_h5,N,showInfo=showInfo)
            
    else:
        f_prior_data_h5 = f_prior_h5

    
    Mname = '/M%d' % im
    Mheight = '/M%d' % im_height
    Dname = '/D%d' % id


    with h5py.File(f_prior_data_h5, 'r') as f_prior_r:
        if im_height>0:
            if (showInfo>1):
                print('Using M%d for height' % im_height)
            tx_height = f_prior_r[Mheight][:]

        # Get thickness
        if 'x' in f_prior_r[Mname].attrs:
            z = f_prior_r[Mname].attrs['x']
        else:
            z = f_prior_r[Mname].attrs['z']
        thickness = np.diff(z)

        # Get conductivity
        if Mname in f_prior_r.keys():
            C = 1 / f_prior_r[Mname][:]
        else:
            print('Could not load %s from %s' % (Mname, f_prior_data_h5))

        N = f_prior_r[Mname].shape[0]

    t1 = time.time()
    if not parallel:
        if (showInfo>-1):
            print("prior_data_gaaem: Using 1 thread /(sequential).")
        # Sequential
        if im_height>0:
            if (showInfo>0):
                print('Using tx_height')
            D = ig.forward_gaaem(C=C,
                                 thickness=thickness,
                                 tx_height=tx_height,
                                 file_gex=file_gex,
                                 stmfiles=stmfiles,
                                 Nhank=Nhank,
                                 Nfreq=Nfreq,
                                 parallel=parallel,
                                 progress_callback=progress_callback, **kwargs)
        else:
            D = ig.forward_gaaem(C=C,
                                 thickness=thickness,
                                 file_gex=file_gex,
                                 stmfiles=stmfiles,
                                 Nhank=Nhank,
                                 Nfreq=Nfreq,
                                 parallel=parallel,
                                 progress_callback=progress_callback, **kwargs)
        if is_log:
            D = np.log10(D)
    else:

        # Make sure STM files are only written once!!! (need for multihreading)
        # D = ig.forward_gaaem(C=C[0:1,:], thickness=thickness, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=False, **kwargs)
        if stmfiles is None or len(stmfiles)==0:
            stmfiles, _ = ig.gex_to_stm(file_gex, Nhank=Nhank, Nfreq=Nfreq, **kwargs)

        # Parallel
        if Ncpu < 1 :
            #Ncpu =  int(multiprocessing.cpu_count()/2)
            Ncpu =  int(multiprocessing.cpu_count())
        if (showInfo>-1):
            print("prior_data_gaaem: Using %d parallel threads." % (Ncpu))

        # 1: Define a function to compute a chunk
        ## OUTSIDE
        # 2: Create chunks
        if progress_callback is None:
            n_chunks = Ncpu
        else:
            # Finer chunking gives smoother live progress updates
            n_chunks = min(C.shape[0], Ncpu * 4)
        C_chunks = np.array_split(C, n_chunks)

        if im_height>0:
            tx_height_chunks = np.array_split(tx_height, n_chunks)

        else:
            # create tx_height_chunks as a list of length n_chunks, where each entry is tx_height=np.array(())
            tx_height_chunks = [np.array(())]*n_chunks


        import os

        # 3: Compute the chunks in parallel
        forward_gaaem_chunk_partial = partial(forward_gaaem_chunk, thickness=thickness, stmfiles=stmfiles, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, **kwargs)

        # On Windows and macOS, multiprocessing uses 'spawn' which normally
        # re-executes the user's __main__ script in every worker process.
        # We prevent this by setting __main__.__spec__ = SimpleNamespace(name='__main__')
        # before creating the Pool.  The spawn bootstrap then calls
        # _fixup_main_from_name('__main__'), which immediately returns because the
        # worker's bootstrap module already has __name__ == '__main__' — so the
        # user's script is never re-run in workers.  No if __name__=='__main__' guard
        # is needed in user scripts on any platform.
        _main_module = sys.modules.get('__main__')
        _spec_patched = _main_module is not None and getattr(_main_module, '__spec__', None) is None
        if _spec_patched:
            _main_module.__spec__ = types.SimpleNamespace(name='__main__')

        is_spawn = os.name == 'nt' or (os.name == 'posix' and os.uname().sysname == 'Darwin')
        try:
            if is_spawn:
                if os.name == 'nt':
                    Ncpu = min(Ncpu, 60)  # Windows handle limit
                ctx = multiprocessing.get_context('spawn')
            else:
                ctx = multiprocessing.get_context('fork')
            with ctx.Pool(processes=Ncpu) as p:
                if progress_callback is None:
                    D_chunks = p.starmap(forward_gaaem_chunk_partial, zip(C_chunks, tx_height_chunks))
                else:
                    # apply_async + ordered get() keeps chunk order for the
                    # concatenate below while reporting per finished chunk
                    async_results = [p.apply_async(forward_gaaem_chunk_partial, args=(Cc, th))
                                     for Cc, th in zip(C_chunks, tx_height_chunks)]
                    D_chunks = []
                    n_total = C.shape[0]
                    n_done = 0
                    for r in async_results:
                        D_chunk = r.get()
                        D_chunks.append(D_chunk)
                        n_done += D_chunk.shape[0]
                        _report_progress(progress_callback, n_done, n_total,
                                         'computing', 'Forward modeling (%d/%d soundings)' % (n_done, n_total))
        finally:
            if _spec_patched:
                _main_module.__spec__ = None

  
        D = np.concatenate(D_chunks)
        
        if is_log:
            D = np.log10(D)

        if os.name == 'nt' and 'get_process_handle_count' in globals():
            # Log handle count after pool is closed
            handle_count_after = get_process_handle_count()
            #   print(f"Handle count after pool: {handle_count_after}")


        # D = ig.forward_gaaem(C=C, thickness=thickness, file_gex=file_gex, Nhank=Nhank, Nfreq=Nfreq, parallel=parallel, **kwargs)

    t2 = time.time()
    t_elapsed = t2 - t1
    if (showInfo>-1):
        print('prior_data_gaaem: Time=%5.1fs/%d soundings. %4.1fms/sounding, %3.1fit/s' % (t_elapsed, N, 1000*t_elapsed/N,N/t_elapsed))

    _report_progress(progress_callback, N, N,
                     'saving', 'Saving forward data to %s' % f_prior_data_h5)

    # Write D to f_prior['/D1']
    with h5py.File(f_prior_data_h5, 'a') as f_prior:
        if Dname in f_prior:
            if force_replace:
                del f_prior[Dname]
            else:
                print("Key '%s' already exists in %s. Use force_replace=True to overwrite." % (Dname, f_prior_data_h5))
                return f_prior_data_h5
        f_prior[Dname] = D

        # Add method, type, file_ex, and im as attributes to '/D1'
        f_prior[Dname].attrs['method'] = method
        f_prior[Dname].attrs['type'] = type
        f_prior[Dname].attrs['im'] = im
        f_prior[Dname].attrs['Nhank'] = Nhank
        f_prior[Dname].attrs['Nfreq'] = Nfreq

    integrate_update_prior_attributes(f_prior_data_h5)

    _report_progress(progress_callback, N, N,
                     'completed', 'Forward data saved to %s' % f_prior_data_h5)

    return f_prior_data_h5


def prior_data_identity(f_prior_h5, id=0, im=1, N=0, doMakePriorCopy=False, **kwargs):
    """
    Generate data D{id} from model M{im} in the prior file f_prior_h5 as an identity of M{im}.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the prior data file in HDF5 format.
    id : int, optional
        Index of the data. If id=0, the next available data id is used. Default is 0.
    im : int, optional
        Index of the model. Default is 1.
    N : int, optional
        Number of soundings to consider. Default is 0 (use all).
    doMakePriorCopy : bool, optional
        Flag indicating whether to make a copy of the prior file. Default is False.
    showInfo : int, optional
        Level of verbosity for output.
    forceDeleteExisting : bool, optional
        Flag to force deletion of existing data. Default is True.

    Returns
    -------
    str
        Path to the HDF5 file containing the updated prior data.
    """
    import integrate as ig
    import time

    type = 'idenity'
    method = '--'
    showInfo = kwargs.get('showInfo', 0)
    forceDeleteExisting = kwargs.get('forceDeleteExisting', True)


    # check keys for the data with max id form 'D1', 'D2', 'D3', ...
    if id==0:
        with h5py.File(f_prior_h5, 'a') as f_prior:
            id = 1
            for id_test in range(999):
                key = '/D%d' % id_test
                if key in f_prior.keys():
                    if showInfo>1:
                        print('Checking key EXISTS: %s' % key)
                    id = id_test+1
                else:                    
                    pass
            if showInfo>0:    
                print('using id = %d' % id)    
        
    
    with h5py.File(f_prior_h5, 'a') as f:
        N_in = f['M1'].shape[0]
    if N==0: 
        N = N_in     
    if N>N_in:
        N=N_in

    if showInfo>2:
        print('N=%d, N_in=%d' % (N,N_in))
    if doMakePriorCopy:
        if N < N_in:
            f_prior_data_h5 = '%s_N%s_IDEN_im%d_id%d.h5' % (os.path.splitext(f_prior_h5)[0], N, im, id)
        else:
            f_prior_data_h5 = '%s_IDEN_im%d_id%d.h5' % (os.path.splitext(f_prior_h5)[0], im, id)
        if (showInfo>0):
            print("Creating a copy of %s as %s" % (f_prior_h5, f_prior_data_h5))
        ig.copy_hdf5_file(f_prior_h5, f_prior_data_h5,N)
        
    else:
        f_prior_data_h5 = f_prior_h5

    Mname = '/M%d' % im
    Dname = '/D%d' % id

    # copy f_prior[Mname] to Dname
    if showInfo>0:
        print('Copying %s to %s in filename=%s' % (Mname, Dname, f_prior_data_h5))

    # f_prior = h5py.File(f_prior_data_h5, 'r+')
    with h5py.File(f_prior_data_h5, 'a') as f:
        D = f[Mname]
        # check if Dname exists, if so, delete it
        if Dname in f.keys():
            if forceDeleteExisting:
                print('Key %s allready exists -- DELETING !!!!' % Dname)
                del f[Dname]
            else:
                print('Key %s allready exists - doing nothing' % Dname)
                return f_prior_data_h5
        
        dataset = f.create_dataset(Dname, data=D)  # 'i4' represents 32-bit integers
        dataset.attrs['description'] = 'Identiy of %s' % Mname
        dataset.attrs['f5_forward'] = 'none'
        dataset.attrs['with_noise'] = 0
        #f_prior.close()
    
    return f_prior_data_h5, id

# %% PRIOR MODEL GENERATORS
def _report_progress(progress_callback, current, total, phase, status):
    """Invoke an optional progress callback as progress_callback(current, total, info_dict).

    Follows the same convention as integrate_rejection(): info_dict carries
    'phase' and 'status' keys, and errors raised by the callback are ignored
    so they can never break the computation.
    """
    if progress_callback is None:
        return
    try:
        progress_callback(current, total, {'phase': phase, 'status': status})
    except Exception:
        pass


def prior_model_layered(lay_dist='uniform', dz = 1, z_max = 90,
                        NLAY_min=3, NLAY_max=6, NLAY_deg=6,
                        RHO_dist='log-uniform', RHO_min=0.1, RHO_max=5000, RHO_mean=100, RHO_std=80,
                        N=100000, save_sparse=True, RHO_threshold=0.001, **kwargs):
    """
    Generate a prior model with layered structure.

    This optimized implementation uses vectorized NumPy operations for improved
    performance, providing ~2x speedup for large N compared to the original
    loop-based implementation.

    Parameters
    ----------
    lay_dist : str, optional
        Distribution of the number of layers. Options are 'chi2' and 'uniform'.
        Default is 'uniform'.
    dz : float, optional
        Depth discretization step. Default is 1.
    z_max : float, optional
        Maximum depth in m. Default is 90.
    NLAY_min : int, optional
        Minimum number of layers. Default is 3.
    NLAY_max : int, optional
        Maximum number of layers. Default is 6.
    NLAY_deg : int, optional
        Degrees of freedom for chi-square distribution. Only applicable if
        lay_dist is 'chi2'. Default is 6.
    RHO_dist : str, optional
        Distribution of resistivity within each layer. Options are 'log-uniform',
        'uniform', 'normal', and 'lognormal'. Default is 'log-uniform'.
    RHO_min : float, optional
        Minimum resistivity value. Default is 0.1.
    RHO_max : float, optional
        Maximum resistivity value. Default is 5000.
    RHO_mean : float, optional
        Mean resistivity value. Only applicable if RHO_dist is 'normal' or
        'lognormal'. Default is 100.
    RHO_std : float, optional
        Standard deviation of resistivity value. Only applicable if RHO_dist is
        'normal' or 'lognormal'. Default is 80.
    N : int, optional
        Number of prior models to generate. Default is 100000.
    save_sparse : bool, optional
        Whether to save the sparse representation (M2: depth-resistivity pairs)
        to the HDF5 file. Setting to False can reduce file size and processing
        time for large priors. Default is True.
    RHO_threshold : float, optional
        Minimum physical resistivity threshold in Ohm·m. Any generated resistivity
        values below this threshold (including zero or negative values from 'normal'
        distribution) will be clamped to this value. Ensures physically realistic
        positive resistivity values. Default is 0.001 Ohm·m.
    f_prior_h5 : str, optional
        Path to the prior model file in HDF5 format. Default is ''.
    showInfo : int, optional
        Level of verbosity for output.

    Returns
    -------
    str
        Filepath of the saved prior model.

    Notes
    -----
    This implementation pre-generates all random values using vectorized NumPy
    operations, significantly improving performance for large N (e.g., N=50000).
    """

    # Safety guard: if somehow called from a worker process, do nothing.
    if multiprocessing.current_process().name != 'MainProcess':
        return None

    import integrate as ig

    showInfo = kwargs.get('showInfo', 0)
    f_prior_h5 = kwargs.get('f_prior_h5', '')
    progress_callback = kwargs.get('progress_callback', None)

    if NLAY_max < NLAY_min:
        NLAY_max = NLAY_min

    if NLAY_min < 1:
        NLAY_min = 1

    # Generate number of layers for all models at once
    if lay_dist == 'uniform':
        NLAY = np.random.randint(NLAY_min, NLAY_max+1, N)
        if len(f_prior_h5)<1:
            f_prior_h5 = 'PRIOR_UNIFORM_NL_%d-%d_%s_N%d.h5' % (NLAY_min, NLAY_max, RHO_dist, N)

    elif lay_dist == 'chi2':
        NLAY = np.random.chisquare(NLAY_deg, N)
        NLAY = np.ceil(NLAY).astype(int)
        if len(f_prior_h5)<1:
            f_prior_h5 = 'PRIOR_CHI2_NF_%d_%s_N%d.h5' % (NLAY_deg, RHO_dist, N)
        NLAY_max = np.max(NLAY)

    # Setup depth discretization
    z_min = 0
    nz = int(np.ceil((z_max - z_min) / dz)) + 1
    z = np.linspace(z_min, z_max, nz)

    # Pre-allocate output arrays
    M_rho = np.zeros((N, nz), dtype=np.float32)
    nm_sparse = NLAY_max + NLAY_max - 1

    if save_sparse:
        M_rho_sparse = np.ones((N, nm_sparse), dtype=np.float32) * np.nan
    else:
        M_rho_sparse = None

    # Generate all layer boundaries at once
    max_boundaries = NLAY_max - 1

    if max_boundaries > 0:
        # Generate random boundaries for all models at once
        # NOTE: Do NOT sort! Boundaries are at random locations
        z_boundaries_all = np.random.random((N, max_boundaries)) * z_max
        # Convert to indices
        i_boundaries_all = np.searchsorted(z, z_boundaries_all)
        i_boundaries_all = np.minimum(i_boundaries_all, nz - 1)
    else:
        z_boundaries_all = np.zeros((N, 0))
        i_boundaries_all = np.zeros((N, 0), dtype=int)

    # Generate all resistivity values at once
    if RHO_dist == 'log-normal':
        rho_all = np.random.lognormal(mean=np.log(RHO_mean), sigma=np.log(RHO_std), size=(N, NLAY_max))
    elif RHO_dist == 'normal':
        rho_all = np.random.normal(loc=RHO_mean, scale=RHO_std, size=(N, NLAY_max))
    elif RHO_dist == 'log-uniform':
        rho_all = np.exp(np.random.uniform(np.log(RHO_min), np.log(RHO_max), (N, NLAY_max)))
    elif RHO_dist == 'uniform':
        rho_all = np.random.uniform(RHO_min, RHO_max, (N, NLAY_max))

    # Ensure physical resistivity values (must be positive)
    # First clamp to threshold to handle zero/negative values
    rho_all = np.maximum(rho_all, RHO_threshold)

    # Then clip to user-specified bounds, ensuring RHO_min is at least threshold
    effective_rho_min = max(RHO_min, RHO_threshold)
    rho_all = np.clip(rho_all, effective_rho_min, RHO_max)

    # Assign resistivity values to depth profiles
    if showInfo > 0:
        from tqdm import tqdm
        iterator = tqdm(range(N), mininterval=1, desc='prior_layered', leave=False)
    else:
        iterator = range(N)

    # Throttle callback to ~100 updates so it does not dominate runtime
    progress_step = max(1, N // 100)

    for i in iterator:
        if progress_callback and ((i + 1) % progress_step == 0 or i + 1 == N):
            _report_progress(progress_callback, i + 1, N,
                             'generating', 'Generating prior realizations')
        n_lay = NLAY[i]
        n_boundaries = n_lay - 1

        # Start with first layer resistivity
        M_rho[i, :] = rho_all[i, 0]

        # Apply boundaries if any exist
        if n_boundaries > 0:
            boundaries = np.sort(i_boundaries_all[i, :n_boundaries])
            for j in range(n_boundaries):
                M_rho[i, boundaries[j]:] = rho_all[i, j + 1]

        # Save sparse representation if requested
        if save_sparse:
            if n_boundaries > 0:
                z_sorted = np.sort(z_boundaries_all[i, :n_boundaries])
                m_current = np.concatenate((z_sorted, rho_all[i, :n_lay]))
            else:
                m_current = rho_all[i, :n_lay]
            M_rho_sparse[i, 0:len(m_current)] = m_current

    if showInfo > 0:
        print("prior_model_layered: Saving prior model to %s" % f_prior_h5)
    _report_progress(progress_callback, N, N,
                     'saving', 'Saving prior model to %s' % f_prior_h5)

    # Save to HDF5 file
    im = 0

    # Extract compression parameters from kwargs if provided
    # Build a dict of save_prior_model kwargs
    save_kwargs = {}
    if 'compression' in kwargs:
        save_kwargs['compression'] = kwargs['compression']
    if 'compression_opts' in kwargs:
        save_kwargs['compression_opts'] = kwargs['compression_opts']

    if showInfo > 1:
        print("Saving '/M1' prior model  %s" % f_prior_h5)
    im = im + 1
    ig.save_prior_model(f_prior_h5, M_rho,
                im=im,
                name='resistivity',
                is_discrete=0,
                x=z,
                z=z,
                delete_if_exist=True,
                force_replace=True,
                showInfo=showInfo,
                **save_kwargs,
                )

    if save_sparse:
        if showInfo > 1:
            print("Saving '/M2' prior model  %s" % f_prior_h5)
        im = im + 1
        ig.save_prior_model(f_prior_h5, M_rho_sparse,
                    im=im,
                    name='sparse - depth-resistivity',
                    is_discrete=0,
                    x=np.arange(0, nm_sparse),
                    z=np.arange(0, nm_sparse),
                    force_replace=True,
                    showInfo=showInfo,
                    **save_kwargs,
                    )

    if showInfo > 1:
        print("Saving '/M%d' prior model  %s" % (im,f_prior_h5))
    im = im + 1
    NLAY_2d = NLAY[:, np.newaxis] if NLAY.ndim == 1 else NLAY
    ig.save_prior_model(f_prior_h5, NLAY_2d.astype(np.float32),
                        im=im,
                        name='Number of layers',
                        is_discrete=0,
                        x=np.array([0]),
                        z=np.array([0]),
                        force_replace=True,
                        showInfo=showInfo,
                        **save_kwargs,
                )

    _report_progress(progress_callback, N, N,
                     'completed', 'Prior model saved to %s' % f_prior_h5)

    return f_prior_h5

def prior_model_workbench_direct(N=100000, RHO_dist='log-uniform', z1=0, z_max= 100,
                          nlayers=0, p=2, NLAY_min=3, NLAY_max=6,
                          RHO_min = 1, RHO_max= 300, RHO_mean=180, RHO_std=80, chi2_deg= 100,
                          RHO_threshold=0.001, **kwargs):
    """
    Generate a prior model with increasingly thick layers.
    
    All models have the same number of layers! See also: prior_model_workbench.
 
    Parameters
    ----------
    N : int, optional
        Number of prior models to generate. Default is 100000.
    RHO_dist : str, optional
        Distribution of resistivity within each layer. Options are 'log-uniform', 
        'uniform', 'normal', 'lognormal', and 'chi2'. Default is 'log-uniform'.
    z1 : float, optional
        Minimum depth value. Default is 0.
    z_max : float, optional
        Maximum depth value. Default is 100.
    nlayers : int, optional
        Number of layers. Default is 0 (uses 30 if less than 1).
    p : int, optional
        Power parameter for thickness increase. Default is 2.
    NLAY_min : int, optional
        Minimum number of layers. Default is 3.
    NLAY_max : int, optional
        Maximum number of layers. Default is 6.
    RHO_min : float, optional
        Minimum resistivity value. Default is 1.
    RHO_max : float, optional
        Maximum resistivity value. Default is 300.
    RHO_mean : float, optional
        Mean resistivity value. Only applicable if RHO_dist is 'normal' or 
        'lognormal'. Default is 180.
    RHO_std : float, optional
        Standard deviation of resistivity value. Only applicable if RHO_dist is
        'normal' or 'lognormal'. Default is 80.
    chi2_deg : int, optional
        Degrees of freedom for chi2 distribution. Only applicable if RHO_dist is
        'chi2'. Default is 100.
    RHO_threshold : float, optional
        Minimum physical resistivity threshold in Ohm·m. Any generated resistivity
        values below this threshold (including zero or negative values from 'normal'
        distribution) will be clamped to this value. Ensures physically realistic
        positive resistivity values. Default is 0.001 Ohm·m.
    f_prior_h5 : str, optional
        Path to the prior model file in HDF5 format. Default is ''.
    showInfo : int, optional
        Level of verbosity for output.

    Returns
    -------
    str
        Filepath of the saved prior model.
    """
    # Safety guard: if somehow called from a worker process, do nothing.
    if multiprocessing.current_process().name != 'MainProcess':
        return None

    import integrate as ig

    showInfo = kwargs.get('showInfo', 0)
    f_prior_h5 = kwargs.get('f_prior_h5', '')
    progress_callback = kwargs.get('progress_callback', None)

    # Fully vectorized: only coarse phase updates are possible
    _report_progress(progress_callback, 0, 100,
                     'generating', 'Generating prior realizations')

    if nlayers<1:
        nlayers = 30

    z2=z_max
    z= z1 + (z2 - z1) * np.linspace(0, 1, nlayers) ** p

    nz = len(z)

    if RHO_dist=='uniform':
        M_rho = np.random.uniform(low=RHO_min, high = RHO_max, size=(N, nz))
        if len(f_prior_h5)<1:
            f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_min, RHO_max)
    elif RHO_dist=='log-uniform':
        M_rho = np.exp(np.random.uniform(low=np.log(RHO_min), high = np.log(RHO_max), size=(N, nz)))
        if len(f_prior_h5)<1:
            f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_min, RHO_max)
    elif RHO_dist=='normal':
        M_rho = np.random.normal(loc=RHO_mean, scale = RHO_std, size=(N, nz))
        if len(f_prior_h5)<1:
            f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_mean, RHO_std)
    elif RHO_dist=='log-normal':
        M_rho = np.random.lognormal(mean=np.log(RHO_mean), sigma = RHO_std/RHO_mean, size=(N, nz))
        if len(f_prior_h5)<1:
            f_prior_h5 = '%s_R%g_%g.h5' % (f_prior_h5, RHO_mean, RHO_std)
    elif RHO_dist=='chi2':
        M_rho = np.random.chisquare(df = chi2_deg, size=(N, nz))
        if len(f_prior_h5)<1:
            f_prior_h5 = '%s_deg%d.h5' % (f_prior_h5,chi2_deg)
    else:
        raise ValueError('RHO_dist=%s not supported' % RHO_dist)

    # Ensure physical resistivity values (must be positive)
    # First clamp to threshold to handle zero/negative values
    M_rho = np.maximum(M_rho, RHO_threshold)

    # Then clip to user-specified bounds, ensuring RHO_min is at least threshold
    effective_rho_min = max(RHO_min, RHO_threshold)
    M_rho = np.clip(M_rho, effective_rho_min, RHO_max)
    
    
    if (showInfo>0):
        print("prior_model_workbench_direct: Saving prior model to %s" % f_prior_h5)
    _report_progress(progress_callback, 80, 100,
                     'saving', 'Saving prior model to %s' % f_prior_h5)

    if (showInfo>1):
        print("Saving '/M1' prior model  %s" % f_prior_h5)
    ig.save_prior_model(f_prior_h5,M_rho.astype(np.float32),
                im=1,
                name='Resistivity',
                is_discrete = 0,
                x = z,
                z = z,
                delete_if_exist = True,
                force_replace=True,
                showInfo=showInfo,
                )

    _report_progress(progress_callback, 100, 100,
                     'completed', 'Prior model saved to %s' % f_prior_h5)

    return f_prior_h5


def prior_model_workbench(N=100000, p=2, z1=0, z_max= 100, dz=1,
                          lay_dist='uniform', nlayers=0, NLAY_min=3, NLAY_max=6, NLAY_deg=5,
                          RHO_dist='log-uniform',
                          RHO_min = 1, RHO_max= 300, RHO_mean=180, RHO_std=80, chi2_deg= 100,
                          RHO_threshold=0.001, **kwargs):
    """
    Generate a prior model with increasingly thick layers.
 
    Parameters
    ----------
    N : int, optional
        Number of prior models to generate. Default is 100000.
    p : int, optional
        Power parameter for thickness increase. Default is 2.
    z1 : float, optional
        Minimum depth value. Default is 0.
    z_max : float, optional
        Maximum depth value. Default is 100.
    dz : float, optional
        Depth discretization step. Default is 1.
    lay_dist : str, optional
        Distribution of the number of layers. Options are 'chi2' and 'uniform'. 
        Default is 'uniform'.
    nlayers : int, optional
        Number of layers. If greater than 0, sets both NLAY_min and NLAY_max 
        to this value. Default is 0.
    NLAY_min : int, optional
        Minimum number of layers. Default is 3.
    NLAY_max : int, optional
        Maximum number of layers. Default is 6.
    NLAY_deg : int, optional
        Degrees of freedom for chi-square distribution. Only applicable if 
        lay_dist is 'chi2'. Default is 5.
    RHO_dist : str, optional
        Distribution of resistivity within each layer. Options are 'log-uniform', 
        'uniform', 'normal', 'lognormal', and 'chi2'. Default is 'log-uniform'.
    RHO_min : float, optional
        Minimum resistivity value. Default is 1.
    RHO_max : float, optional
        Maximum resistivity value. Default is 300.
    RHO_mean : float, optional
        Mean resistivity value. Only applicable if RHO_dist is 'normal' or 
        'lognormal'. Default is 180.
    RHO_std : float, optional
        Standard deviation of resistivity value. Only applicable if RHO_dist is 
        'normal' or 'lognormal'. Default is 80.
    chi2_deg : int, optional
        Degrees of freedom for chi2 distribution. Only applicable if RHO_dist is
        'chi2'. Default is 100.
    RHO_threshold : float, optional
        Minimum physical resistivity threshold in Ohm·m. Any generated resistivity
        values below this threshold (including zero or negative values from 'normal'
        distribution) will be clamped to this value. Ensures physically realistic
        positive resistivity values. Default is 0.001 Ohm·m.
    f_prior_h5 : str, optional
        Path to the prior model file in HDF5 format. Default is ''.
    showInfo : int, optional
        Level of verbosity for output.

    Returns
    -------
    str
        Filepath of the saved prior model.
    """
    # Safety guard: if somehow called from a worker process, do nothing.
    if multiprocessing.current_process().name != 'MainProcess':
        return None

    from tqdm import tqdm
    import integrate as ig

    f_prior_h5 = kwargs.get('f_prior_h5', '')
    showInfo = kwargs.get('showInfo', 0)
    progress_callback = kwargs.get('progress_callback', None)
    if nlayers>0:
        NLAY_min = nlayers
        NLAY_max = nlayers

    if NLAY_max < NLAY_min:
        #raise ValueError('NLAY_max must be greater than or equal to NLAY_min.')
        NLAY_max = NLAY_min

    if NLAY_min < 1:
        #raise ValueError('NLAY_min must be greater than or equal to 1.')
        NLAY_min = 1

    
    if lay_dist == 'chi2':
        NLAY = np.random.chisquare(NLAY_deg, N)
        NLAY = np.ceil(NLAY).astype(int)
        if len(f_prior_h5)<1:
            f_prior_h5 = 'PRIOR_WB_CHI2_NF_%d_%s_N%d.h5' % (NLAY_deg, RHO_dist, N)
        NLAY_max = np.max(NLAY)  # Update NLAY_max to accommodate chi2 distribution
    elif lay_dist == 'uniform':
        NLAY = np.random.randint(NLAY_min, NLAY_max+1, N)
        if NLAY_min == NLAY_max:
            nlayers = NLAY_min
            if len(f_prior_h5)<1:
                f_prior_h5 = 'PRIOR_WB_UNIFORM_%d_N%d_%s' % (nlayers,N,RHO_dist)
        else:   
            if len(f_prior_h5)<1:
                f_prior_h5 = 'PROPR_WB_UNIFORM_%d-%d_N%d_%s' % (NLAY_min,NLAY_max,N,RHO_dist)
        


    # Force NLAY to be a 2 dimensional numpy array (for when exporting to HDF5)
    NLAY = NLAY[:, np.newaxis]
    

    z_min = 0
    # Ensure z_max is included in the array
    nz = int(np.ceil((z_max - z_min) / dz)) + 1
    z = np.linspace(z_min, z_max, nz)
    
    if showInfo>1:
        print('z_min, z_max, dz, nz = %g, %g, %g, %d' % (z_min, z_max, dz, nz))
    M_rho = np.zeros((N, nz))

    nm_sparse = NLAY_max+NLAY_max-1
    if (showInfo>1):
        print("nm_sparse", nm_sparse)
    M_rho_sparse = np.ones((N, nm_sparse))*np.nan
    

    # Throttle callback to ~100 updates so it does not dominate runtime
    progress_step = max(1, N // 100)

    for i in tqdm(range(N), mininterval=1, disable=(showInfo<0), desc='prior_workbench', leave=False):
        if progress_callback and ((i + 1) % progress_step == 0 or i + 1 == N):
            _report_progress(progress_callback, i + 1, N,
                             'generating', 'Generating prior realizations')
        nlayers = NLAY[i][0]
        #print(nlayers)
        z2=z_max
        z_single= z1 + (z2 - z1) * np.linspace(0, 1, nlayers) ** p
        
        if RHO_dist=='uniform':
            M_rho_single = np.random.uniform(low=RHO_min, high = RHO_max, size=(1, nlayers))
        elif RHO_dist=='log-uniform':
            M_rho_single = np.exp(np.random.uniform(low=np.log(RHO_min), high = np.log(RHO_max), size=(1, nlayers)))
        elif RHO_dist=='normal':
            M_rho_single = np.random.normal(loc=RHO_mean, scale = RHO_std, size=(1, nlayers))
        elif RHO_dist=='log-normal' or RHO_dist=='lognormal':
            M_rho_single = np.random.lognormal(mean=np.log(RHO_mean), sigma = RHO_std/RHO_mean, size=(1, nlayers))
        elif RHO_dist=='chi2':
            M_rho_single = np.random.chisquare(df = chi2_deg, size=(1, nlayers))
        else:
            # Default to log-uniform if RHO_dist is not recognized
            M_rho_single = np.exp(np.random.uniform(low=np.log(RHO_min), high = np.log(RHO_max), size=(1, nlayers)))

        # Ensure physical resistivity values (must be positive)
        # First clamp to threshold to handle zero/negative values
        M_rho_single = np.maximum(M_rho_single, RHO_threshold)

        # Then clip to user-specified bounds, ensuring RHO_min is at least threshold
        effective_rho_min = max(RHO_min, RHO_threshold)
        M_rho_single = np.clip(M_rho_single, effective_rho_min, RHO_max)

        for j in range(nlayers):
            ind = np.where(z>=z_single[j])[0]
            M_rho[i,ind]= M_rho_single[0,j]


        m_current = np.concatenate((z_single[0:-1].flatten(), M_rho_single.flatten()))
        M_rho_sparse[i,0:len(m_current)] = m_current



    if (showInfo>0):
        print("prior_model_workbench: Saving prior model to %s" % f_prior_h5)
    _report_progress(progress_callback, N, N,
                     'saving', 'Saving prior model to %s' % f_prior_h5)

    if (showInfo>1):
        print("Saving '/M1' prior model  %s" % f_prior_h5)
    ig.save_prior_model(f_prior_h5,M_rho.astype(np.float32),
                im=1,
                name='Resistivity',
                is_discrete = 0, 
                x = z,
                z = z,
                delete_if_exist = True,
                force_replace=True,
                showInfo=showInfo,
                )

    if (showInfo>1):
        print("Saving '/M2' prior model  %s" % f_prior_h5)
    ig.save_prior_model(f_prior_h5,M_rho_sparse.astype(np.float32),
                im=2,
                name='sparse - depth-resistivity',
                is_discrete = 0, 
                x = np.arange(0,nm_sparse),
                z = np.arange(0,nm_sparse),
                force_replace=True,
                showInfo=showInfo,
                )

    if (showInfo>1):
        print("Saving '/M3' prior model  %s" % f_prior_h5)
    ig.save_prior_model(f_prior_h5,NLAY.astype(np.float32),
                        im=3,
                        name = 'Number of layers',
                        is_discrete=0,
                        x=np.array([0]),
                        z=np.array([0]),
                        force_replace=True,
                        showInfo=showInfo,
                )

    _report_progress(progress_callback, N, N,
                     'completed', 'Prior model saved to %s' % f_prior_h5)

    # return the full filepath to f_prior_h5
    return f_prior_h5



def posterior_cumulative_thickness(f_post_h5, im=2, icat=[0], usePrior=False, **kwargs):
    """
    Calculate the posterior cumulative thickness based on the given inputs.

    Parameters
    ----------
    f_post_h5 : str
        Path to the input h5 file.
    im : int, optional
        Index of model parameter number, M[im]. Default is 2.
    icat : list, optional
        List of category indices. Default is [0].
    usePrior : bool, optional
        Flag indicating whether to use prior. Default is False.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        
        thick_mean : ndarray
            Array of mean cumulative thickness.
        thick_median : ndarray
            Array of median cumulative thickness.
        thick_std : ndarray
            Array of standard deviation of cumulative thickness.
        class_out : list
            List of class names.
        X : ndarray
            Array of X values.
        Y : ndarray
            Array of Y values.
    """

    import h5py
    import integrate as ig

    if isinstance(icat, int):
        icat = np.array([icat])

    with h5py.File(f_post_h5,'r') as f_post:
        f_prior_h5 = f_post['/'].attrs['f5_prior']
        f_data_h5 = f_post['/'].attrs['f5_data']

    X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

    Mstr = '/M%d' % im
    with h5py.File(f_prior_h5,'r') as f_prior:
        if not Mstr in f_prior.keys():
            print('No %s found in %s' % (Mstr, f_prior_h5))
            return -1
        if not f_prior[Mstr].attrs['is_discrete']:
            print('M%d is not discrete' % im)
            return -1



    with h5py.File(f_prior_h5,'r') as f_prior:
        try:
            z = f_prior[Mstr].attrs['z'][:].flatten()
        except:
            z = f_prior[Mstr].attrs['x'][:].flatten()
        is_discrete = f_prior[Mstr].attrs['is_discrete']
        if 'clim' in f_prior[Mstr].attrs.keys():
            clim = f_prior[Mstr].attrs['clim'][:].flatten()
        else:
            # if clim set in kwargs, use it, otherwise use default
            if 'clim' in kwargs:
                clim = kwargs['clim']
            else:
                clim = [.1, 2600]
                clim = [10, 500]
        if 'class_id' in f_prior[Mstr].attrs.keys():
            class_id = f_prior[Mstr].attrs['class_id'][:].flatten()
        else:   
            print('No class_id found')
        if 'class_name' in f_prior[Mstr].attrs.keys():
            class_name = f_prior[Mstr].attrs['class_name'][:].flatten()
        else:
            class_name = []
        n_class = len(class_name)
        if 'cmap' in f_prior[Mstr].attrs.keys():
            cmap = f_prior[Mstr].attrs['cmap'][:]
        else:
            cmap = plt.cm.hot(np.linspace(0, 1, n_class)).T
        from matplotlib.colors import ListedColormap

    with h5py.File(f_post_h5,'r') as f_post:
        #P=f_post[Mstr+'/P'][:]
        i_use = f_post['/i_use'][:]

    ns,nr=i_use.shape

    if usePrior:
        for i in range(ns):
            i_use[i,:]=np.arange(nr)
        

    with h5py.File(f_prior_h5,'r') as f_prior:
        M_prior = f_prior[Mstr][:]
    nz = M_prior.shape[1]

    thick_mean = np.zeros((ns))
    thick_median = np.zeros((ns))
    thick_std = np.zeros((ns))


    thick = np.diff(z)

    for i in range(ns):

        jj = i_use[i,:].astype(int)-1
        m_sample = M_prior[jj,:]
            
        cum_thick = np.zeros((nr))
        for ic in range(len(icat)):

            
            # the number of values of i_cat in the sample

            i_match = (m_sample == class_id[icat[ic]]).astype(int)
            i_match = i_match[:,0:nz-1]
            
            n_cat = np.sum(m_sample==icat[ic], axis=0)
        
            cum_thick = cum_thick + np.sum(i_match*thick, axis=1)

        thick_mean[i] = np.mean(cum_thick)
        thick_median[i] = np.median(cum_thick)
        thick_std[i] = np.std(cum_thick)

    class_out = class_name[icat]

    return thick_mean, thick_median, thick_std, class_out, X, Y


# # Import rejection sampling functions from separate module
# # Note: These imports work when the package is properly installed
# # For development, you may need to modify paths or use try/except
# try:
#     from integrate.integrate_rejection import (
#         integrate_rejection,
#         integrate_rejection_range, 
#         integrate_posterior_main,
#         integrate_posterior_chunk,
#         likelihood_gaussian_diagonal,
#         likelihood_gaussian_full,
#         likelihood_multinomial,
#         select_subset_for_inversion,
#         create_shared_memory,
#         reconstruct_shared_arrays,
#         cleanup_shared_memory
#     )
# except ImportError:
#     # For development when running directly, try relative import
#     from .integrate_rejection import (
#         integrate_rejection,
#         integrate_rejection_range, 
#         integrate_posterior_main,
#         integrate_posterior_chunk,
#         likelihood_gaussian_diagonal,
#         likelihood_gaussian_full,
#         likelihood_multinomial,
#         select_subset_for_inversion,
#         create_shared_memory,
#         reconstruct_shared_arrays,
#         cleanup_shared_memory
#     )


# Functions moved to integrate_rejection.py
# Functions moved to integrate_rejection.py have been removed

# All rejection sampling related functions have been moved to integrate_rejection.py
# This includes:
# - reconstruct_shared_arrays
# - cleanup_shared_memory
# - integrate_rejection
# - integrate_rejection_range
# - integrate_posterior_main
# - integrate_posterior_chunk
# - likelihood_gaussian_diagonal
# - likelihood_gaussian_full
# - likelihood_multinomial
# - select_subset_for_inversion
# moved to integrate_rejection.py


# %% Synthetic data

def _interpolate_resistivity(rho_values, nx):
    """
    Interpolate resistivity values along a profile.

    Parameters
    ----------
    rho_values : array_like
        Resistivity values at control points. Can be:
        - Single value [v]: constant resistivity
        - Two values [v1, v2]: linear from left to right
        - Multiple values [v1, v2, ..., vN]: interpolated through all points
    nx : int
        Number of x positions to interpolate to.

    Returns
    -------
    ndarray
        Interpolated resistivity values of length nx.
    """
    rho_values = np.atleast_1d(rho_values)
    n_control = len(rho_values)

    if n_control == 1:
        # Constant value
        return np.full(nx, rho_values[0])
    else:
        # Interpolate through control points
        control_indices = np.linspace(0, nx-1, n_control)
        x_indices = np.arange(nx)
        return np.interp(x_indices, control_indices, rho_values)


def synthetic_case(case='Wedge', **kwargs):
    """
    Generate synthetic geological models for different cases.

    This function creates synthetic 2D geological models for testing and validation
    purposes. Supports 'Wedge' and '3Layer' model types with customizable parameters.

    Parameters
    ----------
    case : str, optional
        The type of synthetic case to generate. Options are 'Wedge' and '3Layer'.
        Default is 'Wedge'.
    showInfo : int, optional
        If greater than 0, print information about the generated case. Default is 0.
    rho_1 : list or array_like, optional
        Resistivity values for layer 1 along the profile. If a single value [v],
        resistivity is constant at v. If multiple values [v1, v2, ...], resistivity
        varies from v1 (left) to v2 (middle) to vN (right) using interpolation.
        Only used when rho_1, rho_2, and rho_3 are all provided.
    rho_2 : list or array_like, optional
        Resistivity values for layer 2 along the profile. Same format as rho_1.
    rho_3 : list or array_like, optional
        Resistivity values for layer 3 along the profile. Same format as rho_1.
    x_max : int, optional
        Maximum x-dimension size. Default is 1000 for 'Wedge', 100 for '3Layer'.
    dx : float, optional
        Step size in the x-dimension.
    z_max : int, optional
        Maximum z-dimension size. Default is 90 for 'Wedge', 60 for '3Layer'.
    dz : float, optional
        Step size in the z-dimension. Default is 1.
    z1 : float, optional
        Depth at which the wedge starts ('Wedge': default z_max/10) or first layer
        ends ('3Layer': default z_max/3).
    rho : list, optional
        Density values for different layers ('Wedge' case). Default is [100, 200, 120].
        Overridden by rho_1, rho_2, rho_3 if all three are provided.
    wedge_angle : float, optional
        Angle of the wedge in degrees ('Wedge' case). Default is 1.
    x_range : float, optional
        Range in the x-dimension for the cosine function ('3Layer' case).
        Default is x_max/4.
    z_thick : float, optional
        Thickness of the second layer ('3Layer' case). Default is z_max/2.
    rho1_1 : float, optional
        Density at the start of the first layer ('3Layer'). Default is 120.
    rho1_2 : float, optional
        Density at the end of the first layer ('3Layer'). Default is 10.
    rho2_1 : float, optional
        Density at the start of the second layer ('3Layer'). Default is rho1_2.
    rho2_2 : float, optional
        Density at the end of the second layer ('3Layer'). Default is rho1_1.
    rho3 : float, optional
        Density of the third layer ('3Layer'). Default is 120.

    Returns
    -------
    M : ndarray
        The generated synthetic resistivity model of shape (nx, nz).
    x : ndarray
        X-coordinates of the model.
    z : ndarray
        Z-coordinates (depth) of the model.
    M_ref_lith : ndarray
        Lithology/layer number for each pixel, same shape as M. Values are 1, 2, 3
        corresponding to the layer number.
    layer_depths : ndarray
        Depth to the top of layers 1, 2, and 3 for each trace, shape (nx, 3).
        Column 0: depth to top of layer 1 (always 0)
        Column 1: depth to top of layer 2
        Column 2: depth to top of layer 3

    Examples
    --------
    >>> # Constant resistivity in each layer
    >>> M, x, z, M_lith, depths = ig.synthetic_case(case='3layer', rho_1=[10], rho_2=[80], rho_3=[10])
    >>>
    >>> # Linear variation from left to right
    >>> M, x, z, M_lith, depths = ig.synthetic_case(case='3layer', rho_1=[10, 80], rho_2=[80, 10], rho_3=[10, 10])
    >>>
    >>> # Three-point variation (left, middle, right)
    >>> M, x, z, M_lith, depths = ig.synthetic_case(case='3layer', rho_1=[10, 80, 10], rho_2=[50, 100, 50], rho_3=[10, 10, 10])
    """
    
    showInfo = kwargs.get('showInfo', 0)

    # Check if rho_1, rho_2, rho_3 are all provided
    rho_1 = kwargs.get('rho_1', None)
    rho_2 = kwargs.get('rho_2', None)
    rho_3 = kwargs.get('rho_3', None)
    use_rho_arrays = (rho_1 is not None) and (rho_2 is not None) and (rho_3 is not None)

    if case.lower() == 'wedge':
        # Create synthetic wedge model

        # variables
        x_max = kwargs.get('x_max', 1000)
        dx = kwargs.get('dx', 1000./x_max)
        z_max = kwargs.get('z_max', 90)
        dz = kwargs.get('dz', 1)
        z1 = kwargs.get('z1', z_max/10)
        rho = kwargs.get('rho', [100,200,120])
        wedge_angle = kwargs.get('wedge_angle', 1)

        if showInfo>0:
            print('Creating synthetic %s case with wedge angle=%f' % (case,wedge_angle))

        z = np.arange(0,z_max,dz)
        x = np.arange(0,x_max,dx)

        nx = x.shape[0]
        nz = z.shape[0]

        # Initialize M and M_ref_lith
        M = np.zeros((nx,nz))
        M_ref_lith = np.ones((nx,nz), dtype=int)  # Layer 1 by default

        # Initialize layer depths array (nx, 3)
        layer_depths = np.zeros((nx, 3))

        if use_rho_arrays:
            # Convert to numpy arrays
            rho_1 = np.atleast_1d(rho_1)
            rho_2 = np.atleast_1d(rho_2)
            rho_3 = np.atleast_1d(rho_3)

            # Interpolate resistivity values along the profile
            rho1_interp = _interpolate_resistivity(rho_1, nx)
            rho2_interp = _interpolate_resistivity(rho_2, nx)
            rho3_interp = _interpolate_resistivity(rho_3, nx)
        else:
            # Use constant values from rho array
            rho1_interp = np.full(nx, rho[0])
            rho2_interp = np.full(nx, rho[1])
            rho3_interp = np.full(nx, rho[2])

        # Build the model
        for ix in range(nx):
            # Layer 1 (top layer) - always starts at depth 0
            M[ix,:] = rho1_interp[ix]
            M_ref_lith[ix,:] = 1
            layer_depths[ix, 0] = 0  # Layer 1 starts at surface

            # Layer 2 (wedge) - starts at z1
            wedge_angle_rad = np.deg2rad(wedge_angle)
            z2 = z1 + x[ix]*np.tan(wedge_angle_rad)
            iz2 = np.where((z>=z1) & (z<=z2))[0]
            M[ix,iz2] = rho2_interp[ix]
            M_ref_lith[ix,iz2] = 2
            layer_depths[ix, 1] = z1  # Layer 2 starts at z1

            # Layer 3 (bottom layer, below wedge)
            iz3 = np.where(z>=z1)[0]
            M[ix,iz3] = rho3_interp[ix]
            M_ref_lith[ix,iz3] = 3
            layer_depths[ix, 2] = z2  # Layer 3 starts at bottom of wedge

        return M, x, z, M_ref_lith, layer_depths

    elif case.lower() == '3layer':
        # Create synthetic 3 layer model

        # variables
        x_max = kwargs.get('x_max', 100)
        x_range = kwargs.get('x_range', x_max/4)
        dx = kwargs.get('dx', 1)
        z_max = kwargs.get('z_max', 60)
        dz = kwargs.get('dz', 1)
        z1 = kwargs.get('z1', z_max/3)
        z_thick = kwargs.get('z_thick', z_max/2)

        rho1_1 = kwargs.get('rho1_1', 120)
        rho1_2 = kwargs.get('rho1_2', 10)
        rho2_1 = kwargs.get('rho2_1', rho1_2)
        rho2_2 = kwargs.get('rho2_2', rho1_1)
        rho3 = kwargs.get('rho3', 120)

        if showInfo>0:
            print('Creating synthetic %s case' % case)

        z = np.arange(0,z_max,dz)
        x = np.arange(0,x_max,dx)

        nx = x.shape[0]
        nz = z.shape[0]

        # Initialize M and M_ref_lith
        M = np.zeros((nx,nz))
        M_ref_lith = np.zeros((nx,nz), dtype=int)

        # Initialize layer depths array (nx, 3)
        layer_depths = np.zeros((nx, 3))

        if use_rho_arrays:
            # Convert to numpy arrays
            rho_1 = np.atleast_1d(rho_1)
            rho_2 = np.atleast_1d(rho_2)
            rho_3 = np.atleast_1d(rho_3)

            # Interpolate resistivity values along the profile
            rho1_interp = _interpolate_resistivity(rho_1, nx)
            rho2_interp = _interpolate_resistivity(rho_2, nx)
            rho3_interp = _interpolate_resistivity(rho_3, nx)

            # Build model with variable resistivity
            for ix in range(nx):
                # Layer 3 (bottom layer) - default
                M[ix,:] = rho3_interp[ix]
                M_ref_lith[ix,:] = 3

                # Layer 1 (top layer) - starts at surface
                iz1 = np.where(z<=z1)[0]
                M[ix,iz1] = rho1_interp[ix]
                M_ref_lith[ix,iz1] = 1
                layer_depths[ix, 0] = 0  # Layer 1 starts at surface

                # Layer 2 (middle layer with varying thickness) - starts at z1
                z2 = z1 + z_thick*0.5*(1+np.cos(np.pi+x[ix]/(x_range)*np.pi))
                iz2 = np.where((z>=z1) & (z<=z2))[0]
                M[ix,iz2] = rho2_interp[ix]
                M_ref_lith[ix,iz2] = 2
                layer_depths[ix, 1] = z1  # Layer 2 starts at z1

                # Layer 3 depth
                layer_depths[ix, 2] = z2  # Layer 3 starts at z2
        else:
            # Use original linear variation from rho1_1 to rho1_2
            for ix in range(nx):
                # Layer 3 (bottom layer) - default
                M[ix,:] = rho3
                M_ref_lith[ix,:] = 3

                # Layer 1 (top layer) - starts at surface
                iz1 = np.where(z<=z1)[0]
                rho1 = rho1_1 + (rho1_2 - rho1_1) * x[ix]/x_max
                M[ix,iz1] = rho1
                M_ref_lith[ix,iz1] = 1
                layer_depths[ix, 0] = 0  # Layer 1 starts at surface

                # Layer 2 (middle layer with varying thickness) - starts at z1
                z2 = z1 + z_thick*0.5*(1+np.cos(np.pi+x[ix]/(x_range)*np.pi))
                rho2 = rho2_1 + (rho2_2 - rho2_1) * x[ix]/x_max
                iz2 = np.where((z>=z1) & (z<=z2))[0]
                M[ix,iz2] = rho2
                M_ref_lith[ix,iz2] = 2
                layer_depths[ix, 1] = z1  # Layer 2 starts at z1

                # Layer 3 depth
                layer_depths[ix, 2] = z2  # Layer 3 starts at z2

        return M, x, z, M_ref_lith, layer_depths


####################################
## MISC 

def comb_cprob(pA, pAgB, pAgC, tau=1.0):
    """
    Combine conditional probabilities based on permanence of updating ratios.

    This function implements the probability combination method described in 
    Journel's "An Alternative to Traditional Data Independence Hypotheses" 
    (Math Geology, 2004).

    Parameters:
    -----------
    pA : array_like
        Probability of event A
    pAgB : array_like
        Conditional probability of A given B
    pAgC : array_like
        Conditional probability of A given C
    tau : float, optional
        Combination parameter controlling the ratio permanence (default=1.0)

    Returns:
    --------
    ndarray
        Combined conditional probability Prob(A|B,C)

    References:
    -----------
    Journel, An Alternative to Traditional Data Independence Hypotheses, 
    Mathematical Geology, 2002
    """
    # Compute odds ratios
    a = (1 - pA) / pA
    b = (1 - pAgB) / pAgB
    c = (1 - pAgC) / pAgC
    
    # Compute combined probability
    pAgBC = 1 / (1 + b * (c / a) ** tau)
    
    return pAgBC

def kl_divergence(prior_sample, posterior_sample, is_discrete=False, log_base=None):
    """
    Compute KL divergence D_KL(posterior || prior) for one or multiple parameters.

    Parameters
    ----------
    prior_sample : array, shape (n_realizations,) or (n_realizations, n_params)
    posterior_sample : array, shape (n_realizations,) or (n_realizations, n_params)
        If 2D, KL is computed per column and a 1D array of size n_params is returned.
    is_discrete : bool, optional
        If True, treat the parameter as discrete. Default is False.
    log_base : int, float, or None
        Base of the logarithm. None uses natural log (nats).
        For discrete parameters, passing log_base=N (number of classes)
        normalizes the result to [0, 1], where 1 means complete certainty.

    Returns
    -------
    float or numpy.ndarray
        KL divergence value(s).
    """
    from scipy.stats import gaussian_kde

    prior_sample = np.asarray(prior_sample)
    posterior_sample = np.asarray(posterior_sample)

    if prior_sample.ndim == 2:
        return np.array([
            kl_divergence(prior_sample[:, i], posterior_sample[:, i],
                                 is_discrete=is_discrete, log_base=log_base)
            for i in range(prior_sample.shape[1])
        ])

    # Change-of-base divisor: ln(base). If None, divisor=1 (natural log).
    log_divisor = np.log(log_base) if log_base is not None else 1.0

    if is_discrete:
        # 1. Get unique values and counts (empirical distribution)
        all_vals = np.unique(np.concatenate([prior_sample, posterior_sample]))
        p = np.array([np.sum(prior_sample == v) for v in all_vals])
        q = np.array([np.sum(posterior_sample == v) for v in all_vals])

        # 2. Add small smoothing to avoid zero probabilities
        p = (p + 1e-10) / (np.sum(p) + len(all_vals) * 1e-10)
        q = (q + 1e-10) / (np.sum(q) + len(all_vals) * 1e-10)

        # 3. Discrete KL: D_KL(posterior || prior) = Σ q·log(q/p)
        return np.sum(p * np.log(p / q)) / log_divisor
    else:
        # Continuous KL using KDE
        kde_p = gaussian_kde(prior_sample)
        kde_q = gaussian_kde(posterior_sample)

        # Evaluate on a grid
        grid = np.linspace(min(prior_sample.min(), posterior_sample.min()),
                           max(prior_sample.max(), posterior_sample.max()), 1000)
        p_vals = kde_p(grid)
        q_vals = kde_q(grid)

        # Normalize to ensure they are valid PDFs
        p_vals /= p_vals.sum()
        q_vals /= q_vals.sum()

        # Numerical integration: D_KL(posterior || prior) = Σ q·log(q/p)
        return np.sum(p_vals * np.log(p_vals / (q_vals + 1e-10))) / log_divisor


def entropy(P, base = None):
    """
    Calculate the entropy of a discrete probability distribution.

    The entropy is calculated using the formula:
    H(P) = -sum(P_i * log_b(P_i))

    Parameters
    ----------
    P : numpy.ndarray
        Probability distribution. Can be a 1D or 2D array.
        If 2D, each row represents a different distribution.
    base : int, optional
        The logarithm base to use. If None, uses the number of elements
        in the probability distribution (P.shape[1]). Default is None.

    Returns
    -------
    numpy.ndarray
        The entropy value(s). If input P is 2D, returns an array with
        entropy for each row distribution.

    Notes
    -----
    - Input probabilities are assumed to be normalized (sum to 1)
    - Zero probabilities are handled by numpy's log function
    - For 2D input, entropy is calculated row-wise

    Examples
    --------
    >>> P = np.array([0.5, 0.5])
    >>> entropy(P)
    1.0

    >>> P = np.array([[0.5, 0.5], [0.1, 0.9]])
    >>> entropy(P)
    array([1.0, 0.469])
    """
    P = np.atleast_2d(P)    
    if base is None:
        base = P.shape[1]
    H = -np.sum(P*np.log(P)/np.log(base), axis=1)
    return H

def class_id_to_idx(D, class_id=None):
    """
    Convert class identifiers to indices.

    This function takes an array of class identifiers and converts them to 
    corresponding indices. If no class identifiers are provided, it will 
    automatically determine the unique class identifiers from the input array.

    Parameters
    ----------
    D : numpy.ndarray
        Array containing class identifiers.
    class_id : numpy.ndarray, optional
        Array of unique class identifiers. If None, unique class identifiers 
        will be determined from the input array `D`. Default is None.
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        
        D_idx : numpy.ndarray
            Array with class identifiers converted to indices.
        class_id : numpy.ndarray
            Array of unique class identifiers.
        class_id_out : numpy.ndarray
            Array of unique output class identifiers.
    """

    if class_id is None:
        class_id = np.unique(D)
    D_idx = np.zeros(D.shape)
    for i in range(len(class_id)):
        D_idx[D==class_id[i]]=i
    # Make sure the indices are integers
    D_idx = D_idx.astype(int)
    class_id_out = np.unique(D_idx)
    
    return D_idx, class_id, class_id_out



def get_hypothesis_probability(f_post_h5_arr, T=1):
    """
    Calculate hypothesis probabilities and related statistics from posterior files.
    
    This function processes an array of HDF5 file paths containing posterior evidences
    to compute normalized probabilities for each hypothesis, along with evidence values,
    mode hypotheses, and entropy measures.
    
    Parameters
    ----------
    f_post_h5_arr : list of str
        Array of file paths to HDF5 files containing posterior evidence values. 
        Each file should have an '/EV' dataset.
    T : float, optional
        Temperature parameter that applies annealing. Higher temperatures create 
        more uniform distributions. Useful for smoothing distributions from smaller 
        lookup tables. Default is 1.
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        
        P : numpy.ndarray
            Normalized probabilities for each hypothesis (shape: n_hypothesis, n_samples).
        EV_all : numpy.ndarray
            Evidence values for each hypothesis and sample (shape: n_hypothesis, n_samples).
        MODE_hypothesis : numpy.ndarray
            Index of most probable hypothesis per sample (shape: n_samples).
        ENT_hypothesis : numpy.ndarray
            Entropy of hypothesis distribution per sample, normalized by number 
            of hypotheses (shape: n_samples).
    
    Notes
    -----
    The probability normalization uses the log-sum-exp trick to avoid numerical
    underflow issues when working with evidence values.
    """

    from scipy import stats

    n_hypothesis = len(f_post_h5_arr)
    EV_all = []
    for f_post_h5 in f_post_h5_arr:
        with h5py.File(f_post_h5, 'r') as f:
            EV = f['/EV'][()]
        EV_all.append(EV)
    EV_all = np.array(EV_all)
    # subtract the small value on each column form each column
    P  = np.exp(EV_all - np.max(EV_all, axis=0))**(1/T)
    P  = np.exp((1/T)*(EV_all - np.max(EV_all, axis=0)))
    #P_acc = np.exp((1/T) * (logL - np.nanmax(logL)))
    # Normalize each column to sum to 1 using NumPy broadcasting
    P = P / np.sum(P, axis=0, keepdims=True)

    ENT_hypothesis = np.zeros(P.shape[1])
    MODE_hypothesis = np.zeros(P.shape[1])

    for i in range(P.shape[1]):        
        # get the entropy for each hypothesis
        ENT_hypothesis[i] = stats.entropy(P[:,i], base=n_hypothesis)
        # get the is of the hypothesis with the maximum probability
        MODE_hypothesis[i] = np.argmax(P[:,i], axis=0)

    

    return P, EV_all, MODE_hypothesis, ENT_hypothesis 



def sample_posterior_multiple_hypotheses(f_post_h5_arr, P_hypothesis=None):
    """
    Sample posterior models from multiple hypotheses.
    
    This function samples posterior models from multiple hypotheses stored in HDF5 files,
    according to the given hypothesis probabilities.
    
    Parameters
    ----------
    f_post_h5_arr : list of str
        List of paths to HDF5 files containing posterior models for different hypotheses.
    P_hypothesis : numpy.ndarray, optional
        Array of shape (n_hypotheses, n_soundings) containing probability of each 
        hypothesis for each sounding. If None, uniform probabilities are used. 
        Default is None.
    
    Returns
    -------
    list of numpy.ndarray
        List of posterior model arrays. Each array has shape (n_soundings, n_samples, 
        n_parameters), where n_samples is determined by the first hypothesis's number 
        of samples.
    
    Notes
    -----
    The function combines posterior samples from different hypotheses in proportion to their
    probabilities and ensures the total number of samples equals the first hypothesis's 
    sample count.
    """

    import numpy as np
    import h5py
    import integrate as ig
    
    f_prior_h5_arr = []
    M_all = []
    i_use_all = []
    n_use_all = []
    for ip in range(len(f_post_h5_arr)):
            f_post_h5 = f_post_h5_arr[ip]
            with h5py.File(f_post_h5, 'r') as f_post:
                f_prior_h5 = f_post['/'].attrs['f5_prior']
                f_data_h5 = f_post['/'].attrs['f5_data']
                f_prior_h5_arr.append(f_prior_h5)

                i_use = f_post['/i_use'][:]
                i_use_all.append(i_use)
                n_use_all.append(i_use.shape[1])

                print("loading prior model %s" % f_prior_h5)
                M, idx = ig.load_prior_model(f_prior_h5)
                M_all.append(M)
    print(f_prior_h5_arr)
    i_use_all = np.array(i_use_all)
    n_use_all = np.array(n_use_all)

    if P_hypothesis is None:
        print('Using unform hypothesis probability')
        D = ig.load_data(f_data_h5)
        n_soundings = D['d_obs'][0].shape[0]

        P_hypothesis = np.ones((len(f_post_h5_arr), n_soundings))
        P_hypothesis = P_hypothesis/np.sum(P_hypothesis, axis=0)

    # def sample_posterior_multipleh_hypotheses(f_post_h5_arr, P_hypothesis, n_post=100):
    nsoundings = P_hypothesis.shape[1]
    # If different hypothsis have different number of realizations..
    #for i in range(nsoundings):


    M_post_arr = []

    for im in range(len(M)):
        print("im=%d/%d" % (im+1,len(M)))
        nm = M[im].shape[1]
        M_post = np.zeros((nsoundings, n_use_all[0],nm))

        for i in range(nsoundings):
            # get the probabliity of each hypothesis
            P_hypothesis_is = P_hypothesis[:,i]
            i_use = i_use_all[:,i,:]

            n_use = P_hypothesis_is*n_use_all
            n_use = np.round(n_use).astype(int)
            #print(n_use)
            n_sum = np.sum(n_use)
            # make sure that the sum of n_use is equal to the number of realizations
            delta_n = n_use_all[0]-n_sum
            if delta_n > 0:
                n_use[0] = n_use[0] + delta_n
            elif delta_n < 0:
                pass
                #n_use[0] = n_use[0] - np.abs(delta_n)
            n_sum = np.sum(n_use)
            
            M_dummy = []
            for j in range(len(n_use)):  
                # use the first  realizations from i_use[j]
                #i_use_single = i_use[j,:n_use[j]]
                # use n_use[j] random realizations from iuse[j]
                #print('    j=%d, n_use=%d' % (j,n_use[j]))
                if n_use[j]>0:                    
                    i_use_single = np.random.choice(i_use[j], n_use[j], replace=False)                
                    M_dummy.append(M_all[j][im][i_use_single,:])
                    
            M_sounding = np.concatenate(M_dummy, axis=0)        
            try:
                # take the first n_use_all[0] realizations
                M_post[i]=M_sounding[:n_use_all[0]]
            except:
                print('i=%d, [%d], n_sum=%d, delta_n=%d, n_use_all[0]=%d' % (i,M_sounding.shape[0],n_sum,delta_n,n_use_all[0]))
                try:
                    M_post[i]=M_sounding[:,:n_use[0]]
                except:
                    pass
                
        M_post_arr.append(M_post)    
                
    return M_post_arr


# %% TIMING FUNCTIONS
# Functions moved from integrate_timing_cli.py

def allocate_large_page():
    """
    Allocate a 2MB large page if running on Windows.
    
    Returns
    -------
    int or None
        Pointer to allocated memory on success, None on failure or non-Windows systems.
        
    Notes
    -----
    Large pages can improve performance but require specific Windows privileges.
    """
    import os
    import ctypes
    
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        kernel32.VirtualAlloc.restype = ctypes.c_void_p
        
        LARGE_PAGE_SIZE = 2 * 1024 * 1024  # 2MB
        
        MEM_COMMIT = 0x1000
        MEM_LARGE_PAGES = 0x20000000
        PAGE_READWRITE = 0x04
        
        ptr = kernel32.VirtualAlloc(None, LARGE_PAGE_SIZE, MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE)
        
        if not ptr:
            error_code = ctypes.GetLastError()
            print(f"Failed to allocate large page. Error code: {error_code}")
            return None
        
        print(f"Successfully allocated {LARGE_PAGE_SIZE} bytes at address {hex(ptr)}")
        return ptr
    else:
        print("Large pages are only supported on Windows.")
        return None


def timing_compute(N_arr=[], Nproc_arr=[], backend='numpy', NcpuForward=0):
    """
    Execute timing benchmark for INTEGRATE workflow components.
    
    This function benchmarks the performance of the complete INTEGRATE workflow including
    prior model generation, forward modeling, rejection sampling, and posterior statistics
    computation across different dataset sizes and processor counts.
    
    Parameters
    ----------
    N_arr : array_like, optional
        Array of dataset sizes (number of prior models) to test. 
        Default is [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000].
    Nproc_arr : array_like, optional
        Array of processor counts to test. Default is powers of 2 up to available CPUs.
    NcpuForward : int, optional
        Fixed number of CPUs to use for forward modeling. When > 0, forward modeling always
        uses this many CPUs regardless of the current Nproc_arr entry. The inversion
        (rejection sampling) still varies over Nproc_arr. Default is 0 (use Nproc_arr value).

    Returns
    -------
    str
        Filename of the NPZ file containing timing results.
        
    Notes
    -----
    The benchmark tests four main components:
    1. Prior model generation (layered geological models)
    2. Forward modeling using GA-AEM electromagnetic simulation
    3. Rejection sampling for Bayesian inversion  
    4. Posterior statistics computation
    
    Results are saved to an NPZ file with timing arrays and system information.
    The function automatically uses appropriate test data and handles parallel processing
    configuration based on system capabilities.
    """
    import integrate as ig
    # check if parallel computations can be performed
    parallel = ig.use_parallel(showInfo=1)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import loglog
    import time
    import h5py
    # get name of CPU
    import psutil

    # Get hostname and number of processors
    import socket
    hostname = socket.gethostname()
    import platform
    hostname = platform.node()
    system = platform.system()

    ## Get number of processors
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    Ncpu = physical_cores

    print("# TIMING TEST")
    print("Hostname (system): %s (%s) " % (hostname, system))
    print("Number of processors: %d" % Ncpu)
    
    # SELECT THE CASE TO CONSIDER AND DOWNLOAD THE DATA
    files = ig.get_case_data(showInfo=-1)
    f_data_h5 = files[0]
    file_gex= ig.get_gex_file_from_data(f_data_h5)

    print("Using data file: %s" % f_data_h5)
    print("Using GEX file: %s" % file_gex)

    with h5py.File(f_data_h5, 'r') as f:
        nobs = f['D1/d_obs'].shape[0]


    ## Setup the timing test

    #### Set the size of the data sets to test
    if len(N_arr)==0:
        N_arr = np.array([100,500,1000,5000,10000,50000,100000, 500000, 1000000, 5000000])

    # Set the number of cores to test
    if len(Nproc_arr)==0:
        Nproc_arr=2**(np.double(np.arange(1+int(np.log2(Ncpu)))))   

    n1 = len(N_arr)
    n2 = len(Nproc_arr)    

    print("Testing on %d data sets of size(s):" % len(N_arr), N_arr)
    print("Testing on %d sets of core(s):" % len(Nproc_arr), Nproc_arr)
    if NcpuForward > 0:
        print("Forward modeling fixed to %d CPUs (inversion varies over Nproc_arr)" % NcpuForward)


    print("Rejection sampling backend: %s" % backend)
    backend_label = backend
    if backend == 'jax':
        import os
        jax_platform_env = os.environ.get('JAX_PLATFORMS', '').strip().lower()
        if jax_platform_env in ('gpu', 'cuda', 'rocm'):
            backend_label = 'jax_gpu'
        elif jax_platform_env == 'cpu':
            backend_label = 'jax_cpu'
        else:
            try:
                import jax
                jax_device = jax.default_backend()
                backend_label = 'jax_gpu' if jax_device == 'gpu' else 'jax_cpu'
            except Exception:
                backend_label = 'jax_cpu'
        print("JAX backend label: %s (JAX_PLATFORMS='%s')" % (backend_label, jax_platform_env))
    file_out  = 'timing_%s-%s-%dcore_Nproc%d_N%d_%s.npz' % (hostname,system,Ncpu,len(Nproc_arr), len(N_arr), backend_label)
    print("Writing results to %s " % file_out)

    ## TIMING

    showInfo = 0

    T_prior = np.zeros((n1,n2))*np.nan
    T_forward = np.zeros((n1,n2))*np.nan
    T_rejection = np.zeros((n1,n2))*np.nan
    T_poststat = np.zeros((n1,n2))*np.nan

    testRejection = True
    testPostStat = True  
                
    for j in np.arange(n2):
        Ncpu = int(Nproc_arr[j])

        for i in np.arange(len(N_arr)):
            N=int(N_arr[i])
            Ncpu_min = int(np.floor(2**(np.log10(N)-4)))
            
            print('=====================================================')
            print('TIMING: N=%d, Ncpu=%d, Ncpu_min=%d'%(N,Ncpu,Ncpu_min))
            print('=====================================================')
            
            RHO_min = 1
            RHO_max = 800
            z_max = 50 
            useP = 1
            
            if (Ncpu>=Ncpu_min):
                    
                t0_prior = time.time()
                if useP ==1:
                    ## Layered model    
                    f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='chi2', NLAY_deg=5, z_max = z_max, RHO_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max, showInfo=showInfo)
                    #f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', z_max = z_max, NLAY_min=1, NLAY_max=3, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
                    #f_prior_h5 = ig.prior_model_layered(N=N,lay_dist='uniform', z_max = z_max, NLAY_min=1, NLAY_max=8, rho_dist='log-uniform', RHO_min=RHO_min, RHO_max=RHO_max)
                else: 
                    ## N layer model with increasing thickness
                    f_prior_h5 = ig.prior_model_workbench(N=N, z_max = 30, nlayers=20, rho_min = RHO_min, rho_max = RHO_max, showInfo=showInfo)
                #t_prior.append(time.time()-t0_prior)
                T_prior[i,j] = time.time()-t0_prior

            
                #ig.plot_prior_stats(f_prior_h5)
                #% A2. Compute prior DATA
                t0_forward = time.time()
                Ncpu_fwd = NcpuForward if NcpuForward > 0 else Ncpu
                f_prior_data_h5 = ig.prior_data_gaaem(f_prior_h5, file_gex, Ncpu=Ncpu_fwd, showInfo=showInfo)
                T_forward[i,j]=time.time()-t0_forward

                #% READY FOR INVERSION
                N_use = 1000000
                t0_rejection = time.time()
                if testRejection:
                    f_post_h5 = ig.integrate_rejection(f_prior_data_h5, f_data_h5, N_use=N_use, parallel=1, updatePostStat=False, Ncpu=Ncpu, showInfo=showInfo, backend=backend)
                T_rejection[i,j]=time.time()-t0_rejection

                #% Compute some generic statistic of the posterior distribution (Mean, Median, Std)
                t0_poststat = time.time()
                if testPostStat and testRejection:
                    ig.integrate_posterior_stats(f_post_h5,showInfo=showInfo)
                    T_poststat[i,j]=time.time()-t0_poststat

            T_total = T_prior + T_forward + T_rejection + T_poststat
            np.savez(file_out, T_total=T_total, T_prior=T_prior, T_forward=T_forward, T_rejection=T_rejection, T_poststat=T_poststat, N_arr=N_arr, Nproc_arr=Nproc_arr, nobs=nobs, backend=backend)
            
            
    return file_out


def timing_plot(f_timing=''):
    """
    Generate comprehensive timing analysis plots from benchmark results.
    
    This function creates multiple plots analyzing the performance characteristics
    of the INTEGRATE workflow across different dataset sizes and processor counts.
    
    Parameters
    ----------
    f_timing : str
        Path to NPZ file containing timing benchmark results from timing_compute().
        
    Returns
    -------
    None
        Saves multiple PNG files with timing analysis plots.
        
    Notes
    -----
    Generated plots include:
    - Total execution time vs processors and dataset size
    - Forward modeling performance and speedup analysis  
    - Rejection sampling performance and scaling
    - Posterior statistics computation performance
    - Cumulative time breakdowns for different processor counts
    - Comparisons with traditional least squares and MCMC methods
    
    The function handles missing data gracefully and includes reference lines
    for linear scaling to assess parallel efficiency.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    def safe_show():
        """Show plot only if using interactive backend, otherwise do nothing."""
        backend = plt.get_backend()
        if backend.lower() != 'agg':
            safe_show()

    if len(f_timing)==0:
        print('No timing file provided')
        return
    else:
        print('Plotting timing results from %s' % f_timing)

    # file_out is f_timing, without file extension
    file_out = f_timing.split('.')[0]
    
    data = np.load(f_timing)
    T_prior = data['T_prior']
    T_forward = data['T_forward']
    T_rejection = data['T_rejection']
    T_poststat = data['T_poststat']

    N_arr = data['N_arr']
    Nproc_arr = data['Nproc_arr']

    xlim_N = xlim_N = [np.min([800,np.min(N_arr)]),np.max([1.2e+6,np.max(N_arr)])]
    xlim_Nproc = np.min([.95,np.min(Nproc_arr)]),np.max([34,np.max(Nproc_arr)])
    try:
        T_total = data['T_total']
    except:
        T_total = T_prior + T_forward + T_rejection + T_poststat

    try:
        nobs=data['nobs']
    except:
        nobs=11693


    # ############################################
    # TOTAL TIME
    # ############################################

    # Plot
    # LSQ, Assumed time, in seconds, for least squares inversion of a single sounding
    t_lsq = 2.0
    # SAMPLING, Assumed time, in seconds, for an McMC inversion of a single sounding
    t_mcmc = 10.0*60.0 

    total_lsq = np.array([nobs*t_lsq, nobs*t_lsq/Nproc_arr[-1]])
    total_mcmc = np.array([nobs*t_mcmc, nobs*t_mcmc/Nproc_arr[-1]])

    # loglog(T_total.T)
    plt.figure(figsize=(6,6))    
    plt.loglog(Nproc_arr, T_total.T, 'o-',  label=N_arr)
    plt.ylabel(r'Total time - $[s]$')
    plt.xlabel('Number of processors')
    plt.grid()
    total_lsq = np.array([nobs*t_lsq, nobs*t_lsq/Nproc_arr[-1]])
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], total_lsq, 'k--', label='LSQ')
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], total_mcmc, 'r--', label='MCMC')
    plt.legend(loc='upper right')
    plt.xticks(ticks=Nproc_arr, labels=[str(int(x)) for x in Nproc_arr])
    plt.tight_layout()
    plt.ylim(1,1e+8)
    plt.xlim(xlim_Nproc)
    plt.savefig('%s_total_sec_CPU' % file_out)
    safe_show()
    plt.close()

    plt.figure(figsize=(6,6)) 
    plt.loglog(N_arr, T_total, 'o-', label=[f'{int(x)}' for x in Nproc_arr])
    plt.ylabel(r'Total time - $[s]$')
    plt.xlabel('N-prior')
    plt.grid()
    plt.tight_layout()
    plt.plot([N_arr[0], N_arr[-1]], [nobs*t_lsq, nobs*t_lsq], 'k--', label='LSQ')
    plt.plot([N_arr[0], N_arr[-1]], [nobs*t_mcmc, nobs*t_mcmc], 'r--', label='MCMC')
    plt.legend(loc='upper left')
    #plt.xticks(ticks=N_arr, labels=[str(int(x)) for x in Nproc_arr])
    plt.ylim(1,1e+8)
    #plt.xlim(np.min([1000,np.min(N_arr)]),np.max([1e+6,np.max(N_arr)]))
    plt.xlim(xlim_N)
    plt.savefig('%s_total_sec_N' % file_out)
    safe_show()
    plt.close()

    # ############################################
    # FORWARD MODELING
    # ############################################
    
    #### Plot timing results for forward modeling - GAAEM
    # Average timer per sounding 
    T_forward_sounding = T_forward/N_arr[:,np.newaxis]
    T_forward_sounding_per_sec = N_arr[:,np.newaxis]/T_forward
    T_forward_sounding_per_sec_per_cpu = T_forward_sounding_per_sec/Nproc_arr[np.newaxis,:]
    T_forward_sounding_speedup = T_forward_sounding_per_sec/T_forward_sounding_per_sec[0,0]

    ## Forward time per sounding - CPU
    plt.figure(figsize=(6,6))    
    plt.loglog(Nproc_arr, T_forward.T, 'o-', label='A')
    # plot dashed line indicating linear scaling
    for i in range(len(N_arr)):
        # Find index of first non-nan value in T_forward[i,:]
        try:
            idx = np.nonzero(~np.isnan(T_forward[i,:]))[0][0]
            plt.plot([Nproc_arr[0], Nproc_arr[-1]], [T_forward[i,idx]*Nproc_arr[idx]/Nproc_arr[0], T_forward[i,idx]*Nproc_arr[idx]/Nproc_arr[-1]], 'k--', 
                    label='Linear scaling', 
                    linewidth=0.5)   
        except:
            pass
    
    plt.ylabel(r'Forward time - $[s]$')
    plt.xlabel('Number of processors')
    #plt.title('Forward calculation')
    plt.grid()
    plt.legend(N_arr, loc='upper right')
    plt.ylim(1e-1, 1e+5)
    #plt.xlim(Nproc_arr[0], Nproc_arr[-1])
    plt.xlim(xlim_Nproc)
    plt.tight_layout()
    plt.savefig('%s_forward_sec_CPU' % file_out)
    safe_show()
    plt.close()

    ## Forward time per sounding - Nproc
    plt.figure(figsize=(6,6))    
    plt.loglog(N_arr, T_forward, 'o-', label='A')
    # plot dashed line indicating linear scaling
    for i in range(len(N_arr)):
        # Find index of first non-nan value in T_forward[i,:]
        try:
            idx = np.nonzero(~np.isnan(T_forward[i,:]))[0][0]
            ref_time = T_forward[i,idx]
            ref_N = N_arr[i]
            plt.plot([N_arr[0], N_arr[-1]], [ref_time*N_arr[0]/ref_N, ref_time*N_arr[-1]/ref_N], 'k--', label='Linear scaling', linewidth=0.5)   
        except:
            pass
    plt.ylabel(r'Forward time - $[s]$')
    plt.xlabel('Number of models')
    #plt.title('Forward calculation')
    plt.grid()
    plt.legend(Nproc_arr, loc='upper left')
    plt.ylim(1e+0, 1e+5)
    #plt.xlim(Nproc_arr[0], Nproc_arr[-1])
    plt.xlim(xlim_N)
    plt.tight_layout()
    plt.savefig('%s_forward_sec_N' % file_out)
    safe_show()
    plt.close()


    #
    plt.figure(figsize=(6,6))    
    plt.plot(Nproc_arr, T_forward_sounding_per_sec.T, 'o-')
    # plot line 
    plt.ylabel(r'Forward computations per second - $[s^{-1}]$')
    plt.xlabel('Number of processors')
    #plt.title('Forward calculation')
    plt.grid()
    plt.legend(N_arr, loc='lower right')
    plt.xlim(xlim_Nproc)    
    plt.ylim(10,1000)
    plt.tight_layout()
    plt.savefig('%s_forward_sounding_per_sec' % file_out)
    safe_show()
    plt.close()

    #
    plt.figure(figsize=(6,6))    
    plt.plot(Nproc_arr, T_forward_sounding_per_sec_per_cpu.T, 'o-')
    plt.ylabel('Forward computations per second per cpu')
    plt.xlabel('Number of processors')
    #plt.title('Forward calculation')
    plt.grid()
    # Make yaxis start at 0
    plt.ylim(0, 140)    
    plt.xlim(Nproc_arr[0], Nproc_arr[-1])
    plt.xlim(xlim_Nproc)
    plt.legend(N_arr)
    plt.tight_layout()
    plt.savefig('%s_forward_sounding_per_sec_per_cpu' % file_out)
    safe_show()
    plt.close()
    #

    plt.figure(figsize=(6,6))    
    plt.plot(Nproc_arr, T_forward_sounding_speedup.T, 'o-')
    # plot a line from 0,0 tp Nproc_arr[-1], Nproc_arr[-1]
    plt.plot([0, Nproc_arr[-1]], [0, Nproc_arr[-1]], 'k--')
    # set xlim to 1, Nproc_arr[-1]
    plt.xlim(.8, Nproc_arr[-1])
    plt.ylim(.8, Nproc_arr[-1])
    plt.ylabel('gatdaem - speedup compared to 1 processor')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.xlim(xlim_Nproc)    
    plt.ylim(0.5, 30)    
    plt.tight_layout()
    plt.savefig('%s_forward_speedup' % file_out)
    safe_show()
    plt.close()

    # ############################################
    # REJECTION SAMPLING
    # ############################################
    
    # Average timer per sounding
    T_rejection_sounding = T_rejection/N_arr[:,np.newaxis]
    T_rejection_sounding_per_sec = N_arr[:,np.newaxis]/T_rejection
    T_rejection_sounding_per_sec_per_cpu = T_rejection_sounding_per_sec/Nproc_arr[np.newaxis,:]
    T_rejection_sounding_speedup = T_rejection_sounding_per_sec/T_rejection_sounding_per_sec[0,0]
    T_rejection_sounding_speedup = T_rejection_sounding_per_sec*0

    T_rejection_per_data = nobs/T_rejection

    for i in range(len(N_arr)):
        # find index of first value in T_rejection_sounding_per_sec[i,:] that is not nan
        try:
            idx = np.where(~np.isnan(T_rejection_sounding_per_sec[i,:]))[0][0]
            T_rejection_sounding_speedup[i,:] = T_rejection_sounding_per_sec[i,:]/(T_rejection_sounding_per_sec[i,idx]/Nproc_arr[idx]) 
        except:
            T_rejection_sounding_speedup[i,:] = T_rejection_sounding_per_sec[i,:]*0


    ## Rejection total sec - per CPU
    plt.figure(figsize=(6,6))
    plt.loglog(Nproc_arr, T_rejection.T, 'o-')
    for i in range(len(N_arr)):
        # Find index of first non-nan value in T_forward[i,:]
        try:
            idx = np.nonzero(~np.isnan(T_rejection[i,:]))[0][0]
            plt.plot([Nproc_arr[0], Nproc_arr[-1]], [T_rejection[i,idx]*Nproc_arr[idx]/Nproc_arr[0], T_rejection[i,idx]*Nproc_arr[idx]/Nproc_arr[-1]], 'k--',
                        label='Linear scaling', 
                        linewidth=0.5)
        except:
            pass
    plt.ylabel('Rejection sampling - time $[s]$')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.tight_layout()
    plt.ylim(1e-1, 2e+3)
    plt.xlim(xlim_Nproc)
    plt.savefig('%s_rejection_sec_CPU' % file_out)
    safe_show()
    plt.close()


    ## Rejection total sec - per process
    plt.figure(figsize=(6,6))
    plt.loglog(N_arr, T_rejection, 'o-')
    for i in range(len(Nproc_arr)):
        # Find index of first non-nan value in T_forward[i,:]
        try:
            idx = np.nonzero(~np.isnan(T_rejection[:,i]))[0][0]
            ref_time = np.abs(T_rejection[idx,i])
            plt.plot([N_arr[0], N_arr[-1]], [ref_time*N_arr[0]/N_arr[idx], ref_time*N_arr[-1]/N_arr[idx]], 'k--',
                        label='Linear scaling', 
                        linewidth=0.5)
        except:
            pass
    plt.ylabel('Rejection sampling - time $[s]$')
    plt.xlabel('Lookup table size')
    plt.grid()
    plt.legend(Nproc_arr)
    plt.ylim(1e-1, 2e+3)
    plt.xlim(xlim_N)
    plt.tight_layout()
    plt.savefig('%s_rejection_sec_N' % file_out)
    safe_show()
    plt.close()


    ## Rejection speedup
    plt.figure(figsize=(6,6))
    plt.plot(Nproc_arr, T_rejection_sounding_speedup.T, 'o-')
    # plot a line from 0,0 tp Nproc_arr[-1], Nproc_arr[-1]
    plt.plot([0, Nproc_arr[-1]], [0, Nproc_arr[-1]], 'k--')
    # set xlim to 1, Nproc_arr[-1]
    plt.xlim(.8, Nproc_arr[-1])
    plt.ylim(.8, Nproc_arr[-1])
    plt.ylabel('Rejection sampling - speedup compared to 1 processor')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.xlim(xlim_Nproc)
    plt.legend(N_arr)
    plt.savefig('%s_rejection_speedup' % file_out)
    safe_show()
    plt.close()


    ## Rejection sound per sec
    plt.figure(figsize=(6,6))
    plt.loglog(Nproc_arr, T_rejection_per_data.T, 'o-', label=N_arr)
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], [1./t_lsq, 1./t_lsq], 'k--', label='LSQ')
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], [1./t_mcmc, 1./t_mcmc], 'r--', label='MCMC')
    plt.ylabel('Rejection sampling - number of soundings per second - $s^{-1}$')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(loc='lower left')
    plt.ylim(1e-3, 1e+5)
    plt.xlim(xlim_Nproc)
    plt.tight_layout()
    plt.savefig('%s_rejection_sounding_per_sec' % file_out)
    safe_show()
    plt.close()

    ## Rejection sec per sounding
    plt.figure(figsize=(6,6))
    plt.semilogy(Nproc_arr, 1./T_rejection_per_data.T, 'o-', label=N_arr)
    #plt.plot(Nproc_arr, 1./T_rejection_per_data.T, 'o-', label=N_arr)
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], [t_lsq, t_lsq], 'k--', label='LSQ')
    plt.plot([Nproc_arr[0], Nproc_arr[-1]], [t_mcmc, t_mcmc], 'r--', label='MCMC')
    plt.ylabel('Rejection sampling - seconds per sounding - $s$')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(loc='upper right')
    plt.ylim(1e-5, 1e+3)
    plt.xlim(xlim_Nproc)
    plt.tight_layout()
    plt.savefig('%s_rejection_sec_per_sound' % file_out)
    safe_show()
    plt.close()

    ## Rejection sound per sec - N
    plt.figure(figsize=(6,6))
    plt.loglog(N_arr, T_rejection_sounding_per_sec, 'o-')
    #plt.ylim(0, 8000)
    plt.ylabel('Rejection sampling - Soundings per second')
    plt.xlabel('Lookup table size')
    plt.grid()
    plt.legend(Nproc_arr)
    plt.xlim(xlim_N)
    plt.tight_layout()
    plt.savefig('%s_rejection_sounding_per_sec_N' % file_out)
    safe_show()
    plt.close()

    ## Rejection sound per sec - per CPU
    plt.figure(figsize=(6,6))
    plt.loglog(Nproc_arr, T_rejection_sounding_per_sec.T, 'o-')
    #plt.ylim(0, 8000)
    plt.ylabel('Rejection sampling - Soundings per second')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.xlim(xlim_Nproc)
    plt.tight_layout()
    plt.savefig('%s_rejection_sounding_per_sec_CPU' % file_out)
    safe_show()
    plt.close()

    ##  Sound per sec per CPU - N  
    plt.figure(figsize=(6,6))
    plt.loglog(N_arr, T_rejection_sounding_per_sec_per_cpu, 'o-')
    plt.plot([0, Nproc_arr[-1]], [0, Nproc_arr[-1]], 'k--')
    plt.xlim(90, 5000000*1.1)
    #plt.ylim(0, 8000)
    plt.ylabel('Rejection sampling - Soundings per second per cpu')
    plt.xlabel('Lookup table size')
    plt.grid()
    plt.legend(Nproc_arr)
    plt.xlim(xlim_N)
    plt.tight_layout()
    plt.savefig('%s_rejection_sounding_per_sec_per_cpu_N' % file_out)
    safe_show()
    plt.close()


    ##  Sound per sec per CPU - CPU 
    plt.figure(figsize=(6,6))
    plt.semilogx(Nproc_arr, T_rejection_sounding_per_sec_per_cpu.T, 'o-')
    plt.ylim([0, np.nanmax(T_rejection_sounding_per_sec_per_cpu.T)*1.1])
    plt.ylabel('Rejection sampling - Soundings per second per cpu')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.xlim(xlim_Nproc)
    plt.tight_layout()
    plt.savefig('%s_rejection_sounding_per_sec_per_cpu_CPU' % file_out)
    safe_show()
    plt.close()


    # ############################################
    # POSTERIOR STATISTICS
    # ############################################
    
    # Average timer per sounding
    T_poststat_sounding = T_poststat/N_arr[:,np.newaxis]
    T_poststat_sounding_per_sec = N_arr[:,np.newaxis]/T_poststat
    T_poststat_sounding_per_sec_per_cpu = T_poststat_sounding_per_sec/Nproc_arr[np.newaxis,:]
    T_poststat_sounding_speedup = T_poststat_sounding_per_sec/T_poststat_sounding_per_sec[0,0]

    plt.figure(figsize=(6,6))
    plt.plot(Nproc_arr, T_poststat_sounding_per_sec.T, 'o-')
    plt.ylabel('Posterior statistics - Soundings per second - $[s^{-1}]$')
    plt.xlabel('Number of processors')
    plt.grid()
    plt.legend(N_arr)
    plt.xlim(xlim_Nproc)
    plt.tight_layout()
    plt.savefig('%s_poststat_sounding_per_sec' % file_out)
    safe_show()
    plt.close()

    # plt.figure(figsize=(6,6))
    # plt.plot(Nproc_arr, T_poststat_sounding_speedup.T, 'o-')
    # # plot a line from 0,0 tp Nproc_arr[-1], Nproc_arr[-1]
    # plt.plot([0, Nproc_arr[-1]], [0, Nproc_arr[-1]], 'k--')
    # # set xlim to 1, Nproc_arr[-1]
    # plt.xlim(.8, Nproc_arr[-1])
    # plt.ylim(.8, Nproc_arr[-1])
    # plt.ylabel('Posterior statistics - speedup compared to 1 processor')
    # plt.xlabel('Number of processors')
    # plt.grid()
    # plt.legend(N_arr)
    # plt.savefig('%s_poststat_speedup' % file_out)

    #####
    # ## Plot Cumulative Time useage for min and max number of used cores

    i_proc = len(Nproc_arr)-1
    #i_proc= 0

    for i_proc in [0,len(Nproc_arr)-1]:

        T=[T_prior[:,i_proc], T_forward[:,i_proc], T_rejection[:,i_proc], T_poststat[:,i_proc]]

        ### %% Plor cumT as an area plot
        plt.figure(figsize=(6,6))
        plt.stackplot(N_arr, T, labels=['Prior', 'Forward', 'Rejection', 'PostStat'])
        plt.plot(N_arr, T_total[:, i_proc], 'k--')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('$N_{lookup}$')
        plt.ylabel('Time [$s$]')
        plt.title('Cumulative time, using %d processors' % Nproc_arr[i_proc])
        plt.legend(loc='upper left')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig('%s_Ncpu%d_cumT' % (file_out,Nproc_arr[i_proc]))
        safe_show()
        plt.close()

        # The same as thea area plot but normalized to the total time
        plt.figure(figsize=(6,6))
        plt.stackplot(N_arr, T/np.sum(T, axis=0), labels=['Prior', 'Forward', 'Rejection', 'PostStat'])
        plt.xscale('log')
        plt.xlabel('$N_{lookup}$')
        plt.ylabel('Normalized time')
        plt.legend(loc='upper left')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.title('Normalized time, using %d processors' % Nproc_arr[i_proc])
        plt.savefig('%s_Ncpu%d_cumT_norm' % (file_out,Nproc_arr[i_proc]))
        safe_show()
        plt.close()

# Working will well data

def compute_P_obs_from_log(depth_top, depth_bottom, lithology_obs, z, class_id, P_single=0.8, P_prior=None):
    """
    Compute discrete observation probability matrix from depth intervals and lithology observations.
    
    This function creates a probability matrix where each depth point is assigned 
    probabilities based on observed lithology classes within specified depth intervals.
    
    Parameters
    ----------
    depth_top : array-like
        Array of top depths for each observation interval.
    depth_bottom : array-like
        Array of bottom depths for each observation interval.
    lithology_obs : array-like
        Array of observed lithology class IDs for each interval.
    z : array-like
        Array of depth/position values where probabilities are computed.
    class_id : array-like
        Array of unique class identifiers (e.g., [0, 1, 2] for 3 lithology types).
    P_single : float, optional
        Probability assigned to the observed class. Default is 0.8.
    P_prior : ndarray, optional
        Prior probability matrix of shape (nclass, nm). If None, uses uniform distribution
        for depths not covered by observations. Default is None.
    
    Returns
    -------
    P_obs : ndarray
        Probability matrix of shape (nclass, nm) where nclass is the number of classes
        and nm is the number of depth points. For each depth point covered by observations,
        the observed class gets probability P_single and other classes share (1-P_single).
        Depths not covered by any observation contain NaN or prior probabilities if provided.
    
    Examples
    --------
    >>> depth_top = [0, 10, 20]
    >>> depth_bottom = [10, 20, 30]
    >>> lithology_obs = [1, 2, 1]  # clay, sand, clay
    >>> z = np.arange(30)
    >>> class_id = [0, 1, 2]  # gravel, clay, sand
    >>> P_obs = compute_P_obs_from_log(depth_top, depth_bottom, lithology_obs, z, class_id)
    >>> print(P_obs.shape)  # (3, 30)
    """
    import numpy as np
    
    nm = len(z)
    nclass = len(class_id)
    
    # Compute probability for non-hit classes
    P_nohit = (1 - P_single) / (nclass - 1)
    
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
                # Assign probabilities for all classes
                for ic in range(nclass):
                    if class_id[ic] == lithology_obs[i]:
                        P_obs[ic, im] = P_single
                    else: 
                        P_obs[ic, im] = P_nohit
    
    return P_obs

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

# def Pobs_to_datagrid(P_obs, X, Y, f_data_h5, r_data=10, r_dis=100, doPlot=False):
#     """
#     Convert point-based discrete probability observations to gridded data with distance-based weighting.

#     This function distributes discrete probability observations (e.g., from a borehole) across
#     a spatial grid using distance-based weighting. Observations at location (X, Y) are applied
#     to nearby grid points with decreasing influence based on distance. Temperature annealing
#     is used to reduce the strength of observations far from the source point.

#     Parameters
#     ----------
#     P_obs : ndarray
#         Probability matrix of shape (nclass, nm) where nclass is the number of classes
#         and nm is the number of model parameters (e.g., depth points).
#         Each column represents a probability distribution over discrete classes.
#     X : float
#         X coordinate (e.g., UTM Easting) of the observation point.
#     Y : float
#         Y coordinate (e.g., UTM Northing) of the observation point.
#     f_data_h5 : str
#         Path to HDF5 data file containing survey geometry (X, Y coordinates).
#     r_data : float, optional
#         Inner radius in meters within which observations have full strength.
#         Default is 10 meters.
#     r_dis : float, optional
#         Outer radius in meters for distance-based weighting. Beyond this distance,
#         observations are fully attenuated (temperature → ∞). Default is 100 meters.
#     doPlot : bool, optional
#         If True, creates diagnostic plots showing weight distributions.
#         Default is False.

#     Returns
#     -------
#     d_obs : ndarray
#         Gridded observation data of shape (nd, nclass, nm) where nd is the number
#         of spatial locations in the survey. Each location gets temperature-scaled
#         probabilities based on distance from (X, Y).
#     i_use : ndarray
#         Binary mask of shape (nd, 1) indicating which grid points should be used
#         (1) or ignored (0) in the inversion. Points with temperature < 100 are used.
#     T_all : ndarray
#         Array of temperature values of shape (nd,) for each grid point, indicating
#         the strength of observation influence based on distance.

#     Notes
#     -----
#     The function uses distance-based temperature annealing:
#     1. Computes distance-based weights using `get_weight_from_position()`
#     2. Converts distance weight to temperature: T = 1 / w_dis
#     3. Caps maximum temperature at 100 (very weak influence)
#     4. For each grid point:
#        - If T < 100: include point (i_use=1) and apply temperature scaling
#        - If T ≥ 100: exclude point (i_use=0) and set observations to NaN

#     Temperature scaling reduces probability certainty with distance:
#     - T = 1 (close to observation): Original probabilities preserved
#     - T > 1 (far from observation): Probabilities become more uniform
#     - T ≥ 100 (very far): Observations effectively ignored

#     Examples
#     --------
#     >>> # Borehole observation at specific location
#     >>> P_obs = compute_P_obs_from_log(depth_top, depth_bottom, lithology, z, class_id)
#     >>> X_well, Y_well = 543000.0, 6175800.0
#     >>> d_obs, i_use, T_all = Pobs_to_datagrid(P_obs, X_well, Y_well, 'survey_data.h5',
#     ...                                  r_data=10, r_dis=100)
#     >>> # Write to data file
#     >>> ig.write_data_multinomial(d_obs, i_use=i_use, id=2, f_data_h5='survey_data.h5')

#     See Also
#     --------
#     rescale_P_obs_temperature : Temperature scaling function
#     compute_P_obs_from_log : Create P_obs from depth intervals
#     get_weight_from_position : Distance-based weighting function
#     """
#     import numpy as np
#     import integrate as ig

#     # Get grid dimensions from data file
#     X_grid, Y_grid, _, _ = ig.get_geometry(f_data_h5)
#     nd = len(X_grid)
#     nclass, nm = P_obs.shape

#     # Initialize output arrays
#     i_use = np.zeros((nd, 1))
#     d_obs = np.zeros((nd, nclass, nm)) * np.nan

#     # Compute distance-based weights for all grid points
#     w_combined, w_dis, w_data, i_use_from_func = ig.get_weight_from_position(
#         f_data_h5, X, Y, r_data=r_data, r_dis=r_dis, doPlot=doPlot
#     )

#     # Convert distance weight to temperature
#     # w_dis is 1 at observation point, decreases with distance
#     # T = 1/w_dis means T increases with distance (weaker influence)
#     T_all = 1 / w_combined
#     #T_all = 1 / w_dis
#     #T_all = 1 / w_data

#     # Cap maximum temperature at 100 (beyond this, observation has negligible effect)
#     T_all[T_all > 100] = 100

#     # Apply temperature scaling to each grid point
#     for ip in np.arange(nd):
#         T = T_all[ip]

#         # Only use points where temperature is reasonable (< 100)
#         if T < 100:
#             i_use[ip] = 1
#             # Scale probabilities based on distance (higher T = more uniform distribution)
#             P_obs_local = rescale_P_obs_temperature(P_obs, T=T)
#             d_obs[ip, :, :] = P_obs_local
#         # else: i_use[ip] = 0 and d_obs[ip] stays NaN

#     return d_obs, i_use, T_all

