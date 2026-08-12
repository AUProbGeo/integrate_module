"""
Rejection Sampling Module for INTEGRATE

This module contains functions for Bayesian inversion using rejection sampling methodology.
It includes the main rejection sampling algorithm, likelihood calculations, and parallel 
processing support for efficient posterior sampling.

Key Functions:
- integrate_rejection(): Main rejection sampling function
- integrate_rejection_range(): Core rejection sampling for data point ranges  
- likelihood_*(): Various likelihood calculation functions
- Shared memory functions for parallel processing
"""

import numpy as np
import h5py
import os
import time
import multiprocessing
from multiprocessing import shared_memory
from datetime import datetime
from tqdm import tqdm
import logging

# Set up logging
logger = logging.getLogger(__name__)

def integrate_rejection(f_prior_h5='prior.h5', 
                              f_data_h5='DAUGAAD_AVG_inout.h5',
                              f_post_h5='',                              
                              N_use=100000000000, 
                              id_use=[], 
                              ip_range=[], 
                              nr=1000,
                              autoT=1,
                              T_base = 1,
                              Nchunks=0,
                              Ncpu=0,
                              parallel=True,
                              use_N_best=0,
                              T_N_above=None,
                              T_P_acc_level=None,
                              progress_callback=None,
                              console_progress=None,
                              backend='numpy',
                              **kwargs):
    """
    Perform probabilistic inversion using rejection sampling.
    
    This is the main function for Bayesian inversion using rejection sampling methodology.
    It samples the posterior distribution by rejecting prior samples that are inconsistent 
    with observed data within a temperature-controlled tolerance. Supports parallel processing
    and automatic temperature estimation for efficient sampling.
    
    Parameters
    ----------
    f_prior_h5 : str, optional
        Path to HDF5 file containing prior model and data samples.
        Default is 'prior.h5'.
    f_data_h5 : str, optional
        Path to HDF5 file containing observed data for inversion.
        Default is 'DAUGAAD_AVG_inout.h5'.
    f_post_h5 : str, optional
        Output path for posterior samples. If empty, auto-generated from prior filename.
        Default is empty string.
    N_use : int, optional
        Maximum number of prior samples to use for inversion.
        Default is 100000000000.
    id_use : list, optional
        List of data identifiers to use for inversion. If empty, uses all available data.
        Default is empty list.
    ip_range : list, optional
        List of data point indices to invert. If empty, inverts all data points.
        Default is empty list.
    nr : int, optional
        Number of posterior samples to retain per data point.
        Default is 1000.
    autoT : int, optional
        Automatic temperature estimation method (1=enabled, 0=disabled).
        Default is 1.
    T_base : float, optional
        Base temperature for rejection sampling when autoT=0.
        Default is 1.
    Nchunks : int, optional
        Number of chunks for parallel processing. If 0, auto-determined.
        Default is 0.
    Ncpu : int, optional
        Number of CPU cores to use. If 0, auto-determined from system.
        Default is 0.
    parallel : bool, optional
        Enable parallel processing if environment supports it.
        Default is True.
    use_N_best : int, optional
        Use only the N best-fitting samples (0=disabled).
        Default is 0.
    T_N_above : int, optional
        Number of top samples used by ``logl_T_est`` to estimate the annealing
        temperature. Passed as ``N_above`` to :func:`logl_T_est`.
        Default is None (uses ``logl_T_est`` default of 10).
    T_P_acc_level : float, optional
        Target acceptance probability level used by ``logl_T_est`` to estimate
        the annealing temperature. Passed as ``P_acc_lev`` to :func:`logl_T_est`.
        Default is None (uses ``logl_T_est`` default of 0.2).
    progress_callback : callable, optional
        Callback function for progress updates. Called as progress_callback(current, total, info_dict).
        Default is None (no callback).
    console_progress : bool, optional
        Whether to show console TQDM progress bar. If None, auto-detects based on progress_callback.
        Default is None.
    backend : str, optional
        Computation backend to use.  ``'numpy'`` (default) uses the original
        NumPy/multiprocessing implementation.  ``'jax'`` uses a JIT-compiled,
        vmapped JAX implementation that processes data points in batches; pass
        ``Nbatch=<int>`` (via **kwargs, default 64) to tune the batch size.
        JAX must be installed separately: ``pip install jax``.
    **kwargs : dict
        Additional keyword arguments including showInfo, updatePostStat, post_dir,
        Nbatch (batch size for backend='jax'), and normalize_likelihood (bool,
        default False; if True, Gaussian likelihoods include the full
        normalization constant, giving a properly comparable EV/entropy across
        data points and hypotheses, without changing P_acc/T/CHI2).

    Returns
    -------
    str
        Path to the output HDF5 file containing posterior samples and statistics.
    
    Notes
    -----
    The function automatically determines optimal processing parameters based on data size
    and system capabilities. Temperature annealing is used to improve sampling efficiency.
    
    Large datasets may require significant memory and processing time. Monitor system
    resources during execution.
    
    Examples
    --------
    >>> import integrate as ig
    >>> f_post = ig.integrate_rejection('prior.h5', 'data.h5', N_use=10000)
    >>> print(f"Results saved to: {f_post}")
    """
    # Safety guard: if somehow called from a worker process, do nothing.
    if multiprocessing.current_process().name != 'MainProcess':
        return None

    import integrate as ig

    # get optional arguments
    showInfo = kwargs.get('showInfo', 0)
    updatePostStat = kwargs.get('updatePostStat', True)
    # If set, Nproc will be used as the number of processors
    Ncpu = kwargs.get('Nproc', Ncpu)
    Ncpu = kwargs.get('N_cpu', Ncpu) # Allow using N_cpu instead of Ncpu
    Nchunks = kwargs.get('N_chunks', Nchunks) # Allow using N_chunks instead of Nchunks
    posterior_output_path = kwargs.get('post_dir', os.getcwd())
    # The parallel (multiprocessing) path below does not forward **kwargs, so
    # this needs to be extracted here and threaded explicitly into
    # integrate_posterior_main. The backend='jax' and non-parallel branches
    # already forward **kwargs and pick this up on their own.
    normalize_likelihood = kwargs.get('normalize_likelihood', False)
    
    # Setup progress callback functionality
    if console_progress is None:
        # Auto-detect: disable console if callback provided
        console_progress = (progress_callback is None)
    
    def update_progress(current, total, extra_info=None):
        """Update both TQDM and GUI callback"""
        if progress_callback:
            try:
                info = {
                    'data_point': current,
                    'total_points': total,
                    'phase': extra_info.get('phase', 'processing') if extra_info else 'processing',
                    'status': extra_info.get('status', '') if extra_info else ''
                }
                if extra_info:
                    info.update(extra_info)
                progress_callback(current, total, info)
            except Exception as e:
                # Don't break main process on callback error
                if showInfo > 0:
                    print(f"Progress callback error: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Note: TQDM disabling is handled in individual tqdm() calls via disable parameter

    if Ncpu < 1 :
        Ncpu =  int(multiprocessing.cpu_count())
        # Set Ncpu to be min of Ncpu and 8
        # as no gain is expected from using more than 8 processors
        Ncpu = min(Ncpu, 8)
    
    # Initial progress update - starting process
    if progress_callback:
        update_progress(0, 1, {'phase': 'initializing', 'status': 'Starting rejection sampling'})

    # Set default f_post_h5 filename if not set    
    if len(f_post_h5)==0:
        # Extract the base name of f_prior_h5 without its path or extension
        f_prior_basename = os.path.splitext(os.path.basename(f_prior_h5))[0]
        
        # Construct the new filename
        f_post_h5 = os.path.join(posterior_output_path, "POST_%s_Nu%d_aT%d.h5" % (f_prior_basename, N_use, autoT))

    # Check that f_post_h5 allready exists, and warn the user   
    if os.path.isfile(f_post_h5):
        if (showInfo>0):    
            print('File %s allready exists' % f_post_h5)
            print('Overwriting...')    

    
    # Load ALL observed data from f_data_h5, mostly to find out how many data types there are
    # This could be more efficient.
    #DATA = ig.load_data(f_data_h5, showInfo=showInfo)
    #Ndt = len(DATA['d_obs']) # Number of data types
    Ndt_total = ig.get_number_of_datasets(f_data_h5)
    Ndt = ig.get_number_of_datasets(f_data_h5)

    
    # if if_use is not a list, convert it to a list
    if not isinstance(id_use, list):
        id_use = [id_use]

    if len(id_use)==0:
        id_use = np.arange(1,Ndt_total+1).tolist()


    Ndt = len(id_use) # Number of data types used

    if showInfo>1:
        print('-- Number of data types: %d' % Ndt_total)
        
        print('-- Using these data types: %s' % str(id_use))

    
    # Load the observed data from the h5 files
    DATA = ig.load_data(f_data_h5, id_arr=id_use, showInfo=showInfo)
    
    # Load the prior data from the h5 files
    id_data_prior = DATA['id_prior']
    D, idx = ig.load_prior_data(f_prior_h5, id_use=id_data_prior, N_use=N_use, Randomize=True, showInfo=showInfo)
    
    
    if showInfo>1:
        for i in range(len(D)):
            print('Memory size of /D%d: %s' % (id_use[i], str(np.array(D[i]).nbytes)))

        # Print infomration about DATA and D, and make sure the same data types are used in both D and DATA
        print('  Number of data types in DATA: %d' % len(DATA['d_obs']))
        print('  Number of data types in D: %d' % len(D))
        for i in range(len(id_use)):
            print('  Size of data in DATA:/D%d: (%d, %d)' % (id_use[i], DATA['d_obs'][i].shape[0], DATA['d_obs'][i].shape[1]))
            print('  Size of prior data PRIOR:/D%d: (%d, %d)' % (id_use[i], D[i].shape[0], D[i].shape[1]))


    #D, idx = load_prior_data(f_prior_h5, id_use = id_use, N_use = N_use, Randomize=True)
    # M, idx = load_prior_model(f_prior_h5, idx=idx, N_use=N_use, Randomize=True)

    # Get sample size N from f_prior_h5
    N = D[0].shape[0]
    if N_use>N:
        N_use = N

    # Get number of data points from, f_data_h5
    Ndp = DATA['d_obs'][0].shape[0]
    
    # if ip_range is empty then use all data points
    if len(ip_range)==0:
        ip_range = np.arange(Ndp)
    Ndp_invert = len(ip_range)

    # Store the ip_range for later use in integrate_posterior_stats
    ip_range_for_stats = np.copy(ip_range)

    if Ncpu ==1:
        parallel = False
    
    if showInfo>0:
        print('<--INTEGRATE_REJECTION-->')
        print('f_prior_h5=%s, f_data_h5=%s\nf_post_h5=%s' % (f_prior_h5, f_data_h5, f_post_h5))
    
    if showInfo>1:
        print('Number of data points: %d (available), %d (used). Nchunks=%s, Ncpu=%d,use_N_best=%d' % (Ndp,Ndp_invert,Nchunks,Ncpu,use_N_best))    
        print('N_use = %d' % (N_use))
        print('Ndp to invert = %d, ip_range=%s' % (len(ip_range),str(ip_range)))
        print('use_N_best=%d' % use_N_best)
        print('Number of data types: %d' % Ndt)
        print('Using these data types: %s' % str(id_use))
        print('Loaded these prior data model types:', str(id_data_prior))
        
    # set i_use_all to be a 2d Matrix of size (nump,nr) of random integers in range(N)
    i_use_all = np.random.randint(0, N, (Ndp, nr))
    N_UNIQUE_all = np.zeros(Ndp)*np.nan
    T_all = np.zeros(Ndp)*np.nan
    EV_all = np.zeros(Ndp)*np.nan
    # 'posterior' evience - mean posterior likelihood TODO
    EV_post_all  = np.zeros(Ndp)*np.nan
    EV_post_all_mean = np.zeros(Ndp)*np.nan
    CHI2_all = np.zeros((Ndp, Ndt))*np.nan

    date_start = str(datetime.now())
    t_start = datetime.now()
    
    
    # Resolve None to logl_T_est defaults before forwarding into parallel/serial paths
    _T_N_above     = T_N_above     if T_N_above     is not None else 10
    _T_P_acc_level = T_P_acc_level if T_P_acc_level is not None else 0.2

    # Depending in whether parallel processing is used or not,
    # two function are implemented to perform the inversion directly on the loaded data.
    # A third option, backend='jax', uses JAX for likelihood computation.


    if backend == 'jax':
        from integrate.integrate_rejection_jax import integrate_rejection_range_jax
        Nbatch = kwargs.pop('Nbatch', 64)
        i_use, T, EV, EV_post, EV_post_mean, CHI2, N_UNIQUE, ip_range = integrate_rejection_range_jax(
            D=D,
            DATA=DATA,
            idx=idx,
            N_use=N_use,
            id_use=id_use,
            ip_range=ip_range,
            autoT=autoT,
            T_base=T_base,
            nr=nr,
            T_N_above=_T_N_above,
            T_P_acc_level=_T_P_acc_level,
            progress_callback=progress_callback,
            Nbatch=Nbatch,
            **kwargs,
        )
        for i in range(len(ip_range)):
            ip = ip_range[i]
            i_use_all[ip] = i_use[i]
            T_all[ip] = T[i]
            EV_all[ip] = EV[i]
            EV_post_all[ip] = EV_post[i]
            EV_post_all_mean[ip] = EV_post_mean[i]
            CHI2_all[ip, :] = CHI2[i, :]
            N_UNIQUE_all[ip] = N_UNIQUE[i]

    elif parallel:
        # Split the ip_range into Nchunks
        if Nchunks==0:
            if parallel:
                if progress_callback is not None:
                    # Finer chunking gives smoother live progress updates
                    Nchunks = min(len(ip_range), Ncpu * 4)
                else:
                    Nchunks = Ncpu
            else:
                Nchunks = 1
        ip_range_shuffled = ip_range.copy()
        np.random.shuffle(ip_range_shuffled)
        ip_chunks = np.array_split(ip_range_shuffled, Nchunks)

        if showInfo>1:
            print('Ncpu = %d\nNchunks=%d' % (Ncpu, Nchunks))

        i_use_all, T_all, EV_all, EV_post_all, EV_post_all_mean, CHI2_all, N_UNIQUE_all = integrate_posterior_main(
            ip_chunks=ip_chunks,
            D=D,
            DATA = DATA,
            idx = idx,
            N_use=N_use,
            id_use=id_use,
            autoT=autoT,
            T_base=T_base,
            nr=nr,
            Ncpu=Ncpu,
            use_N_best=use_N_best,
            T_N_above=_T_N_above,
            T_P_acc_level=_T_P_acc_level,
            normalize_likelihood=normalize_likelihood,
            progress_callback=update_progress if progress_callback else None,
        )


    else:

            i_use, T, EV, EV_post, EV_post_mean, CHI2, N_UNIQUE, ip_range = integrate_rejection_range(D=D,
                                        DATA = DATA,
                                        idx = idx,
                                        N_use=N_use,
                                        id_use=id_use,
                                        ip_range=ip_range,
                                        autoT=autoT,
                                        T_base = T_base,
                                        nr=nr,
                                        use_N_best=use_N_best,
                                        T_N_above=_T_N_above,
                                        T_P_acc_level=_T_P_acc_level,
                                        progress_callback=progress_callback,
                                        console_progress=console_progress,
                                        **kwargs
                                        )
        
            for i in range(len(ip_range)):
                ip = ip_range[i]
                #print('ip=%d, i=%d' % (ip,i))
                i_use_all[ip] = i_use[i]
                T_all[ip] = T[i]
                EV_all[ip] = EV[i]
                EV_post_all[ip] = EV_post[i]
                EV_post_all_mean[ip] = EV_post_mean[i]
                CHI2_all[ip, :] = CHI2[i, :]
                N_UNIQUE_all[ip] = N_UNIQUE[i]

    # WHere T_all is Inf set it to Nan
    T_all[T_all==np.inf] = np.nan
    EV_all[EV_all==np.inf] = np.nan
    
    date_end = str(datetime.now())
    t_end = datetime.now()
    t_elapsed = (t_end - t_start).total_seconds()
    t_per_sounding = t_elapsed / Ndp_invert
    if (showInfo>-1):
        print('integrate_rejection: Time=%5.1fs/%d soundings, %4.1fms/sounding, %3.1fit/s. ' % (t_elapsed,Ndp_invert,t_per_sounding*1000,Ndp_invert/t_elapsed), end='')
        print('T_av=%3.1f, EV_av=%3.1f' % (np.nanmean(T_all), np.nanmean(EV_all)))

    # SAVE THE RESULTS to f_post_h5
    with h5py.File(f_post_h5, 'w') as f_post:
        f_post.create_dataset('i_use', data=i_use_all)
        f_post.create_dataset('T', data=T_all)
        f_post.create_dataset('EV', data=EV_all)
        f_post.create_dataset('EV_post', data=EV_post_all)
        f_post.create_dataset('EV_post_mean', data=EV_post_all_mean)
        f_post.create_dataset('CHI2', data=CHI2_all)
        f_post.create_dataset('N_UNIQUE', data=N_UNIQUE_all)
        #f_post.create_dataset('ip_range', data=ip_range)
        f_post.attrs['date_start'] = date_start
        f_post.attrs['date_end'] = date_end
        f_post.attrs['inv_time'] = t_elapsed
        f_post.attrs['f5_prior'] = f_prior_h5
        f_post.attrs['f5_data'] = f_data_h5
        f_post.attrs['N_use'] = N_use
    
    # Update progress - saving results
    if progress_callback:
        update_progress(len(ip_range), len(ip_range), {'phase': 'saving', 'status': 'Results saved to HDF5 file'})

    if updatePostStat:
        if progress_callback:
            update_progress(len(ip_range), len(ip_range), {'phase': 'post_processing', 'status': 'Computing posterior statistics'})
        ig.integrate_posterior_stats(f_post_h5, ip_range=ip_range_for_stats, **kwargs)
    
    # Final progress update - completion
    if progress_callback:
        update_progress(len(ip_range), len(ip_range), {'phase': 'completed', 'status': 'Integration completed successfully'})

    #return f_post_h5 T_all, EV_all, i_use_all
    return f_post_h5



def integrate_rejection_range(D, 
                              DATA, 
                              idx = [],
                              N_use=None, 
                              id_use=[], 
                              ip_range=[], 
                              nr=1000,
                              autoT=1,
                              T_base = 1,
                              T_N_above=10,
                              T_P_acc_level=0.2,
                              progress_callback=None,
                              **kwargs):
    """
    Perform rejection sampling for a specific range of data points.
    
    This function implements the core rejection sampling algorithm for a subset of data points.
    It evaluates likelihood for each data point in the range and accepts/rejects prior samples
    based on temperature-controlled criteria. Used internally by integrate_rejection for
    both serial and parallel processing.
    
    Parameters
    ----------
    D : list
        List of forward modeled data arrays for each data type.
    DATA : dict
        Dictionary containing observed data including 'd_obs', 'd_std', and other data arrays.
    idx : list, optional
        Indices of prior samples to use. If empty, uses sequential indexing.
        Default is empty list.
    N_use : int, optional
        Maximum number of prior samples to evaluate.
        Default is 1000.
    id_use : list, optional
        List of data identifiers to use for likelihood calculation.
        Default is [] which use all data types available.
    ip_range : list, optional
        Range of data point indices to process. If empty, processes all data points.
        Default is empty list.
    nr : int, optional
        Number of posterior samples to retain per data point.
        Default is 1000.
    autoT : int, optional
        Automatic temperature estimation method (1=enabled, 0=disabled).
        Default is 1.
    T_base : float, optional
        Base temperature for rejection sampling when autoT=0.
        Default is 1.
    T_N_above : int, optional
        Number of top samples used by ``logl_T_est`` to estimate the annealing
        temperature. Passed as ``N_above`` to :func:`logl_T_est`.
        Default is 10.
    T_P_acc_level : float, optional
        Target acceptance probability level used by ``logl_T_est`` to estimate
        the annealing temperature. Passed as ``P_acc_lev`` to :func:`logl_T_est`.
        Default is 0.2.
    progress_callback : callable, optional
        Optional callback function for progress updates. Called with (current, total).
        Default is None (no callbacks).
    **kwargs : dict
        Additional arguments including useRandomData, showInfo, use_N_best, and
        normalize_likelihood (bool, default False; use the full normalized
        Gaussian log-pdf. Inert for P_acc/T; CHI2 is de-normalized back out;
        EV becomes a properly comparable evidence across data points/hypotheses).
    
    Returns
    -------
    i_use_all : ndarray, shape (nump, nr)
        Indices of accepted posterior samples for each data point.
    T_all : ndarray, shape (nump,)
        Temperature values used for each data point.
    EV_all : ndarray, shape (nump,)
        Evidence values for each data point.
    EV_post_all : ndarray, shape (nump,)
        Posterior evidence values for each data point.
    N_UNIQUE_all : ndarray, shape (nump,)
        Number of unique samples for each data point.
    ip_range : ndarray
        Range of data point indices that were processed.
    
    Notes
    -----
    This function is the computational core of the rejection sampling algorithm.
    It handles temperature annealing and likelihood evaluation for efficient sampling.
    
    The algorithm evaluates the likelihood of observed data given forward modeled data
    for each prior sample, then uses temperature-controlled acceptance criteria to
    select posterior samples that are consistent with observations.
    """
    
    import integrate as ig

    # get optional arguments
    use_N_best = kwargs.get('use_N_best', 0)
    #print("use_N_best=%d" % use_N_best)
    showInfo = kwargs.get('showInfo', 0)
    console_progress = kwargs.get('console_progress', True)
    if (showInfo<0):
        disableTqdm=True
    else:
        disableTqdm=not console_progress
    
    useRandomData = kwargs.get('useRandomData', True)
    #useRandomData = kwargs.get('useRandomData', False)

    # If True, Gaussian likelihoods include the full normalization constant
    # (-0.5*(n*log(2*pi) + log|Cov|)). This is inert for P_acc/T (a constant
    # shift cancels in L - max(L)), but makes EV a properly comparable
    # evidence across data points/hypotheses with differing noise or valid
    # channel counts. CHI2 is de-normalized back out below so it stays a
    # pure misfit statistic.
    normalize_likelihood = kwargs.get('normalize_likelihood', False)


    # Get number of data points
    Ndp = DATA['d_obs'][0].shape[0]
    # if ip_range is empty then use all data points
    if len(ip_range)==0:
        ip_range = np.arange(Ndp)

    nump=len(ip_range)
        
    # Get number of data types used - needed for array initialization
    if len(id_use)==0:
        # Get nmumber of data points from
        Ndt=len(DATA['d_obs'])
        id_use = np.arange(Ndt)
    Ndt = len(id_use)
        
    i_use_all = np.zeros((nump, nr), dtype=np.int32)
    T_all = np.zeros(nump)*np.nan
    EV_all = np.zeros(nump)*np.nan
    EV_post_all = np.zeros(nump)*np.nan
    EV_post_all_mean = np.zeros(nump)*np.nan
    CHI2_all = np.zeros((nump, Ndt))*np.nan
    N_UNIQUE_all = np.zeros(nump)*np.nan
    
    
    # Get the lookup sample size
    N = D[0].shape[0]
    if N_use is None:
        N_use = N

    if N_use>N:
        N_use = N

    if len(idx)==0:
        idx = np.arange(N_use)
    
    noise_model = DATA['noise_model']
    i_use_data = DATA['i_use']
    
    if showInfo>1:
        print('Number of data points to invert: %d' % nump)
        print('Number of data type(s) used, Ndt=%d' % Ndt)
        print('Noise model(s):', noise_model)
        
    # Convert class id to index
    # The class_id_list, could /should be loaded prior_h5:/M1/class_id !!

    # Select whether to convert CLASS to IDX before doing inversion?
    # class_is_idx = True is MUCH faster!!
    class_is_idx = True
    #class_is_idx = False
    
    
    class_id_list = []
    updated_data_ids = []
    for i in range(Ndt):
        i_prior = i
        if (noise_model[i]=='multinomial'):
            Di, class_id, class_id_out = ig.class_id_to_idx(D[i_prior])
            #print(class_id_out)
            if (class_is_idx)&(i_prior not in updated_data_ids):
                updated_data_ids.append(i_prior)
                D[i_prior]=Di
                if showInfo>1:
                    print('Updated prior id %d' % i_prior)
            
            if (class_is_idx):                
                class_id_list.append(class_id_out)
            else:
                class_id_list.append(class_id)

        else:    
            class_id_list.append([])

    if showInfo>2:
        print('class_id_list',class_id_list)
        print('len(class_id_list)',len(class_id_list))

    #print(ip_range)
    # Throttle callback to ~100 updates so it does not dominate runtime
    progress_step = max(1, nump // 100)
    #tqdm(range(nd), mininterval=1, disable=disableTqdm, desc='gatdaem1d', leave=False):
    for j in tqdm(range(len(ip_range)), disable=disableTqdm, desc='Rejection Sampling', leave=False):
        if progress_callback and ((j + 1) % progress_step == 0 or j + 1 == nump):
            try:
                progress_callback(j + 1, nump,
                                  {'phase': 'sampling',
                                   'status': 'Rejection sampling (%d/%d data points)' % (j + 1, nump)})
            except Exception:
                pass
        ip = ip_range[j] # This is the index of the data point to invert
  
        t=[]
        N = D[0].shape[0]
        # Get number of data types used - needed for array initialization
        NDsets = len(id_use)
        L = np.zeros((NDsets, N))

        if showInfo>3:
            print('Ndt=%d, ip=%d/%d, N=%d' % (Ndt, ip, nump, N))

        # Loop over the number of data types Ndt
        total_n_data_non_nan = 0  # Initialize total count for all data types
        n_data_per_type = np.zeros(Ndt)  # Track data count per data type
        log_norm_const_per_type = np.zeros(Ndt)  # Gaussian normalization constant used per data type (0 unless normalize_likelihood)

        for i in range(Ndt):
            use_data_point = i_use_data[i][ip]
            #print(j)
            #print(ip)
            #print('..')
            #print('i=%g, j=%g, ip=%g' % (i,j,ip))
            #print(use_data_point)
            #use_data_point = 1 # FORCE USE OF DATA POINT
            #use_data_point = 0 # FORCE NOT TO USE DATA POINT
            if showInfo>3:    
                print("-i=%d, Using data type %d" % (i,Ndt))
                print("len(D)",len(D))
                
            n_data_non_nan=0
            if (use_data_point==1):
                #if i_use_data[i]==1:
                #    print('Using data %d' % i)
                #
                if showInfo>3:    
                    print('j=%4d Using data %d --> %d' % (j,i,use_data_point))

                # ONLY PERFORM CALUCATION IF I_USE_DATA = 1.. UPDATE LOAD_DATA, TO ALWAY PROVIDE I_USE 
                # Select the proper data types. It is give, the integer in 'D1/', 'D2/' etc, so we need to subtract 1
                # as D1 is the first data types D[0]
                
                i_prior = i 
                
                t0=time.time()
                #id = id_use[i_prior]
                if noise_model[i]=='gaussian':
                    d_obs = DATA['d_obs'][i][ip]
                    n_data_non_nan = np.sum(~np.isnan(d_obs))
                    total_n_data_non_nan += n_data_non_nan
                    n_data_per_type[i] = n_data_non_nan
                        
                    if DATA['Cd'][0] is not None:                    
                        # if Cd is 3 dimensional, take the first slice
                        if len(DATA['Cd'][0].shape) == 3:
                            Cd = DATA['Cd'][0][ip]
                        else:
                            Cd = DATA['Cd'][0][:]

                        L_single, log_norm_const_per_type[i] = likelihood_gaussian_full(
                            D[i_prior], d_obs, Cd, N_app = use_N_best,
                            normalize=normalize_likelihood, return_norm_const=True)

                    elif DATA['d_std'][0] is not None:
                        d_std = DATA['d_std'][i][ip]
                        #print(d_obs)
                        #print(d_std)
                        #print(D[i_prior][0])
                        L_single, log_norm_const_per_type[i] = likelihood_gaussian_diagonal(
                            D[i_prior], d_obs, d_std, use_N_best,
                            normalize=normalize_likelihood, return_norm_const=True)
                        #print(L_single[0:3])
                    else:
                        print('No d_std or Cd in %s' % DS)

                    L[i] = L_single
                    t.append(time.time()-t0)
                elif noise_model[i]=='multinomial':
                    d_obs = DATA['d_obs'][i][ip]
                    n_data_non_nan = np.sum(~np.isnan(d_obs))
                    total_n_data_non_nan += n_data_non_nan
                    n_data_per_type[i] = n_data_non_nan

                    if showInfo>3:
                        print(D[i])
                    
                    class_id = class_id_list[i]
                    #print(class_id)
                    useMultiNomal = True
                    if useMultiNomal:
                        
                        L_single = likelihood_multinomial(D[i_prior],d_obs, np.array(class_id), class_is_idx=class_is_idx)
                        #print(L_single[0])
                    L[i] = L_single           
                    t.append(time.time()-t0)

                else: 
                    # noise model not regcognized
                    # L_single = -1
                    pass
            else:
                L[i] = np.zeros(N)
                    
        t0=time.time()

        # Now we have all the likelihoods for all data types. Combine them into one
        # L is an array of shape (Ndt,1)
        # If we have only one data type, then L is already correct, 
        # and we do not need to sum
        L_single = L
        if Ndt>1:
            L = np.sum(L_single, axis=0)

        # Automatic annealing temperature estimation, if autoT=1, else use T=T_base
        # T_base = 1 indicates no annealing
        t0=time.time()
        # Compute the annealing temperature
        if autoT == 1:
            T = ig.logl_T_est(L, N_above=T_N_above, P_acc_lev=T_P_acc_level)
        else:
            T = T_base        
        # maxlogL = np.nanmax(logL)
        t.append(time.time()-t0)

        # Find ns realizations of the posterior, using the log-likelihood values logL, and the annealing tempetrature T 
        
        P_acc = np.exp((1/T) * (L - np.nanmax(L)))
        P_acc[np.isnan(P_acc)] = 0

        # Select the index of P_acc propportion to the probabilituy given by P_acc
        t0=time.time()
        try:
            if P_acc.shape[0] == 1:
                # This should probably not happen!
                P_acc = P_acc.flatten()
            p=P_acc/np.sum(P_acc)
            i_use = np.random.choice(N, nr, p=p)
        except:       
            print('####################################################################')     
            print('####################################################################')     
            print('Error in np.random.choice for ip=%d' % ip)   
            print('####################################################################')     
            print('####################################################################')     
            i_use = np.random.choice(N, nr)
            

        #print(P_acc.shape)
        #print(p.shape)
        #print(i_use.shape)

        # Store i_use before reordering (for computing CHI2)
        #i_use_before_reordering = i_use.copy()
        # Compute CHI2 (reduced chi-squared) per data type
        # CHI2 = mean(-2 * log-likelihood) / n_data
        CHI2_current = np.zeros(Ndt) * np.nan
        for i in range(Ndt):
            if n_data_per_type[i] > 0:
                # Get log-likelihood for accepted samples for this data type
                # De-normalize (subtract back out the Gaussian normalization constant,
                # if any) so CHI2 remains a pure misfit statistic (~2 for a good fit).
                L_accepted = L_single[i, i_use] - log_norm_const_per_type[i]

                # Convert log-likelihood to chi-squared: chi2 = -2 * logL
                chi2_samples = -2.0 * L_accepted

                # Compute mean chi-squared
                chi2_mean = np.nanmean(chi2_samples)

                # Normalize by number of data points to get reduced chi-squared
                CHI2_current[i] = chi2_mean / n_data_per_type[i]

        if useRandomData:
            # get the correct index of the subset used
            i_use = idx[i_use]

        t.append(time.time()-t0)        

        # Compute the evidence
        # Numerically stable log-mean-exp calculation
        max_L = np.nanmax(L)
        EV = max_L + np.log(np.nanmean(np.exp(L - max_L)))

        # BUG !!!
        # Compute log-'posterior evidence' - mean posterior log-likelihood
        EV_post = np.nan # np.nanmean(exp_logL)
        #EV_post = maxlogL + np.log(np.nansum(exp_logL[i_use])/len(L[i_use]))
        
        # Compute normalized posterior evidence per data point
        if total_n_data_non_nan > 0:
            EV_post_mean = EV_post / total_n_data_non_nan
        else:
            EV_post_mean = np.nan
        
        t.append(time.time()-t0)

        pltDegug = 0
        if pltDegug>0:
            import matplotlib.pyplot as plt
            plt.semilogy(d_obs, 'k', linewidth=4)
            plt.semilogy(D[0][i_use].T, 'r', linewidth=1)
            plt.show()
            print(D[0][10])

        i_use_all[j] = i_use
        T_all[j] = T
        EV_all[j] = EV
        EV_post_all[j] = EV_post
        EV_post_all_mean[j] = EV_post_mean
        CHI2_all[j, :] = CHI2_current
        # find the number of unique indexes
        N_UNIQUE_all[j] = len(np.unique(i_use))

        if showInfo>2:
            for i in range(len(t)):
                if i<Ndt:
                    print(' Time id%d: %f - %s' % (i,t[i],noise_model[i]))
                else:
                    print(' Time id%d, sampling: %f' % (i,t[i]))
            print('Time total: %f' % np.sum(t))
        
    return i_use_all, T_all, EV_all, EV_post_all, EV_post_all_mean, CHI2_all, N_UNIQUE_all, ip_range



def integrate_posterior_main(ip_chunks, D, DATA, idx, N_use, id_use, autoT, T_base, nr, Ncpu, use_N_best, T_N_above=10, T_P_acc_level=0.2, normalize_likelihood=False, progress_callback=None):
    """
    Coordinate parallel processing of posterior sampling across multiple chunks.
    
    This function manages the parallel execution of rejection sampling by distributing
    data point chunks across multiple CPU cores. It handles shared memory management
    for efficient data transfer between processes and aggregates results from all chunks.
    
    Parameters
    ----------
    ip_chunks : list
        List of data point index chunks for parallel processing.
    D : list
        List of forward modeled data arrays shared across processes.
    DATA : dict
        Dictionary containing observed data structures.
    idx : list
        Indices of prior samples to use for inversion.
    N_use : int
        Maximum number of prior samples per chunk.
    id_use : list
        List of data identifiers for likelihood calculation.
    autoT : int
        Automatic temperature estimation flag.
    T_base : float
        Base temperature for rejection sampling.
    nr : int
        Number of posterior samples to retain per data point.
    Ncpu : int
        Number of CPU cores to use for parallel processing.
    use_N_best : int
        Flag to use only the N best-fitting samples.
    T_N_above : int, optional
        Passed through to ``integrate_rejection_range`` / ``logl_T_est``. Default is 10.
    T_P_acc_level : float, optional
        Passed through to ``integrate_rejection_range`` / ``logl_T_est``. Default is 0.2.

    Returns
    -------
    i_use_all : ndarray, shape (Ndp, nr)
        Indices of accepted posterior samples for all data points.
    T_all : ndarray, shape (Ndp,)
        Temperature values used for all data points.
    EV_all : ndarray, shape (Ndp,)
        Evidence values for all data points.
    EV_post_all : ndarray, shape (Ndp,)
        Posterior evidence values for all data points.
    N_UNIQUE_all : ndarray, shape (Ndp,)
        Number of unique samples for all data points.
    
    Notes
    -----
    This function uses shared memory to minimize data copying overhead during parallel processing.
    Shared memory is automatically cleaned up after processing completion.
    
    The function creates a process pool with the specified number of CPUs and distributes
    the work chunks across them. Each worker process operates on a subset of data points
    and returns its results, which are then aggregated by the main process.
    """
    #import integrate as ig
    import multiprocessing
    import os

    #shared_memory_refs = create_shared_memory(D)
    shared_memory_refs, shm_objects = create_shared_memory(D)
    #reconstructed_arrays = reconstruct_shared_arrays(shared_memory_refs)

    # Use spawn on Windows and macOS (required for correctness);
    # fork on Linux for performance
    if os.name == 'nt' or (os.name == 'posix' and os.uname().sysname == 'Darwin'):
        ctx = multiprocessing.get_context('spawn')
    else:
        ctx = multiprocessing.get_context('fork')

    # Prevent spawned workers from re-executing the user's __main__ script.
    # See prior_data_gaaem for a full explanation of this mechanism.
    import sys
    import types as _types
    _main_module = sys.modules.get('__main__')
    _spec_patched = _main_module is not None and getattr(_main_module, '__spec__', None) is None
    if _spec_patched:
        _main_module.__spec__ = _types.SimpleNamespace(name='__main__')

    try:
        with ctx.Pool(Ncpu) as p:
            # New implementation with shared memory
            chunk_args = [(i, ip_chunks, DATA, idx,  N_use, id_use, shared_memory_refs, autoT, T_base, nr, use_N_best, T_N_above, T_P_acc_level, normalize_likelihood) for i in range(len(ip_chunks))]
            if progress_callback is None:
                results = p.map(integrate_posterior_chunk, chunk_args)
            else:
                # imap_unordered yields per finished chunk, enabling live
                # progress. Result order does not matter: each result carries
                # its own ip_range used for aggregation below.
                results = []
                total_dp = sum(len(c) for c in ip_chunks)
                done_dp = 0
                for res in p.imap_unordered(integrate_posterior_chunk, chunk_args):
                    results.append(res)
                    done_dp += len(res[-1])  # last element is the chunk's ip_range
                    progress_callback(done_dp, total_dp,
                                      {'phase': 'sampling',
                                       'status': 'Rejection sampling (%d/%d data points)' % (done_dp, total_dp)})
            # Old implementation where D was copied to each process
            #results = p.map(integrate_posterior_chunk, [(i, ip_chunks, D, DATA, idx,  N_use, id_use, shared_memory_refs, autoT, T_base, nr, use_N_best) for i in range(len(ip_chunks))])
    finally:
        # Always clean up shared memory
        if shm_objects:
            cleanup_shared_memory(shm_objects)
        if _spec_patched:
            _main_module.__spec__ = None

    # Cleanup shared memory
    #cleanup_shared_memory(shared_memory_refs)
    cleanup_shared_memory(shm_objects)
    

    # Get sample size N from f_prior_h5
    N=D[0].shape[0]
    
    # Get number of data points from, f_data_h5
    Ndp = DATA['d_obs'][0].shape[0]
    
    # Get number of data types
    Ndt = len(id_use)

    i_use_all = np.random.randint(0, N, (Ndp, nr))
    T_all = np.zeros(Ndp)*np.nan
    EV_all = np.zeros(Ndp)*np.nan
    EV_post_all = np.zeros(Ndp)*np.nan
    EV_post_all_mean = np.zeros(Ndp)*np.nan
    CHI2_all = np.zeros((Ndp, Ndt))*np.nan
    N_UNIQUE_all = np.zeros(Ndp)*np.nan

    for i, (i_use, T, EV, EV_post, EV_post_mean, CHI2, N_UNIQUE, ip_range) in enumerate(results):
        for i in range(len(ip_range)):
                ip = ip_range[i]
                #print('ip=%d, i=%d' % (ip,i))
                i_use_all[ip] = i_use[i]
                T_all[ip] = T[i]
                EV_all[ip] = EV[i]
                EV_post_all[ip] = EV_post[i]
                EV_post_all_mean[ip] = EV_post_mean[i]
                CHI2_all[ip, :] = CHI2[i, :]
                N_UNIQUE_all[ip] = N_UNIQUE[i]

    return i_use_all, T_all, EV_all, EV_post_all, EV_post_all_mean, CHI2_all, N_UNIQUE_all



def integrate_posterior_chunk(args):
    """
    Process a single chunk of data points for parallel rejection sampling.
    
    This function is called by each worker process in the parallel processing pool.
    It reconstructs shared data arrays, processes the assigned chunk of data points
    using rejection sampling, and returns results for aggregation by the main process.
    
    Parameters
    ----------
    args : tuple
        Packed arguments: (i_chunk, ip_chunks, DATA, idx, N_use, id_use,
        shared_memory_refs, autoT, T_base, nr, use_N_best, T_N_above, T_P_acc_level,
        normalize_likelihood). See ``integrate_rejection_range`` for descriptions
        of individual fields.

    Returns
    -------
    i_use : ndarray, shape (nump, nr)
        Indices of accepted posterior samples for the chunk.
    T : ndarray, shape (nump,)
        Temperature values used for the chunk.
    EV : ndarray, shape (nump,)
        Evidence values for the chunk.
    EV_post : ndarray, shape (nump,)
        Posterior evidence values for the chunk.
    N_UNIQUE : ndarray, shape (nump,)
        Number of unique samples for the chunk.
    ip_range : ndarray
        Range of data point indices that were processed.
    
    Notes
    -----
    This function runs in a separate process and communicates with the main process
    through shared memory for data arrays and return values through the process pool.
    
    The function first reconstructs the shared data arrays from memory references,
    then calls integrate_rejection_range to perform the actual rejection sampling
    on the assigned chunk of data points.
    """
    #import integrate as ig
    
    # New implementation with shared memory
    i_chunk, ip_chunks, DATA, idx, N_use, id_use, shared_memory_refs, autoT, T_base, nr, use_N_best, T_N_above, T_P_acc_level, normalize_likelihood = args
    # Old implementation where D was copied to each process
    #i_chunk, ip_chunks, D, DATA, idx, N_use, id_use, shared_memory_refs, autoT, T_base, nr, use_N_best = args
    #D=reconstruct_shared_arrays(shared_memory_refs)
    
    # Reconstruct shared arrays without copying - returns tuple (arrays, shm_objects)
    D, worker_shm_objects = reconstruct_shared_arrays(shared_memory_refs)

    try:
        # Perhaps truncat according to N_use
        #for i in len(D)
        #    D[i] = D[i][:N_use]

        ip_range = ip_chunks[i_chunk]

        #print(f'Chunk {i_chunk+1}/{len(ip_chunks)}, ndp={len(ip_range)}')

        i_use, T, EV, EV_post, EV_post_mean, CHI2, N_UNIQUE, ip_range = integrate_rejection_range(
            D,
            DATA,
            idx,
            N_use=N_use,
            id_use=id_use,
            ip_range=ip_range,
            autoT=autoT,
            T_base=T_base,
            nr=nr,
            use_N_best=use_N_best,
            T_N_above=T_N_above,
            T_P_acc_level=T_P_acc_level,
            normalize_likelihood=normalize_likelihood,
        )

        return i_use, T, EV, EV_post, EV_post_mean, CHI2, N_UNIQUE, ip_range
    
    finally:
        # Clean up worker's shared memory references
        for shm in worker_shm_objects:
            try:
                shm.close()
            except Exception as e:
                logger.debug(f"Error closing shared memory in worker: {e}")


def select_subset_for_inversion(dd, N_app):
    """
    Select a subset of indices for inversion based on the sum of squared residuals.

    This function calculates the sum of squared values along the specified axis
    for each row in the input array `dd`. It then selects the indices of the 
    `N_app` smallest sums for fastest performance.

    Parameters
    ----------
    dd : numpy.ndarray
        A 2D array of data from which to select the subset.
    N_app : int
        The number of indices to select based on the smallest sums.

    Returns
    -------
    idx : numpy.ndarray
        An array of indices corresponding to the `N_app` smallest L2 norms.

    Notes
    -----
    This function uses squared residuals (L2 norm) for optimal performance,
    avoiding expensive absolute value operations. Uses `np.argpartition` 
    for efficient selection of the smallest sums.
    """
    norms = np.sum(dd**2, axis=1)
    idx = np.argpartition(norms, N_app)[:N_app]
    return idx


def likelihood_gaussian_diagonal(D, d_obs, d_std, N_app=0, normalize=False,
                                  log_norm_const=None, return_norm_const=False):
    """
    Compute the Gaussian likelihood for a diagonal covariance matrix.

    This function calculates the likelihood of observed data given a set of predicted data
    and standard deviations, assuming a Gaussian distribution with a diagonal covariance matrix.

    Parameters
    ----------
    D : ndarray, shape (n_samples, n_features)
        Predicted data array containing forward model predictions.
    d_obs : ndarray, shape (n_features,)
        Observed data array containing measured values.
    d_std : ndarray, shape (n_features)
        Standard deviation array containing measurement uncertainties.
    N_app : int, optional
        Number of data points to use for approximation. If 0, uses all data.
        Default is 0.
    normalize : bool, optional
        If True, include the Gaussian normalization constant
        -0.5*sum(log(2*pi*d_std**2)) in the returned log-likelihood, giving
        the full (normalized) Gaussian log-pdf instead of just the quadratic
        form. Default is False (preserves prior behavior).
    log_norm_const : float, optional
        Precomputed normalization constant to use instead of recomputing it
        from d_std. Only used when normalize=True. Default is None.
    return_norm_const : bool, optional
        If True, return (L, log_norm_const) instead of just L. Default is False.

    Returns
    -------
    ndarray, shape (n_samples,)
        Log-likelihood values for each sample, computed as:
        L[i] = -0.5 * sum((D[i] - d_obs)**2 / d_std**2)
        plus the normalization constant when normalize=True.

    Notes
    -----
    The function assumes independent Gaussian errors with diagonal covariance matrix.
    The log-likelihood is computed using vectorized operations for efficiency.

    When N_app > 0, only the N_app samples with smallest residuals are evaluated,
    and the remaining samples are assigned a very low likelihood (-1e15).

    This implementation is already well-optimized. Micro-optimizations like pre-computing
    inverse variance do not improve performance with modern NumPy.
    """

    if normalize:
        if log_norm_const is None:
            log_norm_const = -0.5 * np.nansum(np.log(2*np.pi*d_std**2))
    else:
        log_norm_const = 0.0

    # Compute the likelihood (fully vectorized)
    dd = D - d_obs

    if N_app > 0:
       L = np.ones(D.shape[0])*-1e+15
       idx = select_subset_for_inversion(dd, N_app)
       L_small = likelihood_gaussian_diagonal(D[idx], d_obs, d_std, 0, normalize, log_norm_const)
       L[idx]=L_small

    else:
        # Vectorized computation - already optimal
        L = -0.5 * np.nansum((dd / d_std)**2, axis=1) + log_norm_const

    if return_norm_const:
        return L, log_norm_const
    return L


def likelihood_gaussian_diagonal_old(D, d_obs, d_std, N_app=0):
    """
    Compute the Gaussian likelihood for a diagonal covariance matrix (original version).

    This is the original implementation kept for reference and backwards compatibility.
    For better performance, use likelihood_gaussian_diagonal() instead.

    Parameters
    ----------
    D : ndarray, shape (n_samples, n_features)
        Predicted data array containing forward model predictions.
    d_obs : ndarray, shape (n_features,)
        Observed data array containing measured values.
    d_std : ndarray, shape (n_features,)
        Standard deviation array containing measurement uncertainties.
    N_app : int, optional
        Number of data points to use for approximation. If 0, uses all data.
        Default is 0.

    Returns
    -------
    ndarray, shape (n_samples,)
        Log-likelihood values for each sample, computed as:
        L[i] = -0.5 * sum((D[i] - d_obs)**2 / d_std**2)

    Notes
    -----
    This is the original implementation. It has been replaced by an optimized
    version that is ~15-25% faster. This function is kept for reference and validation.
    """

    # Compute the likelihood
    dd = D - d_obs

    if N_app > 0:
       L = np.ones(D.shape[0])*-1e+15
       idx = select_subset_for_inversion(dd, N_app)
       L_small = likelihood_gaussian_diagonal_old(D[idx], d_obs, d_std,0)
       L[idx]=L_small

    else:
        # Explicit broadcasting
        L = -0.5 * np.nansum((dd / d_std)**2, axis=1)

    return L

_log_det_cache = {}


def _cached_slogdet(Cd_sub):
    """
    Return log|Cd_sub|, memoized by content so repeated calls with the same
    (typically re-sliced but byte-identical) covariance matrix skip the
    O(n^3) slogdet recomputation.
    """
    key = (Cd_sub.shape, Cd_sub.dtype.str, Cd_sub.tobytes())
    if key not in _log_det_cache:
        if len(_log_det_cache) > 64:
            _log_det_cache.clear()
        _log_det_cache[key] = np.linalg.slogdet(Cd_sub)[1]
    return _log_det_cache[key]


def likelihood_gaussian_full(D, d_obs, Cd, N_app=0, checkNaN=True, useVectorized=True,
                              normalize=False, log_det_Cd=None, return_norm_const=False):
    """
    Calculate the Gaussian likelihood with full covariance matrix.

    This function computes likelihood values for model predictions given observed data
    and a full covariance matrix, handling NaN values appropriately.

    Parameters
    ----------
    D : ndarray, shape (n_samples, n_features)
        Model predictions containing forward model results.
    d_obs : ndarray, shape (n_features,)
        Observed data containing measured values.
    Cd : ndarray, shape (n_features, n_features)
        Full covariance matrix of observed data uncertainties.
    N_app : int, optional
        Number of data points to use for approximation. If 0, uses all data.
        Default is 0.
    checkNaN : bool, optional
        If True, handles NaN values in d_obs by ignoring them in calculations.
        Default is True.
    useVectorized : bool, optional
        If True, uses vectorized computation for better performance.
        Default is False.
    normalize : bool, optional
        If True, include the Gaussian normalization constant
        -0.5*(n*log(2*pi) + log|Cd|) in the returned log-likelihood, giving
        the full (normalized) Gaussian log-pdf instead of just the quadratic
        form. The log-determinant is memoized (see _cached_slogdet) so it is
        only recomputed when the covariance (or the valid-channel subset)
        actually changes. Default is False (preserves prior behavior).
    log_det_Cd : float, optional
        Precomputed log|Cd| (of the same valid-channel subset used internally)
        to use instead of recomputing/looking it up. Only used when
        normalize=True. Default is None.
    return_norm_const : bool, optional
        If True, return (L, log_norm_const) instead of just L. Default is False.

    Returns
    -------
    ndarray, shape (n_samples,)
        Log-likelihood values for each sample, computed as:
        L[i] = -0.5 * (D[i] - d_obs)^T * Cd^(-1) * (D[i] - d_obs)
        plus the normalization constant when normalize=True.

    Notes
    -----
    The function handles full covariance matrices accounting for correlated errors.
    When checkNaN=True, only non-NaN data points are used in the likelihood calculation.

    The vectorized implementation uses einsum for efficient matrix operations.
    When N_app > 0, only the N_app samples with smallest residuals are evaluated.

    TODO: Check that this works when D has NaN values and determine why they occur.
    """

    if checkNaN:
        # find index of non-nan values in d_obs or non-nan values in np.sum(Cd, axis=0)
        #ind = np.where(~np.isnan(d_obs))[0]
        ind = np.where(~np.isnan(d_obs) & ~np.isnan(np.sum(Cd, axis=0)))[0]
        # Exclude also all data for which one Nan Is available.. This is probably not ideal
        ind = np.where(~np.isnan(d_obs) & ~np.isnan(np.sum(Cd, axis=0)) & ~np.isnan(np.sum(D, axis=0)) )[0]
        dd = D[:,ind] - d_obs[ind]
        Cd_sub = Cd[np.ix_(ind, ind)]
        iCd = np.linalg.inv(Cd_sub)
    else:
        dd = D - d_obs
        Cd_sub = Cd
        iCd = np.linalg.inv(Cd)

    if normalize:
        if log_det_Cd is None:
            log_det_Cd = _cached_slogdet(Cd_sub)
        log_norm_const = -0.5 * (Cd_sub.shape[0] * np.log(2*np.pi) + log_det_Cd)
    else:
        log_norm_const = 0.0

    if N_app > 0:
        L = np.ones(D.shape[0])*-1e+15
        idx = select_subset_for_inversion(dd, N_app)
        if useVectorized:
            #print('Using vectorized likelihood calculation -approximation')
            L_small = -.5 * np.einsum('ij,ij->i', dd[idx] @ iCd, dd[idx]) + log_norm_const
        else:
            L_small = np.zeros(idx.shape[0])
            for i in range(idx.shape[0]):
                L_small[i] = -.5 * np.nansum(dd[idx[i]].T @ iCd @ dd[idx[i]]) + log_norm_const
        L[idx] = L_small

        if return_norm_const:
            return L, log_norm_const
        return L

    if useVectorized:
        # vectorized
        #print('Using vectorized likelihood calculation')
        L = -.5 * np.einsum('ij,ij->i', dd @ iCd, dd) + log_norm_const
    else:
        # non-vectorized
        L = np.zeros(D.shape[0])
        for i in range(D.shape[0]):
            L[i] = -.5 * np.nansum(dd[i].T @ iCd @ dd[i]) + log_norm_const

    if return_norm_const:
        return L, log_norm_const
    return L




def likelihood_multinomial(D, P_obs, class_id=None, class_is_idx=False, entropyFilter=False, entropyThreshold=0.99):
    """
    Calculate log-likelihood of multinomial distribution for discrete data.

    This function computes the log-likelihood of multinomial distribution for discrete data
    using fully vectorized array operations for efficient computation.

    Parameters
    ----------
    D : ndarray, shape (N, n_features)
        Matrix of observed discrete data, where each element represents a class ID.
    P_obs : ndarray, shape (n_classes, n_features)
        Matrix of probabilities, where each column represents probability distribution over classes.
    class_id : ndarray, optional
        Array of unique class IDs corresponding to rows in P_obs.
        If None, extracted from unique values in D.
        Default is None.
    class_is_idx : bool, optional
        If True, class_id is already an index. If False, computes index from class_id array.
        Default is False.
    entropyFilter : bool, optional
        If True, applies entropy filtering to select features.
        Default is False.
    entropyThreshold : float, optional
        Threshold for entropy filtering. Features with entropy below this value are selected.
        Default is 0.99.

    Returns
    -------
    ndarray, shape (N,)
        Log-likelihood values for each sample, computed using natural logarithm.
        For each sample i: logL[i] = sum(log(p[i,j])) over all features j.

    Notes
    -----
    This vectorized implementation eliminates Python loops for significant performance gains.
    The log-likelihood is calculated as the sum of natural logarithms of probabilities:
    logL[i] = sum(log(p[i,j])) for all features j

    This means exp(logL[i]) equals the product of probabilities across features.
    For single-feature cases, exp(logL) directly equals the observed probability.

    When entropyFilter is True, only features with entropy below the threshold
    are used in the likelihood calculation, which can improve computational efficiency
    for datasets with many uninformative features.

    Performance: This vectorized version is approximately 5-10x faster than the loop-based
    implementation for large datasets (N > 10,000).

    Examples
    --------
    >>> D = np.array([[1, 2], [2, 1]])  # Sample data with class IDs
    >>> P_obs = np.array([[0.3, 0.7], [0.7, 0.3]])  # Class probabilities
    >>> logL = likelihood_multinomial(D, P_obs)
    """

    from scipy.stats import entropy

    if class_id is None:
        class_id = np.unique(D).astype(int)

    D = np.atleast_2d(D)

    # Filter out columns with NaN values in P_obs before any processing
    valid_features = ~np.any(np.isnan(P_obs), axis=0)

    if not np.any(valid_features):
        # If all features have NaN, return array of NaN
        return np.full(D.shape[0], np.nan)

    # Apply NaN filtering to both D and P_obs
    D = D[:, valid_features]
    P_obs = P_obs[:, valid_features]

    if entropyFilter:
        H = entropy(P_obs.T)
        used = np.where(H < entropyThreshold)[0]
        if len(used) == 0:
            used = np.arange(1)
        D = D[:, used]
        P_obs = P_obs[:, used]

    N, nm = D.shape

    # Convert D to integer indices
    if class_is_idx:
        # D already contains indices
        indices = D.astype(int)
    else:
        # Create vectorized mapping from class_id to indices
        class_id = class_id.astype(int)

        # Create a lookup array for fast vectorized conversion
        # This assumes class IDs are reasonably bounded
        max_class_id = np.max(class_id)
        min_class_id = np.min(class_id)

        # Use a lookup table approach for efficiency
        lookup = np.full(max_class_id + 1, -1, dtype=int)
        lookup[class_id] = np.arange(len(class_id))

        # Vectorized conversion of all class IDs to indices
        indices = lookup[D.astype(int)]

    # Create column indices for advanced indexing
    col_indices = np.arange(nm)

    # Vectorized probability extraction using advanced indexing
    # indices has shape (N, nm), col_indices has shape (nm,)
    # Broadcasting: indices[:, j] selects row, col_indices[j] selects column
    probs = P_obs[indices, col_indices]

    # Vectorized log-likelihood calculation
    # Sum log probabilities along features axis
    # Use np.errstate to suppress divide-by-zero warnings when probs=0
    # log(0) = -inf is mathematically correct (zero probability events)
    with np.errstate(divide='ignore'):
        logL = np.sum(np.log(probs), axis=1)

    return logL


def likelihood_multinomial_old(D, P_obs, class_id=None, class_is_idx=False, entropyFilter=False, entropyThreshold=0.99):
    """
    Calculate log-likelihood of multinomial distribution for discrete data (old loop-based version).

    This is the original loop-based implementation kept for reference and backwards compatibility.
    For better performance, use likelihood_multinomial() instead.

    Parameters
    ----------
    D : ndarray, shape (N, n_features)
        Matrix of observed discrete data, where each element represents a class ID.
    P_obs : ndarray, shape (n_classes, n_features)
        Matrix of probabilities, where each column represents probability distribution over classes.
    class_id : ndarray, optional
        Array of unique class IDs corresponding to rows in P_obs.
        If None, extracted from unique values in D.
        Default is None.
    class_is_idx : bool, optional
        If True, class_id is already an index. If False, computes index from class_id array.
        Default is False.
    entropyFilter : bool, optional
        If True, applies entropy filtering to select features.
        Default is False.
    entropyThreshold : float, optional
        Threshold for entropy filtering. Features with entropy below this value are selected.
        Default is 0.99.

    Returns
    -------
    ndarray, shape (N,)
        Log-likelihood values for each sample, computed using natural logarithm.
        For each sample i: logL[i] = sum(log(p[i,j])) over all features j.

    Notes
    -----
    This is the original loop-based implementation. It has been replaced by a vectorized
    version that is 5-10x faster. This function is kept for reference and validation.

    Examples
    --------
    >>> D = np.array([[1, 2], [2, 1]])  # Sample data with class IDs
    >>> P_obs = np.array([[0.3, 0.7], [0.7, 0.3]])  # Class probabilities
    >>> logL = likelihood_multinomial_old(D, P_obs)
    """

    from scipy.stats import entropy

    if class_id is None:
        class_id =  np.arange(len(np.unique(D))).astype(int)
        class_id =  np.unique(D).astype(int)

    D=np.atleast_2d(D)

    # Filter out columns with NaN values in P_obs before any processing
    # Check each column (feature) for NaN values
    valid_features = ~np.any(np.isnan(P_obs), axis=0)

    if not np.any(valid_features):
        # If all features have NaN, return array of NaN
        return np.full(D.shape[0], np.nan)

    # Apply NaN filtering to both D and P_obs
    D = D[:, valid_features]
    P_obs = P_obs[:, valid_features]

    if entropyFilter:
        H=entropy(P_obs.T)
        used = np.where(H<entropyThreshold)[0]
        if len(used)==0:
            used = np.arange(1)
        D = D[:,used]
        P_obs = P_obs[:,used]

    N, nm = D.shape
    logL = np.zeros((N))
    class_id = class_id.astype(int)

    # Create mapping from class_id to index
    class_to_idx = {cid: idx for idx, cid in enumerate(class_id)}

    for i in range(N):
        # Convert test data to indices using the mapping
        if class_is_idx:
            i_test = D[i]
        else:
            i_test = np.array([class_to_idx[cls] for cls in D[i]])

        # Get probabilities directly using advanced indexing
        p = P_obs[i_test, np.arange(nm)]
        # Calculate log likelihood (natural log of probabilities)
        logL[i] = np.sum(np.log(p))


    return logL


def create_shared_memory(arrays):
    """
    Create shared memory segments for arrays.
    
    This function creates shared memory segments for a list of numpy arrays,
    allowing them to be accessed efficiently across multiple processes without
    copying data. Returns both memory references and objects for cleanup.
    
    Parameters
    ----------
    arrays : list
        List of numpy arrays to place in shared memory.
    
    Returns
    -------
    shared_memories : list
        List of ``(name, shape, dtype)`` tuples identifying shared memory segments.
    shm_objects : list
        List of SharedMemory objects for cleanup.
    
    Notes
    -----
    The returned shm_objects must be cleaned up using cleanup_shared_memory()
    to prevent memory leaks. This should be done in a finally block.
    
    If an error occurs during creation, any successfully created memory segments
    are automatically cleaned up before raising the exception.
    """
    shared_memories = []
    shm_objects = []
    
    try:
        for array in arrays:
            shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
            shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
            shared_array[:] = array[:]
            shared_memories.append((shm.name, array.shape, array.dtype))
            shm_objects.append(shm)
            logger.debug(f"Created shared memory: {shm.name}")
        return shared_memories, shm_objects
    except Exception as e:
        logger.error(f"Error creating shared memory: {e}")
        # Clean up any created memory segments before raising
        for shm in shm_objects:
            shm.close()
            shm.unlink()
        raise

def reconstruct_shared_arrays(shared_memory_refs):
    """
    Reconstruct arrays from shared memory references.
    
    This function takes shared memory references (created by create_shared_memory)
    and reconstructs the original numpy arrays by accessing the shared memory
    segments. Used by worker processes to access shared data.
    
    Parameters
    ----------
    shared_memory_refs : list
        List of (name, shape, dtype) tuples identifying shared memory segments.
    
    Returns
    -------
    reconstructed_arrays : list
        List of numpy arrays reconstructed from shared memory.
    shm_objects : list
        List of shared memory objects that must be closed after use.
    
    Warnings
    --------
    The reconstructed arrays are views into shared memory. Modifications
    will affect the shared data across all processes. Do NOT modify these arrays.
    The shared memory objects must be closed after use to prevent leaks.
    
    Notes
    -----
    If an error occurs during reconstruction, any successfully opened shared memory
    objects are automatically closed before raising the exception.
    """
    reconstructed_arrays = []
    shm_objects = []
    for shm_name, shape, dtype in shared_memory_refs:
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            reconstructed_arrays.append(array)
            shm_objects.append(shm)
        except Exception as e:
            logger.error(f"Error reconstructing array: {e}")
            # Clean up any successfully opened shared memory objects
            for opened_shm in shm_objects:
                opened_shm.close()
            raise
    return reconstructed_arrays, shm_objects

def cleanup_shared_memory(shm_objects):
    """
    Clean up shared memory segments.
    
    This function properly closes and unlinks shared memory objects created during
    parallel processing to prevent memory leaks and system resource exhaustion.
    It handles cleanup gracefully by catching and ignoring errors for objects
    that may have already been cleaned up.
    
    Parameters
    ----------
    shm_objects : list
        List of shared memory objects to clean up.
    
    Returns
    -------
    None
    
    Notes
    -----
    This function should always be called in a finally block or similar
    error-safe context to ensure cleanup occurs even if exceptions are raised.
    
    Each shared memory object is both closed (to release the local reference)
    and unlinked (to remove it from the system). Errors during cleanup are
    silently ignored to prevent cascading failures.
    
    Examples
    --------
    >>> shared_memories, shm_objects = create_shared_memory(arrays)
    >>> try:
    ...     # Use shared memory
    ...     pass
    ... finally:
    ...     cleanup_shared_memory(shm_objects)
    """
    if not shm_objects:
        return
    
    for shm in shm_objects:
        try:
            shm.close()
            shm.unlink()
            logger.debug(f"Cleaned up shared memory: {shm.name}")

        except Exception as e:
            #logger.error(f"Error cleaning up shared memory: {e}")
            pass


def compute_hypothesis_probability(f_post_h5_list, **kwargs):
    """
    Compute hypothesis probabilities from evidence values in posterior files.

    This function reads evidence (EV) values from multiple posterior HDF5 files,
    each representing a different hypothesis/prior model, and computes the
    probability of each hypothesis at each data point using Bayesian model averaging.

    The probability is computed using Bayes' theorem for model selection with the
    assumption of equal prior probabilities for all hypotheses.

    Parameters
    ----------
    f_post_h5_list : list of str
        List of paths to posterior HDF5 files, one for each hypothesis.
        Each file must contain an '/EV' dataset with log-evidence values (natural log).
    showInfo : int, optional
        Level of verbosity for output. Default is 0.

    Returns
    -------
    P : ndarray, shape (n_data_points, n_hypotheses)
        Probability of each hypothesis at each data point.
        P[i, j] is the probability of hypothesis j at data point i.
        Each row sums to 1.0 (within numerical precision).
    mode : ndarray, shape (n_data_points,)
        Index of the most probable hypothesis for each data point.
        Values are 0-based indices in range [0, n_hypotheses-1].
    entropy_values : ndarray, shape (n_data_points,)
        Entropy (uncertainty measure) for each data point.
        Values are in range [0, log_base(n_hypotheses)], where base=n_hypotheses.
        0 = certain (one hypothesis has probability 1), higher values = more uncertain.

    Notes
    -----
    The probability is computed using Bayes' theorem for model selection:

        P(hypothesis_i | data) = P(data | hypothesis_i) * P(hypothesis_i) / P(data)

    Where P(data | hypothesis_i) is the marginal likelihood (evidence), stored
    as log-evidence (EV, natural log) in the HDF5 files. Assuming equal prior
    probabilities for all hypotheses:

        P(hypothesis_i | data) = exp(EV_i) / sum_j(exp(EV_j))

    For numerical stability, the log-sum-exp trick is used:

        P(hypothesis_i | data) = exp(EV_i - log_sum_exp(all EVs))

    where log_sum_exp is computed using np.logaddexp.reduce() for arbitrary
    number of hypotheses.

    The evidence (EV) values are stored as natural logarithms (ln, not log10)
    as computed by integrate_rejection_range().

    Examples
    --------
    >>> import integrate as ig
    >>> # Create three posterior files from different prior models
    >>> f_post_list = ['post_valley.h5', 'post_standard.h5', 'post_merged.h5']
    >>> P, mode, entropy = ig.compute_hypothesis_probability(f_post_list)
    >>> print(P.shape)  # (n_data_points, 3)
    >>> print(P[0])  # Probabilities for first data point: [0.3, 0.5, 0.2]
    >>> print(np.sum(P[0]))  # Should be 1.0
    >>> print(mode[0])  # Most probable hypothesis index: 1 (0-based)
    >>> print(entropy[0])  # Uncertainty measure: ~0.96

    >>> # For two hypotheses (e.g., valley vs standard lithology)
    >>> f_post_list = ['post_valley.h5', 'post_standard.h5']
    >>> P, mode, entropy = ig.compute_hypothesis_probability(f_post_list, showInfo=1)
    >>> P_valley = P[:, 0]  # Probability of valley hypothesis
    >>> P_standard = P[:, 1]  # Probability of standard hypothesis
    >>> most_probable_hypothesis = mode  # Index of most probable hypothesis per data point
    >>> uncertainty = entropy  # Entropy values indicating uncertainty

    See Also
    --------
    integrate_rejection : Main rejection sampling function that creates posterior files
    integrate_rejection_range : Core rejection sampling that computes EV values
    """
    import h5py
    import numpy as np
    import integrate as ig

    showInfo = kwargs.get('showInfo', 0)

    n_hypotheses = len(f_post_h5_list)

    if n_hypotheses < 2:
        raise ValueError("At least two posterior files are required for hypothesis comparison. "
                        f"Received {n_hypotheses} file(s).")

    # Read EV from all files
    EV_list = []
    n_data_points = None

    for i, f_post_h5 in enumerate(f_post_h5_list):
        try:
            with h5py.File(f_post_h5, 'r') as f:
                if '/EV' not in f:
                    raise KeyError(f"'/EV' dataset not found in {f_post_h5}")

                EV = f['/EV'][:]
                EV_list.append(EV)

                if n_data_points is None:
                    n_data_points = len(EV)
                elif len(EV) != n_data_points:
                    raise ValueError(f"Inconsistent number of data points: file '{f_post_h5}' "
                                   f"has {len(EV)} data points, expected {n_data_points}")

                if showInfo > 0:
                    print(f"Hypothesis {i+1}: Loaded EV from {os.path.basename(f_post_h5)}")
                    if showInfo > 1:
                        print(f"  - Data points: {len(EV)}")
                        print(f"  - EV range: [{np.nanmin(EV):.2f}, {np.nanmax(EV):.2f}]")

        except FileNotFoundError:
            raise FileNotFoundError(f"Posterior file not found: {f_post_h5}")

    # Stack EV values: shape (n_data_points, n_hypotheses)
    EV_all = np.stack(EV_list, axis=1)

    if showInfo > 1:
        print(f"\nCombined EV array shape: {EV_all.shape}")
        print(f"Overall EV range: [{np.nanmin(EV_all):.2f}, {np.nanmax(EV_all):.2f}]")

    # Compute probabilities using log-sum-exp trick for numerical stability
    # P(hypothesis_i | data) = exp(EV_i) / sum_j(exp(EV_j))
    #                        = exp(EV_i - log_sum_exp(all EVs))

    # Compute log_sum_exp across hypotheses for each data point
    # np.logaddexp.reduce handles arbitrary number of hypotheses
    log_sum = np.logaddexp.reduce(EV_all, axis=1, keepdims=True)

    # Compute probabilities
    P = np.exp(EV_all - log_sum)

    # Compute mode (most probable hypothesis index for each data point)
    mode = np.argmax(P, axis=1)

    # Compute entropy for each data point using the entropy function from integrate
    entropy_values = ig.entropy(P, base=n_hypotheses)

    if showInfo > 0:
        print(f"\nComputed hypothesis probabilities:")
        print(f"  - Output shape: {P.shape} (n_data_points × n_hypotheses)")
        print(f"  - Probability range: [{np.nanmin(P):.4f}, {np.nanmax(P):.4f}]")
        row_sums = np.sum(P, axis=1)
        print(f"  - Row sums (should be 1.0): mean={np.nanmean(row_sums):.6f}, "
              f"std={np.nanstd(row_sums):.2e}")

        # Print mode and entropy statistics
        print(f"\nMode and entropy statistics:")
        mode_counts = np.bincount(mode, minlength=n_hypotheses)
        print(f"  - Mode distribution:")
        for i in range(n_hypotheses):
            percentage = (mode_counts[i] / n_data_points) * 100
            print(f"    Hypothesis {i+1}: {mode_counts[i]} data points ({percentage:.1f}%)")
        print(f"  - Entropy range: [{np.nanmin(entropy_values):.4f}, {np.nanmax(entropy_values):.4f}]")
        print(f"  - Mean entropy: {np.nanmean(entropy_values):.4f}, Std entropy: {np.nanstd(entropy_values):.4f}")

        if showInfo > 1:
            # Print summary statistics for each hypothesis
            for i in range(n_hypotheses):
                print(f"  - Hypothesis {i+1}: mean P = {np.nanmean(P[:, i]):.4f}, "
                      f"median P = {np.nanmedian(P[:, i]):.4f}")

    return P, mode, entropy_values
