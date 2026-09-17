"""GA-AEM (gatdaem1d) 1D TDEM forward modelling backend.

Public API: ``forward_gaaem`` / ``prior_data_gaaem`` (mirrored by
``integrate.anemone_forward`` for the anemone backend).  ``gatdaem1d`` is
imported lazily so importing ``integrate`` never requires it.
"""
import multiprocessing
import os
import sys
import time
import types
from functools import partial
from multiprocessing import Pool, get_context

import h5py
import numpy as np


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
    from integrate.integrate import _report_progress
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

def prior_data_gaaem(f_prior_h5, file_gex=None, stmfiles=None, N=0, doMakePriorCopy=True, im=1, id=1, im_height=0, Nhank=280, Nfreq=12, is_log=False, parallel=True, force_replace=False, f_prior_data_h5='', randomize=True, **kwargs):
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
    f_prior_data_h5 : str, optional
        Output path for the prior-data copy (only used when
        ``doMakePriorCopy=True``). If empty (default), a name is generated
        automatically as ``'<prior-stem>_<gex/stm-basename>[_N<N>]_Nh<Nhank>_Nf<Nfreq>.h5'``.
        The path actually used is always the return value.
    randomize : bool, optional
        When ``doMakePriorCopy=True`` and ``N < N_in``: draw ``N`` random
        realizations (True, default) or copy the first ``N`` sequentially
        (False). Only used when a copy is made.
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
    from integrate.integrate import _report_progress
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

        # Use the caller-provided output name if given; otherwise auto-generate
        # one from the prior file and the gex/stm basename.
        if f_prior_data_h5:
            if (showInfo > 0):
                print('Using caller-provided f_prior_data_h5=%s' % f_prior_data_h5)
        else:
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
        ig.copy_hdf5_file(f_prior_h5, f_prior_data_h5,N,randomize=randomize,showInfo=showInfo)
            
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

        if os.name == 'nt':
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

    ig.integrate_update_prior_attributes(f_prior_data_h5)

    _report_progress(progress_callback, N, N,
                     'completed', 'Forward data saved to %s' % f_prior_data_h5)

    return f_prior_data_h5
