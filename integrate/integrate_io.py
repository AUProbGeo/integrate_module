"""
INTEGRATE I/O Module - Data Input/Output and File Management

This module provides comprehensive input/output functionality for the INTEGRATE
geophysical data integration package. It handles reading and writing of HDF5 files,
data format conversions, and management of prior/posterior data structures.

Key Features:
    - HDF5 file I/O for prior models, data, and posterior results
    - Support for multiple geophysical data formats (GEX, STM, USF)
    - Automatic data validation and format checking
    - File conversion utilities between different formats
    - Data merging and aggregation functions
    - Checksum verification and file integrity checks

Main Functions:
    - load_*(): Functions for loading prior models, data, and results
    - save_*(): Functions for saving prior models and data arrays
    - read_*(): File format readers (GEX, USF, etc.)
    - write_*(): File format writers and converters
    - merge_*(): Data and posterior merging utilities

File Format Support:
    - HDF5: Primary data storage format
    - GEX: Geometry and survey configuration files
    - STM: System transfer function files
    - USF: Field measurement files
    - CSV: Export format for GIS integration

Author: Thomas Mejer Hansen
Email: tmeha@geo.au.dk
"""

import os
import numpy as np
import h5py
import re
from typing import Dict, List, Union, Any

# Global default compression settings for HDF5 datasets
# These can be modified to change default behavior module-wide
DEFAULT_COMPRESSION = 'gzip'
DEFAULT_COMPRESSION_OPTS = 1  # gzip level 1: optimal balance (78% faster than level 9, only 2% larger files)

def load_prior(f_prior_h5, N_use=0, idx = [], Randomize=False, ii=None):
    """
    Load prior model parameters and data from HDF5 file.

    Loads both model parameters and forward-modeled data from a prior HDF5 file,
    with options for sample selection, indexing, and randomization. This is a
    convenience function that combines model and data loading operations.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the HDF5 file containing prior model realizations and data.
    N_use : int, optional
        Number of samples to load. If 0, loads all available samples (default is 0).
    idx : list, optional
        Specific indices to load. If empty, uses N_use or loads all samples
        (default is []).
    Randomize : bool, optional
        Whether to randomize the order of loaded samples (default is False).
    ii : array-like, optional
        Array of indices specifying which models and data to load. If provided,
        only len(ii) models and data will be loaded from 'M1', 'M2', ... and 
        'D1', 'D2', ... datasets using these indices (default is None).

    Returns
    -------
    D : dict
        Dictionary containing forward-modeled data arrays, with keys corresponding
        to data types (e.g., 'D1', 'D2').
    M : dict
        Dictionary containing model parameter arrays, with keys corresponding
        to model types (e.g., 'M1', 'M2').
    idx : numpy.ndarray
        Array of indices corresponding to the loaded samples.

    Notes
    -----
    This function internally calls load_prior_data() and load_prior_model()
    with consistent indexing to ensure data and model correspondence.
    Sample selection priority: ii > explicit idx > N_use > all samples.
    """
    # If ii is provided, use it as the index selection
    if ii is not None:
        ii = np.asarray(ii)
        D, idx = load_prior_data(f_prior_h5, idx=ii, Randomize=Randomize)
        M, idx = load_prior_model(f_prior_h5, idx=ii, Randomize=Randomize)
    elif len(idx)==0:
        D, idx = load_prior_data(f_prior_h5, N_use=N_use, Randomize=Randomize)
        M, idx = load_prior_model(f_prior_h5, idx=idx, Randomize=Randomize)
    else:
        D, idx = load_prior_data(f_prior_h5, idx=idx, Randomize=Randomize)
        M, idx = load_prior_model(f_prior_h5, idx=idx, Randomize=Randomize)
    return D, M, idx



def load_prior_model(f_prior_h5, im_use=[], idx=[], N_use=0, Randomize=False):
    """
    Load model parameter arrays from prior HDF5 file.

    Loads model parameter arrays (e.g., resistivity, layer thickness, geological units)
    from a prior HDF5 file with flexible model selection and sample indexing options.
    Supports loading specific model types and sample subsets.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the HDF5 file containing prior model parameter realizations.
    im_use : list of int, optional
        Model parameter indices to load (e.g., [1, 2] for M1 and M2).
        If empty, loads all available model parameters (default is []).
    idx : list or array-like, optional
        Specific sample indices to load. If empty, uses N_use and Randomize
        to determine samples (default is []).
    N_use : int, optional
        Number of samples to load. If 0, loads all available samples.
        Ignored if idx is provided (default is 0).
    Randomize : bool, optional
        Whether to randomly select samples when idx is empty.
        If False, uses sequential selection (default is False).

    Returns
    -------
    M : list of numpy.ndarray
        List of model parameter arrays, one for each requested model type.
        Each array has shape (N_samples, N_model_parameters).
    idx : numpy.ndarray
        Array of sample indices that were loaded, useful for consistent
        indexing across related datasets.

    Notes
    -----
    The function automatically detects available model parameters (M1, M2, ...)
    and loads the requested subset. Sample selection priority follows:
    explicit idx > N_use random/sequential > all samples.
    
    When idx length differs from N_use, the function uses len(idx) and
    issues a warning message.
    """
    import h5py
    import numpy as np


    with h5py.File(f_prior_h5, 'r') as f_prior:
        if len(im_use)==0:
            Nmt = sum(1 for key in f_prior.keys() if key[0]=='M')
            im_use = np.arange(1, Nmt+1)

        N = f_prior['/M1'].shape[0]
        if N_use == 0:
            N_use = N

        if len(idx)==0:
            if Randomize:
                idx = np.sort(np.random.choice(N, min(N_use, N), replace=False)) if N_use < N else np.arange(N)
            else:
                idx = np.arange(N_use)
        else:
            # check if length of idx is equal to N_use
            if len(idx)!=N_use:
                print('Length of idx (%d) must be equal to N_use)=%d' % (len(idx), N_use))
                N_use = len(idx)
                print('using N_use=len(idx)=%d' % N_use)

        M = [f_prior[f'/M{id}'][:][idx] for id in im_use]
    
    
    return M, idx

def save_prior_model(f_prior_h5, M_new,
                     im=None,
                     force_replace=False,
                     delete_if_exist=False,
                     compression='gzip',
                     compression_opts=1,
                     **kwargs):
    """
    Save model parameter arrays to prior HDF5 file.

    Saves model parameter realizations (e.g., resistivity, layer thickness) to an
    HDF5 file with automatic model identifier assignment and data type optimization.
    Supports overwriting existing models and file management options.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the HDF5 file where model data will be saved.
    M_new : numpy.ndarray
        Model parameter array to save. Can be 1D or 2D; 1D arrays are
        automatically converted to column vectors.
    im : int, optional
        Model identifier for the dataset key (creates '/M{im}'). If None,
        automatically assigns the next available ID (default is None).
    force_replace : bool, optional
        Whether to overwrite existing model data with the same identifier.
        If False, raises error when key exists (default is False).
    delete_if_exist : bool, optional
        Whether to delete the entire HDF5 file before saving. Use with
        caution as this removes all existing data (default is False).
    compression : str or None, optional
        Compression filter to use. Options: 'gzip', 'lzf', or None.
        - 'gzip': Good compression ratio, moderate speed (default)
        - 'lzf': Faster but lower compression ratio
        - None: No compression, fastest read/write
        Default is 'gzip'. Set to None for temporary files or fast iteration.
    compression_opts : int, optional
        Compression level for gzip (1-9). Higher = better compression but slower.
        - 1: Fast compression, excellent balance (NEW DEFAULT, changed from 9)
        - 4: Good compression, moderate speed
        - 9: Maximum compression, very slow (OLD DEFAULT)
        Only used when compression='gzip'. Default is 1.
    **kwargs : dict
        Additional arguments:
        - showInfo : int, verbosity level (0=silent, >0=verbose)

    Returns
    -------
    im : int
        The model identifier used for saving the data.

    Notes
    -----
    Model data is stored as HDF5 datasets with keys '/M1', '/M2', etc.
    Data type optimization is performed automatically:
    - Floating-point arrays are converted to float32 for memory efficiency
    - Integer arrays are preserved as appropriate integer types

    Compression settings (based on performance tests with N=50000):
    - compression=None: Fastest (baseline), but 3.6x larger files
    - compression='gzip', compression_opts=1: OPTIMAL - 78% faster than level 9, only 2% larger (NEW DEFAULT)
    - compression='gzip', compression_opts=4: 71% faster than level 9, only 0.5% larger
    - compression='gzip', compression_opts=9: Maximum compression, very slow (diminishing returns)

    **Recommendation**: The new default (gzip level 1) provides the best balance:
    - 3.5x file size reduction vs no compression
    - 78% faster write than the old default (level 9)
    - Only 2% larger files than maximum compression

    For temporary files or rapid iteration, use compression=None.
    For maximum compression (archival), use compression_opts=9.

    The function ensures 2D array format with shape (N_samples, N_parameters)
    where 1D arrays are converted to column vectors.
    """
    import h5py
    import numpy as np

    # Handle compression parameter: False means explicitly disable compression
    if compression is False:
        compression = None

    # LZF compression doesn't accept compression_opts
    if compression == 'lzf':
        compression_opts = None

    import os

    showInfo = kwargs.get('showInfo', 0)
    # if f_prior_h5 exists, delete it
    if delete_if_exist:
        
        # Assuming f_prior_h5 already contains the filename
        if os.path.exists(f_prior_h5):
            os.remove(f_prior_h5)
            if showInfo>1:
                print("File %s has been deleted." % f_prior_h5)
        else:
            if showInfo>1:
                print("File %s does not exist." % f_prior_h5)
            pass

        
    # Make sure the data is 2D using atleast_2d
    if M_new.ndim<2:
        M_new = np.atleast_2d(M_new.flatten()).T

    with h5py.File(f_prior_h5, 'a') as f_prior:
        if im is None:
            Nmt = sum(1 for key in f_prior.keys() if key[0]=='M')
            im = Nmt+1

        key = '/M%d' % im
        if showInfo>1:
            print("Saving new prior model '%s' to file: %s " % (key,f_prior_h5))

        if key in f_prior:
            print("Deleting prior model '%s' from file: %s " % (key,f_prior))
            if force_replace:
                del f_prior[key]
            else:
                print("Key '%s' already exists. Use force_replace=True to overwrite." % key)
                return False

        # Write the new data
        # Convert to 32-bit float for better memory efficiency if the data is floating point
        if np.issubdtype(M_new.dtype, np.floating):
            M_new_32 = M_new.astype(np.float32)
            if compression is None:
                f_prior.create_dataset(key, data=M_new_32)
            elif compression_opts is None:
                f_prior.create_dataset(key, data=M_new_32, compression=compression)
            else:
                f_prior.create_dataset(key, data=M_new_32, compression=compression, compression_opts=compression_opts)
        elif np.issubdtype(M_new.dtype, np.integer):
            M_new_32 = M_new.astype(np.int32)
            if compression is None:
                f_prior.create_dataset(key, data=M_new_32)
            elif compression_opts is None:
                f_prior.create_dataset(key, data=M_new_32, compression=compression)
            else:
                f_prior.create_dataset(key, data=M_new_32, compression=compression, compression_opts=compression_opts)
        else:
            if compression is None:
                f_prior.create_dataset(key, data=M_new)
            elif compression_opts is None:
                f_prior.create_dataset(key, data=M_new, compression=compression)
            else:
                f_prior.create_dataset(key, data=M_new, compression=compression, compression_opts=compression_opts)

        # if 'name' is not set in kwargs, set it to 'XXX'
        if 'name' not in kwargs:
            kwargs['name'] = 'Model %d' % (im)
        if 'is_discrete' not in kwargs:
            kwargs['is_discrete'] = 0
        if 'x' not in kwargs:
            kwargs['x'] = np.arange(M_new.shape[1])

        # if kwargs is set print keys
        if showInfo>2:
            for kwargkey in kwargs:
                print('save_prior_model: key=%s, value=%s' % (kwargkey, kwargs[kwargkey]))


        # if kwarg has keyy 'method' then write it to the file as att
        if 'x' in kwargs:
             f_prior[key].attrs['x'] = kwargs['x']
        if 'name' in kwargs:
             f_prior[key].attrs['name'] = kwargs['name']
        if 'method' in kwargs:
             f_prior[key].attrs['method'] = kwargs['method']
        if 'is_discrete' in kwargs:
            f_prior[key].attrs['is_discrete'] = kwargs['is_discrete']
        if 'class_id' in kwargs:
            f_prior[key].attrs['class_id'] = kwargs['class_id']
        if 'class_name' in kwargs:
            f_prior[key].attrs['class_name'] = kwargs['class_name']
        if 'clim' in kwargs:
            f_prior[key].attrs['clim'] = kwargs['clim']
        if 'cmap' in kwargs:
            f_prior[key].attrs['cmap'] = kwargs['cmap']

        if showInfo>1:
            print("New prior data '%s' saved to file: %s " % (key,f_prior_h5))
    
    return im


def load_prior_data(f_prior_h5, id_use=[], idx=[], N_use=0, Randomize=False, **kwargs):
    """
    Load forward-modeled data arrays from prior HDF5 file.

    Loads electromagnetic or other geophysical data predictions from forward
    modeling runs stored in the prior file. Supports selective loading by
    data type, sample indices, and randomization for sampling purposes.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the HDF5 file containing forward-modeled data arrays.
    id_use : list of int, optional
        Data type identifiers to load (e.g., [1, 2] for D1 and D2).
        If empty, loads all available data types (default is []).
    idx : list or array-like, optional
        Specific sample indices to load. If empty, uses N_use and Randomize
        to determine samples (default is []).
    N_use : int, optional
        Number of samples to load. If 0, loads all available samples.
        Automatically limited to available data size (default is 0).
    Randomize : bool, optional
        Whether to randomly select samples when idx is empty.
        If False, uses sequential selection (default is False).

    Returns
    -------
    D : list of numpy.ndarray
        List of forward-modeled data arrays, one for each requested data type.
        Each array has shape (N_samples, N_data_points).
    idx : numpy.ndarray
        Array of sample indices that were loaded, useful for consistent
        indexing with corresponding model parameters.

    Notes
    -----
    Data arrays are stored as HDF5 datasets with keys '/D1', '/D2', etc.,
    representing different data types (e.g., different measurement systems,
    frequencies, or processing stages). The function automatically detects
    available data types and loads the requested subset.
    
    Sample selection follows the same priority as load_prior_model():
    explicit idx > N_use random/sequential > all samples.
    """

    showInfo = kwargs.get('showInfo', 1)

    if showInfo > 0:
        print('Loading prior data from %s. ' % f_prior_h5, end='')
        print('Using prior data ids: %s' % str(id_use))

    import h5py
    import numpy as np

    if len(id_use)==0:        
        Ndt=0
        with h5py.File(f_prior_h5, 'r') as f_prior:
            for key in f_prior.keys():
                if key[0]=='D':
                    Ndt = Ndt+1
        if len(id_use)==0:
            id_use = np.arange(1,Ndt+1) 

    with h5py.File(f_prior_h5, 'r') as f_prior:
        N = f_prior['/D1'].shape[0]
        if N_use == 0:
            N_use = N    
        if N_use>N:
            N_use = N

        if len(idx)==0:
            if Randomize:
                idx = np.sort(np.random.choice(N, min(N_use, N), replace=False)) if N_use < N else np.arange(N)
            else:
                idx = np.arange(N_use)
        else:
            # check if length of idx is equal to N_use
            if len(idx)!=N_use:
                print('Length of idx (%d) must be equal to N_use)=%d' % (len(idx), N_use))
                N_use = len(idx)      
                print('using N_use=len(idx)=%d' % N_use)


        D = [f_prior[f'/D{id}'][:][idx] for id in id_use]

    if showInfo>0:
        for i in range(len(D)):
            print('  - /D%d: ' % (id_use[i]), end='')
            print(' N,nd = %d/%d' % (D[i].shape[0], D[i].shape[1]))

        
    return D, idx

def save_prior_data(f_prior_h5, D_new, id=None, force_delete=False,
                    compression='gzip', compression_opts=1, **kwargs):
    """
    Save forward-modeled data arrays to prior HDF5 file.

    Saves electromagnetic or other geophysical data predictions from forward
    modeling to an HDF5 file with automatic data identifier assignment and
    data type optimization. Supports overwriting existing data arrays.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the HDF5 file where forward-modeled data will be saved.
    D_new : numpy.ndarray
        Forward-modeled data array to save. Should have shape
        (N_samples, N_data_points) for consistency.
    id : int, optional
        Data identifier for the dataset key (creates '/D{id}'). If None,
        automatically assigns the next available ID (default is None).
    force_delete : bool, optional
        Whether to delete existing data with the same identifier before
        saving. If False, raises error when key exists (default is False).
    compression : str or None, optional
        Compression filter to use. Options: 'gzip', 'lzf', or None.
        Default is 'gzip' for good compression with reasonable speed.
        Set to None to disable compression (fastest I/O, largest files).
    compression_opts : int, optional
        Compression level (0-9 for gzip). Default is 1 (optimal balance).
        Level 1 provides 78% faster writes than level 9 with only 2% larger files.
        Only used when compression='gzip'. Ignored if compression is None.
    **kwargs : dict
        Additional arguments:
        - showInfo : int, verbosity level (0=silent, >0=verbose)

    Returns
    -------
    id : int
        The data identifier used for saving the data.

    Notes
    -----
    Forward-modeled data is stored as HDF5 datasets with keys '/D1', '/D2', etc.,
    representing different data types (e.g., electromagnetic frequencies,
    measurement systems, or processing variants).

    Data type optimization is performed automatically:
    - Floating-point arrays are converted to float32 for memory efficiency
    - Integer arrays are preserved as appropriate integer types

    Compression settings (default: gzip level 1):
    - Provides 3.5x file size reduction vs no compression
    - 78% faster write than gzip level 9 (old default)
    - Only 2% larger files than maximum compression

    The function ensures 2D array format with shape (N_samples, N_data_points).
    """
    showInfo = kwargs.get('showInfo', 1)

    import h5py
    import numpy as np

    # Handle compression parameter: False means explicitly disable compression
    if compression is False:
        compression = None

    # LZF compression doesn't accept compression_opts
    if compression == 'lzf':
        compression_opts = None

    with h5py.File(f_prior_h5, 'a') as f_prior:
        if id is None:
            Ndt = sum(1 for key in f_prior.keys() if key[0]=='D')
            id = Ndt+1

        key = '/D%d' % id
        if showInfo>2:
            print("Saving new prior data '%s' to file: %s " % (key,f_prior_h5))

        if key in f_prior:
            print("Deleting prior data '%s' from file: %s " % (key,f_prior))
            if force_delete:
                del f_prior[key]
            else:
                print("Key '%s' already exists. Use force_delete=True to overwrite." % key)
                return False

        # Ensure 2D array: reshape 1D (N,) to (N, 1)
        if D_new.ndim == 1:
            D_new = D_new.reshape(-1, 1)

        # Write the new data
        # Convert to 32-bit float for better memory efficiency if the data is floating point
        if np.issubdtype(D_new.dtype, np.floating):
            D_new_32 = D_new.astype(np.float32)
            if compression is None:
                f_prior.create_dataset(key, data=D_new_32)
            else:
                f_prior.create_dataset(key, data=D_new_32,
                                     compression=compression,
                                     compression_opts=compression_opts)
        else:
            if compression is None:
                f_prior.create_dataset(key, data=D_new)
            else:
                f_prior.create_dataset(key, data=D_new,
                                     compression=compression,
                                     compression_opts=compression_opts)
        if showInfo>1:
            print("New prior data '%s' saved to file: %s " % (key,f_prior_h5))
        # if kwarg has keyy 'method' then write it to the file as att
        if 'method' in kwargs:
             f_prior[key].attrs['method'] = kwargs['method']
        if 'type' in kwargs:
            f_prior[key].attrs['type'] = kwargs['type']
        if 'im' in kwargs:
            f_prior[key].attrs['im'] = kwargs['im']
        if 'Nhank' in kwargs:
            f_prior[key].attrs['Nhank'] = kwargs['Nhank']
        if 'Nfreq' in kwargs:
            f_prior[key].attrs['Nfreq'] = kwargs['Nfreq']
        if 'f5_forward' in kwargs:
            f_prior[key].attrs['f5_forward'] = kwargs['f5_forward']
        if 'with_noise' in kwargs:
            f_prior[key].attrs['with_noise'] = kwargs['with_noise']

    return id


def load_data(f_data_h5, id_arr=[], ii=None, **kwargs):
    """
    Load observational electromagnetic data from HDF5 file.

    Loads observed electromagnetic measurements, uncertainties, covariance matrices,
    and associated metadata from structured HDF5 files. Handles multiple data types
    and noise models with automatic fallback for missing data components.

    Parameters
    ----------
    f_data_h5 : str
        Path to the HDF5 file containing observational electromagnetic data.
    id_arr : list of int, optional
        Dataset identifiers to load (e.g., [1, 2] for D1 and D2).
        Each ID corresponds to a different measurement system or processing
        stage (default is [1]).
    ii : array-like, optional
        Array of indices specifying which data points to load from each dataset.
        If provided, only len(ii) data points will be loaded from each dataset
        using these indices (default is None).
    **kwargs : dict
        Additional arguments:
        - showInfo : int, verbosity level (0=silent, 1=normal, >1=verbose)

    Returns
    -------
    dict
        Dictionary containing loaded observational data with keys:
        
        - 'noise_model' : list of str
            Noise model type for each dataset ('gaussian', 'multinomial', etc.)
        - 'd_obs' : list of numpy.ndarray
            Observed data measurements, shape (N_stations, N_channels) per dataset
        - 'd_std' : list of numpy.ndarray or None
            Standard deviations of observations, same shape as d_obs
        - 'Cd' : list of numpy.ndarray or None
            Full covariance matrices for each dataset
        - 'id_arr' : list of int
            Dataset identifiers that were successfully loaded. If set as empty, all data types will be loaded
        - 'i_use' : list of numpy.ndarray
            Data point usage indicators (1=use, 0=ignore)
        - 'id_prior' : list of int or numpy.ndarray
            Index of prior data type to compare against, used for cross-referencing.
            If 'id_prior' is not present in the file, it defaults to the dataset id_arr

    Notes
    -----
    The function gracefully handles missing data components:
    - Missing 'id_prior' defaults to sequential dataset IDs (1, 2, 3, ...)
    - Missing 'i_use' defaults to ones array (use all data points)
    - Missing 'd_std' and 'Cd' remain as None (diagonal noise assumed)

    Data structure follows INTEGRATE standard format:
    - '/D{id}/d_obs': observed measurements
    - '/D{id}/d_std': measurement uncertainties
    - '/D{id}/Cd': full covariance matrix (optional)
    - '/D{id}/i_use': data usage flags (optional)
    - '/D{id}/id_prior': prior dataset cross-reference IDs (optional)
    
    Each dataset can have a different noise model specified in the 'noise_model'
    attribute, enabling mixed data types in the same file.
    """

    showInfo = kwargs.get('showInfo', 1)

    import h5py
        
    if not isinstance(id_arr, list):
        id_arr = [id_arr]

    if showInfo > 0:
        print('Loading data from %s. ' % f_data_h5, end='')

    # Convert ii to numpy array if provided
    if ii is not None:
        ii = np.asarray(ii)

    with h5py.File(f_data_h5, 'r') as f_data:
        # If id_arr is empty find all '/D{id}' datasets in the file
        if len(id_arr) == 0:
            id_arr = [int(re.search(r'D(\d+)', key).group(1)) for key in f_data.keys() if re.match(r'D\d+', key)]
            id_arr.sort()

        if showInfo > 0:
            print('Using data types: %s' % str(id_arr))
        noise_model = [f_data[f'/D{id}'].attrs.get('noise_model', 'none') for id in id_arr]
        
        # Load data with selective indexing if ii is provided
        if ii is not None:
            d_obs = [f_data[f'/D{id}/d_obs'][ii] for id in id_arr]
            d_std = [f_data[f'/D{id}/d_std'][ii] if 'd_std' in f_data[f'/D{id}'] else None for id in id_arr]
            i_use = [f_data[f'/D{id}/i_use'][ii] if 'i_use' in f_data[f'/D{id}'] else None for id in id_arr]
        else:
            d_obs = [f_data[f'/D{id}/d_obs'][:] for id in id_arr]
            d_std = [f_data[f'/D{id}/d_std'][:] if 'd_std' in f_data[f'/D{id}'] else None for id in id_arr]
            i_use = [f_data[f'/D{id}/i_use'][:] if 'i_use' in f_data[f'/D{id}'] else None for id in id_arr]
        
        # Full covariance matrices and id_prior are typically not indexed by data points
        Cd = [f_data[f'/D{id}/Cd'][:] if 'Cd' in f_data[f'/D{id}'] else None for id in id_arr]
        id_prior = [f_data[f'/D{id}/id_prior'][()] if 'id_prior' in f_data[f'/D{id}'] and f_data[f'/D{id}/id_prior'].shape == () else f_data[f'/D{id}/id_prior'][:] if 'id_prior' in f_data[f'/D{id}'] else None for id in id_arr]

    for i in range(len(id_arr)):
        if id_prior[i] is None:
            #id_prior[i] = i+1
            id_prior[i] = id_arr[i]
        if i_use[i] is None:
            i_use[i] = np.ones((len(d_obs[i]),1))

        
    DATA = {}
    DATA['noise_model'] = noise_model
    DATA['d_obs'] = d_obs
    DATA['d_std'] = d_std
    DATA['Cd'] = Cd
    DATA['id_arr'] = id_arr
    DATA['i_use'] = i_use
    DATA['id_prior'] = id_prior
    # return noise_model, d_obs, d_std, Cd, id_arr

    if showInfo>0:
        for i in range(len(id_arr)):
            print('  - D%d: id_prior=%d, %11s, Using %d/%d data' % (id_arr[i], id_prior[i], noise_model[i],  DATA['d_obs'][i].shape[0],  DATA['d_obs'][i].shape[1]))

    return DATA


## def ###################################################

#def write_stm_files(GEX, Nhank=140, Nfreq=6, Ndig=7, **kwargs):
def write_stm_files(GEX, **kwargs):
    """
    Generate STM (System Transfer Matrix) files from GEX system configuration.

    Creates system transfer matrix files required for electromagnetic forward modeling
    using GA-AEM. Processes both high-moment (HM) and low-moment (LM) configurations
    with customizable frequency content and Hankel transform parameters.

    Parameters
    ----------
    GEX : dict
        Dictionary containing GEX system configuration data with keys:
        - 'General': System description and waveform information
        - Waveform and timing parameters for electromagnetic modeling
    **kwargs : dict
        Additional configuration parameters:
        - Nhank : int, number of Hankel transform coefficients (default 280)
        - Nfreq : int, number of frequencies for transform (default 12)
        - Ndig : int, number of digital filters (default 7)
        - showInfo : int, verbosity level (0=silent, >0=verbose)
        - WindowWeightingScheme : str, weighting scheme ('AreaUnderCurve', 'BoxCar')
        - NumAbsHM : int, number of abscissae for high moment (default Nhank)
        - NumAbsLM : int, number of abscissae for low moment (default Nhank)
        - NumFreqHM : int, number of frequencies for high moment (default Nfreq)
        - NumFreqLM : int, number of frequencies for low moment (default Nfreq)

    Returns
    -------
    list of str
        List of file paths for the generated STM files (typically HM and LM variants).

    Notes
    -----
    STM files contain system transfer functions that describe the electromagnetic
    system response characteristics needed for accurate forward modeling. The
    function generates separate files for high-moment and low-moment configurations
    when applicable.
    
    The generated STM files follow GA-AEM format specifications and include:
    - Frequency domain transfer functions
    - Hankel transform coefficients
    - Digital filter parameters
    - System timing and waveform information
    
    File naming convention follows: {system_description}_{moment_type}.stm
    """
    system_name = GEX['General']['Description']

    # Parse kwargs
    Nhank = kwargs.get('Nhank', 280)
    Nfreq = kwargs.get('Nfreq', 12)
    Ndig = kwargs.get('Ndig', 7)
    showInfo = kwargs.get('showInfo', 0)
    WindowWeightingScheme  = kwargs.get('WindowWeightingScheme', 'AreaUnderCurve')
    #WindowWeightingScheme  = kwargs.get('WindowWeightingScheme', 'BoxCar')


    NumAbsHM = kwargs.get('NumAbsHM', Nhank)
    NumAbsLM = kwargs.get('NumAbsLM', Nhank)
    NumFreqHM = kwargs.get('NumFreqHM', Nfreq)
    NumFreqLM = kwargs.get('NumFreqLM', Nfreq)
    DigitFreq = kwargs.get('DigitFreq', 4E6)
    stm_dir = kwargs.get('stm_dir', os.getcwd())
    file_gex = kwargs.get('file_gex', '')

    windows = GEX['General']['GateArray']

    # Handle both scalar and array values for NumPy 2.x compatibility
    LastWin_LM = int(np.atleast_1d(GEX['Channel1']['NoGates'])[0])
    LastWin_HM = int(np.atleast_1d(GEX['Channel2']['NoGates'])[0])

    SkipWin_LM = int(np.atleast_1d(GEX['Channel1']['RemoveInitialGates'])[0])
    SkipWin_HM = int(np.atleast_1d(GEX['Channel2']['RemoveInitialGates'])[0])

    # Get MeaTimeDelay with default value of 0.0 if not present (Workbench format compatibility)
    MeaTimeDelay_LM = np.atleast_1d(GEX['Channel1'].get('MeaTimeDelay', np.array([0.0])))[0]
    MeaTimeDelay_HM = np.atleast_1d(GEX['Channel2'].get('MeaTimeDelay', np.array([0.0])))[0]

    windows_LM = windows[SkipWin_LM:LastWin_LM, :] + np.atleast_1d(GEX['Channel1']['GateTimeShift'])[0] + MeaTimeDelay_LM
    windows_HM = windows[SkipWin_HM:LastWin_HM, :] + np.atleast_1d(GEX['Channel2']['GateTimeShift'])[0] + MeaTimeDelay_HM

    #windows_LM = GEX['Channel1']['GateFactor'][0] * windows_LM
    #windows_HM = GEX['Channel2']['GateFactor'][0] * windows_HM
    #windows_LM = windows_LM/GEX['Channel1']['GateFactor'][0] 
    #windows_HM = windows_HM/GEX['Channel2']['GateFactor'][0] 
    
    NWin_LM = windows_LM.shape[0]
    NWin_HM = windows_HM.shape[0]

    # PREPARE WAVEFORMS
    LMWF = GEX['General']['WaveformLM']
    HMWF = GEX['General']['WaveformHM']


    LMWFTime1 = LMWF[0, 0]
    LMWFTime2 = LMWF[-1, 0]

    HMWFTime1 = LMWF[0, 0]
    HMWFTime2 = LMWF[-1, 0]

    LMWF_Period = 1. / np.atleast_1d(GEX['Channel1']['RepFreq'])[0]
    HMWF_Period = 1. / np.atleast_1d(GEX['Channel2']['RepFreq'])[0]

    # Check if full waveform is defined
    LMWF_isfull = (LMWFTime2 - LMWFTime1) == LMWF_Period
    HMWF_isfull = (HMWFTime2 - HMWFTime1) == HMWF_Period

    

    if not LMWF_isfull:
        LMWF = np.vstack((LMWF, [LMWFTime1 + LMWF_Period, 0]))

    if not HMWF_isfull:
        HMWF = np.vstack((HMWF, [HMWFTime1 + HMWF_Period, 0]))

    # Make sure the output folder exists
    if not os.path.isdir(stm_dir):
        os.mkdir(stm_dir)

    if len(file_gex) > 0:
        p, gex_f = os.path.split(file_gex)
        # get filename without extension
        gex_f = os.path.splitext(gex_f)[0]
        gex_str = gex_f + '_'
        # Remove next line when working OK
        gex_str = gex_f + '-P-'
    else:
        gex_str = ''

    LM_name = os.path.join(stm_dir, gex_str + system_name + '_LM.stm')
    HM_name = os.path.join(stm_dir, gex_str + system_name + '_HM.stm')

    stm_files = [LM_name, HM_name]
    if (showInfo>0):
        print('writing LM to %s'%(LM_name))
        print('writing HM to %s'%(HM_name))

    # WRITE LM AND HM FILES
    with open(LM_name, 'w') as fID_LM:
        fID_LM.write('System Begin\n')
        
        fID_LM.write('\tName = %s\n' % (GEX['General']['Description']))        
        fID_LM.write("\tType = Time Domain\n\n")

        fID_LM.write("\tTransmitter Begin\n")
        fID_LM.write("\t\tNumberOfTurns = 1\n")
        fID_LM.write("\t\tPeakCurrent = 1\n")
        fID_LM.write("\t\tLoopArea = 1\n")
        fID_LM.write("\t\tBaseFrequency = %f\n" % np.atleast_1d(GEX['Channel1']['RepFreq'])[0])
        fID_LM.write("\t\tWaveformDigitisingFrequency = %s\n" % ('%21.8f' % DigitFreq))
        fID_LM.write("\t\tWaveFormCurrent Begin\n")
        np.savetxt(fID_LM, GEX['General']['WaveformLM'], fmt='%23.6e', delimiter=' ')
        fID_LM.write("\t\tWaveFormCurrent End\n")
        fID_LM.write("\tTransmitter End\n\n")
        
        fID_LM.write("\tReceiver Begin\n")
        fID_LM.write("\t\tNumberOfWindows = %d\n" % NWin_LM)
        fID_LM.write("\t\tWindowWeightingScheme = %s\n" % WindowWeightingScheme)
        fID_LM.write('\t\tWindowTimes Begin\n')
        #np.savetxt(fID_LM, windows_LM, fmt='%23.6e', delimiter=' ')
        np.savetxt(fID_LM, windows_LM[:,1::], fmt='%23.6e', delimiter=' ')
        fID_LM.write('\t\tWindowTimes End\n\n')
        TiBFilt = GEX['Channel1']['TiBLowPassFilter']

        fID_LM.write('\t\tLowPassFilter Begin\n')
        fID_LM.write('\t\t\tCutOffFrequency = %10.0f\n' % (TiBFilt[1]))
        fID_LM.write('\t\t\tOrder = %d\n' % (TiBFilt[0]))
        fID_LM.write('\t\tLowPassFilter End\n\n')
        
        fID_LM.write('\tReceiver End\n\n')
        
        fID_LM.write('\tForwardModelling Begin\n')
        #fID_LM.write('\t\tModellingLoopRadius = %f\n' % (np.sqrt(GEX['General']['TxLoopArea'][0] / np.pi)))
        fID_LM.write('\t\tModellingLoopRadius = %5.4f\n' % (np.sqrt(np.atleast_1d(GEX['General']['TxLoopArea'])[0] / np.pi)))
        fID_LM.write('\t\tOutputType = dB/dt\n')
        fID_LM.write('\t\tSaveDiagnosticFiles = no\n')
        fID_LM.write('\t\tXOutputScaling = 0\n')
        fID_LM.write('\t\tYOutputScaling = 0\n')
        fID_LM.write('\t\tZOutputScaling = 1\n')
        fID_LM.write('\t\tSecondaryFieldNormalisation = none\n')
        fID_LM.write('\t\tFrequenciesPerDecade = %d\n' % NumFreqLM)
        fID_LM.write('\t\tNumberOfAbsiccaInHankelTransformEvaluation = %d\n' % NumAbsLM)
        fID_LM.write('\tForwardModelling End\n\n')

        fID_LM.write('System End\n')

    with open(HM_name, 'w') as fID_HM:
        fID_HM.write('System Begin\n')
        fID_HM.write('\tName = %s\n' % (GEX['General']['Description']))
        fID_HM.write("\tType = Time Domain\n\n")

        fID_HM.write("\tTransmitter Begin\n")
        fID_HM.write("\t\tNumberOfTurns = 1\n")
        fID_HM.write("\t\tPeakCurrent = 1\n")
        fID_HM.write("\t\tLoopArea = 1\n")
        fID_HM.write("\t\tBaseFrequency = %f\n" % np.atleast_1d(GEX['Channel2']['RepFreq'])[0])
        fID_HM.write("\t\tWaveformDigitisingFrequency = %s\n" % ('%21.8f' % DigitFreq))
        fID_HM.write("\t\tWaveFormCurrent Begin\n")
        np.savetxt(fID_HM, GEX['General']['WaveformHM'], fmt='%23.6e', delimiter=' ')
        fID_HM.write("\t\tWaveFormCurrent End\n")
        fID_HM.write("\tTransmitter End\n\n")

        fID_HM.write("\tReceiver Begin\n")
        fID_HM.write("\t\tNumberOfWindows = %d\n" % NWin_HM)
        fID_HM.write("\t\tWindowWeightingScheme = %s\n" % WindowWeightingScheme)
        fID_HM.write('\t\tWindowTimes Begin\n')
        #np.savetxt(fID_HM, windows_HM, fmt='%23.6e', delimiter=' ')
        np.savetxt(fID_HM, windows_HM[:,1::], fmt='%23.6e', delimiter=' ')
        fID_HM.write('\t\tWindowTimes End\n\n')
        TiBFilt = GEX['Channel2']['TiBLowPassFilter']
        
        fID_HM.write('\t\tLowPassFilter Begin\n')
        fID_HM.write('\t\t\tCutOffFrequency = %10.0f\n' % (TiBFilt[1]))
        fID_HM.write('\t\t\tOrder = %d\n' % (TiBFilt[0]))
        fID_HM.write('\t\tLowPassFilter End\n\n')
        
        fID_HM.write('\tReceiver End\n\n')

        fID_HM.write('\tForwardModelling Begin\n')
        #fID_HM.write('\t\tModellingLoopRadius = %f\n' % (np.sqrt(GEX['General']['TxLoopArea'][0] / np.pi)))
        fID_HM.write('\t\tModellingLoopRadius = %5.4f\n' % (np.sqrt(np.atleast_1d(GEX['General']['TxLoopArea'])[0] / np.pi)))
        fID_HM.write('\t\tOutputType = dB/dt\n')
        fID_HM.write('\t\tSaveDiagnosticFiles = no\n')
        fID_HM.write('\t\tXOutputScaling = 0\n')
        fID_HM.write('\t\tYOutputScaling = 0\n')
        fID_HM.write('\t\tZOutputScaling = 1\n')
        fID_HM.write('\t\tSecondaryFieldNormalisation = none\n')
        fID_HM.write('\t\tFrequenciesPerDecade = %d\n' % NumFreqHM)
        fID_HM.write('\t\tNumberOfAbsiccaInHankelTransformEvaluation = %d\n' % NumAbsHM)
        fID_HM.write('\tForwardModelling End\n\n')

        fID_HM.write('System End\n')

    return stm_files


def read_gex(file_gex, **kwargs):
    """
    Parse GEX (Geometry Exchange) file into structured dictionary.

    Reads and parses electromagnetic system configuration files in GEX format,
    which contain survey geometry, system parameters, waveforms, and timing
    information required for electromagnetic forward modeling.

    Parameters
    ----------
    file_gex : str
        Path to the GEX file containing electromagnetic system configuration.
    Nhank : int, optional
        Number of Hankel transform abscissae for both frequency windows.
    Nfreq : int, optional
        Number of frequencies per decade for both frequency windows.
    Ndig : int, optional
        Number of digits for waveform digitizing frequency.
    showInfo : int, optional
        Verbosity level (0=silent, >0=verbose, default 0).

    Returns
    -------
    dict
        Dictionary containing parsed GEX file contents. Keys include
        ``'filename'`` (str), ``'General'`` (dict), section-specific
        parameter dicts, ``'WaveformLM'`` (ndarray), ``'WaveformHM'``
        (ndarray), and ``'GateArray'`` (ndarray).

    Raises
    ------
    FileNotFoundError
        If the specified GEX file does not exist or cannot be accessed.

    Notes
    -----
    GEX files use a section-based format with key=value pairs:
    - [Section] headers define parameter groups
    - Numeric values are automatically converted to numpy arrays
    - String values are preserved as text
    - Waveform and gate timing data are consolidated into arrays
    
    The parser automatically handles:
    - Multi-point waveform definitions (WaveformLMPoint*, WaveformHMPoint*)
    - Gate timing arrays (GateTime*)
    - Numeric array conversion with space-separated values
    - Comments and formatting variations
    
    Output dictionary structure matches INTEGRATE conventions for
    electromagnetic system configuration and GA-AEM compatibility.
    """
    showInfo = kwargs.get('showInfo', 0)
    
    GEX = {}
    GEX['filename']=file_gex
    comment_counter = 1
    current_key = None

    # Check if file_gex exists
    if not os.path.exists(file_gex):
        raise FileNotFoundError(f"Error: file {file_gex} does not exist")

    with open(file_gex, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if line.startswith('/'):
                GEX[f'comment{comment_counter}'] = line[1:].strip()
                comment_counter += 1
            elif line.startswith('['):
                current_key = line[1:-1]
                GEX[current_key] = {}
            else:
                key_value = line.split('=')
                if len(key_value) == 2:
                    key, value = key_value[0].strip(), key_value[1].strip()
                    
                    try:
                        GEX[current_key][key] = np.fromstring(value, sep=' ')
                        # Check if array is empty before converting
                        if isinstance(GEX[current_key][key], np.ndarray) and len(GEX[current_key][key]) == 0:
                            # value is probably a string
                            GEX[current_key][key] = value
                        # Convert single-element arrays to scalars for NumPy 2.x compatibility
                        elif isinstance(GEX[current_key][key], np.ndarray) and GEX[current_key][key].size == 1:
                            GEX[current_key][key] = GEX[current_key][key].item()
                    #except ValueError:
                    except:
                        GEX[current_key][key] = value


    # WaveformLM
    waveform_keys = [key for key in GEX['General'].keys() if 'WaveformLMPoint' in key]
    waveform_keys.sort(key=lambda x: int(x.replace('WaveformLMPoint', '')))

    waveform_values = [GEX['General'][key] for key in waveform_keys]
    GEX['General']['WaveformLM'] = np.vstack(waveform_values)
    
    for key in waveform_keys:
        del GEX['General'][key]

    # WaveformHM
    waveform_keys = [key for key in GEX['General'].keys() if 'WaveformHMPoint' in key]
    waveform_keys.sort(key=lambda x: int(x.replace('WaveformHMPoint', '')))

    waveform_values = [GEX['General'][key] for key in waveform_keys]
    GEX['General']['WaveformHM']=np.vstack(waveform_values)

    for key in waveform_keys:
        del GEX['General'][key]

    # GateArray
    gate_keys = [key for key in GEX['General'].keys() if 'GateTime' in key]
    gate_keys.sort(key=lambda x: int(x.replace('GateTime', '')))

    gate_values = [GEX['General'][key] for key in gate_keys]
    GEX['General']['GateArray']=np.vstack(gate_values)

    for key in gate_keys:
        del GEX['General'][key]

    return GEX



def read_gex_workbench(file_gex, **kwargs):
    """
    Parse Seequent Workbench GEX file into structured dictionary.

    Reads and parses electromagnetic system configuration files in the newer
    Seequent Workbench GEX format, which supports both dual-moment (LM/HM) and
    single-moment configurations. This function handles:
    - Dual-moment systems with GateTimeLM## and GateTimeHM## entries
    - Single-moment systems with GateTime## entries (e.g., Diamond SkyTEM)
    - WaveformLMPoint## and WaveformHMPoint## entries

    Parameters
    ----------
    file_gex : str
        Path to the GEX file containing electromagnetic system configuration.
    **kwargs : dict
        Additional parsing parameters:
        - showInfo : int, verbosity level (0=silent, >0=verbose, default 0)

    Returns
    -------
    dict
        Dictionary containing parsed GEX file contents with structure:
        - 'filename' : str, original file path
        - 'General' : dict, system description and general parameters
        - 'WaveformLM' : numpy.ndarray, low-moment waveform points
        - 'WaveformHM' : numpy.ndarray, high-moment waveform points (if present)
        - 'GateArrayLM' : numpy.ndarray, low-moment gate timing (if dual-moment)
        - 'GateArrayHM' : numpy.ndarray, high-moment gate timing (if dual-moment)
        - 'GateArray' : numpy.ndarray, gate timing (if single-moment)

    Raises
    ------
    FileNotFoundError
        If the specified GEX file does not exist or cannot be accessed.

    Notes
    -----
    This function supersedes read_gex() for newer Workbench-exported files.

    Format detection:
    - Dual-moment: Keys contain 'GateTimeLM' or 'GateTimeHM' suffixes
    - Single-moment: Keys are 'GateTime##' without moment identifier

    Examples
    --------
    >>> GEX = read_gex_workbench('TX08_20201112.gex')
    >>> print(GEX['General']['GateArrayLM'].shape)  # Dual-moment
    (30, 3)
    >>> print(GEX['General']['GateArrayHM'].shape)
    (30, 3)

    >>> GEX = read_gex_workbench('diamond_system.gex')
    >>> print(GEX['General']['GateArray'].shape)  # Single-moment
    (25, 3)
    """
    showInfo = kwargs.get('showInfo', 0)

    GEX = {}
    GEX['filename'] = file_gex
    comment_counter = 1
    current_key = None

    # Check if file_gex exists
    if not os.path.exists(file_gex):
        raise FileNotFoundError(f"Error: file {file_gex} does not exist")

    # Parse the file
    with open(file_gex, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if line.startswith('/'):
                GEX[f'comment{comment_counter}'] = line[1:].strip()
                comment_counter += 1
            elif line.startswith('['):
                current_key = line[1:-1]
                GEX[current_key] = {}
            else:
                key_value = line.split('=')
                if len(key_value) == 2:
                    key, value = key_value[0].strip(), key_value[1].strip()

                    try:
                        GEX[current_key][key] = np.fromstring(value, sep=' ')
                        # Check if array is empty before converting
                        if isinstance(GEX[current_key][key], np.ndarray) and len(GEX[current_key][key]) == 0:
                            # value is probably a string
                            GEX[current_key][key] = value
                        # Convert single-element arrays to scalars for NumPy 2.x compatibility
                        elif isinstance(GEX[current_key][key], np.ndarray) and GEX[current_key][key].size == 1:
                            GEX[current_key][key] = GEX[current_key][key].item()
                    except:
                        GEX[current_key][key] = value

    # Process WaveformLM
    waveform_keys = [key for key in GEX['General'].keys() if 'WaveformLMPoint' in key]
    if waveform_keys:
        waveform_keys.sort(key=lambda x: int(x.replace('WaveformLMPoint', '')))
        waveform_values = [GEX['General'][key] for key in waveform_keys]
        GEX['General']['WaveformLM'] = np.vstack(waveform_values)
        for key in waveform_keys:
            del GEX['General'][key]
        if showInfo > 0:
            print(f"Processed {len(waveform_values)} WaveformLM points")

    # Process WaveformHM
    waveform_keys = [key for key in GEX['General'].keys() if 'WaveformHMPoint' in key]
    if waveform_keys:
        waveform_keys.sort(key=lambda x: int(x.replace('WaveformHMPoint', '')))
        waveform_values = [GEX['General'][key] for key in waveform_keys]
        GEX['General']['WaveformHM'] = np.vstack(waveform_values)
        for key in waveform_keys:
            del GEX['General'][key]
        if showInfo > 0:
            print(f"Processed {len(waveform_values)} WaveformHM points")

    # Process GateTimes - detect format (dual-moment vs single-moment)
    all_gate_keys = [key for key in GEX['General'].keys() if 'GateTime' in key]

    # Check for dual-moment format (GateTimeLM## or GateTimeHM##)
    gate_keys_lm = [key for key in all_gate_keys if 'GateTimeLM' in key]
    gate_keys_hm = [key for key in all_gate_keys if 'GateTimeHM' in key]

    if gate_keys_lm or gate_keys_hm:
        # Dual-moment format
        if gate_keys_lm:
            gate_keys_lm.sort(key=lambda x: int(x.replace('GateTimeLM', '')))
            gate_values_lm = [GEX['General'][key] for key in gate_keys_lm]
            GEX['General']['GateArrayLM'] = np.vstack(gate_values_lm)
            for key in gate_keys_lm:
                del GEX['General'][key]
            if showInfo > 0:
                print(f"Processed {len(gate_values_lm)} GateTimeLM entries")

        if gate_keys_hm:
            gate_keys_hm.sort(key=lambda x: int(x.replace('GateTimeHM', '')))
            gate_values_hm = [GEX['General'][key] for key in gate_keys_hm]
            GEX['General']['GateArrayHM'] = np.vstack(gate_values_hm)
            for key in gate_keys_hm:
                del GEX['General'][key]
            if showInfo > 0:
                print(f"Processed {len(gate_values_hm)} GateTimeHM entries")

        # Create combined GateArray for compatibility with write_stm_files()
        # Use HM gates as the base (typically contains all gates)
        if gate_keys_hm:
            GEX['General']['GateArray'] = GEX['General']['GateArrayHM'].copy()
            if showInfo > 0:
                print(f"Created unified GateArray from GateArrayHM ({GEX['General']['GateArray'].shape[0]} gates)")
    else:
        # Single-moment format (standard GateTime##)
        gate_keys = [key for key in all_gate_keys if key.startswith('GateTime')]
        if gate_keys:
            gate_keys.sort(key=lambda x: int(x.replace('GateTime', '')))
            gate_values = [GEX['General'][key] for key in gate_keys]
            GEX['General']['GateArray'] = np.vstack(gate_values)
            for key in gate_keys:
                del GEX['General'][key]
            if showInfo > 0:
                print(f"Processed {len(gate_values)} GateTime entries")

    return GEX


# gex_to_stm: convert a GEX file to a set of STM files
def gex_to_stm(file_gex, **kwargs):
    """
    Convert GEX system configuration to STM files for electromagnetic modeling.

    Convenience function that combines GEX file reading and STM file generation
    into a single operation. Handles both file paths and pre-loaded GEX dictionaries
    to create system transfer matrix files required for GA-AEM forward modeling.

    Parameters
    ----------
    file_gex : str or dict
        GEX system configuration. Pass a file path (str) to read and process
        a GEX file, or pass a pre-loaded GEX dictionary from a previous
        ``read_gex()`` call.
    Nhank : int, optional
        Number of Hankel transform coefficients.
    Nfreq : int, optional
        Number of frequencies for transform.
    showInfo : int, optional
        Verbosity level.

    Returns
    -------
    stm_files : list of str
        Paths to the generated STM files.
    GEX : dict
        Processed GEX dictionary used for STM generation.

    Raises
    ------
    TypeError
        If file_gex is neither a string nor a dictionary.
    FileNotFoundError
        If file_gex is a string pointing to a non-existent file.

    Notes
    -----
    This function provides a streamlined workflow for electromagnetic system
    setup by automating the GEX→STM conversion process. The generated STM files
    contain system transfer functions needed for accurate forward modeling
    with GA-AEM.

    When file_gex is a string, the function first tries ``read_gex()`` for
    legacy format compatibility. If that fails (e.g., Workbench format), it
    automatically falls back to ``read_gex_workbench()``.

    When file_gex is a dictionary, it is assumed to be a valid GEX structure
    from a previous ``read_gex()`` or ``read_gex_workbench()`` call.

    The write_stm_files() function handles the actual STM file generation
    with the provided or default parameters.

    Examples
    --------
    >>> # Direct file path (automatically detects format)
    >>> stm_files, GEX = gex_to_stm('TX08_20201112.gex')

    >>> # Pre-loaded GEX dictionary
    >>> GEX = read_gex_workbench('TX08_20201112.gex')
    >>> stm_files, _ = gex_to_stm(GEX)
    """
    if isinstance(file_gex, str):
        # Try legacy read_gex first for backward compatibility
        try:
            GEX = read_gex(file_gex)
        except (ValueError, KeyError) as e:
            # Fall back to read_gex_workbench for newer Workbench format
            showInfo = kwargs.get('showInfo', 0)
            if showInfo > 0:
                print(f"Legacy read_gex() failed ({type(e).__name__}), trying read_gex_workbench()...")
            GEX = read_gex_workbench(file_gex, showInfo=showInfo)

        stm_files = write_stm_files(GEX, file_gex=file_gex, **kwargs)
    else:
        GEX = file_gex
        stm_files = write_stm_files(GEX, file_gex=GEX['filename'], **kwargs)

    return stm_files, GEX


def get_gex_file_from_data(f_data_h5, id=1):
    """
    Retrieves the 'gex' attribute from the specified HDF5 file.

    :param str f_data_h5: The path to the HDF5 file.
    :param int id: The ID of the dataset within the HDF5 file. Defaults to 1.
    :return: The value of the 'gex' attribute if found, otherwise an empty string.
    :rtype: str
    """
    with h5py.File(f_data_h5, 'r') as f:
        dname = '/D%d' % id
        if 'gex' in f[dname].attrs:
            file_gex = f[dname].attrs['gex']
        else:
            print('"gex" attribute not found in %s:%s' % (f_data_h5,dname))
            file_gex = ''
    return file_gex


def get_geometry(f_data_h5):
    """
    Extract survey geometry data from HDF5 file.

    Retrieves spatial coordinates, survey line identifiers, and elevation data
    from an INTEGRATE data file. Automatically handles both direct data files
    and posterior files that reference data files.

    Parameters
    ----------
    f_data_h5 : str
        Path to the HDF5 file containing geometry data. Can be either a data
        file or posterior file (function automatically detects and uses correct file).

    Returns
    -------
    X : numpy.ndarray
        UTM X coordinates in meters, shape (N_points,).
    Y : numpy.ndarray  
        UTM Y coordinates in meters, shape (N_points,).
    LINE : numpy.ndarray
        Survey line identifiers, shape (N_points,).
    ELEVATION : numpy.ndarray
        Ground surface elevation in meters, shape (N_points,).

    Raises
    ------
    IOError
        If the HDF5 file cannot be opened or required datasets are missing.

    Examples
    --------
    >>> X, Y, LINE, ELEVATION = get_geometry('data.h5')
    >>> print(f"Survey covers {X.max()-X.min():.0f}m x {Y.max()-Y.min():.0f}m")

    Notes
    -----
    The function expects geometry data to be stored in standard INTEGRATE format:
    - '/UTMX': UTM X coordinates
    - '/UTMY': UTM Y coordinates  
    - '/LINE': Survey line numbers
    - '/ELEVATION': Ground elevation
    
    When passed a posterior file, automatically extracts the reference to the
    original data file from the 'f5_data' attribute.
    """

    with h5py.File(f_data_h5, 'r') as f_data:
        if 'f5_data' in f_data.attrs:
            f_data_h5 = f_data.attrs['f5_data']
            print('Using f5_data_h5: %s' % f_data_h5)
        else:
            X = f_data['/UTMX'][:].flatten()
            Y = f_data['/UTMY'][:].flatten()
            LINE = f_data['/LINE'][:].flatten()
            ELEVATION = f_data['/ELEVATION'][:].flatten()
            return X, Y, LINE, ELEVATION

    with h5py.File(f_data_h5, 'r') as f_data:
        X = f_data['/UTMX'][:].flatten()
        Y = f_data['/UTMY'][:].flatten()
        LINE = f_data['/LINE'][:].flatten()
        ELEVATION = f_data['/ELEVATION'][:].flatten()

    return X, Y, LINE, ELEVATION


def extract_feature_at_elevation(f_post_h5, elevation, im=1, key='', iz=None, ic=None, iclass=None):
    """
    Extract model parameter feature values at a specific elevation for all data points.

    This function extracts values from a posterior model parameter at a specified
    elevation (e.g., 40m above sea level) across all data points. The function
    performs linear interpolation between model layers to obtain values at the
    exact requested elevation. For each data point, it uses the ELEVATION from
    the data file and the depth discretization from the prior model to compute
    the interpolated value.

    Parameters
    ----------
    f_post_h5 : str
        Path to the HDF5 file containing posterior sampling results.
    elevation : float
        Elevation in meters at which to extract the feature values. This is
        an absolute elevation value (e.g., 40 means 40m above sea level).
    im : int, optional
        Model index to extract from (e.g., 1 for M1, 2 for M2, default is 1).
    key : str, optional
        Dataset key within the model group to extract. If empty string,
        automatically selects appropriate statistic based on parameter type:

        **Continuous parameters**: 'Mean', 'Median', 'Std'
        - Default: 'Median'

        **Discrete parameters**: 'Mode', 'Entropy', 'P' (probability)
        - Default: 'Mode'
        - For 'P': requires ic/iclass parameter to specify which class

    iz : int or None, optional
        Specific layer/feature index to extract. If None, attempts to find the
        appropriate depth layer automatically based on the elevation and model
        discretization (default is None). This parameter is primarily for
        advanced use when you want to extract a specific indexed feature rather
        than interpolating at an elevation.
    ic : int or None, optional
        Class index for probability extraction when key='P'. Specifies which
        class probability to extract. If None and key='P', defaults to 0
        (first class). Alias for iclass parameter (default is None).
    iclass : int or None, optional
        Alternative name for ic parameter. Class index for probability extraction
        when key='P' (default is None).

    Returns
    -------
    numpy.ndarray
        Array of feature values at the specified elevation for all data points.
        Shape is (N_points,) where N_points is the number of data locations.
        Values are interpolated from the model layers surrounding the requested
        elevation. Returns NaN for data points where the requested elevation is
        outside the model domain (above surface or below maximum depth).

    Raises
    ------
    FileNotFoundError
        If the specified HDF5 file does not exist.
    KeyError
        If the requested model index (im) or key is not found in the file.
    ValueError
        If the elevation is invalid or cannot be interpolated from the model.

    Notes
    -----
    **Elevation and Depth Calculation:**

    The function uses the following coordinate system:
    - ELEVATION: Ground surface elevation for each data point (from data file)
    - z: Depth below surface from the prior model (e.g., 0, 1, 2, ... meters)
    - Absolute elevation = ELEVATION - z

    For example, if a data point has ELEVATION=50m and the model has z=[0,10,20,30]:
    - At z=0: absolute elevation = 50m (surface)
    - At z=10: absolute elevation = 40m (10m below surface)
    - At z=20: absolute elevation = 30m (20m below surface)

    To extract a value at elevation=40m, the function:
    1. Computes depth below surface: depth = ELEVATION - elevation = 50 - 40 = 10m
    2. Interpolates the feature value at depth=10m from the model

    **Interpolation:**

    Linear interpolation is used between model layers. If the requested elevation
    falls exactly on a model layer boundary, that layer's value is returned.
    If the elevation is between two layers, values are linearly interpolated.

    **Automatic Key Selection:**

    When key='', the function automatically selects an appropriate statistic:
    - Discrete parameters: defaults to 'Mode' (most probable class)
    - Continuous parameters: defaults to 'Median' (robust central estimate)

    **Valid Keys by Parameter Type:**

    Continuous parameters:
    - 'Mean': Average value
    - 'Median': Median value (default)
    - 'Std': Standard deviation

    Discrete parameters:
    - 'Mode': Most probable class (default)
    - 'Entropy': Uncertainty measure
    - 'P': Probability for a specific class (requires ic/iclass parameter)

    **Probability Extraction:**

    When extracting probabilities (key='P'), the function requires a class index
    specified by ic or iclass. The P array has shape (nd, n_classes, nz) where:
    - nd = number of data points
    - n_classes = number of discrete classes
    - nz = number of depth layers

    The ic/iclass parameter selects which class probability to extract.

    Examples
    --------
    Extract median resistivity at 40m elevation (continuous):

    >>> values = extract_feature_at_elevation('post.h5', elevation=40, im=1, key='Median')
    >>> print(values.shape)  # (N_points,)

    Extract mean and standard deviation (continuous):

    >>> mean_vals = extract_feature_at_elevation('post.h5', elevation=40, im=1, key='Mean')
    >>> std_vals = extract_feature_at_elevation('post.h5', elevation=40, im=1, key='Std')

    Extract mode (most probable class) at 25m elevation (discrete):

    >>> classes = extract_feature_at_elevation('post.h5', elevation=25, im=2, key='Mode')

    Extract entropy (uncertainty) for discrete parameter:

    >>> entropy = extract_feature_at_elevation('post.h5', elevation=25, im=2, key='Entropy')

    Extract probability for first class (discrete):

    >>> prob_class0 = extract_feature_at_elevation('post.h5', elevation=25, im=2, key='P', ic=0)

    Extract probability for second class using iclass parameter:

    >>> prob_class1 = extract_feature_at_elevation('post.h5', elevation=25, im=2, key='P', iclass=1)

    Use automatic key selection (Mode for discrete, Median for continuous):

    >>> values = extract_feature_at_elevation('post.h5', elevation=30, im=1)

    Extract mean values at sea level (elevation=0):

    >>> values = extract_feature_at_elevation('post.h5', elevation=0, im=1, key='Mean')
    """
    import integrate as ig

    # Load file references
    with h5py.File(f_post_h5, 'r') as f_post:
        f_prior_h5 = f_post['/'].attrs['f5_prior']
        f_data_h5 = f_post['/'].attrs['f5_data']

    # Get geometry (ELEVATION for each data point)
    X, Y, LINE, ELEVATION = get_geometry(f_data_h5)
    nd = len(ELEVATION)

    # Get model information
    Mstr = '/M%d' % im

    with h5py.File(f_prior_h5, 'r') as f_prior:
        # Get depth array (z or x)
        if 'z' in f_prior[Mstr].attrs.keys():
            z = f_prior[Mstr].attrs['z'][:].flatten()
        elif 'x' in f_prior[Mstr].attrs.keys():
            z = f_prior[Mstr].attrs['x'][:].flatten()
        else:
            raise KeyError(f"Neither 'z' nor 'x' found in attributes of {Mstr}")

        is_discrete = f_prior[Mstr].attrs['is_discrete']

    # Determine key if not provided
    if len(key) == 0:
        with h5py.File(f_post_h5, 'r') as f_post:
            available_keys = list(f_post[Mstr].keys())
            if is_discrete:
                key = 'Mode' if 'Mode' in available_keys else available_keys[0]
            else:
                key = 'Median' if 'Median' in available_keys else available_keys[0]
        print(f"No key provided. Using default key for {'discrete' if is_discrete else 'continuous'} parameter: {key}")

    # Handle class index for probability extraction
    if key == 'P' or key == 'p':
        key = 'P'  # Normalize to uppercase
        # Determine class index - ic takes priority over iclass
        class_idx = ic if ic is not None else iclass
        if class_idx is None:
            class_idx = 0  # Default to first class
            print(f"No class index provided for key='P'. Using default class index: {class_idx} (first class)")
        else:
            print(f"Extracting probability for class index: {class_idx}")
    else:
        class_idx = None

    # Load the feature data
    with h5py.File(f_post_h5, 'r') as f_post:
        if Mstr not in f_post:
            raise KeyError(f"Model {Mstr} not found in {f_post_h5}")
        if key not in f_post[Mstr].keys():
            raise KeyError(f"Key '{key}' not found in {Mstr} (available: {list(f_post[Mstr].keys())})")

        # Load feature data
        if key == 'P':
            # P has shape (nd, n_classes, nz) - extract specific class
            P_data = f_post[Mstr][key][:]
            if P_data.ndim != 3:
                raise ValueError(f"Expected P data to have 3 dimensions (nd, n_classes, nz), got {P_data.ndim}")

            n_classes = P_data.shape[1]
            if class_idx < 0 or class_idx >= n_classes:
                raise ValueError(f"Class index {class_idx} out of range. Valid range: 0 to {n_classes-1}")

            # Extract probability for the specified class: shape (nd, nz)
            feature_data = P_data[:, class_idx, :]
        else:
            # Load feature data: shape is (nd, nz) where nd=number of data points, nz=number of depth layers
            feature_data = f_post[Mstr][key][:]

    # Initialize output array
    values_at_elevation = np.full(nd, np.nan)

    # For each data point, interpolate at the requested elevation
    for i in range(nd):
        elev_i = ELEVATION[i]

        # Compute absolute elevations for this data point
        # elevation_abs = elev_i - z (surface is at z=0)
        elevation_abs = elev_i - z

        # Check if requested elevation is within the model domain
        if elevation > elev_i:
            # Requested elevation is above ground surface
            continue  # Leave as NaN
        if elevation < elevation_abs[-1]:
            # Requested elevation is below the deepest model layer
            continue  # Leave as NaN

        # Interpolate feature value at the requested elevation
        # Note: elevation_abs is in descending order (surface to depth)
        # so we need to be careful with interpolation
        values_at_elevation[i] = np.interp(elevation, elevation_abs[::-1], feature_data[i, :][::-1])

    return values_at_elevation


def get_discrete_classes(f_h5, im=1):
    """
    Get class IDs and class names for a discrete model parameter.

    Retrieves the classification information (class IDs and class names) for
    a discrete model parameter from either a prior or posterior HDF5 file.
    This function is useful for understanding the categorical classes used in
    discrete parameter inversion (e.g., geological units, lithology types).

    Parameters
    ----------
    f_h5 : str
        Path to the HDF5 file. Can be a prior file (reads classes directly)
        or a posterior file (extracts the prior file reference first, then
        reads classes from the prior).

    im : int, optional
        Model index to get classes for (e.g., 1 for M1, 2 for M2, default is 1).

    Returns
    -------
    class_id : numpy.ndarray or list
        Array of class IDs. Empty list if the model parameter is not discrete
        or if class_id attribute is not set.
    class_name : numpy.ndarray or list
        Array of class names corresponding to the class IDs. Empty list if
        the model parameter is not discrete or if class_name attribute is not set.

    Examples
    --------
    Get classes from a prior file:

    >>> class_id, class_name = get_discrete_classes('PRIOR.h5', im=2)
    >>> for cid, cname in zip(class_id, class_name):
    ...     print(f"Class {cid}: {cname}")

    Get classes from a posterior file (automatically finds prior):

    >>> class_id, class_name = get_discrete_classes('POST.h5', im=2)
    >>> if len(class_id) > 0:
    ...     print(f"Found {len(class_id)} classes")

    Check if parameter is discrete:

    >>> class_id, class_name = get_discrete_classes('POST.h5', im=1)
    >>> if len(class_id) == 0:
    ...     print("Model parameter M1 is continuous")
    ... else:
    ...     print(f"Model parameter M1 is discrete with {len(class_id)} classes")

    Notes
    -----
    The function automatically determines whether the input file is a prior or
    posterior file. For posterior files, it extracts the prior file reference
    from the file attributes and reads the class information from the prior.

    Class information is stored in the prior file attributes:
    - 'class_id': Numeric identifiers for each class (e.g., [0, 1, 2, 3])
    - 'class_name': Text labels for each class (e.g., ['Clay', 'Sand', 'Gravel', 'Bedrock'])
    - 'is_discrete': Boolean flag indicating if the parameter is discrete

    If the model parameter is continuous (is_discrete=False) or if the class
    attributes are not set, the function returns empty lists.
    """

    import os

    Mstr = '/M%d' % im

    # Try to open as posterior first, if it fails, assume it's a prior file
    try:
        with h5py.File(f_h5, 'r') as f:
            # Check if this is a posterior file by looking for typical posterior attributes
            if 'f5_prior' in f['/'].attrs:
                # It's a posterior file, get the prior file reference
                f_prior_h5_rel = f['/'].attrs['f5_prior']

                # Make path absolute relative to the posterior file location
                if not os.path.isabs(f_prior_h5_rel):
                    # Get directory of the posterior file
                    post_dir = os.path.dirname(os.path.abspath(f_h5))
                    # Construct absolute path to prior file
                    f_prior_h5 = os.path.join(post_dir, f_prior_h5_rel)
                else:
                    f_prior_h5 = f_prior_h5_rel
            else:
                # It's a prior file
                f_prior_h5 = f_h5
    except Exception:
        # If something goes wrong, assume it's the prior file
        f_prior_h5 = f_h5

    # Now read class information from the prior file
    class_id = []
    class_name = []

    try:
        with h5py.File(f_prior_h5, 'r') as f_prior:
            if Mstr not in f_prior:
                # Model parameter doesn't exist
                return class_id, class_name

            # Check if the parameter is discrete
            if 'is_discrete' in f_prior[Mstr].attrs:
                is_discrete = f_prior[Mstr].attrs['is_discrete']
                if not is_discrete:
                    # Parameter is continuous, return empty lists
                    return class_id, class_name
            else:
                # Attribute not set, assume continuous
                return class_id, class_name

            # Get class_id if available
            if 'class_id' in f_prior[Mstr].attrs.keys():
                class_id = f_prior[Mstr].attrs['class_id'][:].flatten()
            else:
                class_id = []

            # Get class_name if available
            if 'class_name' in f_prior[Mstr].attrs.keys():
                class_name = f_prior[Mstr].attrs['class_name'][:].flatten()
            else:
                class_name = []

    except Exception as e:
        # If any error occurs, return empty lists
        print(f"Warning: Could not read class information: {e}")
        return [], []

    return class_id, class_name


def get_number_of_datasets(f_data_h5, return_ids=False):
    """
    Get the number of datasets (D1, D2, D3, etc.) in an INTEGRATE data HDF5 file.

    Counts the number of dataset groups with names following the pattern 'D1', 'D2', 'D3', etc.
    in an INTEGRATE HDF5 data file. This function is useful for determining how many different
    data types or measurement systems are stored in a single file.

    Parameters
    ----------
    f_data_h5 : str
        Path to the HDF5 file containing INTEGRATE data with dataset groups.
    return_ids : bool, optional
        If True, returns the list of dataset IDs instead of just the count (default is False).

    Returns
    -------
    int or list
        If return_ids=False: Number of datasets found in the file. Returns 0 if no datasets are found.
        If return_ids=True: List of dataset IDs (e.g., [1, 2, 3] for D1, D2, D3). Returns empty list if none found.
        
    Raises
    ------
    FileNotFoundError
        If the specified HDF5 file does not exist.
    IOError
        If the HDF5 file cannot be opened or read.
        
    Examples
    --------
    >>> # Get number of datasets
    >>> n_datasets = get_number_of_datasets('data.h5')
    >>> print(f"File contains {n_datasets} datasets")
    File contains 3 datasets

    >>> # Get dataset IDs
    >>> dataset_ids = get_number_of_datasets('data.h5', return_ids=True)
    >>> print(f"Dataset IDs: {dataset_ids}")
    Dataset IDs: [1, 2, 3]
    
    Notes
    -----
    This function looks for HDF5 groups with names starting with 'D' followed by digits.
    The typical INTEGRATE data file structure includes:
    - '/D1/': First dataset (e.g., high moment data)
    - '/D2/': Second dataset (e.g., low moment data)  
    - '/D3/': Third dataset (e.g., processed data)
    - And so on...
    
    The function only counts top-level groups that match the 'D{number}' pattern,
    ignoring other groups like geometry data (UTMX, UTMY, etc.).
    """
    dataset_ids = []
    try:
        with h5py.File(f_data_h5, 'r') as f:
            for key in f.keys():
                if key[0] == 'D' and key[1:].isdigit():
                    dataset_ids.append(int(key[1:]))
        dataset_ids.sort()
    except (FileNotFoundError, IOError) as e:
        raise e
    except Exception:
        # Return appropriate empty value for any other errors (e.g., corrupted file)
        return [] if return_ids else 0

    return dataset_ids if return_ids else len(dataset_ids)


def get_number_of_data(f_data_h5, id=None, count_nan=False):
    """
    Get the number of data per location for datasets in an INTEGRATE data HDF5 file.

    Returns a 2D numpy array of size (Ndataset, Ndatapoints) containing the number
    of valid (non-NaN) or total data values at each measurement location for each dataset.

    Parameters
    ----------
    f_data_h5 : str
        Path to the HDF5 file containing INTEGRATE data with dataset groups.
    id : int or list of int, optional
        Dataset identifier(s) to query (e.g., 1 for D1, [1,2] for D1 and D2).
        If None, returns data for all datasets found in the file.
    count_nan : bool, optional
        If False (default), counts only non-NaN values at each location.
        If True, counts total number of data channels regardless of NaN values.

    Returns
    -------
    numpy.ndarray
        2D array of shape (Ndataset, Ndatapoints) where:
        - Ndataset: number of datasets
        - Ndatapoints: maximum number of data locations across all datasets
        - Values: number of valid data channels per location (or total if count_nan=True)

    Raises
    ------
    FileNotFoundError
        If the specified HDF5 file does not exist.
    IOError
        If the HDF5 file cannot be opened or read.
    KeyError
        If the specified dataset ID does not exist in the file.

    Examples
    --------
    >>> # Get non-NaN data counts for all datasets
    >>> data_counts = get_number_of_data('data.h5')
    >>> print(f"Shape: {data_counts.shape}")  # (3, 4000) for 3 datasets, 4000 locations
    Shape: (3, 4000)

    >>> # Get total data counts (including NaN) for specific dataset
    >>> counts_d1 = get_number_of_data('data.h5', id=1, count_nan=True)
    >>> print(f"Shape: {counts_d1.shape}")  # (1, 4000) for 1 dataset, 4000 locations
    Shape: (1, 4000)

    Notes
    -----
    This function analyzes d_obs arrays in each dataset:
    - d_obs shape: (N_locations, N_data_per_location)
    - By default, counts non-NaN values: np.sum(~np.isnan(d_obs), axis=1)
    - With count_nan=True, returns total data channels: d_obs.shape[1] for each location

    The returned 2D array allows easy comparison across datasets and locations.
    Missing datasets are filled with zeros in the output array.
    """
    import h5py
    import numpy as np

    try:
        # Get dataset IDs using get_number_of_datasets (avoids code duplication)
        if id is None:
            dataset_ids = get_number_of_datasets(f_data_h5, return_ids=True)
        else:
            # Handle single id or list of ids
            if not isinstance(id, list):
                dataset_ids = [id]
            else:
                dataset_ids = id

        with h5py.File(f_data_h5, 'r') as f:

            if not dataset_ids:
                return np.array([]).reshape(0, 0)

            # First pass: determine maximum number of data points across all datasets
            max_datapoints = 0
            valid_datasets = []

            for dataset_id in dataset_ids:
                d_obs_path = f'D{dataset_id}/d_obs'
                if d_obs_path in f:
                    data_shape = f[d_obs_path].shape
                    if len(data_shape) >= 2:
                        max_datapoints = max(max_datapoints, data_shape[0])
                        valid_datasets.append(dataset_id)
                    elif len(data_shape) == 1:
                        max_datapoints = max(max_datapoints, 1)
                        valid_datasets.append(dataset_id)

            if not valid_datasets:
                return np.array([]).reshape(0, 0)

            # Initialize result array
            result = np.zeros((len(valid_datasets), max_datapoints), dtype=int)

            # Second pass: fill the result array
            for i, dataset_id in enumerate(valid_datasets):
                d_obs_path = f'D{dataset_id}/d_obs'
                if d_obs_path in f:
                    d_obs = f[d_obs_path][:]

                    if len(d_obs.shape) >= 2:
                        n_locations = d_obs.shape[0]

                        if count_nan:
                            # Count total data channels per location
                            n_data_per_location = d_obs.shape[1]
                            result[i, :n_locations] = n_data_per_location
                        else:
                            # Count non-NaN values per location
                            for loc in range(n_locations):
                                result[i, loc] = np.sum(~np.isnan(d_obs[loc, :]))
                    else:
                        # Handle 1D case
                        if count_nan:
                            result[i, 0] = d_obs.shape[0]
                        else:
                            result[i, 0] = np.sum(~np.isnan(d_obs))

            return result

    except (FileNotFoundError, IOError) as e:
        raise e
    except Exception as e:
        raise IOError(f"Error reading HDF5 file: {str(e)}")


def post_to_csv(f_post_h5='', Mstr='/M1'):
    """
    Export posterior results to CSV format for GIS integration.

    Converts posterior sampling results to CSV files containing spatial coordinates
    and model parameter statistics. Creates files suitable for import into GIS
    software or other analysis tools.

    Parameters
    ----------
    f_post_h5 : str, optional
        Path to the HDF5 file containing posterior results. If empty string,
        uses a default example file (default is '').
    Mstr : str, optional
        Model parameter dataset path within the HDF5 file (e.g., '/M1', '/M2').
        Specifies which model parameter to export (default is '/M1').

    Returns
    -------
    str
        Path to the generated CSV file.

    Raises
    ------
    KeyError
        If the specified model parameter dataset does not exist in the HDF5 file.
    FileNotFoundError
        If the specified HDF5 file does not exist or cannot be accessed.

    Notes
    -----
    The exported CSV file contains:
    - X, Y: UTM coordinates
    - ELEVATION: Ground surface elevation
    - Model statistics: Mean, Median, Mode, Standard deviation
    - For discrete models: probability distributions across classes
    - For continuous models: quantile values and uncertainty measures
    
    The function automatically handles both discrete and continuous model types
    based on the 'is_discrete' attribute in the prior file. Output format is
    optimized for GIS applications with appropriate coordinate reference systems.
    
    TODO: Future enhancements planned for LINE number export and separate
    functions for grid vs. point data export.
    """
    
    # TODO: Would be nice if also the LINE number was exported (to allow filter by LINE)
    # Perhaps this function should be split into two functions, 
    #   one for exporting the grid data and one for exporting the point data.
    # Also, split into a function the generates the points scatter data, and one that stores them as a csv file


    import pandas as pd
    import integrate as ig

    #Mstr = '/M1'
    # if f_post_h5 is Null then use the last f_post_h5 file

    if len(f_post_h5)==0:
        f_post_h5 = 'POST_PRIOR_Daugaard_N2000000_TX07_20230731_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5'

    with h5py.File(f_post_h5, 'r') as f_post:
        f_prior_h5 = f_post.attrs['f5_prior']
        f_data_h5 = f_post.attrs['f5_data']
        with h5py.File(f_prior_h5, 'r') as f_prior:
            if 'x' in f_prior[Mstr].attrs.keys():
                z = f_prior[Mstr].attrs['x']
            else:
                z = f_prior[Mstr].attrs['z']
            is_discrete = f_prior[Mstr].attrs['is_discrete']

            X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)

            # Check that Dstr exist in f_poyt_h5
            if Mstr not in f_post:
                print("ERROR: %s not in %s" % (Mstr, f_post_h5))
                sys.exit(1)

            D_mul = []
            D_name = []
            if is_discrete:
                D_mul.append(f_post[Mstr+'/Mode'])
                D_name.append('Mode')
                D_mul.append(f_post[Mstr+'/Entropy'])
                D_name.append('Entropy')
            else:
                D_mul.append(f_post[Mstr+'/Median'])
                D_name.append('Median')
                D_mul.append(f_post[Mstr+'/Mean'])
                D_name.append('Mean')
                D_mul.append(f_post[Mstr+'/Std'])
                D_name.append('Std')


            # replicate z[1::] to be a 2D matric of zie ndx89
            ZZ = np.tile(z[1::], (D_mul[0].shape[0], 1))

            #
            df = pd.DataFrame(data={'X': X, 'Y': Y, 'Line': LINE, 'ELEVATION': ELEVATION})

            dataframes = [df]

            for i in range(len(D_mul)):
                D = D_mul[i][:]

                for j in range(D.shape[1]):
                    temp_df = pd.DataFrame(D[:,j], columns=[D_name[i]+'_'+str(j)])
                    dataframes.append(temp_df)

            for j in range(ZZ.shape[1]):
                temp_df = pd.DataFrame(ZZ[:,j], columns=['zbot_'+str(j)])
                dataframes.append(temp_df)

            df = pd.concat(dataframes, axis=1)
            f_post_csv='%s_%s.csv' % (os.path.splitext(f_post_h5)[0],Mstr[1::])
            #f_post_csv='%s.csv' % (os.path.splitext(f_post_h5)[0])
            #f_post_csv = f_post_h5.replace('.h5', '.csv')
            print('Writing to %s' % f_post_csv)
            df.to_csv(f_post_csv, index=False)


            #%% Store point data sets of varianle in D_name
            # # Save a file with columns, x, y, z, and the median.
            print("----------------------------------------------------")
            D_mul_out = []
            for icat in range(len(D_name)):
                #icat=0
                Vstr = D_name[icat]
                print('Creating point data set: %s'  % Vstr)
                D=f_post[Mstr+'/'+Vstr]
                nd,nz=D.shape
                n = nd*nz

                Xp = np.zeros(n)
                Yp = np.zeros(n)
                Zp = np.zeros(n)
                LINEp = np.zeros(n)
                Dp = np.zeros(n)

                for i in range(nd):
                    for j in range(nz):
                        k = i*nz+j
                        Xp[k] = X[i]
                        Yp[k] = Y[i]
                        Zp[k] = ELEVATION[i]-z[j]
                        LINEp[k] = LINE[i]
                        Dp[k] = D[i,j]
                D_mul_out.append(Dp)

            if is_discrete:
                df = pd.DataFrame(data={'X': Xp, 'Y': Yp, 'Z': Zp, 'LINE': LINEp, D_name[0]: D_mul_out[0], D_name[1]: D_mul_out[1] })
            else:
                df = pd.DataFrame(data={'X': Xp, 'Y': Yp, 'Z': Zp, 'LINE': LINEp, D_name[0]: D_mul_out[0], D_name[1]: D_mul_out[1], D_name[2]: D_mul_out[2] })

            f_csv = '%s_%s_point.csv' % (os.path.splitext(f_post_h5)[0],Mstr[1::])
            print('- saving to : %s'  % f_csv)

            df.to_csv(f_csv, index=False)

    return f_post_csv, f_csv


'''
HDF% related functions
'''
def copy_hdf5_file(input_filename, output_filename, N=None, loadToMemory=True, compress=True, **kwargs):
    """
    Copy the contents of an HDF5 file to another HDF5 file.

    :param input_filename: The path to the input HDF5 file.
    :type input_filename: str
    :param output_filename: The path to the output HDF5 file.
    :type output_filename: str
    :param N: The number of elements to copy from each dataset. If not specified, all elements will be copied.
    :type N: int, optional
    :param loadToMemory: Whether to load the entire dataset to memory before slicing. Default is True.
    :type loadToMemory: bool, optional
    :param compress: Whether to compress the output dataset. Default is True.
    :type compress: bool, optional

    :return: output_filename
    """
    import time
    
    showInfo = kwargs.get('showInfo', 0)
    delay_after_close = kwargs.get('delay_after_close', 0.1)
    
    # Check and close any open HDF5 file handles for these files
    try:
        # Get all open file objects
        open_files = list(h5py.h5f.get_obj_ids())
        for fid in open_files:
            try:
                f_temp = h5py.h5i.get_name(fid)
                if f_temp:
                    f_temp = f_temp.decode('utf-8') if isinstance(f_temp, bytes) else f_temp
                    if f_temp == input_filename or f_temp == output_filename:
                        if showInfo > 0:
                            print(f'Closing open HDF5 file handle for: {f_temp}')
                        h5py.h5f.close(fid)
            except (ValueError, OSError, RuntimeError):
                # Handle cases where the file ID might be invalid or already closed
                continue
    except (ValueError, OSError):
        # If we can't get file IDs, continue without closing
        if showInfo > 0:
            print('Could not check for open HDF5 file handles')

    # Open the input file
    if showInfo > 0:
        print('Trying to copy %s to %s' % (input_filename, output_filename))

    # Remove any stale/corrupted output file before writing (important on
    # Windows/NTFS and WSL mounts where h5py 'w' mode may fail to truncate)
    import os as _os
    if _os.path.exists(output_filename):
        try:
            _os.remove(output_filename)
        except OSError:
            pass

    try:
        with h5py.File(input_filename, 'r') as input_file, h5py.File(output_filename, 'w') as output_file:
            # Copy each group/dataset from the input file to the output file
            i_use = []
            for name in input_file:
                if showInfo > 2:
                    print('Copying %s. ' % name, end='')
                if isinstance(input_file[name], h5py.Dataset):
                    # If N is specified, only copy the first N elements

                    if len(i_use) == 0:
                        N_in = input_file[name].shape[0]
                        if N is None:
                            N = N_in
                        if N > N_in:
                            N = N_in
                        if N == N_in:
                            i_use = np.arange(N)
                        else:
                            i_use = np.sort(np.random.choice(N_in, N, replace=False))

                    if N < 20000:
                        loadToMemory = False

                    # Read full dataset into memory
                    if loadToMemory:
                        # Load all data to memory, before slicing
                        if showInfo > 1:
                            print('%s ' % name, end='')
                        data_in = input_file[name][:]
                        data = data_in[i_use]
                    else:
                        # Read directly from HDF5 file
                        data = input_file[name][i_use]

                    # Create new dataset in output file with compression
                    # Convert floating point data to 32-bit precision
                    if data.dtype.kind == 'f':
                        data = data.astype(np.float32)

                    if compress:
                        output_dataset = output_file.create_dataset(name, data=data, compression="gzip", compression_opts=4)
                    else:
                        output_dataset = output_file.create_dataset(name, data=data)
                    # Copy the attributes of the dataset
                    for key, value in input_file[name].attrs.items():
                        output_dataset.attrs[key] = value
                else:
                    input_file.copy(name, output_file)

            # Copy the attributes of the input file to the output file
            for key, value in input_file.attrs.items():
                output_file.attrs[key] = value

    except Exception as e:
        # Remove partially created output file
        try:
            import os
            if os.path.exists(output_filename):
                os.remove(output_filename)
        except:
            pass
        raise e

    # Add small delay to ensure file handles are fully released
    if delay_after_close > 0:
        time.sleep(delay_after_close)

    if showInfo > 1:
        print('')

    return output_filename

def copy_prior(input_filename, output_filename, idx=None, N_use=None, loadtomem=False, **kwargs):
    """
    Copy a PRIOR file, optionally subsetting the data.

    This function copies an HDF5 PRIOR file, which may contain model parameters
    (M1, M2, ...) and forward-modeled data (D1, D2, ...). It allows for
    copying only a specific subset of samples using either an index array (`idx`)
    or a specified number of random samples (`N_use`).

    Parameters
    ----------
    input_filename : str
        Path to the input PRIOR HDF5 file.
    output_filename : str
        Path to the output PRIOR HDF5 file.
    idx : array-like, optional
        An array of indices to copy. If provided, only the data corresponding
        to these indices will be included in the new file. This takes
        precedence over `N_use`. Default is None (copy all data).
    N_use : int, optional
        The number of random samples to select and copy. This is ignored if
        `idx` is provided. Default is None.
    loadtomem : bool, optional
        If True, datasets are loaded entirely into memory before slicing.
        This can significantly speed up copying large subsets of data but
        increases memory consumption. Default is False.
    **kwargs : dict
        Additional keyword arguments (e.g., `showInfo`, `compress`).

    Returns
    -------
    str
        The path to the output HDF5 file (`output_filename`).

    Raises
    ------
    ValueError
        If `N_use` is greater than the total number of samples in the file,
        or if no datasets are found to determine the size for random sampling.
    """
    import time
    import numpy as np

    showInfo = kwargs.get('showInfo', 0)
    delay_after_close = kwargs.get('delay_after_close', 0.1)
    compress = kwargs.get('compress', True)

    # Open the input file to determine dataset size if needed, then copy
    try:
        with h5py.File(input_filename, 'r') as input_file:
            # Handle N_use parameter: generate random indices if N_use is set and idx is not
            if idx is None and N_use is not None:
                # Find the first dataset to determine the total number of samples
                first_dataset_name = None
                for name in input_file:
                    if isinstance(input_file[name], h5py.Dataset) and input_file[name].ndim > 0:
                        first_dataset_name = name
                        break

                if first_dataset_name is None:
                    raise ValueError("Could not find any dataset in the prior file to determine size")

                N_total = input_file[first_dataset_name].shape[0]

                if N_use > N_total:
                    raise ValueError(f"N_use ({N_use}) exceeds total number of samples ({N_total})")

                # Generate random indices
                idx = np.random.choice(N_total, size=N_use, replace=False)
                idx = np.sort(idx)  # Sort for better HDF5 read performance

                if showInfo > 0:
                    print(f'Randomly selected {N_use} samples from {N_total} total samples')

            if showInfo > 0:
                print('Copying PRIOR file %s to %s' % (input_filename, output_filename))
                if idx is not None:
                    print('Using subset with %d indices' % len(idx))

            # Convert idx to numpy array if provided
            if idx is not None:
                idx = np.asarray(idx)

            with h5py.File(output_filename, 'w') as output_file:
                # Copy each group/dataset from the input file to the output file
                for name in input_file:
                    if showInfo > 0:
                        print('Copying %s' % name)

                    if isinstance(input_file[name], h5py.Dataset):
                        # Determine if this is a dataset that should be subset
                        dataset = input_file[name]

                        if idx is not None and dataset.ndim > 0:
                            if len(idx) > dataset.shape[0]:
                                raise ValueError(f"Index array length ({len(idx)}) exceeds dataset size ({dataset.shape[0]}) for {name}")

                            # Get the subset of data
                            if loadtomem:
                                if showInfo > 1:
                                    print(f"Loading '{name}' to memory before slicing.", end='')
                                data = dataset[:][idx]
                            else:
                                data = dataset[idx]
                        else:
                            # Copy all data
                            data = dataset[:]

                        # Convert floating point data to 32-bit precision
                        if data.dtype.kind == 'f':
                            data = data.astype(np.float32)

                        # Create new dataset in output file with compression
                        if compress:
                            output_dataset = output_file.create_dataset(name, data=data, compression="gzip", compression_opts=4)
                        else:
                            output_dataset = output_file.create_dataset(name, data=data)

                        # Copy all attributes of the dataset
                        for key, value in dataset.attrs.items():
                            output_dataset.attrs[key] = value

                    else:
                        # Copy groups and other non-dataset objects directly
                        input_file.copy(name, output_file)

                # Copy all attributes of the input file to the output file
                for key, value in input_file.attrs.items():
                    output_file.attrs[key] = value

    except Exception as e:
        # Remove partially created output file
        try:
            import os
            if os.path.exists(output_filename):
                os.remove(output_filename)
        except:
            pass
        raise e

    # Add small delay to ensure file handles are fully released
    if delay_after_close > 0:
        time.sleep(delay_after_close)

    return output_filename


def filter_prior(f_prior_h5, type='nonnegative_data', id=1,
                 f_prior_filtered_h5='', **kwargs):
    """
    Filter prior realizations and write the result to a new HDF5 file.

    Removes rows (realizations) from all M and D datasets in a prior file
    based on a criterion evaluated on a chosen D dataset. The filtered file
    is a complete, self-consistent prior that can be used directly in place
    of the original.

    Parameters
    ----------
    f_prior_h5 : str
        Path to the input prior HDF5 file.
    type : str, optional
        Filter criterion to apply. Supported values:

        ``'nonnegative_data'``
            Keep only realizations where every value in ``/D{id}`` is >= 0.
            Useful after forward modelling to remove unphysical responses.

        Default is ``'nonnegative_data'``.
    id : int, optional
        Index of the D dataset used for filtering (e.g. ``id=1`` uses ``/D1``).
        Default is 1.
    f_prior_filtered_h5 : str, optional
        Output filename. If empty, auto-generates as
        ``<stem>_filtered_<type>.h5``. Default is ``''``.
    **kwargs
        showInfo : int, optional
            Verbosity level (default 0). Passed through to ``copy_prior``.

    Returns
    -------
    str
        Path to the filtered output HDF5 file.

    Raises
    ------
    KeyError
        If ``/D{id}`` is not found in the input file.
    ValueError
        If an unknown ``type`` is specified.

    Examples
    --------
    >>> f_prior_filtered = ig.filter_prior(f_prior_h5, type='nonnegative_data', id=1)

    Notes
    -----
    Filtering is delegated to ``copy_prior``, which preserves all dataset
    attributes and applies compression. New filter types can be added by
    extending the ``if/elif`` block that computes ``idx``.
    """
    import numpy as np
    import os

    showInfo = kwargs.get('showInfo', 0)

    if not f_prior_filtered_h5:
        stem = os.path.splitext(f_prior_h5)[0]
        f_prior_filtered_h5 = '%s_filtered_%s.h5' % (stem, type)

    Dname = '/D%d' % id

    with h5py.File(f_prior_h5, 'r') as f:
        if Dname not in f:
            raise KeyError("Dataset '%s' not found in %s" % (Dname, f_prior_h5))
        D = f[Dname][:]

    if type == 'nonnegative_data':
        idx = np.where(np.all(D >= 0, axis=1))[0]
    else:
        raise ValueError("Unknown filter type: '%s'" % type)

    N_in = D.shape[0]
    N_out = len(idx)
    if showInfo >= 0:
        print("filter_prior [%s on %s]: keeping %d / %d realizations (%.1f%%)"
              % (type, Dname, N_out, N_in, 100.0 * N_out / N_in))

    copy_prior(f_prior_h5, f_prior_filtered_h5, idx=idx, **kwargs)

    return f_prior_filtered_h5


def hdf5_scan(file_path):
    """
    Scans an HDF5 file and prints information about datasets (including their size) and attributes.

    Args:
        file_path (str): The path to the HDF5 file.

    """
    import h5py
    with h5py.File(file_path, 'r') as f:
        def print_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f"  Shape: {obj.shape}")
                print(f"  Data type: {obj.dtype}")
                if obj.attrs:
                    print("  Attributes:")
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"    {attr_name}: {attr_value}")
            elif isinstance(obj, h5py.Group):
                if obj.attrs:
                    print(f"Group: {name}")
                    print("  Attributes:")
                    for attr_name, attr_value in obj.attrs.items():
                        print(f"    {attr_name}: {attr_value}")

        f.visititems(print_info)




def file_checksum(file_path):
    """
    Calculate the MD5 checksum of a file.

    :param file_path: The path to the file.
    :type file_path: str
    :return: The MD5 checksum of the file.
    :rtype: str
    """
    import hashlib
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def download_file(url, download_dir, use_checksum=False, **kwargs):
    """
    Download a file from a URL to a specified directory.

    :param url: The URL of the file to download.
    :type url: str
    :param download_dir: The directory to save the downloaded file.
    :type download_dir: str
    :param use_checksum: Whether to verify the file checksum after download.
    :type use_checksum: bool
    :param kwargs: Additional keyword arguments.
    :return: None
    """
    import requests
    import os
    showInfo = kwargs.get('showInfo', 0)
    # Extract the file name from the URL
    file_name = os.path.basename(url)
    file_path = os.path.join(download_dir, file_name)

    # Check if the file already exists locally
    if os.path.exists(file_path):
        if showInfo>0:
            print(f'File {file_name} already exists. Skipping download.')
        return

    # Check if the remote file exists
    if showInfo>1:
        print('Checking if file exists on the remote server...')
    head_response = requests.head(url)
    if head_response.status_code != 200:
        if showInfo>-1:
            print(f'File {file_name} does not exist on the remote server. Skipping download.')
        return

    # Download and save the file
    if showInfo>0:
        print(f'Downloading {file_name}')
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    with open(file_path, 'wb') as file:
        file.write(response.content)
    if showInfo>-1:
        print(f'Downloaded {file_name}')

    # Check if checksum verification is enabled
    if use_checksum:
        # Calculate the MD5 checksum of the downloaded file
        downloaded_checksum = file_checksum(file_path)

        # Get the remote file checksum
        remote_checksum = head_response.headers.get('Content-MD5')

        # Compare checksums
        if downloaded_checksum != remote_checksum:
            print(f'Checksum verification failed for {file_name}. Downloaded file may be corrupted.')
            os.remove(file_path)
        else:
            print(f'Checksum verification successful for {file_name}.')
    else:
        pass
        # print(f'Checksum verification disabled for {file_name}.')

def download_file_old(url, download_dir, **kwargs):
    """
    Download a file from a URL to a specified directory (old version).

    :param url: The URL of the file to download.
    :type url: str
    :param download_dir: The directory to save the downloaded file.
    :type download_dir: str
    :param kwargs: Additional keyword arguments.
    :return: None
    """
    import requests
    import os
    showInfo = kwargs.get('showInfo', 0)
    # Extract the file name from the URL
    file_name = os.path.basename(url)
    file_path = os.path.join(download_dir, file_name)

    # Check if the remote file exists
    head_response = requests.head(url)
    if head_response.status_code != 200:
        print(f'File {file_name} does not exist on the remote server. Skipping download.')
        return

    # Check if the file already exists locally
    if os.path.exists(file_path):
        # Get the local file checksum
        local_checksum = file_checksum(file_path)

        # Download the remote file to a temporary location to compare checksums
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        remote_temp_path = os.path.join(download_dir, f'temp_{file_name}')
        with open(remote_temp_path, 'wb') as temp_file:
            temp_file.write(response.content)

        # Get the remote file checksum
        remote_checksum = file_checksum(remote_temp_path)

        # Compare checksums
        if local_checksum == remote_checksum:
            print(f'File {file_name} already exists and is identical. Skipping download.')
            os.remove(remote_temp_path)
            return
        else:
            print(f'File {file_name} exists but is different. Downloading new version.')
            os.remove(remote_temp_path)

    # Download and save the file
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    print(f'Downloading {file_name}')
    with open(file_path, 'wb') as file:
        file.write(response.content)
    print(f'Downloaded {file_name}')


def get_case_data(case='DAUGAARD', loadAll=False, loadType='', filelist=None, **kwargs):
    """
    Get case data for a specific case.

    :param case: The case name. Default is 'DAUGAARD'. Options are 'DAUGAARD', 'GRUSGRAV', 'FANGEL', 'HALD', 'ESBJERG', and 'OERUM.
    :type case: str
    :param loadAll: Whether to load all files for the case. Default is False.
    :type loadAll: bool
    :param loadType: The type of files to load. Options are '', 'prior', 'prior_data', 'post', and 'inout'.
    :type loadType: str
    :param filelist: A list of files to load. Default is None (creates new empty list).
    :type filelist: list or None
    :param kwargs: Additional keyword arguments.
    :return: A list of file names for the case.
    :rtype: list
    """
    if filelist is None:
        filelist = []

    showInfo = kwargs.get('showInfo', 0)

    if showInfo>-1:
        print('Getting data for case: %s' % case)

    if case=='DAUGAARD':

        if len(filelist)==0:
            filelist.append('DAUGAARD_AVG.h5')
            filelist.append('TX07_20231016_2x4_RC20-33.gex')
            filelist.append('daugaard_12boreholes.json')
            filelist.append('README_DAUGAARD')

        if loadAll:
            filelist.append('DAUGAARD_RAW.h5')
            filelist.append('TX07_20230731_2x4_RC20-33.gex')
            filelist.append('TX07_20230828_2x4_RC20-33.gex')
            filelist.append('TX07_20230906_2x4_RC20-33.gex')
            filelist.append('tTEM_20230727_AVG_export.h5')
            filelist.append('tTEM_20230814_AVG_export.h5')
            filelist.append('tTEM_20230829_AVG_export.h5')
            filelist.append('tTEM_20230913_AVG_export.h5')
            filelist.append('tTEM_20231109_AVG_export.h5')
            filelist.append('DAUGAARD_AVG_inout.h5')

        if (loadAll or loadType=='xyz'):            
            filelist.append('tTEM_20230727_AVG_export.xyz')
            filelist.append('tTEM_20230814_AVG_export.xyz')
            filelist.append('tTEM_20230829_AVG_export.xyz')
            filelist.append('tTEM_20230913_AVG_export.xyz')
            filelist.append('tTEM_20231109_AVG_export.xyz')
            filelist.append('tTEM_20230727_RAW_export.xyz')
            filelist.append('tTEM_20230814_RAW_export.xyz')
            filelist.append('tTEM_20230829_RAW_export.xyz')
            filelist.append('tTEM_20230913_RAW_export.xyz')
            filelist.append('tTEM_20231109_RAW_export.xyz')
            filelist.append('TX07_20230731_2x4_RC20-33.gex')
            filelist.append('TX07_20230828_2x4_RC20-33.gex')
            filelist.append('TX07_20230906_2x4_RC20-33.gex')
            filelist.append('TX07_20231016_2x4_RC20-33.gex')

            
        if (loadAll or loadType=='shapefiles'):            
            #filelist.append('Begravet dal.zip')
            filelist.append('Begravet dal.shp')
            filelist.append('Begravet dal.shx')
            #filelist.append('Erosion øvre.zip')
            filelist.append('Erosion øvre.shp')
            filelist.append('Erosion øvre.shx')
            
        
        if (loadAll or loadType=='prior'):            
            filelist.append('prior_detailed_general_N2000000_dmax90.h5')
        
        if (loadAll or loadType=='prior_data' or loadType=='post'):            
            filelist.append('prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
            filelist.append('prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
            filelist.append('prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
            filelist.append('daugaard_valley_new_N1000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
            filelist.append('daugaard_standard_new_N1000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
            
            
        if (loadAll or loadType=='post'):
            filelist.append('POST_DAUGAARD_AVG_prior_detailed_general_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5')
                    
        if (loadAll or loadType=='inout'):
            filelist.append('prior_detailed_invalleys_N2000000_dmax90.h5')
            filelist.append('prior_detailed_outvalleys_N2000000_dmax90.h5')
            filelist.append('prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
            filelist.append('prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')
            filelist.append('POST_DAUGAARD_AVG_prior_detailed_invalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5')
            filelist.append('POST_DAUGAARD_AVG_prior_detailed_outvalleys_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12_Nu2000000_aT1.h5')    
            filelist.append('prior_detailed_inout_N4000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')

    elif case=='ESBJERG':

        if len(filelist)==0:
            filelist.append('ESBJERG_ALL.h5')
            filelist.append('TX07_20230906_2x4_RC20-33.gex')
            filelist.append('README_ESBJERG')
            

        if (loadAll or loadType=='gex'):  
            filelist.append('TX07_20230906_2x4_RC20-33.gex')
            filelist.append('TX07_20231016_2x4_RC20-33.gex')
            filelist.append('TX07_20231127_2x4x1_RC20_33.gex')
            filelist.append('TX07_20240125_2x4_RC20-33.gex')
            filelist.append('TX07_20230906_2x4_RC20-33_merged.h5')  
            filelist.append('TX07_20231016_2x4_RC20-33_merged.h5')
            filelist.append('TX07_20231127_2x4x1_RC20_33_merged.h5')
            filelist.append('TX07_20240125_2x4_RC20-33_merged.h5')
        
        if (loadAll or loadType=='premerge'):
            filelist.append('20230921_AVG_export.h5')
            filelist.append('20230922_AVG_export.h5')
            filelist.append('20230925_AVG_export.h5')
            filelist.append('20230926_AVG_export.h5')
            filelist.append('20231026_AVG_export.h5')
            filelist.append('20231027_AVG_export.h5')
            filelist.append('20240109_AVG_export.h5')
            filelist.append('20240313_AVG_export.h5')
            filelist.append('TX07_20230906_2x4_RC20-33.gex')
            filelist.append('TX07_20231016_2x4_RC20-33.gex')
            filelist.append('TX07_20231127_2x4x1_RC20_33.gex')
            filelist.append('TX07_20240125_2x4_RC20-33.gex')

        if (loadAll or loadType=='ESBJERG_ALL' or len(filelist)==0):
            filelist.append('ESBJERG_ALL.h5')
            filelist.append('TX07_20230906_2x4_RC20-33.gex')
            filelist.append('README_ESBJERG')
   
        if (loadAll or loadType=='prior' or len(filelist)==0):
            filelist.append('prior_Esbjerg_claysand_N2000000_dmax90.h5')
            filelist.append('prior_Esbjerg_piggy_N2000000.h5')
            
        if (loadAll or loadType=='priordata' or len(filelist)==0):
            filelist.append('prior_Esbjerg_piggy_N2000000_TX07_20230906_2x4_RC20-33_Nh280_Nf12.h5')
            filelist.append('prior_Esbjerg_claysand_N2000000_dmax90_TX07_20231016_2x4_RC20-33_Nh280_Nf12.h5')


    elif case=='GRUSGRAV':

        filelist.append('GRUSGRAV_AVG.h5')
        filelist.append('TX07_20230425_2x4_RC20_33.gex')
        filelist.append('README_GRUSGRAV')                    

        if (loadAll or loadType=='prior'):            
            filelist.append('DJURSLAND_P01_N1000000_NB-13_NR03_PRIOR.h5') 
            filelist.append('DJURSLAND_P03_N1000000_NB-13_NR03_PRIOR.h5')
            filelist.append('DJURSLAND_P13_N1000000_NB-13_NR03_PRIOR.h5')  
            filelist.append('DJURSLAND_P40_N1000000_NB-13_NR03_PRIOR.h5')
            filelist.append('DJURSLAND_P02_N1000000_NB-13_NR03_PRIOR.h5')  
            filelist.append('DJURSLAND_P12_N1000000_NB-13_NR03_PRIOR.h5')  
            filelist.append('DJURSLAND_P34_N1000000_NB-13_NR03_PRIOR.h5')  
            filelist.append('DJURSLAND_P60_N1000000_NB-13_NR03_PRIOR.h5')    
            
    elif case=='FANGEL':
        
        filelist.append('FANGEL_AVG.h5')
        filelist.append('TX07_20230828_2x4_RC20-33.gex')
        filelist.append('README_FANGEL')

    elif case=='HALD':

        filelist.append('HALD_AVG.h5')
        filelist.append('TX07_20230731_2x4_RC20-33.gex')
        filelist.append('README_HALD')
        if loadAll:
            filelist.append('TX07_20231016_2x4_RC20-33.gex')
            filelist.append('HALD_RAW.h5')
            filelist.append('tTEM_20230801_AVG_export.h5')
            filelist.append('tTEM_20230815_AVG_export.h5')
            filelist.append('tTEM_20230905_AVG_export.h5')
            filelist.append('tTEM_20231018_AVG_export.h5')

    elif case=='OERUM':
        filelist.append('OERUM_AVG.h5')
        filelist.append('TX07_20240802_2x4_RC20-39.gex')
        filelist.append('README_OERUM')
        if loadAll:
            filelist.append('OERUM_RAW.h5')
            filelist.append('20240827_AVG_export.h5')
            filelist.append('20240828_AVG_export.h5')
            filelist.append('20240903_AVG_export.h5')
            filelist.append('20240827_RAW_export.h5')
            filelist.append('20240828_RAW_export.h5')
            filelist.append('20240903_RAW_export.h5')
                  

    elif case=='HJOELLUND':
        filelist.append('HJOELLUND_AVG.h5')
        filelist.append('TX07_20241014_2x4_RC20_33_and_57_EksternGPS.gex')
        filelist.append('README_HJOELLUND')
        if loadAll:
            filelist.append('HJOELLUND_RAW.h5')

    elif case=='HADERUP':
        filelist.append('HADERUP_MEAN_ALL.h5')
        filelist.append('TX07_Haderup_mean.gex')
        filelist.append('README_HADERUP')
        filelist.append('prior_haderup_dec25.xlsx')
        if loadAll:
            filelist.append('haderup_N1000000_dmax90_dz1.h5')
                  
    elif case=='SOENDER_FELDING':
        filelist.append('README_SOENDER_FELDING')
        filelist.append('TX07_20240802_2x4_RC20-39.gex')
        filelist.append('TX07_20240802_2x4_RC20-39_eksternGPS.gex')
        filelist.append('TX07_20240912_2x4_RC20-39_eksterngps.gex')
        filelist.append('TX07_20241014_2x4_RC20_33_and_57_EksternGPS.gex')
        filelist.append('TX07_20241202_2x4_RC20_57.gex')
        filelist.append('TX07_20241202_2x4_RC20_57_EksternGPS.gex')
        filelist.append('20240819_AVG_export.xyz')
        filelist.append('20240820_AVG_export.xyz')
        filelist.append('20240821_AVG_export.xyz')
        filelist.append('20240911_AVG_export.xyz')
        filelist.append('20240924_AVG_export.xyz')
        filelist.append('20240924_test_AVG_export.xyz')
        filelist.append('20241007_AVG_export.xyz')
        filelist.append('20241008_AVG_export.xyz')
        filelist.append('20241029_AVG_export.xyz')
        filelist.append('20240911_eksterngps_AVG_export.xyz')      
        filelist.append('20241210_AVG_export.xyz')
        filelist.append('20241210_InternGPS_AVG_export.xyz')
        filelist.append('Sdr_Felding_prior_standard_N1000000_dmax90_20260417_0929.h5')
        filelist.append('SdrFelding_boreholes.json')


        
    else:
        
        filelist = []
        print('Case %s not found' % case)

    if showInfo>2:
        print("filelist to download:")
        print(filelist)

    urlErda = 'https://anon.erda.au.dk/share_redirect/dxOLKDtoul'
    urlErdaCase = '%s/%s' % (urlErda,case)
    from tqdm import tqdm
    for remotefile in tqdm(filelist, desc='Downloading %s' % case):
        remoteurl = '%s/%s' % (urlErdaCase,remotefile)
        download_file(remoteurl,'.',showInfo=showInfo)
    if showInfo>-1:
        print('--> Got data for case: %s' % case)

    return [os.path.basename(f) for f in filelist]



def save_data_gaussian(D_obs, D_std = [], d_std=[], Cd=[], id=1, id_prior=None, i_use=None, is_log = 0, f_data_h5='data.h5', UTMX=None, UTMY=None, LINE=None, ELEVATION=None, delete_if_exist=False, name=None, compression=None, compression_opts=None, **kwargs):
    """
    Save observational data with Gaussian noise model to HDF5 file.

    Creates HDF5 datasets for electromagnetic or other geophysical measurements
    assuming Gaussian-distributed uncertainties. Handles both diagonal and full
    covariance representations of measurement errors.

    Parameters
    ----------
    D_obs : numpy.ndarray
        Observed data measurements with shape (N_stations, N_channels).
        Each row represents a measurement location, each column a data channel.
    D_std : list, optional
        Standard deviations of observed data, same shape as D_obs.
        If empty, computed from d_std parameter (default is []).
    d_std : list, optional
        Default standard deviation values or multipliers for uncertainty
        calculation when D_std is not provided (default is []).
    Cd : list, optional
        Full covariance matrices for measurement uncertainties.
        If provided, takes precedence over D_std (default is []).
    id : int, optional
        Dataset identifier for HDF5 group naming ('/D{id}', default is 1).
    id_prior : int, optional
        Prior dataset identifier to compare against during inversion. If specified,
        observed data in /D{id} will be compared with prior data in /D{id_prior}.
        If None, defaults to same ID (D1 compares with D1, D2 with D2, etc.) (default is None).
    i_use : numpy.ndarray, optional
        Binary mask indicating which data points to use in inversion, shape (N_stations,) or (N_stations,1).
        Values of 1 indicate data should be used, 0 indicates data should be excluded.
        If None, creates array of ones (all data used by default, default is None).
    is_log : int, optional
        Flag indicating logarithmic data scaling (0=linear, 1=log, default is 0).
    f_data_h5 : str, optional
        Path to output HDF5 file (default is 'data.h5').
    UTMX : numpy.ndarray, optional
        UTM X coordinates in meters, shape (N_stations,) or (N_stations,1).
        If None, creates sequential integers (default is None).
    UTMY : numpy.ndarray, optional
        UTM Y coordinates in meters, shape (N_stations,) or (N_stations,1).
        If None, creates zeros array (default is None).
    LINE : numpy.ndarray, optional
        Survey line identifiers, shape (N_stations,) or (N_stations,1).
        If None, creates array filled with 1s (default is None).
    ELEVATION : numpy.ndarray, optional
        Ground surface elevation in meters, shape (N_stations,) or (N_stations,1).
        If None, creates zeros array (default is None).
    delete_if_exist : bool, optional
        Whether to delete the entire HDF5 file if it exists before creating
        new data. Use with caution as this removes all existing data (default is False).
    name : str, optional
        Optional name attribute to be written to the data group. If provided,
        this string will be stored as an attribute alongside 'noise_model' (default is None).
    compression : str or None, optional
        Compression filter to use. Options: 'gzip', 'lzf', or None.
        If None (default), uses global DEFAULT_COMPRESSION setting.
        Set to False to explicitly disable compression.
    compression_opts : int, optional
        Compression level (0-9 for gzip). If None (default), uses global
        DEFAULT_COMPRESSION_OPTS setting. Level 1 provides 78% faster writes
        than level 9 with only 2% larger files.
    **kwargs : dict
        Additional metadata parameters:
        - showInfo : int, verbosity level
        - Other dataset attributes for electromagnetic processing

    Returns
    -------
    str
        Path to the HDF5 file where data was written.

    Notes
    -----
    The function creates HDF5 structure following INTEGRATE conventions:
    - '/D{id}/d_obs': observed measurements
    - '/D{id}/d_std': measurement standard deviations (if available)
    - '/D{id}/Cd': full covariance matrix (if provided)
    - Dataset attributes include 'noise_model'='gaussian'
    
    Uncertainty handling priority: Cd > D_std > computed from d_std
    The Gaussian noise model assumes independent, normally distributed
    measurement errors with specified standard deviations or covariances.

    Compression settings default to module-wide DEFAULT_COMPRESSION and
    DEFAULT_COMPRESSION_OPTS values (gzip level 1 by default), providing
    3.5x file size reduction with good performance.
    
    .. note::
        **Additional Parameters (kwargs):**
        
        - showInfo (int): Level of verbosity for printing information. Default is 0.
        - f_gex (str): Name of the GEX file associated with the data. Default is empty string.
        
        **Behavior:**
        
        - If D_std is not provided, it is calculated as d_std * D_obs
        - If coordinate parameters (UTMX, UTMY, LINE, ELEVATION) are provided, uses check_data() to create/update geometry datasets
        - If coordinate parameters are not provided, creates default geometry datasets if they don't exist
        - If a group with name 'D{id}' exists, it is removed before adding new data
        - Writes attributes 'noise_model' and 'is_log' to the dataset group
    """
    
    showInfo = kwargs.get('showInfo', 0)
    f_gex = kwargs.get('f_gex', '')

    # Handle compression parameters
    if compression is None:
        compression = DEFAULT_COMPRESSION
    elif compression is False:
        compression = None
    if compression_opts is None:
        compression_opts = DEFAULT_COMPRESSION_OPTS

    # Delete entire file if requested
    if delete_if_exist:
        import os
        if os.path.exists(f_data_h5):
            os.remove(f_data_h5)
            if showInfo > 1:
                print("File %s has been deleted." % f_data_h5)
        else:
            if showInfo > 1:
                print("File %s does not exist." % f_data_h5)

    # Ensure D_obs is 2D with shape (N_stations, N_channels)
    D_obs = np.atleast_2d(D_obs)
    #if D_obs.shape[0] == 1 and D_obs.shape[1] > 1:
    #    D_obs = D_obs.T

    if len(D_std)==0:
        if len(d_std)==0:
            d_std = 0.01
        D_std = np.abs(d_std * D_obs)
    else:
        # Ensure D_std is 2D with same shape as D_obs
        D_std = np.atleast_2d(D_std)
        #if D_std.shape[0] == 1 and D_std.shape[1] > 1:
        #    D_std = D_std.T

    D_str = 'D%d' % id
    ns,nd=D_obs.shape
    if showInfo>-1:
        print('Data has %d stations and %d channels' % (ns,nd))

    # Handle i_use parameter
    if i_use is None:
        i_use = np.ones((ns, 1))
    else:
        i_use = np.atleast_2d(i_use)
        if i_use.shape[0] == 1 and i_use.shape[1] > 1:
            i_use = i_use.T

    # Handle geometry data
    coord_provided = any(coord is not None for coord in [UTMX, UTMY, LINE, ELEVATION])
    
    if coord_provided:
        # Use check_data to handle geometry with provided coordinates
        import integrate as ig
        check_kwargs = {'showInfo': showInfo}
        if UTMX is not None:
            check_kwargs['UTMX'] = UTMX
        if UTMY is not None:
            check_kwargs['UTMY'] = UTMY
        if LINE is not None:
            check_kwargs['LINE'] = LINE
        if ELEVATION is not None:
            check_kwargs['ELEVATION'] = ELEVATION
        ig.check_data(f_data_h5, **check_kwargs)
    else:
        # Original behavior: create default geometry datasets if they don't exist
        with h5py.File(f_data_h5, 'a') as f:
            # check if '/UTMX' exists and create it if it does not
            if 'UTMX' not in f:
                if showInfo>0:
                    print('Creating %s:/UTMX' % f_data_h5) 
                UTMX_default = np.atleast_2d(np.arange(ns)).T
                f.create_dataset('UTMX' , data=UTMX_default) 
            if 'UTMY' not in f:
                if showInfo>0:
                    print('Creating %s:/UTMY' % f_data_h5)
                UTMY_default = f['UTMX'][:]*0
                f.create_dataset('UTMY', data=UTMY_default)
            if 'LINE' not in f:
                if showInfo>0:
                    print('Creating %s:/LINE' % f_data_h5)
                LINE_default = f['UTMX'][:]*0+1
                f.create_dataset('LINE', data=LINE_default)
            if 'ELEVATION' not in f:
                if showInfo>0:
                    print('Creating %s:/ELEVATION' % f_data_h5)
                ELEVATION_default = f['UTMX'][:]*0
                f.create_dataset('ELEVATION', data=ELEVATION_default)

    # check if group 'D{id}/' exists and remove it if it does
    with h5py.File(f_data_h5, 'a') as f:
        if D_str in f:
            if showInfo>-1:
                print('Removing group %s:%s ' % (f_data_h5,D_str))
            del f[D_str]

    # Write DATA
    with h5py.File(f_data_h5, 'a') as f:
        if showInfo>-1:
            print('Adding group %s:%s ' % (f_data_h5,D_str))

        # Helper function to create dataset with optional compression
        def create_ds(name, data):
            # Scalar datasets cannot have compression
            data_arr = np.asarray(data)
            if compression is None or data_arr.ndim == 0:
                f.create_dataset(name, data=data)
            elif compression_opts is None:
                f.create_dataset(name, data=data, compression=compression)
            else:
                f.create_dataset(name, data=data, compression=compression, compression_opts=compression_opts)

        create_ds('/%s/d_obs' % D_str, D_obs)
        create_ds('/%s/i_use' % D_str, i_use)

        # Write id_prior if specified
        if id_prior is not None:
            create_ds('/%s/id_prior' % D_str, id_prior)

        # Write either Cd or d_std
        if len(Cd) == 0:
            create_ds('/%s/d_std' % D_str, D_std)
        else:
            create_ds('/%s/Cd' % D_str, Cd)

        # wrote attribute noise_model
        f['/%s/' % D_str].attrs['noise_model'] = 'gaussian'
        f['/%s/' % D_str].attrs['is_log'] = is_log
        if name is not None:
            f['/%s/' % D_str].attrs['name'] = name
        if len(f_gex)>0:
            f['/%s/' % D_str].attrs['gex'] = f_gex
    
    return f_data_h5


def xyz_to_h5(file_xyz, file_gex, f_data_h5=None, i_lm_skip=None, i_hm_skip=None, nan_value=None, showInfo=0, disregardFullNan=True, data_obs=None, data_std=None):
    """
    Convert Aarhus Workbench XYZ export file(s) to an INTEGRATE HDF5 data file.

    Reads one or more tTEM/SkyTEM XYZ files exported from Aarhus Workbench and
    writes a Gaussian-noise HDF5 data file suitable for use with
    ``integrate_rejection()``.  The GEX file is used to determine which initial
    gates to skip per channel (``RemoveInitialGates``) and the total gate count
    (``NoGates``).

    Parameters
    ----------
    file_xyz : str or list of str
        Path(s) to Aarhus Workbench XYZ export file(s).  Multiple files are
        concatenated in order (e.g. several flight days sharing one GEX).
    file_gex : str
        Path to the GEX file describing the EM system configuration.  Gate
        selection and channel count are read from this file.
    f_data_h5 : str, optional
        Output HDF5 file path.  If None, derived by joining the XYZ
        basename(s) with ``'_'`` and appending ``'.h5'``.
    i_lm_skip : list of int, optional
        Workbench LM gate numbers to exclude from inversion (1-indexed, same
        numbering as in the XYZ file header, e.g. ``DBDT_Ch1GT3`` = gate 3).
        Gates already removed by ``RemoveInitialGates`` in the GEX are silently
        ignored.  Excluded gates have their ``d_obs`` set to NaN and
        ``d_std`` set to 100.
    i_hm_skip : list of int, optional
        Same as ``i_lm_skip`` but for HM (channel 2) Workbench gate numbers.
    nan_value : float or None, optional
        Value used as a missing-data sentinel in the XYZ file.  If None
        (default), the value is read from the XYZ file header (``/DUMMY``
        field, via ``model_info['dummy']``); falls back to 9999 if the
        header field is absent.  Pass an explicit value to override the
        header (e.g. ``nan_value=9999``).
    showInfo : int, optional
        Verbosity level (default 0).
        ``-1``: suppress all output.
        ``0``: minimal — print only the output file summary ("Adding group …").
        ``>=1``: verbose — also print each XYZ file as it is read.
    disregardFullNan : bool, optional
        If True (default), soundings where all gates are NaN are excluded
        from the output HDF5 file.
    data_obs : list of str, optional
        Flightlines column names (case-insensitive) to write as additional
        data blocks.  The first entry becomes ``/D2/d_obs``, the second
        ``/D3/d_obs``, and so on.  Example: ``['RX_ALTITUDE', 'TX_ALTITUDE']``.
    data_std : list of str or None, optional
        Flightlines column names for the corresponding standard deviations,
        same length as ``data_obs``.  Use ``None`` for an individual entry to
        fall back to ``0.05 * |d_obs|`` for that column.  If the whole
        parameter is omitted, all columns default to ``0.05 * |d_obs|``.

    Returns
    -------
    str
        Path to the written HDF5 file.

    Notes
    -----
    - ``d_std`` in the XYZ file is a relative (fractional) uncertainty;
      absolute ``d_std`` is computed as ``relative_std * d_obs``.
    - Geometry (UTMX, UTMY, LINE, ELEVATION) is taken from channel-1 rows.
    - The ``/D1/i_lm`` and ``/D1/i_hm`` datasets store the 0-indexed gate
      arrays that were used (mirrors the MATLAB output).
    - The GEX file path is stored as the ``gex`` attribute on ``/D1``.
    - Requires ``libaarhusxyz`` (``pip install libaarhusxyz``).

    Examples
    --------
    Single file:

    >>> f = xyz_to_h5('tTEM_20230727_AVG_export.xyz',
    ...                          'TX07_20230731_2x4_RC20-33.gex')

    Multiple files merged into one HDF5:

    >>> f = xyz_to_h5(
    ...     ['tTEM_20230727_AVG_export.xyz', 'tTEM_20230814_AVG_export.xyz'],
    ...     'TX07_20230731_2x4_RC20-33.gex'
    ... )

    Skip Workbench LM gates 2, 3 and HM gates 27-30:

    >>> f = xyz_to_h5('data.xyz', 'system.gex',
    ...                          i_lm_skip=[2, 3], i_hm_skip=[27, 28, 29, 30])
    """
    try:
        import libaarhusxyz
    except ImportError:
        raise ImportError(
            "libaarhusxyz is required for xyz_to_h5(). "
            "Install it with: pip install libaarhusxyz"
        )
    import contextlib, io
    import pandas as pd

    if showInfo > -1:
        print('Converting XYZ to HDF5: %s → %s' % (file_xyz, f_data_h5 if f_data_h5 else '(derived from XYZ)'))

    _sink = contextlib.redirect_stdout(io.StringIO())

    if isinstance(file_xyz, str):
        file_xyz = [file_xyz]
    if i_lm_skip is None:
        i_lm_skip = []
    if i_hm_skip is None:
        i_hm_skip = []

    # --- default output filename ---
    if f_data_h5 is None:
        basenames = [os.path.splitext(os.path.basename(f))[0] for f in file_xyz]
        f_data_h5 = '_'.join(basenames) + '.h5'

    # --- read GEX for gate/channel configuration ---
    with (_sink if showInfo < 1 else contextlib.nullcontext()):
        gex = libaarhusxyz.GEX(file_gex)
    n_channels = int(gex.number_channels)
    i_lm_start = int(gex.remove_initial_gates(1))
    i_lm_end   = int(gex.no_gates(1))
    if n_channels >= 2:
        i_hm_start = int(gex.remove_initial_gates(2))
        i_hm_end   = int(gex.no_gates(2))

    if showInfo >= 1:
        print('LM gates used: %d (gates %d-%d)' % (i_lm_end - i_lm_start, i_lm_start + 1, i_lm_end))
        if n_channels >= 2:
            print('HM gates used: %d (gates %d-%d)' % (i_hm_end - i_hm_start, i_hm_start + 1, i_hm_end))

    # --- read and concatenate XYZ file(s) ---
    xyz_list = []
    for f in file_xyz:
        if showInfo >= 1:
            print('Reading %s' % f)
        with (_sink if showInfo < 1 else contextlib.nullcontext()):
            xyz_list.append(libaarhusxyz.XYZ(f))
    fl = pd.concat([xyz.flightlines for xyz in xyz_list], ignore_index=True)
    ld = {k: pd.concat([xyz.layer_data[k] for xyz in xyz_list], ignore_index=True)
          for k in xyz_list[0].layer_data}

    # Handle XYZ files that use 'x'/'y' instead of 'utmx'/'utmy'
    if 'utmx' not in fl.columns and 'x' in fl.columns:
        fl = fl.rename(columns={'x': 'utmx', 'y': 'utmy'})

    # Determine dummy/missing value: explicit arg > XYZ header > fallback 9999
    if nan_value is None:
        nan_value = xyz_list[0].model_info.get('dummy', 9999)
    for k in ld:
        ld[k] = ld[k].replace(nan_value, np.nan)

    # --- pair ch1 / ch2 rows (mirrors MATLAB logic) ---
    # Every ch1 row becomes a sounding.  HM data is filled where the
    # immediately following row is ch2; otherwise those columns stay NaN.
    channel_arr = fl['channel_no'].values
    ch1_pos = np.where(channel_arr == 1)[0]

    # geometry from channel-1 rows (all of them)
    UTMX      = fl['utmx'].values[ch1_pos].astype(float)
    UTMY      = fl['utmy'].values[ch1_pos].astype(float)
    LINE      = fl['line_no'].values[ch1_pos].astype(float)
    ELEVATION = fl['elevation'].values[ch1_pos].astype(float)

    # --- LM data (channel 1) ---
    lm_obs = ld['dbdt_ch1gt'].values[ch1_pos, i_lm_start:i_lm_end].astype(float)
    lm_rel = ld['dbdt_std_ch1gt'].values[ch1_pos, i_lm_start:i_lm_end].astype(float)
    lm_std = lm_rel * lm_obs  # relative → absolute

    # --- HM data (channel 2, dual-channel systems only) ---
    if n_channels >= 2:
        n_hm = i_hm_end - i_hm_start
        hm_obs = np.full((len(ch1_pos), n_hm), np.nan)
        hm_std = np.full((len(ch1_pos), n_hm), np.nan)
        ch2_pos = ch1_pos + 1  # candidate ch2 row index for each ch1 sounding
        has_ch2 = channel_arr[np.minimum(ch2_pos, len(channel_arr) - 1)] == 2
        ch2_valid = ch2_pos[has_ch2]
        raw_obs = ld['dbdt_ch2gt'].values[:, i_hm_start:i_hm_end].astype(float)
        raw_rel = ld['dbdt_std_ch2gt'].values[:, i_hm_start:i_hm_end].astype(float)
        hm_obs[has_ch2] = raw_obs[ch2_valid]
        hm_std[has_ch2] = raw_rel[ch2_valid] * raw_obs[ch2_valid]
        d_obs = np.hstack([lm_obs, hm_obs])
        d_std = np.hstack([lm_std, hm_std])
    else:
        d_obs = lm_obs
        d_std = lm_std

    n_lm = lm_obs.shape[1]

    # --- apply gate skipping using Workbench gate numbers (mirrors MATLAB) ---
    # Workbench gate number → column index in d_obs:
    #   LM col j = Workbench gate (i_lm_start + 1 + j)
    #   HM col j = Workbench gate (i_hm_start + 1 + j)
    d_std_high = 100.0
    lm_wb_gates = np.arange(i_lm_start + 1, i_lm_end + 1)  # Workbench gate nums used
    for il in i_lm_skip:
        j_arr = np.where(lm_wb_gates == il)[0]
        if len(j_arr):
            d_obs[:, j_arr[0]] = np.nan
            d_std[:, j_arr[0]] = d_std_high
    if n_channels >= 2:
        hm_wb_gates = np.arange(i_hm_start + 1, i_hm_end + 1)
        for ih in i_hm_skip:
            j_arr = np.where(hm_wb_gates == ih)[0]
            if len(j_arr):
                d_obs[:, n_lm + j_arr[0]] = np.nan
                d_std[:, n_lm + j_arr[0]] = d_std_high

    # --- exclude all-NaN soundings ---
    keep = ~np.all(np.isnan(d_obs), axis=1) if disregardFullNan else np.ones(len(d_obs), dtype=bool)
    if disregardFullNan and showInfo >= 1:
        n_removed = np.sum(~keep)
        if n_removed > 0:
            print('Removed %d all-NaN soundings (%d remaining)' % (n_removed, np.sum(keep)))
    d_obs     = d_obs[keep]
    d_std     = d_std[keep]
    UTMX      = UTMX[keep]
    UTMY      = UTMY[keep]
    LINE      = LINE[keep]
    ELEVATION = ELEVATION[keep]

    # --- write HDF5 ---
    save_data_gaussian(
        d_obs, D_std=d_std,
        f_data_h5=f_data_h5,
        UTMX=UTMX, UTMY=UTMY, LINE=LINE, ELEVATION=ELEVATION,
        delete_if_exist=True,
        f_gex=file_gex,
        showInfo=showInfo,
    )

    # store gate index arrays (0-indexed; mirrors MATLAB i_lm / i_hm)
    with h5py.File(f_data_h5, 'a') as hf:
        hf.create_dataset('/D1/i_lm', data=np.arange(i_lm_start, i_lm_end))
        if n_channels >= 2:
            hf.create_dataset('/D1/i_hm', data=np.arange(i_hm_start, i_hm_end))

    # --- write additional data columns as D2, D3, ... ---
    if data_obs is not None:
        _data_std = data_std if data_std is not None else [None] * len(data_obs)
        for i, col_obs in enumerate(data_obs):
            obs = fl[col_obs.lower()].values[ch1_pos][keep].reshape(-1, 1).astype(float)
            std_col = _data_std[i]
            if std_col is not None:
                std = fl[std_col.lower()].values[ch1_pos][keep].reshape(-1, 1).astype(float)
            else:
                std = 0.05 * np.abs(obs)
            save_data_gaussian(
                obs, D_std=std,
                f_data_h5=f_data_h5,
                id=i + 2,
                name=col_obs,
                delete_if_exist=False,
                showInfo=showInfo,
            )

    return f_data_h5


def save_data_multinomial(D_obs, i_use=None, id=[],  id_prior=None, f_data_h5='data.h5', name=None, compression=None, compression_opts=None, **kwargs):
    """
    Save observed data to an HDF5 file in a specified group with a multinomial noise model.

    :param D_obs: The observed data array to be written to the file.
    :type D_obs: numpy.ndarray
    :param id: The ID of the group to write the data to. If not provided, the function will find the next available ID.
    :type id: list, optional
    :param id_prior: The ID of PRIOR data to compare against this data. If not set, id_prior=id
    :type id_prior: int, optional
    :param f_data_h5: The path to the HDF5 file where the data will be written. Default is 'data.h5'.
    :type f_data_h5: str, optional
    :param name: Optional human-readable name for this dataset (e.g. 'Lithology'). Stored as
        the ``name`` attribute on the HDF5 group and used by plotting routines for titles.
    :type name: str, optional
    :param kwargs: Additional keyword arguments.
    :return: The path to the HDF5 file where the data was written.
    :rtype: str
    """
    showInfo = kwargs.get('showInfo', 0)

    # Handle compression parameters
    if compression is None:
        compression = DEFAULT_COMPRESSION
    elif compression is False:
        compression = None
    if compression_opts is None:
        compression_opts = DEFAULT_COMPRESSION_OPTS

    # LZF compression doesn't accept compression_opts
    if compression == 'lzf':
        compression_opts = None

    if np.ndim(D_obs)==1:
        D_obs = np.atleast_2d(D_obs).T

    # Handle 2D input: assume shape (ns, nclass) and expand to (ns, nclass, 1)
    if np.ndim(D_obs)==2:
        if showInfo>0:
            print(f"Converting 2D input with shape {D_obs.shape} to 3D with shape {D_obs.shape + (1,)}")
            print("Assuming input has shape (ns, nclass) and setting nm=1")
        D_obs = D_obs[:, :, np.newaxis]

    # f_data_h5 is a HDF% file grousp "/D1/", "/D2".
    # FInd the is with for the maximum '/D*' group
    if not id:
        with h5py.File(f_data_h5, 'a') as f:
            for id in range(1, 100):
                D_str = 'D%d' % id
                if D_str not in f:
                    break
        if showInfo>0:
            print('Using id=%d' % id)


    D_str = 'D%d' % id

    if showInfo>-1:
        print("Trying to write %s to %s" % (D_str,f_data_h5))

    ns,nclass,nm=D_obs.shape

    if i_use is None:
        i_use = np.ones((ns,1))
    if np.ndim(D_obs)==1:
        i_use = np.atleast_2d(i_use).T
    
    if id_prior is None:
        id_prior = id
        
    # check if group 'D{id}/' exists and remove it if it does
    with h5py.File(f_data_h5, 'a') as f:
        if D_str in f:
            if showInfo>-1:
                print('Removing group %s:%s ' % (f_data_h5,D_str))
            del f[D_str]


    # Write DATA
    with h5py.File(f_data_h5, 'a') as f:
        if showInfo>-1:
            print('Adding group %s:%s ' % (f_data_h5,D_str))

        # Helper function to create dataset with optional compression
        def create_ds(name, data):
            # Scalar datasets cannot have compression
            data_arr = np.asarray(data)
            if compression is None or data_arr.ndim == 0:
                f.create_dataset(name, data=data)
            elif compression_opts is None:
                f.create_dataset(name, data=data, compression=compression)
            else:
                f.create_dataset(name, data=data, compression=compression, compression_opts=compression_opts)

        create_ds('/%s/d_obs' % D_str, D_obs)
        create_ds('/%s/i_use' % D_str, i_use)
        create_ds('/%s/id_prior' % D_str, id_prior)
            

        # write attribute noise_model as 'multinomial'
        f['/%s/' % D_str].attrs['noise_model'] = 'multinomial'
        if name is not None:
            f['/%s/' % D_str].attrs['name'] = name

    return id, f_data_h5


def check_data(f_data_h5='data.h5', **kwargs):
    """
    Validate and complete INTEGRATE data file structure.

    Ensures HDF5 data files contain required geometry datasets (UTMX, UTMY, LINE,
    ELEVATION) for electromagnetic surveys. Creates missing datasets using provided
    values or sensible defaults based on existing data dimensions.

    Parameters
    ----------
    f_data_h5 : str, optional
        Path to the HDF5 data file to validate and update (default is 'data.h5').
    **kwargs : dict
        Dataset values and configuration options:
        - UTMX : array-like, UTM X coordinates
        - UTMY : array-like, UTM Y coordinates  
        - LINE : array-like, survey line identifiers
        - ELEVATION : array-like, ground elevation values
        - showInfo : int, verbosity level (0=silent, >0=verbose)

    Returns
    -------
    None
        Function modifies the HDF5 file in place, adding missing datasets.

    Raises
    ------
    KeyError
        If 'D1/d_obs' dataset is missing and geometry dimensions cannot be determined.
    FileNotFoundError
        If the specified HDF5 file does not exist.

    Notes
    -----
    The function ensures INTEGRATE data files have complete geometry information:
    - UTMX, UTMY: Spatial coordinates (required for mapping and modeling)
    - LINE: Survey line identifiers (required for data organization) 
    - ELEVATION: Ground surface elevation (required for depth calculations)
    
    **Behavior:**
    
    - If coordinate parameters are provided (UTMX, UTMY, LINE, ELEVATION):
      * Updates existing datasets with new values
      * Creates datasets if they don't exist
    - If coordinate parameters are not provided:
      * Leaves existing datasets unchanged
      * Creates missing datasets with default values
    
    Default value generation when datasets are missing and no values provided:
    - UTMX: Sequential values 0, 1, 2, ... (placeholder coordinates)
    - UTMY: Zeros array with same length as UTMX
    - LINE: All values set to 1 (single survey line)
    - ELEVATION: All values set to 0 (sea level reference)
    
    Dataset dimensions are inferred from existing 'D1/d_obs' observations
    when no coordinate data is provided.
    """

    showInfo = kwargs.get('showInfo', 0)

    if showInfo>0:
        print('Checking INTEGRATE data in %s' % f_data_h5)
    
    # Check which coordinate parameters were provided
    coord_params = ['UTMX', 'UTMY', 'LINE', 'ELEVATION']
    provided_coords = {param: kwargs.get(param, None) for param in coord_params}
    coords_provided = any(coord is not None for coord in provided_coords.values())
    
    with h5py.File(f_data_h5, 'a') as f:
        # Handle each coordinate dataset
        for coord_name in coord_params:
            coord_data = provided_coords[coord_name]
            
            if coord_data is not None:
                # Coordinate data provided - update or create
                # Ensure coordinate data is 2D with shape (N_stations, 1)
                coord_data_2d = np.atleast_2d(coord_data)
                if coord_data_2d.shape[0] == 1 and coord_data_2d.shape[1] > 1:
                    coord_data_2d = coord_data_2d.T
                
                if coord_name in f:
                    if showInfo > 0:
                        print('Updating %s' % coord_name)
                    # Delete existing dataset and recreate with new data
                    del f[coord_name]
                    f.create_dataset(coord_name, data=coord_data_2d)
                else:
                    if showInfo > 0:
                        print('Creating %s' % coord_name)
                    f.create_dataset(coord_name, data=coord_data_2d)
            else:
                # No coordinate data provided - create defaults only if missing
                if coord_name not in f:
                    if showInfo > 0:
                        print('Creating default %s' % coord_name)
                    
                    # Determine size from existing data or other coordinates
                    ns = None
                    if 'D1/d_obs' in f:
                        ns = f['D1/d_obs'].shape[0]
                    elif 'UTMX' in f:
                        ns = f['UTMX'].shape[0]
                    elif any(provided_coords[c] is not None for c in coord_params):
                        # Use size from first provided coordinate
                        for c in coord_params:
                            if provided_coords[c] is not None:
                                ns = len(np.atleast_1d(provided_coords[c]))
                                break
                    
                    if ns is None:
                        print('Warning: Cannot determine dataset size for %s' % coord_name)
                        continue
                        
                    # Create default data based on coordinate type
                    if coord_name == 'UTMX':
                        default_data = np.atleast_2d(np.arange(ns)).T
                    elif coord_name == 'UTMY':
                        default_data = np.zeros((ns, 1))
                    elif coord_name == 'LINE':
                        default_data = np.ones((ns, 1))
                    elif coord_name == 'ELEVATION':
                        default_data = np.zeros((ns, 1))
                    
                    f.create_dataset(coord_name, data=default_data)
                else:
                    if showInfo > 0:
                        print('%s already exists - leaving unchanged' % coord_name)




def merge_data(f_data, f_gex='', delta_line=0, f_data_merged_h5='', **kwargs):
    """
    Merge multiple data files into a single HDF5 file.

    :param f_data: List of input data files to merge.
    :type f_data: list
    :param f_gex: Path to geometry exchange file, by default ''.
    :type f_gex: str, optional
    :param delta_line: Line number increment for each merged file, by default 0.
    :type delta_line: int, optional
    :param f_data_merged_h5: Output merged HDF5 file path, by default derived from f_gex.
    :type f_data_merged_h5: str, optional
    :param kwargs: Additional keyword arguments.
    :return: Filename of the merged HDF5 file.
    :rtype: str
    :raises ValueError: If f_data is not a list.
    """
    
    import h5py
    import numpy as np
    import integrate as ig

    showInfo = kwargs.get('showInfo', 0)

    if len(f_data_merged_h5) == 0:
        f_data_merged_h5 = f_gex.split('.')[0] + '_merged.h5'
    

    # CHeck the f_data is a list. If so return a error
    if not isinstance(f_data, list):
        raise ValueError('f_data must be a list of strings')

    nd = len(f_data)
    if showInfo:
        print('Merging %d data sets to %s ' % (nd, f_data_merged_h5))
    
    f_data_h5 = f_data[0]
    if showInfo>1:
        print('.. Merging ', f_data_h5)    
    Xc, Yc, LINEc, ELEVATIONc = ig.get_geometry(f_data_h5)
    Dc = ig.load_data(f_data_h5, showInfo=showInfo-1)
    d_obs_c = Dc['d_obs']
    d_std_c = Dc['d_std']
    noise_model = Dc['noise_model']

    for i in range(1, len(f_data)):
        f_data_h5 = f_data[i]                   
        if showInfo>1:
            print('.. Merging ', f_data_h5)    
        X, Y, LINE, ELEVATION = ig.get_geometry(f_data_h5)
        D = ig.load_data(f_data_h5, showInfo=showInfo)

        # attempt data merge before touching geometry
        merge_ok = True
        d_obs_new = list(d_obs_c)
        d_std_new = list(d_std_c)
        for id in range(len(d_obs_c)):
            try:
                d_obs_new[id] = np.vstack((d_obs_c[id], np.atleast_2d(D['d_obs'][id])))
                d_std_new[id] = np.vstack((d_std_c[id], np.atleast_2d(D['d_std'][id])))
            except:
                merge_ok = False
                if showInfo>-1:
                    print("!!!!! Could not merge %s" % f_data_h5)
                break

        if not merge_ok:
            continue

        # only append geometry when data merged successfully
        d_obs_c = d_obs_new
        d_std_c = d_std_new
        Xc = np.append(Xc, X)
        Yc = np.append(Yc, Y)
        LINEc = np.append(LINEc, LINE+i*delta_line)
        ELEVATIONc = np.append(ELEVATIONc, ELEVATION)

    Xc = np.atleast_2d(Xc).T
    Yc = np.atleast_2d(Yc).T
    LINEc = np.atleast_2d(LINEc).T
    ELEVATIONc = np.atleast_2d(ELEVATIONc).T

    with h5py.File(f_data_merged_h5, 'w') as f:
        f.create_dataset('UTMX', data=Xc)
        f.create_dataset('UTMY', data=Yc)
        f.create_dataset('LINE', data=LINEc)
        f.create_dataset('ELEVATION', data=ELEVATIONc)

    for id in range(len(d_obs_c)):
        write_data_gaussian(d_obs_c[id], D_std = d_std_c[id], noise_model = noise_model, f_data_h5=f_data_merged_h5, id=id+1, f_gex = f_gex)

    return f_data_merged_h5




## 

def merge_posterior(f_post_h5_files, f_data_h5_files, f_post_merged_h5='', showInfo=0):
    """
    Merge multiple posterior sampling results into unified datasets.

    Combines posterior results from separate electromagnetic survey areas or
    time periods into single merged files for comprehensive regional analysis.
    Handles both model parameter statistics and observational data consolidation.

    Parameters
    ----------
    f_post_h5_files : list of str
        List of paths to posterior HDF5 files containing sampling results
        from different survey areas or processing runs.
    f_data_h5_files : list of str
        List of paths to corresponding observational data HDF5 files.
        Must have same length as f_post_h5_files with matching order.
    f_post_merged_h5 : str, optional
        Output path for merged posterior file. If empty, generates default
        name based on input files (default is '').

    Returns
    -------
    tuple
        Tuple containing (merged_posterior_path, merged_data_path) where:
        - merged_posterior_path : str, path to merged posterior HDF5 file
        - merged_data_path : str, path to merged observational data HDF5 file

    Raises
    ------
    ValueError
        If f_data_h5_files and f_post_h5_files have different lengths.
    FileNotFoundError
        If any input files do not exist or cannot be accessed.

    Notes
    -----
    The merging process combines:
    - Model parameter statistics (Mean, Median, Mode, Std, Entropy)
    - Temperature and evidence fields from sampling
    - Geometry and observational data from all survey areas
    - Metadata and file references for traceability
    
    Spatial coordinates are preserved to maintain geographic relationships
    between different survey areas. The merged files retain full compatibility
    with INTEGRATE analysis and visualization functions.
    
    If f_post_merged_h5 is not provided, the output uses the format
    ``'POST_merged_N{N}.h5'`` and the data file uses ``'DATA_merged_N{N}.h5'``.
    Posterior files must have compatible structure for merging.
    """
    import h5py
    import integrate as ig

    nf = len(f_post_h5_files)
    # Check that legth of f_data_h5_files is the same as f_post_h5_files
    if len(f_data_h5_files) != nf:
        raise ValueError('Length of f_data_h5_files must be the same as f_post_h5_files')

    if len(f_post_merged_h5) == 0:
        f_post_merged_h5 = 'POST_merged_N%d.h5' % nf

    f_data_merged_h5 = 'DATA_merged_N%d.h5' % nf

    if showInfo>0:
        print('Merging %d posterior files to %s' % (nf, f_post_merged_h5))
        print('Merging %d data files to %s' % (nf, f_data_merged_h5))

    f_data_merged_h5 = ig.merge_data(f_data_h5_files, f_data_merged_h5=f_data_merged_h5)


    for i in range(len(f_post_h5_files)):
        #  get 'i_sample' from the merged file
        f_post_h5 = f_post_h5_files[i]
        with h5py.File(f_post_h5, 'r') as f:
            i_use_s = f['i_use'][:]
            T_s = f['T'][:]
            EV_s = f['EV'][:]
            f_prior_h5 = f['/'].attrs['f5_prior']
            f_data_h5 = f['/'].attrs['f5_data']
            if i == 0:
                i_use = i_use_s
                T = T_s
                EV = EV_s 
            else:
                i_use = np.concatenate((i_use,i_use_s))
                T = np.concatenate((T,T_s))
                EV = np.concatenate((EV,EV_s))

    # Write the merged data to             
    with h5py.File(f_post_merged_h5, 'w') as f:
        f.create_dataset('i_use', data=i_use)
        f.create_dataset('T', data=T)
        f.create_dataset('EV', data=EV)
        f.attrs['f5_prior'] = f_prior_h5
        f.attrs['f5_data'] = f_data_merged_h5
        # ALSOE WRITE AN ATTRIBUET 'f5_data_mul' to the merged file
        #f.attrs['f5_data_files'] = f_data_h5_files


    return f_post_merged_h5, f_data_merged_h5


def merge_prior(f_prior_h5_files, f_prior_merged_h5='', shuffle=True, showInfo=0):
    """
    Merge multiple prior model files into a single combined HDF5 file.

    Combines prior model parameters and forward-modeled data from multiple
    HDF5 files into a unified dataset. Creates a new model parameter (MX where
    X is the next available number) that tracks the source file index for each
    sample, enabling traceability of merged data origins.

    Parameters
    ----------
    f_prior_h5_files : list of str
        List of paths to prior HDF5 files to merge. Each file must contain
        compatible model parameters (M1, M2, M3, ...) and data arrays (D1, D2, ...).
    f_prior_merged_h5 : str, optional
        Output path for the merged prior file. If empty, generates default
        name 'PRIOR_merged_N{number_of_files}.h5' (default is '').
    shuffle : bool, optional
        If True (default), randomly shuffle the order of realizations in the merged
        output. The same permutation is applied to all datasets (M1, M2, D1, D2, etc.)
        to maintain consistency. This is useful for ensuring realizations from different
        source files are well-mixed. If False, realizations are concatenated in order.
    showInfo : int, optional
        Verbosity level for progress information. Higher values provide more
        detailed output (default is 0).

    Returns
    -------
    str
        Path to the merged prior HDF5 file.

    Raises
    ------
    ValueError
        If f_prior_h5_files is not a list or is empty.
    FileNotFoundError
        If any input files do not exist or cannot be accessed.

    Notes
    -----
    The merging process:
    - Concatenates all model parameters (M1, M2, M3, ...) across files
    - Concatenates all data arrays (D1, D2, D3, ...) across files
    - Creates new MX parameter (where X is next available number) containing source file indices (1-based)
    - Optionally shuffles realizations using a consistent permutation across all arrays
    - Preserves HDF5 attributes that are identical across all input files
    - Updates metadata to reflect merged status

    **Shuffling Behavior:**
    When shuffle=True (default), a random permutation is applied to all realizations:
    - A single permutation is generated and applied to ALL datasets (M1, M2, D1, D2, etc.)
    - This ensures realizations remain synchronized across all parameters
    - Uses fixed random seed (42) for reproducibility
    - Useful for mixing realizations from different source files
    - The source file tracking parameter (MX) is also shuffled to maintain traceability

    **Attribute Preservation:**
    The function intelligently copies dataset attributes from input files to the merged file:
    - Only attributes that are **identical** across all input files are copied
    - This includes important attributes like `class_name`, `class_id`, `is_discrete`, `clim`, `cmap`, etc.
    - Attributes for data arrays (D1, D2, ...) like `method`, `type`, `Nfreq`, etc. are preserved
    - Special handling for `x` and `z` attributes to match potentially padded dimensions

    **Source File Tracking:**
    The new MX parameter is a DISCRETE integer array with shape (Ntotal, 1) where
    each value indicates which input file the corresponding sample originated from:
    - 1: samples from first file in f_prior_h5_files
    - 2: samples from second file in f_prior_h5_files
    - etc.
    
    The MX parameter is properly marked with:
    - is_discrete = 1 (discrete parameter type)
    - shape = (Ntotal, 1) (consistent with other model parameters)
    - class_name = meaningful names derived from filenames
    - class_id = [1, 2, 3, ...] (class identifiers)

    **File Compatibility:**
    Input files can have different model parameter dimensions (e.g., different
    numbers of layers). Arrays with fewer parameters will be padded with NaN
    values to match the maximum dimensions. Data arrays should ideally have
    the same dimensions, but padding is applied if they differ.

    Examples
    --------
    >>> # Default: merge with shuffling
    >>> f_files = ['prior1.h5', 'prior2.h5', 'prior3.h5']
    >>> merged_file = merge_prior(f_files, 'combined_prior.h5')
    >>> print(f"Merged {len(f_files)} files into {merged_file}")

    >>> # Merge without shuffling (preserves original order)
    >>> merged_file = merge_prior(f_files, 'combined_prior.h5', shuffle=False)

    >>> # Merge with verbose output
    >>> merged_file = merge_prior(f_files, 'combined_prior.h5', showInfo=1)
    """
    import h5py
    import numpy as np
    
    # Input validation
    if not isinstance(f_prior_h5_files, list):
        raise ValueError('f_prior_h5_files must be a list of strings')
    
    if len(f_prior_h5_files) == 0:
        raise ValueError('f_prior_h5_files cannot be empty')
    
    nf = len(f_prior_h5_files)
    
    # Generate output filename if not provided
    if len(f_prior_merged_h5) == 0:
        f_prior_merged_h5 = 'PRIOR_merged_N%d.h5' % nf
    
    if showInfo > 0:
        print('Merging %d prior files to %s' % (nf, f_prior_merged_h5))
    
    # Initialize storage for merged data
    M_merged = {}  # Model parameters
    D_merged = {}  # Data arrays
    source_file_values = []  # Source file indices
    sample_counts = []  # Track samples per file
    
    # First pass: collect all model parameters and data arrays
    for i, f_prior_h5 in enumerate(f_prior_h5_files):
        if showInfo > 1:
            print('.. Processing file %d/%d: %s' % (i+1,nf, f_prior_h5))
        
        with h5py.File(f_prior_h5, 'r') as f:
            # Count samples in this file (use M1 as reference)
            if 'M1' in f:
                n_samples = f['M1'].shape[0]
                sample_counts.append(n_samples)
                source_file_values.extend([i + 1] * n_samples)  # Add file index for each sample (1-based for discrete compatibility)
            else:
                raise ValueError(f'File {f_prior_h5} does not contain M1 dataset')
            
            # Process model parameters (M1, M2, M3, ...)
            for key in f.keys():
                if key.startswith('M'):
                    if key not in M_merged:
                        M_merged[key] = []
                    M_merged[key].append(f[key][:])
            
            # Process data arrays (D1, D2, D3, ...)
            for key in f.keys():
                if key.startswith('D'):
                    if key not in D_merged:
                        D_merged[key] = []
                    D_merged[key].append(f[key][:])
    
    # Concatenate all arrays (handle different dimensions)
    if showInfo > 1:
        print('.. Concatenating arrays')
    
    # Concatenate model parameters (handle different parameter dimensions)
    for key in M_merged:
        arrays = M_merged[key]
        if len(arrays) == 1:
            M_merged[key] = arrays[0]
        else:
            # Find maximum dimensions across all arrays
            max_cols = max(arr.shape[1] for arr in arrays)
            
            # Pad arrays to match maximum dimensions
            padded_arrays = []
            for arr in arrays:
                if arr.shape[1] < max_cols:
                    # Pad with NaN values for missing parameters
                    pad_width = ((0, 0), (0, max_cols - arr.shape[1]))
                    padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=np.nan)
                    padded_arrays.append(padded_arr)
                else:
                    padded_arrays.append(arr)
            
            M_merged[key] = np.vstack(padded_arrays)
    
    # Concatenate data arrays (should have same dimensions)
    for key in D_merged:
        arrays = D_merged[key]
        if len(arrays) == 1:
            D_merged[key] = arrays[0]
        else:
            # Check if all data arrays have same dimensions
            shapes = [arr.shape[1] for arr in arrays]
            if len(set(shapes)) > 1:
                if showInfo > 0:
                    print(f'Warning: Data arrays for {key} have different dimensions: {shapes}')
                # Pad data arrays to match maximum dimensions
                max_cols = max(shapes)
                padded_arrays = []
                for arr in arrays:
                    if arr.shape[1] < max_cols:
                        pad_width = ((0, 0), (0, max_cols - arr.shape[1]))
                        padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=np.nan)
                        padded_arrays.append(padded_arr)
                    else:
                        padded_arrays.append(arr)
                D_merged[key] = np.vstack(padded_arrays)
            else:
                D_merged[key] = np.vstack(arrays)
    
    # Determine next available model parameter number
    existing_m_params = [key for key in M_merged.keys() if key.startswith('M') and key[1:].isdigit()]
    if existing_m_params:
        param_numbers = [int(key[1:]) for key in existing_m_params]
        next_param_num = max(param_numbers) + 1
    else:
        next_param_num = 1
    
    next_param_key = f'M{next_param_num}'
    
    # Create the new model parameter array (source file indices) - must be shape (Ntotal, 1)
    M_merged[next_param_key] = np.array(source_file_values).reshape(-1, 1)

    # Apply shuffling if requested
    if shuffle:
        if showInfo > 1:
            print('.. Shuffling realizations')

        # Get total number of samples
        total_samples = sum(sample_counts)

        # Create a single random permutation to apply to all arrays
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        shuffle_indices = rng.permutation(total_samples)

        if showInfo > 0:
            print(f'Shuffling {total_samples} realizations (seed=42 for reproducibility)')

        # Apply the same permutation to all M arrays
        for key in M_merged.keys():
            M_merged[key] = M_merged[key][shuffle_indices]

        # Apply the same permutation to all D arrays
        for key in D_merged.keys():
            D_merged[key] = D_merged[key][shuffle_indices]

    # Write merged file
    if showInfo > 1:
        print('.. Writing merged file')

    with h5py.File(f_prior_merged_h5, 'w') as f_out:
        # Write all model parameters including M4
        for key, data in M_merged.items():
            f_out.create_dataset(key, data=data)
        
        # Write all data arrays
        for key, data in D_merged.items():
            f_out.create_dataset(key, data=data)
        
        # Copy attributes from first file and update
        with h5py.File(f_prior_h5_files[0], 'r') as f_first:
            for attr_name, attr_value in f_first.attrs.items():
                f_out.attrs[attr_name] = attr_value
        
        # Set the new model parameter as discrete parameter with proper attributes
        if next_param_key in f_out:
            f_out[next_param_key].attrs['is_discrete'] = 1  # Mark as discrete
            f_out[next_param_key].attrs['name'] = 'Source File Index'
            f_out[next_param_key].attrs['x'] = np.array([0])  # Single feature dimension (like morrill example)
            f_out[next_param_key].attrs['clim'] = [0.5, nf + 0.5]  # Colormap limits for 1-based indexing
            
            # Create class names from filenames
            class_names = []
            for f_name in f_prior_h5_files:
                # Extract meaningful name from filename
                base_name = f_name.replace('.h5', '').replace('PRIOR_', '')
                class_names.append(base_name)
            
            f_out[next_param_key].attrs['class_name'] = [name.encode('utf-8') for name in class_names]
            f_out[next_param_key].attrs['class_id'] = np.arange(1, nf + 1)  # 1-based class IDs
        
        # Copy attributes from existing model parameters to maintain consistency
        # First, check which attributes are identical across all files
        common_attrs = {}  # key -> {attr_name: attr_value}

        for key in list(M_merged.keys()) + list(D_merged.keys()):
            if key == next_param_key:
                continue  # Skip the new tracking parameter

            common_attrs[key] = {}

            # Collect attributes from first file
            with h5py.File(f_prior_h5_files[0], 'r') as f_first:
                if key not in f_first:
                    continue

                for attr_name, attr_value in f_first[key].attrs.items():
                    # Check if this attribute is the same in all files
                    is_common = True

                    for f_path in f_prior_h5_files[1:]:
                        with h5py.File(f_path, 'r') as f_other:
                            if key not in f_other or attr_name not in f_other[key].attrs:
                                is_common = False
                                break

                            other_value = f_other[key].attrs[attr_name]

                            # Compare values (handle arrays and scalars)
                            try:
                                if isinstance(attr_value, np.ndarray) and isinstance(other_value, np.ndarray):
                                    if not np.array_equal(attr_value, other_value):
                                        is_common = False
                                        break
                                else:
                                    if attr_value != other_value:
                                        is_common = False
                                        break
                            except (ValueError, TypeError):
                                # If comparison fails, don't include this attribute
                                is_common = False
                                break

                    if is_common:
                        common_attrs[key][attr_name] = attr_value

        # Now copy the common attributes to the merged file
        for key, attrs in common_attrs.items():
            if key in f_out:
                for attr_name, attr_value in attrs.items():
                    # Special handling for x/z attributes - update to match padded dimensions
                    if attr_name in ['x', 'z'] and key in M_merged:
                        new_dim = M_merged[key].shape[1]
                        f_out[key].attrs[attr_name] = np.arange(new_dim)
                    else:
                        f_out[key].attrs[attr_name] = attr_value

        # Add merge-specific attributes
        f_out.attrs['merged_from_files'] = [f.encode('utf-8') for f in f_prior_h5_files]
        f_out.attrs['n_merged_files'] = nf
        f_out.attrs['samples_per_file'] = sample_counts
        f_out.attrs[f'{next_param_key}_description'] = 'Source file index (1-based) - DISCRETE parameter'
    
    if showInfo > 0:
        total_samples = sum(sample_counts)
        print('Successfully merged %d samples from %d files' % (total_samples, nf))
        print(f'Added {next_param_key} parameter tracking source file indices')
    
    return f_prior_merged_h5


def read_usf(file_path: str) -> Dict[str, Any]:
    """
    Parse Universal Sounding Format (USF) electromagnetic data file.

    Reads and parses USF files containing electromagnetic survey data including
    measurement sweeps, timing information, and system parameters. USF is a
    standard format for time-domain electromagnetic data exchange.

    Parameters
    ----------
    file_path : str
        Path to the USF file to be parsed.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing parsed USF file contents with keys:
        - 'sweeps' : list of dict, measurement sweep data with timing and values
        - 'header' : dict, file header information and metadata
        - 'parameters' : dict, system and acquisition parameters
        - 'dummy_value' : float, placeholder value for missing data
        - Additional keys for file-specific parameters and settings

    Notes
    -----
    USF files contain structured electromagnetic data with sections for:
    - Header information (file version, date, system type)
    - Acquisition parameters (timing, frequencies, coordinates)
    - Measurement sweeps with data points and uncertainties
    - System configuration and processing parameters
    
    The parser handles various USF format variations and automatically
    converts numeric data while preserving text metadata. Sweep data
    includes timing gates, measured values, and quality indicators.
    
    This function is compatible with USF files from various electromagnetic
    systems and processing software, following standard format specifications
    for time-domain electromagnetic data exchange.
    """
    # Initialize result dictionary
    usf_data = {}
    # Current sweep being processed
    current_sweep = None
    # List to store all sweeps
    sweeps = []
    # Flag to indicate if we're reading data points
    reading_points = False
    # Store data points for current sweep
    data_points = []
    # Store the dummy value
    dummy_value = None
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")
    
    # Process each line in the file
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Process variable declarations in comment lines (//XXX: YYY)
        if line.startswith('//') and ': ' in line and not line.startswith('//USF:'):
            # Extract variable name and value
            var_match = re.match(r"//([^:]+):\s*(.+)", line)
            if var_match:
                var_name, var_value = var_match.groups()
                var_name = var_name.strip()
                var_value = var_value.strip()
                
                # Process dummy value
                if var_name == 'DUMMY':
                    try:
                        dummy_value = float(var_value)
                    except ValueError:
                        dummy_value = var_value
                    usf_data[var_name] = dummy_value
                else:
                    # Try to convert to numeric if possible
                    try:
                        usf_data[var_name] = float(var_value)
                    except ValueError:
                        usf_data[var_name] = var_value
        
        # Process lines starting with a single '/'
        elif line.startswith('/') and not line.startswith('//'):
            # Check if it's an END marker
            if line == '/END':
                # This doesn't actually end the data reading - it just marks the end of the sweep header
                # We'll now be expecting a header line followed by data points
                reading_points = True
                continue
                
            # Check if it's a SWEEP_NUMBER marker
            if line.startswith('/SWEEP_NUMBER:'):
                # If we already have a sweep, add it to our list
                if current_sweep is not None:
                    sweeps.append(current_sweep)
                
                # Start a new sweep
                current_sweep = {}
                reading_points = False
                data_points = []
                
                # Extract sweep number
                sweep_match = re.match(r"/SWEEP_NUMBER:\s*(\d+)", line)
                if sweep_match:
                    sweep_number = int(sweep_match.group(1))
                    current_sweep['SWEEP_NUMBER'] = sweep_number
                continue
            
            # Check if it's a POINTS marker
            if line.startswith('/POINTS:'):
                points_match = re.match(r"/POINTS:\s*(\d+)", line)
                if points_match and current_sweep is not None:
                    current_sweep['POINTS'] = int(points_match.group(1))
                continue
                
            # Process other parameters
            param_match = re.match(r"/([^:]+):\s*(.+)", line)
            if param_match:
                param_name, param_value = param_match.groups()
                param_name = param_name.strip()
                param_value = param_value.strip()
                
                # Check if this is TX_RAMP which contains a complex list
                if param_name == 'TX_RAMP':
                    values = []
                    pairs = param_value.split(',')
                    for i in range(0, len(pairs), 2):
                        if i+1 < len(pairs):
                            try:
                                time_val = float(pairs[i].strip())
                                amp_val = float(pairs[i+1].strip())
                                values.append((time_val, amp_val))
                            except ValueError:
                                pass
                    if current_sweep is not None:
                        current_sweep[param_name] = values
                # Check if parameter contains multiple values
                elif ',' in param_value:
                    values = []
                    for val in param_value.split(','):
                        val = val.strip()
                        try:
                            values.append(float(val))
                        except ValueError:
                            values.append(val)
                    
                    if current_sweep is not None:
                        current_sweep[param_name] = values
                    else:
                        usf_data[param_name] = values
                else:
                    # Try to convert to numeric if possible
                    try:
                        value = float(param_value)
                        if current_sweep is not None:
                            current_sweep[param_name] = value
                        else:
                            usf_data[param_name] = value
                    except ValueError:
                        if current_sweep is not None:
                            current_sweep[param_name] = param_value
                        else:
                            usf_data[param_name] = param_value
            
            # Check if we should start reading data points
            if line == '/CHANNEL: 1' or line == '/CHANNEL: 2':
                reading_points = True
                channel_match = re.match(r"/CHANNEL:\s*(\d+)", line)
                if channel_match and current_sweep is not None:
                    current_sweep['CHANNEL'] = int(channel_match.group(1))
                continue
                
        # Process data points
        elif reading_points and current_sweep is not None:
            # Check for the header line that comes after /END
            if line.strip().startswith('TIME,'):
                # Store the header names for this data block
                headers = [h.strip() for h in line.split(',')]
                current_sweep['DATA_HEADERS'] = headers
                
                # Initialize arrays for each data column
                for header in headers:
                    current_sweep[header] = []
                
                continue
                
            # Parse data point values
            values = line.split(',')
            if len(values) >= 6:
                try:
                    # Add each value to the corresponding array
                    for i, val in enumerate(values):
                        if i < len(headers):
                            # Try to convert to appropriate type
                            try:
                                if headers[i] == 'QUALITY':
                                    current_sweep[headers[i]].append(int(val.strip()))
                                else:
                                    current_sweep[headers[i]].append(float(val.strip()))
                            except ValueError:
                                current_sweep[headers[i]].append(val.strip())
                                
                except (ValueError, IndexError, NameError) as e:
                    # Skip problematic lines
                    pass
    
    # Add the last sweep if there is one
    if current_sweep is not None:
        sweeps.append(current_sweep)
    
    # Add sweeps to the result
    usf_data['SWEEP'] = sweeps


    # Extract d_obs as an array of usf_data['SWEEP'][0]['VOLTAGE'],usf_data['SWEEP'][1]['VOLTAGE'] ...
    # and store it a single 1D numpy array
    d_obs = np.concatenate([sweep['VOLTAGE'] for sweep in usf_data['SWEEP']])
    d_obs = np.array(d_obs)
    usf_data['d_obs'] = d_obs
    d_rel_err = np.concatenate([sweep['ERROR_BAR'] for sweep in usf_data['SWEEP']])
    d_rel_err = np.array(d_rel_err)
    usf_data['d_rel_err'] = d_rel_err
    time = np.concatenate([sweep['TIME'] for sweep in usf_data['SWEEP']])
    time = np.array(time)   
    usf_data['time'] = time
    # Add usf_data['id'] that is '0' for SWEEP1 and '1' for SWEEP2  etc
    # so, usf_data['id'] = [0,0,0,0,1,1,1,1,1] for 2 sweeps with 4 and 5 data points
    usf_data['id'] = np.concatenate([[i] * sweep['POINTS'] for i, sweep in enumerate(usf_data['SWEEP'])])
    usf_data['id'] = 1+np.array(usf_data['id'])
    # Add usf_data['dummy'] that is the dummy value
    usf_data['dummy'] = dummy_value
    # Add usf_data['file_name'] that is the file name
    usf_data['file_name'] = file_path.split('/')[-1]
    # Add usf_data['file_path'] that is the file path
    usf_data['file_path'] = file_path
    
    
    return usf_data


def test_read_usf(file_path: str) -> None:
    """
    Test function to read a USF file and print some key values.
    
    Args:
        file_path: Path to the USF file
    """
    usf = read_usf(file_path)
    
    print(f"DUMMY: {usf.get('DUMMY')}")
    print(f"SWEEPS: {usf.get('SWEEPS')}")
    
    for i, sweep in enumerate(usf.get('SWEEP', [])):
        print(f"\nSWEEP {i}:")
        print(f"CURRENT: {sweep.get('CURRENT')}")
        print(f"FREQUENCY: {sweep.get('FREQUENCY')}")
        print(f"POINTS: {sweep.get('POINTS')}")
        
        if 'TIME' in sweep and len(sweep['TIME']) > 0:
            print(f"First TIME value: {sweep['TIME'][0]}")
            print(f"First VOLTAGE value: {sweep['VOLTAGE'][0]}")
            print(f"Number of data points: {len(sweep['TIME'])}")
            print(f"Data headers: {sweep.get('DATA_HEADERS', [])}")
    



    return usf


def read_usf_mul(directory: str = ".", ext: str = ".usf") -> List[Dict[str, Any]]:
    """
    Read all USF files in a specified directory and return a list of USF data structures.
    
    Args:
        directory: Path to the directory containing USF files (default: current directory)
        ext: File extension to look for (default: ".usf")
        
    Returns:
        tuple containing:
            - np.ndarray: Array of observed data (d_obs) from all USF files
            - np.ndarray: Array of relative errors (d_rel_err) from all USF files
            - List[Dict[str, Any]]: List of USF data structures, each representing a single USF file

    """
    import os
    import glob
    from typing import List, Dict, Any

    # Make sure the extension starts with a period
    if not ext.startswith('.'):
        ext = '.' + ext
    
    # Get all matching files in the directory
    file_pattern = os.path.join(directory, f"*{ext}")
    usf_files = sorted(glob.glob(file_pattern))
    
    if not usf_files:
        print(f"No files with extension '{ext}' found in '{directory}'")
        return []
    
    # List to hold all USF data structures
    usf_list = []


    D_obs = []
    D_rel_err = []
    # Process each file
    for file_path in usf_files:
        try:
            # Read the USF file
            usf_data = read_usf(file_path)
            
            # Add the file name to the USF data structure
            usf_data['FILE_NAME'] = os.path.basename(file_path)

            D_obs.append(usf_data['d_obs'])
            D_rel_err.append(usf_data['d_rel_err'])

            # Add to the list
            usf_list.append(usf_data)
            
            print(f"Successfully read: {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    D_obs = np.array(D_obs)
    D_rel_err = np.array(D_rel_err)

    print(f"Read {len(usf_list)} out of {len(usf_files)} files.")
    return D_obs, D_rel_err, usf_list


# ============================================================================
# DEPRECATED FUNCTIONS - Maintained for backward compatibility
# ============================================================================

def write_data_gaussian(*args, **kwargs):
    """
    [DEPRECATED] Use save_data_gaussian() instead.

    This function has been renamed to save_data_gaussian() to maintain consistency
    with the HDF5 I/O naming convention (load_* / save_* for HDF5 operations).

    The write_data_gaussian() function will be removed in a future version.
    Please update your code to use save_data_gaussian() instead.

    See Also
    --------
    save_data_gaussian : The new function name for this functionality
    """
    import warnings
    warnings.warn(
        "write_data_gaussian() is deprecated and will be removed in a future version. "
        "Please use save_data_gaussian() instead. "
        "This change maintains consistency with HDF5 I/O naming conventions (load_*/save_*).",
        DeprecationWarning,
        stacklevel=2
    )
    return save_data_gaussian(*args, **kwargs)


def write_data_multinomial(*args, **kwargs):
    """
    [DEPRECATED] Use save_data_multinomial() instead.

    This function has been renamed to save_data_multinomial() to maintain consistency
    with the HDF5 I/O naming convention (load_* / save_* for HDF5 operations).

    The write_data_multinomial() function will be removed in a future version.
    Please update your code to use save_data_multinomial() instead.

    See Also
    --------
    save_data_multinomial : The new function name for this functionality
    """
    import warnings
    warnings.warn(
        "write_data_multinomial() is deprecated and will be removed in a future version. "
        "Please use save_data_multinomial() instead. "
        "This change maintains consistency with HDF5 I/O naming conventions (load_*/save_*).",
        DeprecationWarning,
        stacklevel=2
    )
    return save_data_multinomial(*args, **kwargs)


def hdf5_info(f_h5, verbose=True, load_data=False):
    """
    Get and print comprehensive information about an HDF5 file.

    This function reads an HDF5 file (DATA, PRIOR, POST, or FORWARD) and prints
    detailed information about its contents, including datasets, dimensions,
    attributes, and file-type-specific metadata.

    By default, only metadata (shapes, dtypes, attributes) is read for fast
    analysis. Set load_data=True to also compute data ranges and statistics.

    Parameters
    ----------
    f_h5 : str
        Path to the HDF5 file to analyze.
    verbose : bool, optional
        If True, prints detailed information. If False, returns dictionary only
        (default is True).
    load_data : bool, optional
        If True, loads actual data to compute ranges and statistics. If False,
        only reads metadata (much faster, default is False).

    Returns
    -------
    info : dict
        Dictionary containing file information with keys:
        - 'file_type': Detected file type ('DATA', 'PRIOR', 'POST', 'FORWARD', or 'UNKNOWN')
        - 'datasets': List of dataset paths
        - 'attributes': Dictionary of root-level attributes
        - 'structure': Nested dictionary of file structure

    Examples
    --------
    >>> hdf5_info('PRIOR.h5')
    >>> info = hdf5_info('DATA.h5', verbose=False)
    >>> info = hdf5_info('POST.h5', load_data=True)  # Include data ranges

    Notes
    -----
    The function determines file type based on the presence of characteristic
    datasets:
    - DATA files: contain /UTMX, /UTMY, /ELEVATION, /LINE and /D1/, /D2/, etc.
    - PRIOR files: contain /M1, /M2, /D1, /D2 arrays
    - POST files: contain /i_use, /T, /EV attributes
    - FORWARD files: contain /method attribute

    Performance:
    - With load_data=False (default): Very fast, only reads file metadata
    - With load_data=True: Slower, reads all data to compute ranges/statistics

    See Also
    --------
    load_prior : Load prior model and data
    load_data : Load observational data
    load_posterior : Load posterior results
    """
    import os

    if not os.path.exists(f_h5):
        print(f"ERROR: File not found: {f_h5}")
        return None

    info = {
        'file_type': 'UNKNOWN',
        'datasets': [],
        'attributes': {},
        'structure': {},
        'load_data': load_data
    }

    def print_line(text='', indent=0):
        """Helper function to print with indentation."""
        if verbose:
            print('  ' * indent + text)

    def get_attrs_dict(h5_obj):
        """Convert HDF5 attributes to dictionary."""
        attrs = {}
        for key, val in h5_obj.attrs.items():
            if isinstance(val, bytes):
                attrs[key] = val.decode('utf-8')
            else:
                attrs[key] = val
        return attrs

    def visit_item(name, obj):
        """Visitor function to catalog all datasets."""
        if isinstance(obj, h5py.Dataset):
            info['datasets'].append(name)

    try:
        with h5py.File(f_h5, 'r') as f:
            # Get root attributes
            info['attributes'] = get_attrs_dict(f)

            # Catalog all datasets
            f.visititems(visit_item)

            # Determine file type
            file_type = _detect_file_type(f, info['datasets'])
            info['file_type'] = file_type

            # Print header
            print_line()
            print_line("=" * 80)
            print_line(f"HDF5 FILE ANALYSIS: {os.path.basename(f_h5)}")
            print_line("=" * 80)
            print_line(f"File path: {f_h5}")
            print_line(f"File type: {file_type}")
            print_line(f"File size: {os.path.getsize(f_h5) / (1024**2):.2f} MB")
            if not load_data:
                print_line("Mode: Metadata only (use load_data=True for data ranges)")
            else:
                print_line("Mode: Full analysis with data ranges")
            print_line()

            # Root-level attributes
            if info['attributes']:
                print_line("Root Attributes:")
                print_line("-" * 80)
                for key, val in info['attributes'].items():
                    print_line(f"  {key}: {val}", 0)
                print_line()

            # Type-specific analysis
            if file_type == 'DATA':
                _analyze_data_file(f, print_line, load_data)
            elif file_type == 'PRIOR':
                _analyze_prior_file(f, print_line, load_data)
            elif file_type == 'POST':
                _analyze_post_file(f, print_line, load_data)
            elif file_type == 'FORWARD':
                _analyze_forward_file(f, print_line, load_data)
            else:
                _analyze_unknown_file(f, print_line, load_data)

            print_line("=" * 80)
            print_line()

    except Exception as e:
        print(f"ERROR analyzing file: {e}")
        import traceback
        traceback.print_exc()
        return None

    return info


def _detect_file_type(f, datasets):
    """Detect the type of HDF5 file (DATA, PRIOR, POST, FORWARD)."""
    # Check for DATA file characteristics
    has_geometry = any(ds in datasets for ds in ['UTMX', 'UTMY', 'ELEVATION', 'LINE'])
    has_data_groups = any(ds.startswith('D') and '/' in ds for ds in datasets)

    # Check for PRIOR file characteristics
    has_model = any(ds.startswith('M') and len(ds) == 2 for ds in datasets)

    # Check for POST file characteristics
    has_post = any(ds in datasets for ds in ['i_use', 'T', 'EV'])

    # Check for FORWARD file characteristics
    has_forward_method = 'method' in f.attrs

    if has_post:
        return 'POST'
    elif has_model:
        # PRIOR files can have model parameters with or without forward data
        return 'PRIOR'
    elif has_geometry and has_data_groups:
        return 'DATA'
    elif has_forward_method:
        return 'FORWARD'
    else:
        return 'UNKNOWN'


def _analyze_data_file(f, print_line, load_data=False):
    """Analyze DATA.h5 file structure."""
    print_line("DATA FILE CONTENTS:")
    print_line("-" * 80)

    # Geometry datasets
    print_line("Geometry Information:", 0)
    for ds_name in ['UTMX', 'UTMY', 'ELEVATION', 'LINE']:
        if ds_name in f:
            ds = f[ds_name]
            print_line(f"  /{ds_name}: shape={ds.shape}, dtype={ds.dtype}", 1)
            if load_data and ds.size > 0:
                print_line(f"    Range: [{np.min(ds[...]):.2f}, {np.max(ds[...]):.2f}]", 1)

    Np = f['UTMX'].shape[0] if 'UTMX' in f else 0
    print_line(f"  Number of data locations (Np): {Np}", 1)
    print_line()

    # Data groups
    data_groups = sorted([key for key in f.keys() if key.startswith('D') and len(key) == 2])
    print_line(f"Data Groups: {len(data_groups)} found", 0)
    print_line()

    for dg in data_groups:
        print_line(f"/{dg}/ - Data Type {dg}", 0)
        group = f[dg]

        # Attributes
        attrs = dict(group.attrs)
        if attrs:
            print_line("  Attributes:", 1)
            for key, val in attrs.items():
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                print_line(f"    {key}: {val}", 1)

        # Datasets in group
        print_line("  Datasets:", 1)
        for ds_name in sorted(group.keys()):
            ds = group[ds_name]
            if isinstance(ds, h5py.Dataset):
                print_line(f"    {ds_name}: shape={ds.shape}, dtype={ds.dtype}", 1)

                # Show attributes of dataset
                ds_attrs = dict(ds.attrs)
                if ds_attrs:
                    for key, val in ds_attrs.items():
                        if isinstance(val, bytes):
                            val = val.decode('utf-8')
                        print_line(f"      @{key}: {val}", 1)
        print_line()


def _analyze_prior_file(f, print_line, load_data=False):
    """Analyze PRIOR.h5 file structure."""
    print_line("PRIOR FILE CONTENTS:")
    print_line("-" * 80)

    # Determine number of realizations
    N = None
    model_keys = sorted([key for key in f.keys() if key.startswith('M') and len(key) == 2])
    if model_keys:
        N = f[model_keys[0]].shape[0]

    print_line(f"Number of realizations (N): {N}", 0)
    print_line()

    # Model parameters
    print_line(f"Model Parameters: {len(model_keys)} found", 0)
    print_line()

    for mk in model_keys:
        dataset = f[mk]
        print_line(f"/{mk}/ - Model Parameter", 0)
        print_line(f"  Shape: {dataset.shape} (N x Nm{mk[1:]})", 1)
        print_line(f"  Dtype: {dataset.dtype}", 1)

        # Attributes
        attrs = dict(dataset.attrs)
        if 'name' in attrs:
            name = attrs['name']
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            print_line(f"  name: {name}", 1)

        if 'is_discrete' in attrs:
            is_discrete = attrs['is_discrete']
            print_line(f"  is_discrete: {is_discrete}", 1)

            if is_discrete:
                if 'class_id' in attrs:
                    class_ids = attrs['class_id']
                    print_line(f"  class_id: {class_ids}", 1)

                if 'class_name' in attrs:
                    class_names = attrs['class_name']
                    if isinstance(class_names, bytes):
                        class_names = class_names.decode('utf-8')
                    print_line(f"  class_name: {class_names}", 1)

        if 'x' in attrs:
            x = attrs['x']
            if hasattr(x, '__len__'):
                print_line(f"  x: {len(x)} values, range=[{np.min(x):.2f}, {np.max(x):.2f}]", 1)
            else:
                print_line(f"  x: {x}", 1)

        if 'clim' in attrs:
            clim = attrs['clim']
            print_line(f"  clim: {clim}", 1)

        # Show data range only if load_data=True
        if load_data and dataset.size > 0:
            data_min = np.min(dataset[...])
            data_max = np.max(dataset[...])
            print_line(f"  Data range: [{data_min:.4f}, {data_max:.4f}]", 1)

        print_line()

    # Data groups
    data_groups = sorted([key for key in f.keys() if key.startswith('D') and len(key) == 2])
    print_line(f"Data Realizations: {len(data_groups)} found", 0)
    print_line()

    for dg in data_groups:
        dataset = f[dg]
        print_line(f"/{dg}/ - Forward Data", 0)
        print_line(f"  Shape: {dataset.shape} (N x Nd{dg[1:]})", 1)
        print_line(f"  Dtype: {dataset.dtype}", 1)

        # Attributes
        attrs = dict(dataset.attrs)
        if 'f5_forward' in attrs:
            forward_file = attrs['f5_forward']
            if isinstance(forward_file, bytes):
                forward_file = forward_file.decode('utf-8')
            print_line(f"  f5_forward: {forward_file}", 1)

        if 'with_noise' in attrs:
            with_noise = attrs['with_noise']
            print_line(f"  with_noise: {with_noise}", 1)

        # Show data range only if load_data=True
        if load_data and dataset.size > 0:
            data_min = np.min(dataset[...])
            data_max = np.max(dataset[...])
            print_line(f"  Data range: [{data_min:.4e}, {data_max:.4e}]", 1)

        print_line()


def _analyze_post_file(f, print_line, load_data=False):
    """Analyze POST.h5 file structure."""
    print_line("POSTERIOR FILE CONTENTS:")
    print_line("-" * 80)

    # Core posterior datasets
    print_line("Core Posterior Information:", 0)

    if 'i_use' in f:
        i_use = f['i_use']
        N, Nr = i_use.shape
        print_line(f"  Posterior indices (/i_use): shape={i_use.shape}", 1)
        print_line(f"    Number of data locations (N): {N}", 1)
        print_line(f"    Number of realizations per location (Nr): {Nr}", 1)

    if 'T' in f:
        print_line(f"  Temperature (/T): shape={f['T'].shape}", 1)
        if load_data:
            T = f['T'][...]
            if T.size > 0:
                print_line(f"    Range: [{np.min(T):.4f}, {np.max(T):.4f}]", 1)

    if 'EV' in f:
        print_line(f"  Evidence (/EV): shape={f['EV'].shape}", 1)
        if load_data:
            EV = f['EV'][...]
            if EV.size > 0:
                print_line(f"    Range: [{np.min(EV):.4e}, {np.max(EV):.4e}]", 1)

    if 'CHI2' in f:
        print_line(f"  Reduced Chi-Squared (/CHI2): shape={f['CHI2'].shape}", 1)
        if load_data:
            chi2 = f['CHI2'][...]
            if chi2.size > 0:
                print_line(f"    CHI2 range: [{np.nanmin(chi2):.3f}, {np.nanmax(chi2):.3f}]", 1)
                print_line(f"    CHI2 mean: {np.nanmean(chi2):.3f}", 1)
                print_line(f"    (CHI2≈1 indicates good fit)", 1)

    # Attributes
    attrs = dict(f.attrs)
    if 'f5_data' in attrs:
        data_file = attrs['f5_data']
        if isinstance(data_file, bytes):
            data_file = data_file.decode('utf-8')
        print_line(f"  f5_data: {data_file}", 1)

    if 'f5_prior' in attrs:
        prior_file = attrs['f5_prior']
        if isinstance(prior_file, bytes):
            prior_file = prior_file.decode('utf-8')
        print_line(f"  f5_prior: {prior_file}", 1)

    print_line()

    # Model statistics
    model_keys = sorted([key for key in f.keys() if key.startswith('M') and len(key) == 2])
    print_line(f"Model Parameter Statistics: {len(model_keys)} found", 0)
    print_line()

    for mk in model_keys:
        group = f[mk]
        print_line(f"/{mk}/ - Model Parameter Statistics", 0)

        # Check if discrete or continuous
        has_mode = 'Mode' in group
        has_mean = 'Mean' in group

        if has_mode:
            print_line("  Type: Discrete parameter", 1)

            for stat_name in ['Mode', 'Entropy', 'P', 'M_N']:
                if stat_name in group:
                    ds = group[stat_name]
                    print_line(f"  {stat_name}: shape={ds.shape}, dtype={ds.dtype}", 1)
                    if load_data and ds.size > 0 and stat_name != 'P':  # P is 3D, skip range
                        data = ds[...]
                        print_line(f"    Range: [{np.min(data):.4f}, {np.max(data):.4f}]", 1)

        elif has_mean:
            print_line("  Type: Continuous parameter", 1)

            for stat_name in ['Mean', 'Median', 'Std']:
                if stat_name in group:
                    ds = group[stat_name]
                    print_line(f"  {stat_name}: shape={ds.shape}, dtype={ds.dtype}", 1)
                    if load_data and ds.size > 0:
                        data = ds[...]
                        print_line(f"    Range: [{np.min(data):.4f}, {np.max(data):.4f}]", 1)

        # Check for attributes
        attrs = dict(group.attrs)
        if attrs:
            print_line("  Attributes:", 1)
            for key, val in attrs.items():
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                print_line(f"    {key}: {val}", 1)

        print_line()


def _analyze_forward_file(f, print_line, load_data=False):
    """Analyze FORWARD.h5 file structure."""
    print_line("FORWARD FILE CONTENTS:")
    print_line("-" * 80)

    # Method and type
    attrs = dict(f.attrs)
    if 'method' in attrs:
        method = attrs['method']
        if isinstance(method, bytes):
            method = method.decode('utf-8')
        print_line(f"Forward method: {method}", 0)

    if 'type' in attrs:
        fwd_type = attrs['type']
        if isinstance(fwd_type, bytes):
            fwd_type = fwd_type.decode('utf-8')
        print_line(f"Forward type: {fwd_type}", 0)

    print_line()

    # List all datasets
    print_line("Datasets:", 0)
    for key in sorted(f.keys()):
        obj = f[key]
        if isinstance(obj, h5py.Dataset):
            print_line(f"  /{key}: shape={obj.shape}, dtype={obj.dtype}", 1)
        elif isinstance(obj, h5py.Group):
            print_line(f"  /{key}/ [Group]", 1)

    print_line()

    # All attributes
    if attrs:
        print_line("All Attributes:", 0)
        for key, val in sorted(attrs.items()):
            if isinstance(val, bytes):
                val = val.decode('utf-8')
            print_line(f"  {key}: {val}", 1)


def _analyze_unknown_file(f, print_line, load_data=False):
    """Analyze unknown HDF5 file structure."""
    print_line("UNKNOWN FILE TYPE - GENERIC STRUCTURE:")
    print_line("-" * 80)

    def print_structure(name, obj, indent=0):
        """Recursively print HDF5 structure."""
        if isinstance(obj, h5py.Dataset):
            print_line(f"/{name} [Dataset]: shape={obj.shape}, dtype={obj.dtype}", indent)

            # Show attributes
            attrs = dict(obj.attrs)
            if attrs:
                for key, val in attrs.items():
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    print_line(f"  @{key}: {val}", indent)

        elif isinstance(obj, h5py.Group):
            print_line(f"/{name}/ [Group]", indent)

            # Show attributes
            attrs = dict(obj.attrs)
            if attrs:
                for key, val in attrs.items():
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    print_line(f"  @{key}: {val}", indent)

            # Recursively show contents
            for key in sorted(obj.keys()):
                print_structure(name + '/' + key if name else key, obj[key], indent + 1)

    # Print full structure
    for key in sorted(f.keys()):
        print_structure(key, f[key], 0)


def write_borehole(W, filename, **kwargs):
    """
    Write one or more borehole dictionaries to a JSON file.

    Parameters
    ----------
    W : dict or list of dict
        A single borehole dict, or a list of borehole dicts.
        Each dict may contain any combination of the standard borehole
        fields:

        * ``name`` (str) – identifier
        * ``X``, ``Y`` (float) – UTM coordinates
        * ``depth_top``, ``depth_bottom`` (list of float) – interval boundaries (m)
        * ``class_obs`` (list of int) – observed lithology class per interval
        * ``class_prob`` (list of float) – confidence per interval (0–1)
        * ``method`` (str, optional) – likelihood method (default ``'mode_probability'``)
        * ``elevation`` (float, optional) – ground-surface elevation (m a.s.l.).
          Used only by :func:`plot_boreholes` to place the well on a shared
          elevation axis.  Has no effect on inversion.

        numpy arrays and scalars are automatically converted to plain
        Python lists/numbers so the file is human-readable JSON.
    filename : str
        Output JSON file path (e.g. ``'borehole.json'``).
    **kwargs
        showInfo : int, optional
            Verbosity level. Default is 0.

    Returns
    -------
    str
        The filename written.

    Examples
    --------
    >>> W = {'name': 'BH1', 'X': 498832.5, 'Y': 6250843.1,
    ...      'depth_top': [0, 10], 'depth_bottom': [10, 20],
    ...      'class_obs': [1, 2], 'class_prob': [0.9, 0.9]}
    >>> ig.write_borehole(W, 'borehole.json')
    'borehole.json'

    >>> WELLS = [W1, W2, W3]
    >>> ig.write_borehole(WELLS, 'all_boreholes.json')
    'all_boreholes.json'
    """
    import json
    import numpy as np

    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    showInfo = kwargs.get('showInfo', 0)
    data = _convert(W)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    if showInfo > 0:
        n = len(data) if isinstance(data, list) else 1
        print(f'write_borehole: wrote {n} borehole(s) to {filename}')

    return filename


def read_borehole(filename, **kwargs):
    """
    Read one or more borehole dictionaries from a JSON file.

    The file must have been written by :func:`write_borehole`.
    If the file contains a single borehole (JSON object) a single dict
    is returned.  If it contains many boreholes (JSON array) a list of
    dicts is returned.

    The returned dict(s) preserve all fields that were written, including
    the optional ``elevation`` key (ground-surface elevation in m a.s.l.)
    used by :func:`plot_boreholes` for elevation-axis plotting.

    Parameters
    ----------
    filename : str
        JSON file path to read.
    **kwargs
        showInfo : int, optional
            Verbosity level. Default is 0.

    Returns
    -------
    dict or list of dict
        Single borehole dict, or list of borehole dicts.

    Examples
    --------
    >>> W = ig.read_borehole('borehole.json')
    >>> print(W['name'], W['depth_top'])

    >>> WELLS = ig.read_borehole('all_boreholes.json')
    >>> for W in WELLS:
    ...     print(W['name'])
    """
    import json

    showInfo = kwargs.get('showInfo', 0)

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if showInfo > 0:
        n = len(data) if isinstance(data, list) else 1
        print(f'read_borehole: read {n} borehole(s) from {filename}')

    return data

