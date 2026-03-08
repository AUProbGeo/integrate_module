=============
Data format
=============

The HDF5_ file format is used as a container for all data in INTEGRATE. 
Each HDF5 file can contain multiple data sets (typically arranged in matrix format) with associated attributes that describe the data. `HDF View`_ is useful to inspect the content of HDF5 files.

.. _HDF View: https://www.hdfgroup.org/downloads/hdfview/
.. _HDF5: https://www.hdfgroup.org/solutions/hdf5/


The following HDF files are used for any INTEGRATE project 

**DATA.h5** : Stores observed data, and its associated geometry.

**PRIOR.h5**: Stores realizations of the prior model, and corresponding forward response

**FORWARD.h5**: Stores information needed to solve the forward problem, and/or needed to describe the observed data in DATA.h5

**POST.h5**: stores index of posterior realizations, as well as posterior statistics 


DATA.h5
=======
DATA.h5 contains observed data, and its associated geometry. 
The observed data can be of many types, such as TEM data and well-log data


By default, observed data in ``/D1`` is compared with prior data in ``/D1``, ``/D2`` with ``/D2``, etc.
To compare observed data with a different prior dataset, set the ``/D1/id_prior`` field.
For example, setting ``/D1/id_prior=2`` will compare observed ``/D1`` with prior ``/D2``. 

  
  ``Np``: Number of data locations (typically one set data per unique X-Y location)
  
  ``Ndi``: Number of data points ``Nd`` per data type ``i`` per location.
  
  ``Nclass``: Number of classes

The datasets ``UTMX``, ``UTMY``, ``ELEVATION``, and ``LINE`` are mandatory for most plotting routines in INTEGRATE, 
but are not used in the inversion itself.

The attribute ``D1/noise_model`` is mandatory for all data types, and describes the noise model used for the data.

.. list-table:: Data and attributes for DATA.h5
   :widths: 10 10 5 5 70 
   :header-rows: 1

   * - Dataset
     - Format
     - Attribute
     - Mandatory
     - Description
   * - /UTMX
     - [Np,1]
     - 
     - (*)
     - X - location of data points

   * - /UTMY
     - [Np,1]
     - 
     - (*)
     - Y - location of data points    
   * - /ELEVATION
     - [Np,1]
     - 
     - (*)
     - Elevation at data points    
   * - /LINE
     - [Np,1]
     - 
     - (*)
     - Linenumber at data points    
   * - /D1/noise_model
     - [string]
     - *
     - *
     - A string describing the noise model used for the data. 
   * - /D1/id_prior
     - [integer]
     -
     -
     - The prior dataset ID to compare against. Observed data in /D1 will be compared with prior data in /D{id_prior} during inversion. If not set, defaults to the same ID as the observed data (D1→D1, D2→D2, etc.) 
   * - /D1/i_use
     - [NP,1] int [0/1]
     - 
     - 
     - Determines whether a data point should be used or not. All data are used by default

The format of the observed data, and the associate uncertainty, depends on the type of data, and the choice of noise model.

See the function :func:`integrate.load_data()` for an example on how read DATA.h5 files.

"""""""""""""""""""""""""""""""""
Gaussian noise  - continuous data
"""""""""""""""""""""""""""""""""

For continuous data and the multivariate Gaussian noise model can be chosen by setting the attribute ``D1/noise_model=gaussian`` 

.. list-table:: Data and attributes in DATA.h5 for continuous data and multivariate Gaussian noise model
   :widths: 10 10 5 5 70 
   :header-rows: 1

   * - Dataset
     - Format
     - Attribute
     - Mandatory
     - Description
   * - /D1/noise_model
     - [string]='gaussian'
     - *
     - *
     - A string describing the noise model used for the data. Here ``'gaussian'`` to represent a multivariate Gaussian noise model.
   * - /D1/d_obs
     - [Np,Nd1]
     - 
     - *
     - Observed data (#1)
   * - /D1/d_std
     - [Np,Nd1]
     - 
     - *
     - Standard deviation of observed data (db/dT). Is the size is [1,Nd], the same ``d_std`` is used for all data.
   * - /D1/Cd
     - [Nd1,Nd1]
     - 
     - 
     - Correlated noise matrix. ``Cd`` is the same for all data
   * - /D1/Cd
     - [Np,Nd1,Nd1]
     - 
     - 
     - Correlated noise matrix; each data observation has its own correlated noise matrix 
   * - /gatetimes
     - [Ndata,1]
     - 
     - 
     - Gate times (in seconds) for each data point
   * - /i_lm
     - [Nlm,1]
     - 
     - 
     - Index (rel to /gatetimes) of Nlm gates for the low moment. 
   * - /i_hm
     - [Nhm,1]
     - 
     - 
     - Index (rel to /gatetimes) of Nhm gates for the high moment. 
   * - --
     - 
     - 
     - 
     - 

"""""""""""""""""""""""""""""""""
Multinomial noise - discrete data
"""""""""""""""""""""""""""""""""

For discrete data the multinomial distribution can be used as likelihood by setting the attribute ``D1/noise_model=multinomial`` 

.. list-table:: Data and attributes in DATA.h5 for data and multinomial noise model
   :widths: 10 10 5 5 70 
   :header-rows: 1

   * - Dataset
     - Format
     - Attribute
     - Mandatory
     - Description
   * - /D2/noise_model
     - [string]='multinomial'
     - *
     - *
     - The multinomial distribution is used as likelihood model for the data.
   * - /D2/d_obs
     - [Np,Nclass,Nm]
     - 
     - *
     - Observed data (class probabilities)
   * - /D2/i_use
     - [Np,1]
     - 
     - 
     - Binary indicator of whether a data point should be used or not



PRIOR.h5 
========

PRIOR.h5 contains ``N`` realizations of a prior model (represented as potentially multiple types of model parameters, such as resistivity, lithology, grainsize,....), and corresponding data (consisting of potentially multiple types of data, such as tTEM, SkyTEM, WellData,..)

``N``: Number of realizations of the prior model

``Nm1``: Number of model parameters of type 1

``Nm2``: Number of model parameters of type 2

``NmX``: Number of model parameters of type X


.. list-table:: PRIOR model realizations in PRIOR.h5
   :widths: 10 10 5 5 70 
   :header-rows: 1

   * - Dataset
     - Format
     - attribute
     - Mandatory
     - Description
   * - /M1
     - [N,Nm1]
     - 
     - *
     - N realizations of model parameter 1, 
       each consisting of Nm1 model param1eters
   * - /M1/x
     - [nm]
     - *
     - *
     - Array of values describing each value in M1 (e.g. depth to top of layer)
   * - /M1/name
     - [string]
     - *
     - 
     - Name of model parameter /M1
   * - /M1/is_discrete
     - [nm]
     - *
     - *
     - [0/1] described whether /M1 is a discrete or continuous parameter
   * - /M1/class_id
     - [1,n_class]
     - *
     - 
     - A list of  ``n_class`` class id; only when case /M1/is_discrete=1.
   * - /M1/class_name
     - [1,n_class]
     - *
     - 
     - A list of ``n_class`` strings describing each class; only when case /M1/is_discrete=1.
   * - /M1/clim
     - [1,2]
     - *
     - 
     - Min and maximum value for colorbar
   * - /M1/cmap
     - [3,nlev]
     - *
     - 
     - Colormap with ``nlev`` levels.
   * - /M2
     -  [N,Nm2]
     - 
     - 
     - N realizations of model parameter 2, 
       each consisting of Nm2 model parameters
   * - /Mx
     -  [N,NmX]
     - 
     - 
     - N realizations of model parameter X, 
       each consisting of NmX model parameters



.. list-table:: prior data realizations in PRIOR.h5
   :widths: 10 10 5 5 70 
   :header-rows: 1

   * - Dataset
     - Format
     - attribute
     - Mandatory
     - Description
   * - /D1
     - [N,Nd1]
     - 
     - *
     - N realizations of data number 1, 
       each consisting of ``Nd`` model parameters
   * - /D1/f5_forward
     - [string]
     - *
     - 
     - HDF file describing the forward model used to compute prior data.
   * - /D1/with_noise
     - [1]
     - *
     - 
     - Indicates whether noise was added to the data[1] or not[0].
   * - /D2
     -  [N,Nd2]
     - 
     - 
     - N realizations of data number 2, 
       each consisting of ``Nd2`` model parameters
     

``/D1`` is only mandatory when PRIOR.h5 is used for inversion

All the mandatory attributes specified for ``/M1`` are also mandatory for other attributes, i.e.  ``/M1``,  ``/M2``, ... . 


f_forward_h5 [string]: Defines the name of the HDF5 file that contains information need to solved the forward problem...



FORWARD.h5
==========
The FORWARD.h5 needs to hold as much information as needed to define the use of a specific forward model.

The attribute ``/method`` refer to a specific choice of forward method.


.. list-table:: posterior data realizations in PRIOR.h5
   :widths: 10 10 5 5 70 
   :header-rows: 1

   * - Dataset
     - Format
     - attribute
     - Mandatory
     - Description
   * - /method
     - [string]
     - *
     - 
     - Defines the type of forward model def:'TDEM'.
   * - /type
     - [string]
     - *
     - 
     - Define the algorithm used to solve the forward model. def:'GA-AEM'.
     

``/method`` can, for example, be ``TDEM`` for Time Domain EM (The default in INTEGRATE),
or can be ``identity`` for an identity mapping (useful to represent log data).

TDEM: Time domain EM, method='tdem'.
"""""""""""""""""""""""""""""""""""""

``/method='TDEM'`` make use of time-domain EM forward modeling. 
The following three types of forward models will (eventually) be available:


``/type='GA-AEM'`` [DEFAULT].
[GA-AEM]_. Available for both Linux and Windows, Matlab and Python.


``/type='AarhusInv'``.
[AarhusInv]_. Windows only.
Not yet implemented


``/type='SimPEG'``.
[SimPEG]_. Python only.

LOG: Well log conditioning, method='log'
""""""""""""""""""""""""""""""""""""""""

``/method='identity'`` maps attributes of a specific model (realizations of the prior) directly into data. 
  

POST.h5
=======

At the very minimum POST.h5 needs to contain the index (in PRIOR.h5) of realizations from the posterior.
Statistics are written by :func:`integrate.integrate_posterior_stats`.

``Np``: Number of data locations.

``Nr``: Number of posterior realizations per location.

``Nm``: Number of model parameters (e.g. depth layers) for a given model type.

``Nclass``: Number of discrete classes for a given model type.

.. list-table:: Core data and attributes in POST.h5
   :widths: 15 10 5 5 55
   :header-rows: 1

   * - Dataset
     - Format
     - Attribute
     - Mandatory
     - Description
   * - /i_use
     - [Np, Nr]
     -
     - *
     - Indices (into PRIOR.h5) of posterior realizations for each data location.
   * - /T
     - [Np, 1]
     -
     - *
     - Annealing temperature used during inversion.
   * - /EV
     - [Np, 1]
     -
     - *
     - Log-evidence at each data location.
   * - /f5_data
     - string
     - *
     - *
     - Filename of the DATA HDF5 file.
   * - /f5_prior
     - string
     - *
     - *
     - Filename of the PRIOR HDF5 file.
   * - /CHI2
     - [Np, Nd]
     - float
     -
     - Reduced chi-squared (χ²/ν) goodness-of-fit metric per data type. Values near 1 indicate good fit; >1 underfit; <1 overfit.
   * - /N_UNIQUE
     - [Np]
     -
     -
     - Number of unique prior realizations used at each data location. Written by :func:`integrate.integrate_posterior_stats`.
   * - /UTMX
     - [Np]
     -
     -
     - X-coordinate, copied from DATA.h5 (written when ``updateGeometryFromData=True``).
   * - /UTMY
     - [Np]
     -
     -
     - Y-coordinate, copied from DATA.h5.
   * - /ELEVATION
     - [Np]
     -
     -
     - Elevation, copied from DATA.h5.
   * - /LINE
     - [Np]
     -
     -
     - Line number, copied from DATA.h5.


Continuous parameters
"""""""""""""""""""""

Written for each continuous model parameter ``/Mx`` (``is_discrete=0``) by
:func:`integrate.integrate_posterior_stats`.

.. list-table:: Statistics for continuous model parameters in POST.h5
   :widths: 15 10 10 65
   :header-rows: 1

   * - Dataset
     - Format
     - Written
     - Description
   * - /Mx/Mean
     - [Np, Nm]
     - always
     - Arithmetic mean of the posterior realizations at each location and depth.
   * - /Mx/LogMean
     - [Np, Nm]
     - always
     - Geometric mean: exp(mean(log(values))). Appropriate for log-normally distributed parameters such as resistivity.
   * - /Mx/Median
     - [Np, Nm]
     - always
     - Median of the posterior realizations.
   * - /Mx/Std
     - [Np, Nm]
     - always
     - Standard deviation of log10(posterior). Measures spread on the logarithmic scale.
   * - /Mx/KL
     - [Np, Nm]
     - ``computeKL_continuous=True``
     - KL divergence D_KL(posterior ∥ prior) in bits (log base 2), estimated from log10-space histograms (50 bins). No fixed upper bound for continuous parameters.


Discrete parameters
"""""""""""""""""""

Written for each discrete model parameter ``/Mx`` (``is_discrete=1``) by
:func:`integrate.integrate_posterior_stats`.

.. list-table:: Statistics for discrete model parameters in POST.h5
   :widths: 15 10 10 65
   :header-rows: 1

   * - Dataset
     - Format
     - Written
     - Description
   * - /Mx/Mode
     - [Np, Nm]
     - always
     - Most probable class (class_id value) at each location and depth.
   * - /Mx/Entropy
     - [Np, Nm]
     - always
     - Shannon entropy of the posterior class probabilities, normalised by log(Nclass). Range [0, 1]: 0 = certain, 1 = maximally uncertain.
   * - /Mx/P
     - [Np, Nclass, Nm]
     - always
     - Posterior probability of each class at each location and depth.
   * - /Mx/KL
     - [Np, Nm]
     - ``computeKL_discrete=True``
     - KL divergence D_KL(posterior ∥ prior) normalised to [0, 1] using log_base = Nclass (read from ``class_name`` attribute if available, otherwise from ``class_id``). 0 = posterior equals prior; 1 = completely certain about one class.


Compression in HDF5 files
=========================
 
All HDF5 files created by INTEGRATE use compression by default to reduce file sizes while maintaining reasonable I/O performance. The default compression settings are optimized based on extensive benchmarking:

- **Default**: ``gzip`` level 1 (provides 3.5× file size reduction, 78% faster than level 9)
- **Performance**: Write overhead of ~3× compared to no compression, but results in significantly smaller files
- **Customizable**: Compression can be configured per-function call or globally

Per-function Configuration
"""""""""""""""""""""""""""

Example usage::

    import integrate as ig

    # Use default compression (gzip level 1)
    ig.prior_model_layered(N=50000)

    # Custom compression level
    ig.prior_model_layered(N=50000, compression='gzip', compression_opts=4)

    # Fast LZF compression
    ig.prior_model_layered(N=50000, compression='lzf')

    # Disable compression for temporary files
    ig.prior_model_layered(N=50000, compression=False)

    # Same parameters work for data functions
    ig.save_data_gaussian(D_obs, compression='gzip', compression_opts=9)

Global Default Configuration
"""""""""""""""""""""""""""""

You can modify the module-wide compression defaults in ``integrate_io.py``::

    import integrate.integrate_io as io

    # Change global defaults (affects all subsequent saves)
    io.DEFAULT_COMPRESSION = 'lzf'        # Options: 'gzip', 'lzf', or None
    io.DEFAULT_COMPRESSION_OPTS = 4       # For gzip: 1-9 (1=fastest, 9=smallest)

    # Now all functions use the new defaults
    ig.prior_model_layered(N=50000)  # Will use lzf compression

Compression Performance Comparison
"""""""""""""""""""""""""""""""""""

Based on benchmarks with N=50,000 models:

+-----------------------------------+-------------+-----------+------------------------+
| Setting                           | Write Speed | File Size | Best For               |
+===================================+=============+===========+========================+
| ``compression=False``             | Fastest     | 19.7 MB   | Temporary files        |
+-----------------------------------+-------------+-----------+------------------------+
| ``compression='gzip'``            | Fast        | 5.6 MB    | **Default** (best      |
| ``compression_opts=1``            |             | (3.5×)    | balance)               |
+-----------------------------------+-------------+-----------+------------------------+
| ``compression='lzf'``             | Very Fast   | ~6-7 MB   | Speed-critical         |
|                                   |             | (2-3×)    | workflows              |
+-----------------------------------+-------------+-----------+------------------------+
| ``compression='gzip'``            | Medium      | 5.5 MB    | Good alternative       |
| ``compression_opts=4``            |             | (3.6×)    |                        |
+-----------------------------------+-------------+-----------+------------------------+
| ``compression='gzip'``            | Slowest     | 5.5 MB    | Long-term archival     |
| ``compression_opts=9``            | (1.51s)     | (3.6×)    | (diminishing returns)  |
+-----------------------------------+-------------+-----------+------------------------+

**Note**: The difference between gzip levels 1 and 9 is only ~2% in file size but 4.6× difference in write time. Level 1 is recommended for most use cases.

Functions Supporting Compression
"""""""""""""""""""""""""""""""""

All HDF5 write functions accept ``compression`` and ``compression_opts`` parameters:

- ``save_prior_model()`` - Model parameter arrays
- ``save_prior_data()`` - Forward-modeled data
- ``save_data_gaussian()`` - Observed data with Gaussian noise model
- ``save_data_multinomial()`` - Observed data with multinomial noise model
- ``prior_model_layered()`` - Passes compression settings to internal saves



A typical workflow
==================

1. Setup DATA.h5
   
   * Store the observed data and its associated uncertainty in DATA.h5

2. Setup FORWARD.h5

   * Define the forward problem for data type A in FORWARD_A.h5.
   * Define the forward problem for data type B in FORWARD_B.h5.

3. Setup PRIOR.h5

   * Generate prior model realizations of model parameter 1 in in /M1
   * Generate prior model realizations of model parameter 2 in in /M2
   * Use FORWARD_A.h5 to compute prior data of the prior realizations for data type A
   * Use FORWARD_A.h5 to compute prior data of the prior realizations for data type B
  
4. Sample the posterior and output POST.h5

5. Update POST.h5 with some statistics computed from the posterior.
