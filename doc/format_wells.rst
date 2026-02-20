.. _format_wells:

Well Log Data Format
====================

Overview
--------

INTEGRATE supports integration of well log (borehole) data with geophysical surveys such as electromagnetic (EM) data. Well logs provide direct observations of subsurface lithology at discrete depth intervals, which can be combined with spatially extensive geophysical data to improve characterization across the entire survey area.

This document describes:

1. How well log data is structured and stored
2. The workflow for incorporating wells into probabilistic inversion
3. Distance-based weighting for spatial extrapolation
4. File formats and data structures

Core Functionality
------------------

Well log handling is implemented in the ``integrate_borehole`` module with the following key functions:

* ``compute_P_obs_discrete()``: Compute observation probability from discrete lithology intervals
* ``compute_P_obs_sparse()``: Extract mode lithology from prior models and compute probabilities
* ``rescale_P_obs_temperature()``: Apply temperature annealing for distance-based weighting
* ``Pobs_to_datagrid()``: Extrapolate point observations to survey grid with distance weighting
* ``get_weight_from_position()``: Calculate spatial and data-similarity weights

Well Data Structure
-------------------

Well Dictionary Format
~~~~~~~~~~~~~~~~~~~~~~

Wells are represented as Python dictionaries containing lithology observations and spatial coordinates:

.. code-block:: python

    W = {
        'depth_top':    [0, 8, 12, 16, 34],           # Top depths (m)
        'depth_bottom': [8, 12, 16, 28, 36],           # Bottom depths (m)
        'class_obs':    [1, 2, 1, 5, 4],               # Lithology class IDs
        'class_prob':   [0.9, 0.9, 0.9, 0.9, 0.9],    # Confidence (0-1)
        'X': 498832.5,                                  # UTM Easting
        'Y': 6250843.1,                                 # UTM Northing
        'name': '65.795',                               # Well identifier
        'method': 'mode_probability'                    # Integration method
    }

**Field Descriptions:**

``depth_top``
    Array of top depths for each lithology interval (meters below surface). Must be strictly increasing.

``depth_bottom``
    Array of bottom depths for each lithology interval (meters below surface). Must match length of ``depth_top``.

``class_obs``
    Array of observed lithology class IDs corresponding to each depth interval. Values must match ``class_id`` defined in prior model.

``class_prob``
    Confidence level (0-1) for each observation. Can be:

    * Scalar: applies same confidence to all intervals
    * Array: per-interval confidence specification
    * Default: 0.9 (high confidence)

``X``, ``Y``
    Spatial coordinates in UTM projection (meters). Used for distance-based weighting.

``name``
    Optional string identifier for the well.

``method``
    Integration method. Options: ``'mode_probability'``, ``'layer_probability'``,
    ``'class_exact'``, ``'layer_probability_independent'``. Default is ``'mode_probability'``.

JSON File Format
~~~~~~~~~~~~~~~~

Borehole dictionaries can be saved to and loaded from human-readable JSON files
using ``ig.write_borehole()`` and ``ig.read_borehole()``. A JSON file may contain
either a **single borehole** (JSON object) or **many boreholes** (JSON array).

Single borehole file (``borehole_65.795.json``):

.. code-block:: json

    {
      "name": "65.795",
      "X": 498832.5,
      "Y": 6250843.1,
      "depth_top":    [0, 8, 12, 16, 34],
      "depth_bottom": [8, 12, 16, 28, 36],
      "class_obs":    [1, 2, 1, 5, 4],
      "class_prob":   [0.9, 0.9, 0.9, 0.9, 0.9],
      "method": "mode_probability"
    }

Multi-borehole file (``all_boreholes.json``):

.. code-block:: json

    [
      {
        "name": "65.795",
        "X": 498832.5,
        "Y": 6250843.1,
        "depth_top":    [0, 8, 12, 16, 34],
        "depth_bottom": [8, 12, 16, 28, 36],
        "class_obs":    [1, 2, 1, 5, 4],
        "class_prob":   [0.9, 0.9, 0.9, 0.9, 0.9],
        "method": "mode_probability"
      },
      {
        "name": "65.732",
        "X": 499100.0,
        "Y": 6251200.0,
        "depth_top":    [0, 5, 15, 25],
        "depth_bottom": [5, 15, 25, 40],
        "class_obs":    [2, 1, 3, 2],
        "class_prob":   [0.8, 0.8, 0.8, 0.8],
        "method": "mode_probability"
      }
    ]

``ig.read_borehole()`` automatically detects the format: it returns a single dict
for a single-borehole file, or a list of dicts for a multi-borehole file.

numpy arrays stored in the dict are automatically converted to plain Python lists
when writing, so the file is fully human-readable and editable in any text editor.

Writing and reading boreholes:

.. code-block:: python

    import integrate as ig

    # Write a single borehole
    ig.write_borehole(W, 'borehole_65.795.json', showInfo=1)

    # Write a list of boreholes (all in one file)
    ig.write_borehole([W1, W2, W3], 'all_boreholes.json', showInfo=1)

    # Read back a single borehole
    W = ig.read_borehole('borehole_65.795.json', showInfo=1)

    # Read back all boreholes
    WELLS = ig.read_borehole('all_boreholes.json', showInfo=1)
    for W in WELLS:
        print(W['name'], W['X'], W['Y'])

Visualising boreholes:

.. code-block:: python

    # Plot lithology sticks (one subplot per borehole)
    ig.plot_boreholes(WELLS)

    # With class names and colours from the prior HDF5 file
    ig.plot_boreholes(WELLS, f_prior_h5='PRIOR.h5')

    # Load directly from a JSON file
    ig.plot_boreholes('all_boreholes.json', f_prior_h5='PRIOR.h5')

HDF5 File Structure
-------------------

Prior File (``f_prior_h5``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The prior file contains lithology models and their spatial discretization:

.. code-block:: none

    f_prior_h5
    ├── /M1                           # Dense parameters (e.g., resistivity)
    ├── /M2                           # Lithology models (discrete)
    │   ├── shape: (N_realizations, N_depth_points)
    │   └── attributes:
    │       ├── 'x'          → depth array [m]
    │       ├── 'class_id'   → [0, 1, 2, ...] lithology identifiers
    │       ├── 'class_name' → ['sand', 'clay', 'gravel', ...]
    │       └── 'cmap'       → colormap for visualization
    ├── /D1, /D2, ...                 # Forward modeled data (e.g., tTEM)
    └── /D3, /D4, ...                 # Well lithology prior data
        ├── shape: (N_realizations, N_intervals)
        └── stores mode lithology for each well interval

**Key Points:**

* ``/M2`` contains the lithology models sampled from the prior distribution
* ``M2.attrs['x']`` provides the depth discretization (uniform spacing)
* ``M2.attrs['class_id']`` defines valid lithology class identifiers
* ``/D3, /D4, ...`` store extracted lithology mode for each well

Data File (``f_data_h5``)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The data file contains observed well log data and survey geometry:

.. code-block:: none

    f_data_h5
    ├── /D1                           # tTEM observed data
    │   ├── d_obs                     # Observed data values
    │   ├── d_std                     # Data uncertainties
    │   └── id_prior                  # Reference to /D1 in prior file
    ├── /D2, /D3, ...                 # Well log observations
    │   ├── d_obs                     # Probability matrix (nd × nclass × nm)
    │   ├── i_use                     # Binary mask (nd × 1)
    │   ├── id_prior                  # Reference to /D{id} in prior file
    │   └── noise_model               # 'multinomial'
    ├── /UTMX                         # Survey Easting coordinates
    ├── /UTMY                         # Survey Northing coordinates
    ├── /LINE                         # Survey line identifiers
    └── /ELEVATION                    # Ground surface elevation

**Data Array Dimensions:**

* ``d_obs``: shape (nd, nclass, nm) where:
    * nd = number of survey data points
    * nclass = number of lithology classes
    * nm = number of depth intervals per well
* ``i_use``: shape (nd, 1), binary mask (1 = use, 0 = ignore)
* ``id_prior``: scalar or array, references which prior data to compare against

Workflow
--------

Complete Workflow Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is a complete workflow for integrating well log data with electromagnetic data:

.. code-block:: python

    import integrate as ig
    import numpy as np

    # 1. Define well data
    W1 = {
        'depth_top': [0, 8, 12, 16, 34],
        'depth_bottom': [8, 12, 16, 28, 36],
        'class_obs': [1, 2, 1, 5, 4],
        'class_prob': 0.9,
        'X': 498832.5,
        'Y': 6250843.1,
        'name': 'Well_1',
        'method': 'mode_probability'
    }

    # 2. Load prior lithology models
    M, idx = ig.load_prior_model(f_prior_h5)
    M_lithology = M[1]  # Extract M2 (lithology)

    # Get depth array and class information
    z = M_lithology.attrs['x'][:]
    class_id = M_lithology.attrs['class_id']
    class_name = M_lithology.attrs['class_name']

    # 3. Compute observation probability from prior
    P_obs, lithology_mode = ig.compute_P_obs_sparse(
        M_lithology,
        depth_top=W1['depth_top'],
        depth_bottom=W1['depth_bottom'],
        class_obs=W1['class_obs'],
        z=z,
        class_id=class_id,
        class_prob=W1['class_prob'],
        parallel=True,
        Ncpu=-1
    )

    # 4. Save prior lithology data for this well
    id_prior = ig.save_prior_data(f_prior_h5, lithology_mode)

    # 5. Extrapolate to survey grid with distance weighting
    d_obs, i_use, T_use = ig.Pobs_to_datagrid(
        P_obs,
        W1['X'],
        W1['Y'],
        f_data_h5,
        r_data=10,      # Full strength within 10m
        r_dis=100,      # Fade to zero by 100m
        doPlot=False
    )

    # 6. Save observed well data
    id_well, f_data = ig.save_data_multinomial(
        D_obs=d_obs,
        i_use=i_use,
        id_prior=id_prior,
        f_data_h5=f_data_h5
    )

    # 7. Run joint inversion (tTEM + Well)
    f_post_h5 = ig.integrate_rejection(
        f_prior_data_h5,
        f_data_h5,
        id_use=[1, id_well],  # Combine tTEM and well data
        parallel=True
    )

Step-by-Step Explanation
~~~~~~~~~~~~~~~~~~~~~~~~~

Step 1: Define Well Data
^^^^^^^^^^^^^^^^^^^^^^^^^

Create a dictionary containing:

* Depth intervals (top/bottom)
* Observed lithology class IDs
* Confidence levels
* Spatial coordinates (UTM)

Step 2: Load Prior Lithology Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extract the lithology model array (``M2``) and its metadata:

* Depth discretization (``z``)
* Valid class identifiers (``class_id``)
* Class names for interpretation (``class_name``)

Step 3: Compute Observation Probability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two approaches are available:

**Approach A: Direct Discrete Observations**

.. code-block:: python

    P_obs = ig.compute_P_obs_discrete(
        depth_top=W['depth_top'],
        depth_bottom=W['depth_bottom'],
        class_obs=W['class_obs'],
        z=z,
        class_id=class_id,
        class_prob=W['class_prob']
    )

This creates a probability matrix where each observed interval has high probability (e.g., 0.9) for the observed class and low probability for other classes.

**Approach B: Mode from Prior (Recommended)**

.. code-block:: python

    P_obs, lithology_mode = ig.compute_P_obs_sparse(
        M_lithology,
        depth_top=W['depth_top'],
        depth_bottom=W['depth_bottom'],
        class_obs=W['class_obs'],
        z=z,
        class_id=class_id,
        class_prob=W['class_prob'],
        parallel=True
    )

This extracts the mode lithology from prior realizations within each depth interval, providing more realistic probability distributions that account for prior model variability.

**Output:**

* ``P_obs``: Probability matrix, shape (nclass, nm) where nm = number of intervals
* ``lithology_mode``: Mode lithology array, shape (N_realizations, nm)

Step 4: Save Prior Lithology Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Store the extracted lithology mode in the prior file:

.. code-block:: python

    id_prior = ig.save_prior_data(f_prior_h5, lithology_mode)

This creates a new dataset (``/D{id_prior}``) containing the well lithology data that can be referenced during inversion.

Step 5: Extrapolate to Survey Grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convert point-based well observations to gridded data using distance-based weighting:

.. code-block:: python

    d_obs, i_use, T_use = ig.Pobs_to_datagrid(
        P_obs,
        W['X'],
        W['Y'],
        f_data_h5,
        r_data=10,
        r_dis=100,
        doPlot=False
    )

**Parameters:**

``r_data``
    Inner radius (meters). Observations within this distance receive full strength (weight ≈ 1).

``r_dis``
    Outer radius (meters). Observations are weighted by distance from well, approaching zero at this distance.

**Distance Weighting Function:**

The weight decreases with distance using a Gaussian-like function:

.. math::

    w_{dis}(d) = \exp\left(-\frac{1}{2}\left(\frac{d - r_{data}}{r_{dis} - r_{data}}\right)^2\right)

This weight is converted to temperature for probability scaling:

.. math::

    T = \frac{1}{w_{dis}}

**Output:**

* ``d_obs``: Gridded probability data, shape (nd, nclass, nm)
* ``i_use``: Binary mask indicating which survey points are influenced

Step 6: Save Observed Well Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Store the gridded well data in the data file:

.. code-block:: python

    id_well, f_data = ig.save_data_multinomial(
        D_obs=d_obs,
        i_use=i_use,
        id_prior=id_prior,
        f_data_h5=f_data_h5
    )

This creates dataset ``/D{id_well}`` with the multinomial noise model.

Step 7: Run Joint Inversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Combine multiple data types in the inversion:

.. code-block:: python

    f_post_h5 = ig.integrate_rejection(
        f_prior_data_h5,
        f_data_h5,
        id_use=[1, id_well],  # e.g., [1, 3] for D1 (tTEM) + D3 (Well)
        parallel=True
    )

The ``id_use`` parameter specifies which observed datasets to include in the joint inversion.

Distance-Based Weighting
-------------------------

Spatial Weighting
~~~~~~~~~~~~~~~~~

Well observations influence survey points based on distance:

**Weight Calculation:**

.. code-block:: python

    w_combined, w_dis, w_data, i_ref = ig.get_weight_from_position(
        f_data_h5,
        x_well=W['X'],
        y_well=W['Y'],
        r_dis=300,
        r_data=2,
        doPlot=True
    )

**Weight Components:**

``w_dis``
    Distance-based weight (spatial proximity)

``w_data``
    Data-similarity weight (optional, requires reference data)

``w_combined``
    Combined weight: w_dis × w_data

**Temperature Annealing:**

Distance converts to temperature for probability scaling:

.. code-block:: python

    P_obs_scaled = ig.rescale_P_obs_temperature(P_obs, T=temperature)

* T = 1.0: No scaling (full confidence)
* T > 1.0: Flattens distribution (less confident)
* T >> 1.0: Approaches uniform distribution (observation ignored)

**Behavior by Distance:**

* d < r_data: T ≈ 1, full observation strength
* r_data < d < r_dis: T increases gradually
* d > r_dis: T >> 1, observation effectively ignored

Visualization
~~~~~~~~~~~~~

Visualize weight distribution:

.. code-block:: python

    w, _, _, _ = ig.get_weight_from_position(
        f_data_h5,
        x_well=W['X'],
        y_well=W['Y'],
        r_dis=300,
        doPlot=True
    )

This creates a map showing how well influence decreases with distance across the survey area.

Multiple Wells
--------------

Handling Multiple Boreholes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Process multiple wells sequentially:

.. code-block:: python

    # Define multiple wells
    wells = [W1, W2, W3]
    well_ids = []

    for i, W in enumerate(wells):
        # Compute P_obs
        P_obs, lithology_mode = ig.compute_P_obs_sparse(
            M_lithology, **W, z=z, class_id=class_id, parallel=True
        )

        # Save prior data
        id_prior = ig.save_prior_data(f_prior_h5, lithology_mode)

        # Extrapolate to grid
        d_obs, i_use, T_use = ig.Pobs_to_datagrid(
            P_obs, W['X'], W['Y'], f_data_h5,
            r_data=10, r_dis=100
        )

        # Save observed data
        id_well, _ = ig.save_data_multinomial(
            d_obs, i_use=i_use, id_prior=id_prior,
            f_data_h5=f_data_h5
        )

        well_ids.append(id_well)

    # Joint inversion with tTEM + all wells
    f_post_h5 = ig.integrate_rejection(
        f_prior_data_h5,
        f_data_h5,
        id_use=[1] + well_ids,  # e.g., [1, 2, 3, 4] for D1, D2, D3, D4
        parallel=True
    )

Overlapping Influence Zones
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When multiple wells have overlapping influence zones (r_dis), the inversion automatically handles this through the multinomial noise model. Each well observation is treated as an independent constraint, and the posterior distribution reflects the combined information from all wells and geophysical data.

Performance Considerations
--------------------------

Parallel Processing
~~~~~~~~~~~~~~~~~~~

For large prior ensembles (N > 100,000), enable parallel processing:

.. code-block:: python

    P_obs, lithology_mode = ig.compute_P_obs_sparse(
        M_lithology,
        ...,
        parallel=True,
        Ncpu=-1  # Auto-detect all available CPUs
    )

**Speedup:**

* 1 core: baseline
* 4 cores: ~3-4× faster
* 8 cores: ~6-8× faster

Memory usage scales with number of processes.

Optimization Tips
~~~~~~~~~~~~~~~~~

**Reduce Prior Ensemble Size:**

If memory is limited, subsample the prior:

.. code-block:: python

    # Use random subset
    N_use = 50000
    idx = np.random.choice(M_lithology.shape[0], N_use, replace=False)
    M_subset = M_lithology[idx, :]

**Adjust Distance Parameters:**

Smaller ``r_dis`` reduces the area influenced by wells:

* Faster processing (fewer survey points affected)
* Less memory required
* More localized well influence

Examples
--------

Complete Example: Haderup Survey
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the complete working example:

* **Script:** ``examples/integrate_haderup_wells.py``
* **Notebook:** ``examples/integrate_haderup_wells.ipynb``

This example demonstrates:

* Loading tTEM survey data
* Defining two borehole observations
* Computing lithology probabilities from prior
* Extrapolating wells to survey grid
* Running joint inversion (tTEM + 2 wells)
* Visualizing results

Key code sections:

.. code-block:: python

    # Lines 107-131: Define well dictionaries
    W1 = {...}
    W2 = {...}

    # Lines 220-230: Read lithology metadata
    z = f['/M2'].attrs['x'][:]
    class_id = f['/M2'].attrs['class_id']
    class_name = f['/M2'].attrs['class_name']

    # Lines 275-288: Compute P_obs
    P_obs_1, lithology_mode_1 = ig.compute_P_obs_sparse(
        M[1], **W1, z=z, class_id=class_id, parallel=True
    )

    # Lines 295-315: Save and extrapolate
    id_prior_1 = ig.save_prior_data(f_prior_h5, lithology_mode_1)
    d_obs_1, i_use_1, T_use_1 = ig.Pobs_to_datagrid(
        P_obs_1, W1['X'], W1['Y'], f_data_h5, r_dis=100
    )

API Reference
-------------

Quick Reference
~~~~~~~~~~~~~~~

**Core Functions:**

.. code-block:: python

    from integrate import (
        compute_P_obs_discrete,      # Direct probability from observations
        compute_P_obs_sparse,         # Extract mode from prior realizations
        rescale_P_obs_temperature,    # Temperature-based scaling
        Pobs_to_datagrid,             # Extrapolate to survey grid
        get_weight_from_position,     # Calculate spatial weights
        save_prior_data,              # Save lithology to prior file
        save_data_multinomial,        # Save discrete observations
        load_prior_model              # Load lithology models
    )

**Function Aliases:**

* ``Pobs_discrete_compute`` → ``compute_P_obs_discrete``
* ``Pobs_rescale_temperature`` → ``rescale_P_obs_temperature``

See Also
--------

* :doc:`format` - General data format specifications
* :doc:`workflow` - Complete inversion workflow
* :doc:`notebooks` - Jupyter notebook examples

References
----------

For more information on the theoretical background:

* Hansen et al. (2021): Localized rejection sampling for Bayesian inversion
* Madsen et al. (2023): Probabilistic lithology modeling

