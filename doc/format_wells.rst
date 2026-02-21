.. _format_wells:

Borehole Data Format
====================

Overview
--------

INTEGRATE supports integration of borehole (well log) data with geophysical surveys such as electromagnetic (EM) data. Boreholes provide direct observations of subsurface lithology at discrete depth intervals, which can be combined with spatially extensive geophysical data to improve characterization across the entire survey area.

This document describes:

1. How borehole data is structured and stored
2. The workflow for incorporating boreholes into probabilistic inversion
3. Distance-based weighting for spatial extrapolation
4. File formats and data structures

Core Functionality
------------------

Borehole handling is implemented in the ``integrate_borehole`` module with the following key functions:

* ``prior_data_borehole()``: Compute and save prior borehole data (dispatcher — recommended entry point)
* ``prior_data_borehole_class_mode()``: Prior data using mode class extraction from prior realizations
* ``prior_data_borehole_class_layer()``: Prior data using direct layer-probability approach
* ``save_borehole_data()``: **One-call wrapper** — compute and save both prior and observed data for a borehole
* ``compute_P_obs_discrete()``: Low-level: compute observation probability from discrete lithology intervals
* ``rescale_P_obs_temperature()``: Apply temperature annealing for distance-based weighting
* ``Pobs_to_datagrid()``: Extrapolate point observations to survey grid with distance weighting
* ``get_weight_from_position()``: Calculate spatial and data-similarity weights

Borehole Data Structure
-----------------------

Borehole Dictionary Format
~~~~~~~~~~~~~~~~~~~~~~~~~~

Boreholes are represented as Python dictionaries containing lithology observations and spatial coordinates:

.. code-block:: python

    BH = {
        'depth_top':    [0, 8, 12, 16, 34],           # Top depths (m)
        'depth_bottom': [8, 12, 16, 28, 36],           # Bottom depths (m)
        'class_obs':    [1, 2, 1, 5, 4],               # Lithology class IDs
        'class_prob':   [0.9, 0.9, 0.9, 0.9, 0.9],    # Confidence (0-1)
        'X': 498832.5,                                  # UTM Easting
        'Y': 6250843.1,                                 # UTM Northing
        'name': '65.795',                               # Borehole identifier
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
    Optional string identifier for the borehole.

``method``
    Integration method. Options: ``'mode_probability'`` (recommended), ``'layer_probability'``.

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
    ig.write_borehole(BH, 'borehole_65.795.json', showInfo=1)

    # Write a list of boreholes (all in one file)
    ig.write_borehole([BH1, BH2, BH3], 'all_boreholes.json', showInfo=1)

    # Read back a single borehole
    BH = ig.read_borehole('borehole_65.795.json', showInfo=1)

    # Read back all boreholes
    BHOLES = ig.read_borehole('all_boreholes.json', showInfo=1)
    for BH in BHOLES:
        print(BH['name'], BH['X'], BH['Y'])

Visualising boreholes:

.. code-block:: python

    # Plot lithology sticks (one subplot per borehole)
    ig.plot_boreholes(BHOLES)

    # With class names and colours from the prior HDF5 file
    ig.plot_boreholes(BHOLES, f_prior_h5='PRIOR.h5')

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
    └── /D3, /D4, ...                 # Borehole lithology prior data
        ├── shape: (N_realizations, N_intervals)
        └── stores mode lithology for each borehole interval

**Key Points:**

* ``/M2`` contains the lithology models sampled from the prior distribution
* ``M2.attrs['x']`` provides the depth discretization (uniform spacing)
* ``M2.attrs['class_id']`` defines valid lithology class identifiers
* ``/D3, /D4, ...`` store extracted lithology mode for each borehole

Data File (``f_data_h5``)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The data file contains observed borehole data and survey geometry:

.. code-block:: none

    f_data_h5
    ├── /D1                           # tTEM observed data
    │   ├── d_obs                     # Observed data values
    │   ├── d_std                     # Data uncertainties
    │   └── id_prior                  # Reference to /D1 in prior file
    ├── /D2, /D3, ...                 # Borehole observations
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
    * nm = number of depth intervals per borehole
* ``i_use``: shape (nd, 1), binary mask (1 = use, 0 = ignore)
* ``id_prior``: scalar or array, references which prior data to compare against

Workflow
--------

Complete Workflow Example
~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended workflow uses ``ig.save_borehole_data()`` to handle all borehole
processing in a single call per borehole:

.. code-block:: python

    import integrate as ig

    # 1. Define boreholes
    BH1 = {
        'depth_top':    [0, 8, 12, 16, 34],
        'depth_bottom': [8, 12, 16, 28, 36],
        'class_obs':    [1, 2, 1, 5, 4],
        'class_prob':   0.9,
        'X': 498832.5,
        'Y': 6250843.1,
        'name': 'BH_1',
        'method': 'mode_probability'
    }
    BH2 = {
        'depth_top':    [0, 5, 15, 25],
        'depth_bottom': [5, 15, 25, 40],
        'class_obs':    [2, 1, 3, 2],
        'class_prob':   0.8,
        'X': 499100.0,
        'Y': 6251200.0,
        'name': 'BH_2',
        'method': 'mode_probability'
    }
    BHOLES = [BH1, BH2]

    # 2. Process all boreholes — one call per borehole
    im_prior = 2     # index of lithology model parameter (M2)
    r_data   = 2     # full-strength radius (m)
    r_dis    = 300   # fade-out radius (m)

    id_borehole_list = []
    for BH in BHOLES:
        id_prior, id_out = ig.save_borehole_data(
            f_prior_h5, f_data_h5, BH,
            im_prior=im_prior, r_data=r_data, r_dis=r_dis,
            parallel=False, showInfo=1)
        id_borehole_list.append(id_out)

    # 3. Run joint inversion (tTEM + boreholes)
    f_post_h5 = ig.integrate_rejection(
        f_prior_data_h5,
        f_data_h5,
        id_use=[1] + id_borehole_list,  # e.g. [1, 2, 3]
        parallel=True
    )

``save_borehole_data`` internally performs these three steps automatically:

1. **Compute prior borehole data** — calls ``prior_data_borehole()`` which extracts
   mode lithology from prior realizations and saves it to ``f_prior_h5``
2. **Extrapolate to survey grid** — calls ``Pobs_to_datagrid()`` to spatially
   distribute the point observation with distance-based weighting
3. **Save observed data** — calls ``save_data_multinomial()`` to write the gridded
   observations to ``f_data_h5``

``save_borehole_data`` Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    id_prior, id_out = ig.save_borehole_data(
        f_prior_h5,          # Path to prior HDF5 file
        f_data_h5,           # Path to observed-data HDF5 file
        BH,                  # Borehole dictionary
        im_prior=2,          # Model parameter index (e.g. 2 → /M2)
        parallel=False,      # Parallel mode extraction
        r_data=2,            # Full-strength radius (m)
        r_dis=300,           # Fade-out radius (m)
        doPlot=False,        # Plot distance-weight maps
        showInfo=1           # Verbosity (0=silent, 1=summary line)
    )

**Returns:**

``id_prior``
    Dataset index of the new ``/D`` entry added to ``f_prior_h5``.

``id_out``
    Dataset index of the new ``/D`` entry added to ``f_data_h5``.
    Append to ``id_borehole_list`` for use in ``id_use`` during inversion.

Integration Methods
~~~~~~~~~~~~~~~~~~~~

``BH['method']`` controls how lithology observations are converted to probabilities:

``'mode_probability'`` (recommended)
    Extracts the mode lithology from prior realizations within each observed
    depth interval, providing realistic probability distributions that account
    for prior model variability. Uses ``prior_data_borehole_class_mode()``
    internally.

``'layer_probability'``
    Converts depth-interval observations directly to a probability matrix
    without querying the prior ensemble. Faster but less prior-aware. Uses
    ``prior_data_borehole_class_layer()`` internally.

Distance-Based Weighting
-------------------------

Spatial Weighting
~~~~~~~~~~~~~~~~~

Borehole observations influence survey points based on distance:

**Weight Calculation:**

.. code-block:: python

    w_combined, w_dis, w_data, i_ref = ig.get_weight_from_position(
        f_data_h5,
        x_well=BH['X'],
        y_well=BH['Y'],
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

**Distance Weighting Function:**

The weight decreases with distance using a Gaussian-like function:

.. math::

    w_{dis}(d) = \exp\left(-\frac{1}{2}\left(\frac{d - r_{data}}{r_{dis} - r_{data}}\right)^2\right)

This weight is converted to temperature for probability scaling:

.. math::

    T = \frac{1}{w_{dis}}

Visualization
~~~~~~~~~~~~~

Visualize weight distribution:

.. code-block:: python

    w, _, _, _ = ig.get_weight_from_position(
        f_data_h5,
        x_well=BH['X'],
        y_well=BH['Y'],
        r_dis=300,
        doPlot=True
    )

This creates a map showing how borehole influence decreases with distance across the survey area.

Multiple Boreholes
------------------

Processing Multiple Boreholes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``save_borehole_data()`` in a loop — one call per borehole:

.. code-block:: python

    BHOLES = [BH1, BH2, BH3]
    id_borehole_list = []

    for BH in BHOLES:
        id_prior, id_out = ig.save_borehole_data(
            f_prior_h5, f_data_h5, BH,
            im_prior=2, r_data=2, r_dis=300,
            showInfo=1)
        id_borehole_list.append(id_out)

    # Joint inversion: tTEM (D1) + all boreholes
    f_post_h5 = ig.integrate_rejection(
        f_prior_data_h5,
        f_data_h5,
        id_use=[1] + id_borehole_list,
        parallel=True
    )

Overlapping Influence Zones
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When multiple boreholes have overlapping influence zones (r_dis), the inversion automatically handles this through the multinomial noise model. Each borehole observation is treated as an independent constraint, and the posterior distribution reflects the combined information from all boreholes and geophysical data.

Performance Considerations
--------------------------

Parallel Processing
~~~~~~~~~~~~~~~~~~~

For large prior ensembles (N > 100,000), enable parallel processing in ``save_borehole_data``:

.. code-block:: python

    id_prior, id_out = ig.save_borehole_data(
        f_prior_h5, f_data_h5, BH,
        parallel=True   # Parallel mode class extraction
    )

**Speedup (mode_probability method):**

* 1 core: baseline
* 4 cores: ~3-4× faster
* 8 cores: ~6-8× faster

Memory usage scales with number of processes.

Optimization Tips
~~~~~~~~~~~~~~~~~

**Adjust Distance Parameters:**

Smaller ``r_dis`` reduces the area influenced by each borehole:

* Faster processing (fewer survey points affected)
* Less memory required
* More localized borehole influence

Examples
--------

Complete Example: Workflow Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the complete working example in ``examples/integrate_workflow.py``, which demonstrates:

* Defining borehole dictionaries
* Processing boreholes with ``ig.save_borehole_data()``
* Running joint inversion (tTEM + boreholes)
* Visualizing results

Key code section:

.. code-block:: python

    im_prior = 2     # lithology model index (M2)
    r_data   = 2     # full-strength radius (m)
    r_dis    = 300   # fade-out radius (m)

    id_borehole_list = []
    for BH in BHOLES:
        id_prior, id_out = ig.save_borehole_data(
            f_prior_h5, f_data_h5, BH,
            im_prior=im_prior, r_data=r_data, r_dis=r_dis,
            parallel=parallel, showInfo=1)
        id_borehole_list.append(id_out)

    f_post_h5 = ig.integrate_rejection(
        f_prior_data_h5, f_data_h5,
        id_use=[1] + id_borehole_list,
        parallel=True)

API Reference
-------------

Quick Reference
~~~~~~~~~~~~~~~

**High-level Functions (recommended):**

.. code-block:: python

    from integrate import (
        save_borehole_data,              # One-call: prior + observed data for a borehole
        prior_data_borehole,             # Dispatcher: compute and save prior borehole data
        prior_data_borehole_class_mode,  # Prior data via mode extraction from ensemble
        prior_data_borehole_class_layer, # Prior data via direct layer probability
    )

**Low-level / utility functions:**

.. code-block:: python

    from integrate import (
        compute_P_obs_discrete,          # Direct probability from observations
        rescale_P_obs_temperature,       # Temperature-based scaling
        Pobs_to_datagrid,                # Extrapolate to survey grid
        get_weight_from_position,        # Calculate spatial weights
        save_prior_data,                 # Save data to prior file
        save_data_multinomial,           # Save discrete observations to data file
        write_borehole,                  # Write borehole dict(s) to JSON
        read_borehole,                   # Read borehole dict(s) from JSON
        plot_boreholes,                  # Plot lithology sticks
    )

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
