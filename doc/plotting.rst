.. _plotting:

Plotting Profiles
=================

:func:`integrate.integrate_plot.plot_profile` plots posterior results along a
profile (sounding index, data ID, or X/Y coordinate). It reads the posterior
HDF5 file, computes the posterior statistics if they are missing, and
dispatches to the discrete or continuous plotter depending on the model type.

.. code-block:: python

    import integrate as ig

    ig.plot_profile(f_post_h5, i1=1, i2=2000, im=1)

Model type and panels
---------------------

The panels shown depend on whether the model is continuous (for example
resistivity) or discrete (for example lithology class):

.. list-table::
   :header-rows: 1

   * - Model type
     - Panels
   * - Continuous
     - ``'value'`` (median/mean), ``'std'``, ``'stats'``
   * - Discrete
     - ``'mode'``, ``'entropy'``, ``'stats'``
   * - Both
     - ``'realization'`` (see below)

Select panels with ``panels=[...]``. The default ``panels=None`` shows all
panels. For example, to show only the median resistivity:

.. code-block:: python

    ig.plot_profile(f_post_h5, im=1, panels=['value'])

Plot all models in the file with ``im=0``. Each model is plotted in turn; an
error for one model is printed and the others are still plotted.

Posterior realizations
----------------------

The ``'realization'`` panel replaces the first panel with one realization per
location instead of a statistic. By default a random entry from ``/i_use`` is
drawn for each sounding. Use ``seed`` for a reproducible draw, or
``plot_prior=True`` to draw from the prior instead. ``i_plot_realization``
(an array of prior indices, one per location) selects specific realizations.

.. code-block:: python

    ig.plot_profile(f_post_h5, i1=1, i2=2000, im=1, panels=['realization'], seed=1)
    ig.plot_profile(f_post_h5, i1=1, i2=2000, im=1, panels=['realization'], seed=1, plot_prior=True)

Selecting data points
---------------------

- ``i1`` and ``i2`` give the first and last data point (1-based, inclusive).
  ``i2=1e+9`` means "to the end".
- ``ii`` is an explicit array of indices. It overrides ``i1`` and ``i2``.

X-axis
------

``xaxis`` sets what is plotted along the horizontal axis:

- ``'index'`` (default): sequential index, 0, 1, 2, ...
- ``'id'``: data point ID from the file
- ``'x'``: X coordinate (easting)
- ``'y'``: Y coordinate (northing)

.. code-block:: python

    ig.plot_profile(f_post_h5, im=1, xaxis='x')

Gaps and transparency
---------------------

- ``gap_threshold`` (float): where the distance between consecutive data
  points exceeds this value, the region is drawn transparent. ``None``
  (default) applies no gap handling.
- ``alpha`` (0.0 to 1.0): applies uncertainty-based transparency to the main
  panel (median/mean for continuous models, mode for discrete models).

.. code-block:: python

    ig.plot_profile(f_post_h5, im=1, alpha=0.5)

Titles and output files
-----------------------

- ``title`` adds a figure title above all panels.
- ``hardcopy=True`` saves a PNG. ``f_png`` sets the file name; otherwise it is
  generated automatically, with ``txt`` as an optional suffix.

Borehole overlay
----------------

Boreholes near the profile can be drawn on the mode, value, or realization
panel. ``BHOLES`` accepts a borehole dictionary, a list of dictionaries from
:func:`integrate.integrate_io.read_borehole`, or a JSON path.

- ``bhole_max_dist`` (default 200): include boreholes within this distance of the profile
- ``bhole_width`` (default 10): column width in pixels
- ``bhole_im``: model to use for the borehole colours
- ``bhole_label``, ``bhole_alpha``: label and transparency

.. code-block:: python

    BHOLES = ig.read_borehole('boreholes.json')
    ig.plot_profile(f_post_h5, im=2, xaxis='x', BHOLES=BHOLES, bhole_max_dist=150, bhole_width=12)

Functions
---------

- :func:`integrate.integrate_plot.plot_profile` dispatches on model type.
- :func:`integrate.integrate_plot.plot_profile_continuous` and
  :func:`integrate.integrate_plot.plot_profile_discrete` take the same
  arguments and expose the full set of panel options.

Examples
--------

See the example gallery:

* :doc:`Profile visualization <auto_examples/80_plotting/integrate_profiles>`
* :doc:`Borehole overlay on profiles <auto_examples/30_data/integrate_boreholes>`
* :doc:`All plotting examples <auto_examples/80_plotting/index>`
