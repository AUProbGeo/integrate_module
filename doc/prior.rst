.. _prior:

Prior Models
============

Overview
--------

The first step of an INTEGRATE inversion is a sample of the prior: ``N``
realizations of the model parameters, written to a prior HDF5 file (layout in
:doc:`format`). INTEGRATE has four generators for 1D resistivity priors:

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Generator
     - Layer interfaces
     - Number of layers
     - Resistivity
   * - :func:`~integrate.integrate.prior_model_layered`
     - random depths
     - uniform or chi-squared distributed
     - independent per layer
   * - :func:`~integrate.integrate.prior_model_smooth`
     - fixed, thickening with depth
     - fixed (``nlayers``)
     - correlated with depth (L2 / Smooth)
   * - :func:`~integrate.integrate.prior_model_blocky`
     - fixed, thickening with depth
     - fixed (``nlayers``)
     - random walk with Laplace steps (L1 / Blocky)
   * - :func:`~integrate.integrate.prior_model_sharp`
     - random depths
     - Poisson distributed
     - independent per layer (MGS / Sharp)

``prior_model_smooth``, ``prior_model_blocky`` and ``prior_model_sharp`` give
sample ensembles that resemble the Smooth, Blocky and Sharp model types of the
Aarhus Workbench. ``prior_model_blocky`` and ``prior_model_sharp`` are
``prior_model_smooth`` with ``regularization='L1'`` and ``regularization='MGS'``.

Each generator returns the path of the prior file it wrote. Set the path with
``f_prior_h5=...``. Otherwise a name is made from the parameters.

Output
------

All four generators write the same three model parameters:

* ``/M1`` -- ``Resistivity``: each realization on a regular depth grid
  ``z = 0, dz, ..., z_max``. This is the parameter that
  :func:`integrate.integrate.prior_data_em` forward-models.
* ``/M2`` -- ``sparse - depth-resistivity``: each realization in the form it
  was generated in. The interface depths (number of layers - 1) come first,
  followed by one resistivity per layer. Rows are padded with NaN to the
  largest number of layers.
* ``/M3`` -- ``Number of layers`` of each realization.

``prior_model_layered(save_sparse=False)`` leaves out ``/M2``, and the number
of layers is then written to ``/M2``.

Resistivity is clipped to ``[RHO_min, RHO_max]``. Values that fall outside
this range pile up at the bounds.

``prior_model_layered`` -- random layers
----------------------------------------

The number of layers is drawn uniformly from ``[NLAY_min, NLAY_max]``
(``lay_dist='uniform'``) or as ``ceil`` of a chi-squared variable with
``NLAY_deg`` degrees of freedom (``lay_dist='chi2'``). The interfaces are
placed at random depths in ``[0, z_max]``. Each layer gets a resistivity from
``RHO_dist``:

* ``'log-uniform'`` (default) or ``'uniform'`` on ``[RHO_min, RHO_max]``
* ``'normal'``, with mean ``RHO_mean`` and standard deviation ``RHO_std``
* ``'lognormal'`` (or ``'log-normal'``), with median ``RHO_mean`` and
  log-standard deviation ``RHO_std / RHO_mean``

.. code-block:: python

    import integrate as ig

    f_prior_h5 = ig.prior_model_layered(N=100000, lay_dist='chi2', NLAY_deg=4,
                                        RHO_min=1, RHO_max=3000,
                                        f_prior_h5='PRIOR.h5')

Defaults: ``z_max=90``, ``dz=1``, ``RHO_min=0.1``, ``RHO_max=5000``. The
``compression`` and ``compression_opts`` keywords are passed on to the HDF5
writer (see :doc:`format`).

``prior_model_smooth`` -- Smooth (L2)
-------------------------------------

``nlayers`` layers (default 30) with tops at
``z1 + (z_max - z1) * linspace(0, 1, nlayers)**p``, so the layers get thicker
with depth. The interfaces are the same in all realizations. Log-resistivity
is a Gaussian process with mean ``log(RHO_ref)`` and exponential covariance
``sigma_logrho**2 * exp(-|dz| / corr_length)``. Use ``corr_length <= 0`` for
independent layers.

``RHO_dist`` controls the resistivity of each layer:

* ``'lognormal'`` (default): the Gaussian process above.
* ``'log-uniform'`` or ``'uniform'`` on ``[RHO_min, RHO_max]``: each layer has
  this distribution, and the depth correlation of the Gaussian process is
  kept (Gaussian copula).

.. code-block:: python

    f_prior_h5 = ig.prior_model_smooth(N=100000, nlayers=30, z_max=90,
                                       corr_length=15.0, sigma_logrho=0.25,
                                       RHO_ref=100.0, RHO_min=1, RHO_max=500)

The default ``sigma_logrho=0.25`` (natural log) gives a narrow prior: nearly
all resistivities lie within a factor of about 2 of ``RHO_ref``. Increase ``sigma_logrho``, or use
``RHO_dist='log-uniform'``, for a less informative prior.

``prior_model_blocky`` -- Blocky (L1)
-------------------------------------

The same layers as ``prior_model_smooth``. The top layer's resistivity is
log-uniform on ``[RHO_min, RHO_max]``. Below it, log-resistivity is a random
walk: from one layer to the next it changes by a Laplace-distributed step with
scale ``blocky_scale``. ``RHO_dist`` and ``RHO_ref`` are not used.

.. code-block:: python

    f_prior_h5 = ig.prior_model_blocky(N=100000, nlayers=30, z_max=90,
                                       blocky_scale=0.3, RHO_min=1, RHO_max=500)

``prior_model_sharp`` -- Sharp (MGS)
------------------------------------

A Poisson-distributed number of interfaces (mean ``n_jumps_mean``), placed at
random depths in ``[z1, z_max]``. Each layer gets a resistivity from
``RHO_dist`` (default ``'log-uniform'``; also ``'uniform'``, ``'normal'``,
``'lognormal'``).

.. code-block:: python

    f_prior_h5 = ig.prior_model_sharp(N=100000, z_max=90, n_jumps_mean=3.0,
                                      RHO_dist='log-uniform',
                                      RHO_min=1, RHO_max=500)

Checking a prior
----------------

Plot summary statistics and realizations of a prior with
:func:`integrate.integrate_plot.plot_prior_stats`:

.. code-block:: python

    ig.plot_prior_stats(f_prior_h5)

Next step
---------

Compute the data for each prior realization (``/D<id>``) as described in
:doc:`prior_data`.

Examples
--------

* :doc:`Prior model generation <auto_examples/90_other/integrate_priors>` --
  all four generators
* :doc:`Getting started <auto_examples/10_getting_started/integrate_getting_started>`
* :doc:`Merging prior files <auto_examples/90_other/integrate_merge_prior>`
