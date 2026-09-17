.. _prior_data:

Prior Data Generation
=======================

Overview
--------

Before running :doc:`rejection sampling <rejection>`, every prior realization
in a prior HDF5 file (``/M<im>`` -- see :doc:`format`) needs a matching
``/D<id>`` dataset: the data that would be observed for that realization,
against which the *actually observed* data (in a data HDF5 file) is later
compared. Three functions cover the ways ``/D<id>`` gets created:

* :func:`ig.prior_data_em` -- forward-model EM (TDEM) data from a resistivity
  model, e.g. ``/M1`` (resistivity) -> ``/D<id>``.
* :func:`ig.prior_data_borehole` -- turn a borehole log into class-probability
  prior data conditioned on a discrete model, e.g. ``/M2`` (lithology) ->
  ``/D<id>``.
* :func:`ig.prior_data_identity` -- copy a model parameter directly into
  ``/D<id>`` unchanged (identity mapping), used e.g. to compare/condition
  directly on model values rather than forward-modelled data.

All three add a new ``/D<id>`` dataset to the prior HDF5 file and return the
id(s) used, which then feed into ``id_use``/``id_prior`` of
:func:`ig.integrate_rejection`.

``prior_data_em`` -- EM forward modelling
-------------------------------------------

.. code-block:: python

    import integrate as ig

    f_prior_data_h5 = ig.prior_data_em(f_prior_h5, file_gex='system.gex')

``prior_data_em`` is a thin dispatcher: it forward-models ``/M<im>`` through
one of the supported EM backends and writes the result as ``/D<id>``. The
backend is chosen by ``method=``, or, if not given, by the
``EM_FORWARD_METHOD`` environment variable, falling back to ``'ga-aem'``.

.. important::

    **Only ``method='ga-aem'`` is currently a fully supported, production
    backend.** ``method='anemone'`` (a PyTorch-based forward model, see
    :mod:`integrate.anemone_forward`) is **integration in development**: it
    requires a local, unpublished checkout of the ``anemone`` package
    (``pip install -e /path/to/anemone``) and has not yet been validated to
    the same standard as GA-AEM. Do not rely on it for production results
    yet -- it is provided for early testing only.

``method='simpeg'`` is also available (:mod:`integrate.simpeg_forward`,
wrapping SimPEG's ``Simulation1DLayered``); see :doc:`format` and
``SIMPEG_VS_GAAEM.md`` for its validation against GA-AEM/AarhusInv. Treat it,
like ``anemone``, as newer and less battle-tested than the ``ga-aem``
default.

.. code-block:: python

    # Default backend (ga-aem) -- recommended for production use
    ig.prior_data_em(f_prior_h5, file_gex='system.gex')
    ig.prior_data_em(f_prior_h5, file_gex='system.gex', method='ga-aem')

    # Experimental / in development -- do not use for production results
    ig.prior_data_em(f_prior_h5, file_gex='system.gex', method='anemone')
    ig.prior_data_em(f_prior_h5, file_gex='system.gex', method='anemone', device='cuda')
    ig.prior_data_em(f_prior_h5, file_gex='system.gex', method='simpeg')

For ``method='anemone'``, ``device=`` selects the Torch device (``'cpu'`` or
``'cuda'``); if not given, the ``EM_FORWARD_DEVICE`` environment variable is
used, falling back to ``'cpu'``. It is ignored for ``'ga-aem'`` and
``'simpeg'``. This is the same environment-variable pattern used by
``EM_FORWARD_METHOD`` above -- an explicit keyword argument always overrides
the environment variable, which in turn overrides the built-in default.

If ``method='anemone'`` or ``method='simpeg'`` is requested but the
corresponding package is not installed, ``prior_data_em`` raises
``ImportError`` naming the required ``pip install`` command.

``prior_data_borehole`` -- well-log conditioning
----------------------------------------------------

.. code-block:: python

    P_obs, id_prior = ig.prior_data_borehole(f_prior_h5, im_prior=2, BH=BH, parallel=True)

``prior_data_borehole`` dispatches on ``BH['method']`` (default
``'mode_probability'``):

* ``'mode_probability'`` (recommended -- fast and robust) ->
  :func:`ig.prior_data_borehole_class_mode`: extracts the most frequent
  lithology class per observed depth interval, across all prior realizations.
* ``'layer_probability'`` -> :func:`ig.prior_data_borehole_class_layer`: a
  direct layer-probability approach using :func:`ig.prior_data_identity`
  internally (no per-realization mode extraction).
* ``'class_exact'`` / ``'layer_probability_independent'`` -- not yet
  implemented; raises ``NotImplementedError``.

The full ``BH`` borehole dictionary format (``depth_top``, ``depth_bottom``,
``class_obs``, ``class_prob``, ``X``, ``Y``, ``name``, ``method``, ...), the
distance-weighted extrapolation to the survey grid
(:func:`ig.Pobs_to_datagrid`), and the one-call
:func:`ig.save_borehole_data` wrapper are documented in full in
:doc:`format_wells` -- this section only covers the ``prior_data_borehole``
entry point itself.

``prior_data_identity`` -- identity mapping
-----------------------------------------------

.. code-block:: python

    f_prior_data_h5, id = ig.prior_data_identity(f_prior_h5, im=1)

Copies ``/M<im>`` directly into a new ``/D<id>`` dataset, unchanged -- no
forward modelling. Used internally by
:func:`ig.prior_data_borehole_class_layer`, and directly whenever you need to
condition on/compare against a model parameter's own values (e.g. a resistivity
or class-id log) rather than simulated data.

Key behaviour:

* ``id=0`` (default) auto-assigns the next free ``/D<id>`` slot in the file.
* ``doMakePriorCopy=True`` writes to a *new* HDF5 file (name derived from
  ``f_prior_h5``, ``im`` and ``id``) instead of modifying ``f_prior_h5`` in
  place; ``N`` optionally truncates to the first ``N`` realizations in that
  copy.
* If ``/D<id>`` already exists, it is deleted and overwritten by default
  (``forceDeleteExisting=True``); pass ``forceDeleteExisting=False`` to leave
  it untouched and return early instead.

Returns ``(f_prior_data_h5, id)`` -- the (possibly copied) file path and the
data id actually used.

See also
--------

* :doc:`format` -- PRIOR.h5 layout, ``/M<im>``/``/D<id>`` conventions, and the
  ``TDEM``/``GA-AEM``/``SimPEG`` forward-model types.
* :doc:`format_wells` -- full borehole (``BH``) dictionary format and
  well-log integration workflow.
* :doc:`rejection` -- what happens next: running rejection sampling against
  the ``/D<id>`` datasets created here.
