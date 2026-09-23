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
one of the supported EM backends and writes the result as ``/D<id>``.
:func:`ig.forward_em` (forward response for a resistivity array, without
writing a file) uses exactly the same backend and device selection.

Backends
^^^^^^^^

* ``'ga-aem'`` -- GA-AEM (``gatdaem1d``), see :mod:`integrate.gaaem_forward`.
  **The only fully supported, production backend.**
* ``'anemone'`` -- PyTorch-based forward model with optional GPU support, see
  :mod:`integrate.anemone_forward`.
* ``'simpeg'`` -- SimPEG's ``Simulation1DLayered``, see
  :mod:`integrate.simpeg_forward`, :doc:`format` and ``SIMPEG_VS_GAAEM.md``
  for its validation against GA-AEM/AarhusInv.

.. important::

    ``method='anemone'`` is **integration in development**: it requires a
    local, unpublished checkout of the ``anemone`` package
    (``pip install -e /path/to/anemone``) and has not yet been validated to
    the same standard as GA-AEM. ``method='simpeg'`` is likewise newer and
    less battle-tested than ``ga-aem``. Do not rely on either for production
    results yet.

Choosing the backend
^^^^^^^^^^^^^^^^^^^^

The backend is resolved in this order:

1. the ``method=`` keyword (``'ga-aem'``/``'gaaem'``, ``'anemone'`` or
   ``'simpeg'``; case-insensitive);
2. the ``EM_FORWARD_METHOD`` environment variable (same values);
3. otherwise, auto-selection of the first *installed* backend, in the order
   ``'anemone'`` -> ``'ga-aem'`` -> ``'simpeg'``.

A backend counts as installed if its package imports: ``torch`` and
``anemone`` for anemone; ``gatdaem1d`` including its compiled library for
ga-aem (a broken GA-AEM install counts as unavailable); ``simpeg`` >= 0.22 for
simpeg. Call with ``showInfo=1`` to see which backend was used -- an
auto-selected one is reported as e.g.
``Using EM forward method: anemone (auto-selected)``.

Because auto-selection prefers ``anemone`` whenever it is installed, pass
``method='ga-aem'`` (or set ``EM_FORWARD_METHOD=ga-aem``) when you need the
production backend.

.. code-block:: python

    # Auto-selected backend (first installed of anemone, ga-aem, simpeg)
    ig.prior_data_em(f_prior_h5, file_gex='system.gex')

    # GA-AEM -- recommended for production use
    ig.prior_data_em(f_prior_h5, file_gex='system.gex', method='ga-aem')

    # Experimental / in development -- do not use for production results
    ig.prior_data_em(f_prior_h5, file_gex='system.gex', method='anemone')
    ig.prior_data_em(f_prior_h5, file_gex='system.gex', method='simpeg')

.. code-block:: bash

    # or select the backend once for a whole session / script
    export EM_FORWARD_METHOD=ga-aem

Choosing the device (anemone only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ``method='anemone'``, the Torch device is resolved in this order:

1. the ``device=`` keyword (any string ``torch.device`` accepts, e.g.
   ``'cpu'``, ``'cuda'``, ``'cuda:1'``, ``'mps'``);
2. the ``EM_FORWARD_DEVICE`` environment variable;
3. otherwise, auto-detection of the first usable device:

   a. ``'cuda'`` if ``torch.cuda.is_available()`` -- i.e. a CUDA-enabled
      torch build, an NVIDIA driver and a GPU are all present (merely having
      ``torch`` installed is not enough);
   b. ``'mps'`` (Apple Silicon GPU) if ``torch.backends.mps.is_available()``
      *and* the device accepts float64 tensors. anemone computes in float64,
      which MPS has so far not supported, so in practice Macs currently fall
      through to ``'cpu'``; this switches to ``'mps'`` automatically once
      float64 is supported;
   c. ``'cpu'``.

``device`` is ignored for ``'ga-aem'`` and ``'simpeg'``. The chosen device is
printed with ``showInfo=1`` and stored as the ``device`` attribute of the
written ``/D<id>`` dataset.

.. code-block:: python

    ig.prior_data_em(f_prior_h5, file_gex='system.gex', method='anemone')                 # auto: cuda -> mps -> cpu
    ig.prior_data_em(f_prior_h5, file_gex='system.gex', method='anemone', device='cpu')   # force CPU
    ig.prior_data_em(f_prior_h5, file_gex='system.gex', method='anemone', device='cuda')  # force GPU

An explicitly requested device is used as given, without any check: e.g.
``device='cuda'`` on a machine without a usable GPU, or ``device='mps'``
while MPS lacks float64, fails with the corresponding Torch error rather than
falling back to CPU.

Errors
^^^^^^

* An unknown ``method`` (keyword or ``EM_FORWARD_METHOD``) raises
  ``ValueError``.
* If the requested backend is not installed, ``ImportError`` is raised
  immediately, naming the required install command; there is **no silent
  fallback** to another backend.
* If no backend is requested and none of the three is installed, the
  ``ImportError`` lists the install hint for each.

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
