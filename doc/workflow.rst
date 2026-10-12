=========
Workflows
=========

An INTEGRATE inversion has five stages. Each stage reads and writes HDF5 files
(layout in :doc:`format`).

1. **Prior models** (``/M1``, ``/M2``, ...): generate layered realizations with
   ``prior_model_layered``. See :doc:`gettingstarted`.
2. **Prior data** (``/D<id>``): forward-model the prior with ``prior_data_em``,
   or use identity or borehole data. See :doc:`prior_data`.
3. **Observed data**: store the data and its uncertainty in ``DATA.h5``. See
   :doc:`format`.
4. **Rejection sampling and posterior statistics**: ``integrate_rejection``
   writes ``POST.h5``, then ``integrate_posterior_stats`` adds statistics. See
   :doc:`rejection`.
5. **Plots**: see :doc:`plotting`.

For the step-by-step tutorial with code, see :doc:`gettingstarted`.

Examples
--------

Runnable versions of this workflow are in the example gallery:

* :doc:`Complete workflow <auto_examples/20_workflow/integrate_workflow>`
* :doc:`Getting started <auto_examples/10_getting_started/integrate_getting_started>`
* :doc:`Synthetic case with a known true model <auto_examples/70_synthetic/integrate_synthetic_case>`
* :doc:`Merging several survey files <auto_examples/90_other/integrate_merge_data>`
