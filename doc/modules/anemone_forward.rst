anemone_forward
===============

anemone (PyTorch) 1D TDEM forward modelling backend, the default EM forward
in INTEGRATE. Provided by the ``anemone-core`` package
(`PyPI <https://pypi.org/project/anemone-core/>`_,
`GitHub <https://github.com/hydrogeophysicsgroup/anemone>`_), a core
dependency; see :doc:`../install`.

anemone setups (one per flight height bin for a varying flight height) are
cached between calls; on a GPU the cache is bounded by free GPU memory. See
the Notes of :func:`integrate.anemone_forward.forward_anemone` and
:doc:`../prior_data`.

.. automodule:: integrate.anemone_forward
   :members:
   :undoc-members:
   :show-inheritance:
