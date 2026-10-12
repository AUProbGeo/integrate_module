Command-line tools
==================

INTEGRATE installs four commands. Run any of them with ``--help`` for the full options.

integrate_rejection
-------------------

Run rejection sampling from the command line. See :doc:`rejection` for the options
and backends.

::

    integrate_rejection --prior PRIOR.h5 --data DATA.h5 --output POST.h5 --samples 1000000

hdf5_info
---------

Print the structure of INTEGRATE HDF5 files (datasets, attributes and shapes).

::

    hdf5_info PRIOR.h5 POST.h5
    hdf5_info -d PRIOR.h5      # also load the data

Options: ``-d``/``--load-data`` loads the data arrays, ``-q``/``--quiet`` reduces the output.

integrate_timing
----------------

Benchmark the INTEGRATE workflow, or plot saved benchmark results.

::

    integrate_timing time small --backend numpy --forward ga-aem
    integrate_timing plot timing_results.npz

- ``time SIZE`` runs the benchmark for ``small``, ``medium`` or ``large``.
  Options include ``--backend numpy|jax``, ``--forward ga-aem|anemone|simpeg``,
  ``--device``, ``--Ncpu`` and ``--N``.
- ``plot FILE`` plots the results in an ``.npz`` file. ``--all`` plots everything,
  ``--no-summary`` skips the summary.

integrate (web frontend)
------------------------

Start the INTEGRATE web interface in a browser.

::

    integrate --workspace /path/to/project_folder

By default the app runs at http://127.0.0.1:8051. Use ``--host`` and ``--port`` to
change the address. ``--workspace`` sets the folder with the project HDF5 files
(default: the current folder).

Examples
--------

The gallery example for ``integrate_timing``:

* :doc:`Timing benchmark in the gallery <auto_examples/90_other/integrate_timing_example>`
