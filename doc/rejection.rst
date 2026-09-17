.. _rejection:

Rejection Sampling
===================

Overview
--------

``integrate_rejection`` performs probabilistic inversion by rejection
sampling: prior samples (and their pre-computed forward-modelled data,
stored in a prior HDF5 file) are compared against observed data (stored in
a data HDF5 file), and samples inconsistent with the observation -- within
a temperature-controlled tolerance -- are rejected. What remains is a
posterior ensemble of ``nr`` samples per data point, written to a posterior
HDF5 file.

It can be run either from Python, via :func:`ig.integrate_rejection`, or
from the command line, via the ``integrate_rejection`` script installed
with the package.

Running from Python
--------------------

Minimal call, using all defaults (``numpy`` backend, automatic temperature
estimation, one posterior file per prior file)::

    import integrate as ig

    f_post_h5 = ig.integrate_rejection('prior.h5', 'data.h5')

A more complete call, matching the options available on the command line::

    f_post_h5 = ig.integrate_rejection(
        f_prior_h5='prior.h5',
        f_data_h5='data.h5',
        f_post_h5='posterior.h5',
        N_use=1_000_000,          # max number of prior samples to use
        id_use=[1, 2],            # which /D<id> data sets to invert
        nr=1000,                  # posterior samples retained per data point
        autoT=1,                  # automatic temperature estimation
        Ncpu=4,                   # CPU cores (0 = auto-detect)
        parallel=True,
        backend='numpy',          # 'numpy' (default) or 'jax'
        showInfo=1,
    )

See the ``Parameters`` section of :func:`ig.integrate_rejection` (or
``help(ig.integrate_rejection)``) for the full, current list of keyword
arguments -- including less commonly used ones such as ``id_prior``,
``use_N_best``, ``T_N_above``/``T_P_acc_level``, and ``progress_callback``
for GUI integration.

Running from the command line
-------------------------------

The same functionality is exposed by the ``integrate_rejection`` console
script (installed as part of the package, see ``[project.scripts]`` in
``pyproject.toml``)::

    integrate_rejection --prior prior.h5 --data data.h5 --output post.h5

Common options::

    integrate_rejection --prior prior.h5 --data data.h5 \
        --samples 1000000 \
        --nr 1000 \
        --auto-temp \
        --cpus 4 \
        --backend numpy \
        --verbose

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Option
     - Short
     - Description
   * - ``--prior``
     - ``-p``
     - Path to prior HDF5 file (required)
   * - ``--data``
     - ``-d``
     - Path to observed-data HDF5 file (required)
   * - ``--output``
     - ``-o``
     - Output posterior HDF5 path (auto-generated if omitted)
   * - ``--samples``
     - ``-n``
     - Max number of prior samples to use
   * - ``--auto-temp``
     - ``-T``
     - Enable automatic temperature estimation
   * - ``--temp-base``
     -
     - Base temperature when ``--auto-temp`` is not set
   * - ``--nr``
     -
     - Posterior samples retained per data point (default 400 on the CLI)
   * - ``--cpus``
     - ``-c``
     - Number of CPU cores (0 = auto-detect)
   * - ``--no-parallel``
     -
     - Disable parallel processing
   * - ``--chunks``
     -
     - Number of chunks for parallel processing (0 = auto)
   * - ``--id-use``
     -
     - Comma-separated list of data IDs to invert
   * - ``--use-n-best``
     -
     - Restrict to the N best-fitting samples (0 = disabled)
   * - ``--backend``
     -
     - ``numpy`` (default) or ``jax``
   * - ``--verbose``
     - ``-v``
     - Print configuration and full tracebacks on error

Run ``integrate_rejection --help`` for the authoritative, up-to-date list.

Backends: ``numpy`` (default) vs. ``jax`` (experimental)
-----------------------------------------------------------

Choosing the default backend: ``REJECTION_BACKEND``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passing ``backend=`` explicitly (Python) or ``--backend`` (CLI) always
wins. If it is *not* given, the backend is taken from the
``REJECTION_BACKEND`` environment variable, and only falls back to
``'numpy'`` if that is unset too. This is useful to switch the default
backend for a whole script, notebook, or shell session without touching
every ``integrate_rejection()`` call::

    import os
    os.environ["REJECTION_BACKEND"] = "jax"   # or "numpy"

    import integrate as ig
    ig.integrate_rejection('prior.h5', 'data.h5')  # uses 'jax'

or from the shell::

    REJECTION_BACKEND=jax integrate_rejection --prior prior.h5 --data data.h5

``numpy`` backend
~~~~~~~~~~~~~~~~~~

The default. Likelihood evaluation and resampling run on the CPU via
NumPy, parallelised across processes with ``multiprocessing`` (see
``--cpus``/``Ncpu`` and ``--chunks``/``Nchunks``). This is the
well-tested, recommended path for routine use and does not require any
GPU or extra dependency.

``jax`` backend (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selected with ``backend='jax'`` (Python) or ``--backend jax`` (CLI).
Implemented in :mod:`integrate.integrate_rejection_jax`. Likelihood
evaluation *and* post-processing (temperature estimation, weighted
sampling, evidence, CHI2) run on-device (GPU, if available), and only the
small final result per batch is transferred back to the host -- this
avoids a PCIe bottleneck that otherwise makes a naive GPU path much slower
than the CPU path. Data points are processed in batches; tune the batch
size with ``Nbatch=<int>`` (default 64), passed through ``**kwargs``.

This backend requires JAX to be installed separately, e.g.::

    pip install jax          # CPU only
    pip install jax[cuda12]  # GPU (CUDA 12)

It is marked experimental: it is newer and less exercised than the
``numpy`` backend, and its main advantage only materialises on GPU
hardware and/or very large ``N``.

**First-run compile time.** JAX JIT-compiles its likelihood kernel the
first time it sees a given input shape (roughly, the number of prior
samples and features). On GPU this compilation -- in particular for the
large reduction fusions involved (sort/cumsum/searchsorted over
``N`` ~ 1e6 samples) -- can take tens of seconds, and XLA will print a
``slow_operation_alarm`` warning if a single fusion takes unusually long
to compile. A per-shape compiled-kernel cache is already wired up
automatically (``~/.cache/jax_xla_gpu``, configured in
``integrate_rejection_jax.py`` via
``jax.config.update("jax_compilation_cache_dir", ...)``), so repeated
runs with the *same* shape reuse the compiled kernel and skip this cost.
For a new shape, or to speed up the first compile itself, set the
following before importing ``jax`` (i.e. before ``import integrate``)::

    import os
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_autotune_level=1 "        # skip exhaustive GEMM/conv autotuning
        "--xla_backend_optimization_level=1" # lower LLVM codegen effort for GPU kernels
    )

``xla_gpu_autotune_level=1`` reduces time spent benchmarking GEMM/
convolution algorithm choices; ``xla_backend_optimization_level=1``
reduces the optimization effort XLA's LLVM-based codegen spends per
kernel, which is what actually dominates compile time for the reduction
fusions used here. Both must be set as a single ``XLA_FLAGS`` string
*before* JAX is imported anywhere in the process to take effect.

If compile time is still a bottleneck after this, consider keeping ``N``
(and other shape-determining parameters) constant across runs so the
per-shape cache is reused, rather than lowering optimization further.
