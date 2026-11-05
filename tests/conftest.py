"""Shared pytest setup for integrate_module.

Environment defaults are set *before* ``integrate`` (and hence matplotlib /
jax) is imported:

- ``MPLBACKEND=Agg``: no GUI windows from plotting code.
- ``JAX_PLATFORMS=cpu``: JAX tests run on CPU by default, which makes the
  numpy-vs-JAX comparison deterministic (the JAX CPU path draws samples with
  the same ``np.random`` stream as the numpy backend).  To run the GPU tests::

      JAX_PLATFORMS=cuda pytest -m gpu
"""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

# make ``import cases`` work from test modules
sys.path.insert(0, os.path.dirname(__file__))
import cases  # noqa: E402


def _jax_platform():
    try:
        import jax
    except ImportError:
        return None
    try:
        return jax.devices()[0].platform
    except RuntimeError:
        return None


def pytest_collection_modifyitems(config, items):
    platform = _jax_platform()
    skip_nojax = pytest.mark.skip(reason="jax is not installed")
    skip_nogpu = pytest.mark.skip(
        reason="no JAX GPU device (JAX_PLATFORMS=%s); run with JAX_PLATFORMS=cuda"
        % os.environ.get("JAX_PLATFORMS", ""))
    for item in items:
        if platform is None and ("jax" in item.keywords or "gpu" in item.keywords):
            item.add_marker(skip_nojax)
        elif "gpu" in item.keywords and platform == "cpu":
            item.add_marker(skip_nogpu)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def case_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("cases")


@pytest.fixture(scope="session")
def build_case(case_dir):
    """Session cache: build each synthetic case once, return a cases.Case."""
    cache = {}

    def _get(name):
        if name not in cache:
            cache[name] = cases.build_case(name, case_dir)
        return cache[name]

    return _get


@pytest.fixture(params=sorted(cases.CASES))
def case(request, build_case):
    """Parametrized over every registered synthetic case."""
    return build_case(request.param)


@pytest.fixture
def jax_x64():
    """Enable JAX 64-bit mode for one test, then restore the previous setting."""
    jax = pytest.importorskip("jax")
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


POST_KEYS = ("i_use", "T", "T_all", "EV", "EV_all", "CHI2", "CHI2_all", "N_UNIQUE")


def run_rejection(case, backend, out_dir, seed=1, nr=400, tag="", **kwargs):
    """Run ig.integrate_rejection reproducibly and return the POST arrays.

    Uses Ncpu=1 / parallel=False: the multiprocessing path of the numpy
    backend shuffles data points across workers with their own RNG state and
    therefore cannot be reproduced against another backend.

    Returns a dict with POST_KEYS, plus ``M{im}_Mean`` / ``M{im}_Std`` for
    every model with those statistics when ``updatePostStat=True``.
    """
    import integrate as ig

    f_post_h5 = str(out_dir / ("POST_%s_%s%s.h5" % (case.name, backend, tag)))
    kwargs.setdefault("updatePostStat", False)
    np.random.seed(seed)
    ig.integrate_rejection(
        f_prior_h5=case.f_prior_h5,
        f_data_h5=case.f_data_h5,
        f_post_h5=f_post_h5,
        backend=backend,
        nr=nr,
        Ncpu=1,
        parallel=False,
        showInfo=-1,
        console_progress=False,
        **kwargs,
    )
    out = {}
    with h5py.File(f_post_h5, "r") as f:
        for k in POST_KEYS:
            out[k] = f[k][:]
        for key in f.keys():
            if key.startswith("M") and isinstance(f[key], h5py.Group):
                for stat in ("Mean", "Std", "Mode", "Entropy"):
                    if stat in f[key]:
                        out["%s_%s" % (key, stat)] = f[key][stat][:]
    return out
