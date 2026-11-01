"""numpy vs JAX backend of ig.integrate_rejection must give the same results.

Two kinds of output:

- **Deterministic**: the evidence ``EV`` and the annealing temperature ``T``
  depend only on the likelihood, not on any random draw.  They must be the
  same for every backend and device (CPU and GPU).
- **Sampled**: ``i_use``, ``N_UNIQUE``, ``CHI2`` and the posterior
  statistics depend on the random draws.  On CPU the JAX backend draws with
  the same ``np.random`` stream as the numpy backend, so with the same seed
  they are identical as well.  On GPU JAX uses ``jax.random``, so only
  statistical agreement can be tested.

Tolerances: JAX computes likelihoods in float32 by default, so with x64 off
``EV``/``T`` agree to ~1e-7 relative (``rtol=1e-6`` below).  With
``jax_enable_x64`` (``jax_x64`` fixture) they are the same number to
~1e-15 and the sampled indices are bit-identical.
"""
import numpy as np
import pytest

import cases
from conftest import run_rejection

pytestmark = pytest.mark.jax

AUTOT = {
    "autoT": dict(autoT=1),
    "fixedT5": dict(autoT=0, T_base=5),
}

# float32 JAX vs float64 numpy
EV_TOL_F32 = dict(rtol=1e-6, atol=1e-5)
T_TOL_F32 = dict(rtol=1e-5)
# JAX in x64 mode: same arithmetic, only summation order may differ
EV_TOL_X64 = dict(rtol=1e-12, atol=1e-10)
T_TOL_X64 = dict(rtol=1e-12)


@pytest.fixture(params=sorted(AUTOT))
def tkw(request):
    return AUTOT[request.param]


def _both(case, tmp_path, **kw):
    a = run_rejection(case, "numpy", tmp_path, **kw)
    b = run_rejection(case, "jax", tmp_path, **kw)
    return a, b


def _assert_same_nan(a, b, what):
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b),
                                  err_msg="%s: NaN pattern differs" % what)


# ---------------------------------------------------------------------------
# Deterministic outputs: evidence and temperature
# ---------------------------------------------------------------------------

def test_evidence_identical(case, tkw, tmp_path):
    a, b = _both(case, tmp_path, **tkw)
    _assert_same_nan(a["EV"], b["EV"], "EV")
    _assert_same_nan(a["T"], b["T"], "T")
    np.testing.assert_allclose(b["EV"], a["EV"], equal_nan=True, **EV_TOL_F32)
    np.testing.assert_allclose(b["T"], a["T"], equal_nan=True, **T_TOL_F32)


# Known places where a backend computes in float32 even with x64 enabled, so
# EV agrees only to float32 precision (~1e-7 relative).  strict=True: once a
# cause is fixed, the XPASS fails the run so the entry gets removed.
KNOWN_X64_ISSUES = {
    "gauss_fullcd": "JAX backend stores the NumPy full-Cd likelihood in a "
                    "float32 buffer (integrate_rejection_jax.py, L_np)",
    "multinomial": "JAX backend stores the NumPy multinomial likelihood in a "
                   "float32 buffer (integrate_rejection_jax.py, L_np)",
}


def test_evidence_identical_x64(case, tkw, tmp_path, jax_x64, request):
    if case.name in KNOWN_X64_ISSUES:
        request.applymarker(pytest.mark.xfail(
            reason=KNOWN_X64_ISSUES[case.name], strict=True))
    a, b = _both(case, tmp_path, **tkw)
    _assert_same_nan(a["EV"], b["EV"], "EV")
    _assert_same_nan(a["T"], b["T"], "T")
    np.testing.assert_allclose(b["EV"], a["EV"], equal_nan=True, **EV_TOL_X64)
    np.testing.assert_allclose(b["T"], a["T"], equal_nan=True, **T_TOL_X64)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("name", ["gauss_diag", "gauss_em_scaled"])
def test_evidence_matches_reference(name, normalize, build_case, tmp_path):
    """Both backends match an EV computed directly in the test (log-mean-exp)."""
    import integrate as ig
    from integrate.integrate_rejection import likelihood_gaussian_diagonal

    case = build_case(name)
    D = ig.load_prior_data(case.f_prior_h5, id_use=[1], showInfo=-1)[0][0]
    DATA = ig.load_data(case.f_data_h5, id_arr=[1], showInfo=-1)
    d_obs, d_std = DATA["d_obs"][0], DATA["d_std"][0]
    EV_ref = np.empty(d_obs.shape[0])
    for ip in range(d_obs.shape[0]):
        L = likelihood_gaussian_diagonal(D, d_obs[ip], d_std[ip],
                                         normalize=normalize)
        Lmax = np.max(L)
        EV_ref[ip] = Lmax + np.log(np.mean(np.exp(L - Lmax)))

    for backend in ("numpy", "jax"):
        r = run_rejection(case, backend, tmp_path,
                          normalize_likelihood=normalize)
        tol = dict(rtol=1e-12, atol=1e-10) if backend == "numpy" else EV_TOL_F32
        np.testing.assert_allclose(r["EV"], EV_ref, err_msg=backend, **tol)


# ---------------------------------------------------------------------------
# Sampled outputs (CPU: same np.random stream -> same samples)
# ---------------------------------------------------------------------------

def test_samples_match(case, tkw, tmp_path):
    """float32 JAX: rounding could in principle flip a draw on a CDF boundary."""
    a, b = _both(case, tmp_path, **tkw)
    frac = np.mean(a["i_use"] == b["i_use"])
    assert frac >= 0.999, "only %.4f of i_use identical" % frac
    np.testing.assert_allclose(b["N_UNIQUE"], a["N_UNIQUE"], rtol=0.02)


def test_samples_identical_x64(case, tkw, tmp_path, jax_x64):
    a, b = _both(case, tmp_path, **tkw)
    np.testing.assert_array_equal(b["i_use"], a["i_use"])
    np.testing.assert_array_equal(b["N_UNIQUE"], a["N_UNIQUE"])


def test_chi2_match(case, tkw, tmp_path):
    a, b = _both(case, tmp_path, **tkw)
    assert a["CHI2"].shape == b["CHI2"].shape == (cases.NDP, case.n_types)
    _assert_same_nan(a["CHI2"], b["CHI2"], "CHI2")
    np.testing.assert_allclose(b["CHI2"], a["CHI2"], rtol=1e-5, equal_nan=True)


def test_posterior_stats_match(case, tmp_path):
    a, b = _both(case, tmp_path, updatePostStat=True)
    stats = sorted(k for k in a if k.startswith("M"))
    assert stats, "no posterior statistics written"
    assert stats == sorted(k for k in b if k.startswith("M"))
    for k in stats:
        np.testing.assert_allclose(b[k], a[k], rtol=1e-6, atol=1e-10,
                                   equal_nan=True, err_msg=k)


# ---------------------------------------------------------------------------
# numpy backend dtype option (float32 likelihood computation)
# ---------------------------------------------------------------------------

def _assert_ev_t_f32(a, b):
    _assert_same_nan(a["EV"], b["EV"], "EV")
    _assert_same_nan(a["T"], b["T"], "T")
    np.testing.assert_allclose(b["EV"], a["EV"], equal_nan=True, **EV_TOL_F32)
    np.testing.assert_allclose(b["T"], a["T"], equal_nan=True, **T_TOL_F32)


def test_numpy_float32_matches_float64(case, tkw, tmp_path):
    a = run_rejection(case, "numpy", tmp_path, **tkw)
    b = run_rejection(case, "numpy", tmp_path, tag="_f32", dtype="float32", **tkw)
    _assert_ev_t_f32(a, b)
    frac = np.mean(a["i_use"] == b["i_use"])
    assert frac >= 0.999, "only %.4f of i_use identical" % frac
    np.testing.assert_allclose(b["CHI2"], a["CHI2"], rtol=1e-5, equal_nan=True)


def test_numpy_float32_matches_jax(case, tkw, tmp_path):
    a = run_rejection(case, "numpy", tmp_path, dtype="float32", **tkw)
    b = run_rejection(case, "jax", tmp_path, **tkw)
    _assert_ev_t_f32(a, b)
    frac = np.mean(a["i_use"] == b["i_use"])
    assert frac >= 0.999, "only %.4f of i_use identical" % frac


@pytest.mark.parametrize("name", ["gauss_diag", "multi_type_masked"])
def test_numpy_dtype_parallel(name, build_case, tmp_path):
    """dtype reaches the multiprocessing workers (EV/T are RNG-independent).

    The parallel run is done in a fresh Python process: on Linux the numpy
    backend forks its workers, and forking the pytest process after other
    tests have started JAX's threads can deadlock.
    """
    import subprocess
    import sys
    import h5py

    case = build_case(name)
    serial = run_rejection(case, "numpy", tmp_path, dtype="float32")
    f_post = str(tmp_path / "POST_parallel.h5")
    code = (
        "import integrate as ig\n"
        "ig.integrate_rejection(%r, %r, f_post_h5=%r, backend='numpy', nr=400,"
        " Ncpu=2, parallel=True, dtype='float32', updatePostStat=False,"
        " showInfo=-1, console_progress=False)\n"
        % (case.f_prior_h5, case.f_data_h5, f_post))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    with h5py.File(f_post, "r") as f:
        EV, T = f["EV"][:], f["T"][:]
    # same dtype in both runs -> identical arithmetic
    np.testing.assert_allclose(EV, serial["EV"], rtol=1e-12)
    np.testing.assert_allclose(T, serial["T"], rtol=1e-12)


def test_backend_env_var(build_case, tmp_path, monkeypatch):
    """REJECTION_BACKEND=jax with backend=None dispatches to the JAX backend."""
    case = build_case("gauss_diag")
    ref = run_rejection(case, "jax", tmp_path)
    monkeypatch.setenv("REJECTION_BACKEND", "jax")
    env = run_rejection(case, None, tmp_path, tag="_env")
    for k in ("i_use", "T", "EV", "CHI2"):
        np.testing.assert_array_equal(env[k], ref[k], err_msg=k)


# ---------------------------------------------------------------------------
# GPU: different RNG, so samples agree only statistically; EV/T still exact
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_gpu_agreement(case, tmp_path):
    nr = 2000
    a = run_rejection(case, "numpy", tmp_path, nr=nr, updatePostStat=True)
    b = run_rejection(case, "jax", tmp_path, nr=nr, updatePostStat=True)

    # deterministic: same tolerance as on CPU
    _assert_same_nan(a["EV"], b["EV"], "EV")
    np.testing.assert_allclose(b["EV"], a["EV"], equal_nan=True, **EV_TOL_F32)
    np.testing.assert_allclose(b["T"], a["T"], equal_nan=True, **T_TOL_F32)

    # statistical: posterior mean within a few Monte-Carlo standard errors.
    # Two independent estimates, effective sample size <= N_UNIQUE.
    if "M1_Mean" in a:
        n_eff = np.minimum(a["N_UNIQUE"], b["N_UNIQUE"])[:, None]
        se = np.sqrt(2.0) * a["M1_Std"] / np.sqrt(n_eff)
        z = np.abs(b["M1_Mean"] - a["M1_Mean"]) / np.maximum(se, 1e-12)
        assert np.nanmax(z) < 6, "posterior mean differs by %.1f std. errors" % np.nanmax(z)
