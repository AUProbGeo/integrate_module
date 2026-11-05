"""Correctness of the Gaussian likelihoods (diagonal and full covariance).

Reference: ``scipy.stats.multivariate_normal.logpdf``, an implementation
independent of integrate.  With missing data (NaN in d_obs) the correct
likelihood is the marginal over the observed channels: the Gaussian of the
valid channels only, with covariance Cd[ok, ok].  This applies to both the
misfit term and the normalization constant -0.5*(n*log(2*pi) + log|Cd|).

Diagonal and full covariance must agree when Cd = diag(d_std**2), both in
the likelihood functions and end-to-end in integrate_rejection (EV), for
both backends.
"""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

import cases
from conftest import run_rejection
from integrate.integrate_rejection import (
    likelihood_gaussian_diagonal,
    likelihood_gaussian_full,
)

NAN_PATTERNS = {
    "no_nan": [],
    "nan_gates": [2, 7, 8],      # missing gates, d_std/Cd still set there
}


def _problem(seed=0, N=50, nd=10, std=0.5):
    rng = np.random.default_rng(seed)
    D = rng.normal(size=(N, nd))
    d_obs = rng.normal(size=nd)
    d_std = rng.uniform(0.5, 2.0, size=nd) * std
    return D, d_obs, d_std


def _correlated_cd(nd, seed=1):
    """Random symmetric positive definite covariance (non-diagonal)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(nd, nd))
    s = rng.uniform(0.5, 2.0, size=nd)
    k = np.arange(nd)
    C = np.exp(-np.abs(k[:, None] - k[None, :]) / 3.0) * np.outer(s, s)
    return C + 0.1 * (A @ A.T) / nd


def _with_nan(d_obs, pattern):
    d = d_obs.copy()
    d[NAN_PATTERNS[pattern]] = np.nan
    return d


def _reference_logpdf(D, d_obs, Cd, normalize):
    """Marginal Gaussian log-density over the valid (non-NaN) channels."""
    ok = ~np.isnan(d_obs)
    C = Cd[np.ix_(ok, ok)]
    r = D[:, ok] - d_obs[ok]
    if normalize:
        return multivariate_normal(mean=np.zeros(ok.sum()), cov=C).logpdf(r)
    return -0.5 * np.einsum("ij,ij->i", r @ np.linalg.inv(C), r)


# ---------------------------------------------------------------------------
# Full covariance: correct against scipy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("pattern", sorted(NAN_PATTERNS))
def test_full_cov_matches_reference(pattern, normalize):
    D, d_obs, _ = _problem()
    Cd = _correlated_cd(D.shape[1])
    d_obs = _with_nan(d_obs, pattern)
    L = likelihood_gaussian_full(D, d_obs, Cd, normalize=normalize)
    np.testing.assert_allclose(L, _reference_logpdf(D, d_obs, Cd, normalize),
                               rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("pattern", sorted(NAN_PATTERNS))
def test_full_cov_norm_const(pattern):
    """return_norm_const gives -0.5*(n*log(2pi) + log|Cd|) over valid channels."""
    D, d_obs, _ = _problem()
    Cd = _correlated_cd(D.shape[1])
    d_obs = _with_nan(d_obs, pattern)
    ok = ~np.isnan(d_obs)
    _, c = likelihood_gaussian_full(D, d_obs, Cd, normalize=True,
                                    return_norm_const=True)
    sign, logdet = np.linalg.slogdet(Cd[np.ix_(ok, ok)])
    assert sign > 0
    np.testing.assert_allclose(c, -0.5 * (ok.sum() * np.log(2 * np.pi) + logdet),
                               rtol=1e-12)


# ---------------------------------------------------------------------------
# Diagonal: correct against scipy, and equal to full with Cd = diag(d_std^2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("pattern", sorted(NAN_PATTERNS))
def test_diagonal_matches_reference(pattern, normalize):
    D, d_obs, d_std = _problem()
    d_obs = _with_nan(d_obs, pattern)
    L = likelihood_gaussian_diagonal(D, d_obs, d_std, normalize=normalize)
    ref = _reference_logpdf(D, d_obs, np.diag(d_std**2), normalize)
    np.testing.assert_allclose(L, ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("pattern", sorted(NAN_PATTERNS))
def test_diagonal_equals_full(pattern, normalize):
    D, d_obs, d_std = _problem()
    d_obs = _with_nan(d_obs, pattern)
    L_diag, c_diag = likelihood_gaussian_diagonal(
        D, d_obs, d_std, normalize=normalize, return_norm_const=True)
    L_full, c_full = likelihood_gaussian_full(
        D, d_obs, np.diag(d_std**2), normalize=normalize, return_norm_const=True)
    np.testing.assert_allclose(c_diag, c_full, rtol=1e-12, atol=1e-12,
                               err_msg="normalization constant")
    np.testing.assert_allclose(L_diag, L_full, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("pattern", sorted(NAN_PATTERNS))
def test_diagonal_equals_full_em_scale(pattern):
    """EM-like noise (std ~1e-13): each wrongly counted gate shifts L by ~29."""
    D, d_obs, d_std = _problem()
    s = 1e-13
    D, d_obs, d_std = D * s, d_obs * s, d_std * s
    d_obs = _with_nan(d_obs, pattern)
    L_diag = likelihood_gaussian_diagonal(D, d_obs, d_std, normalize=True)
    L_full = likelihood_gaussian_full(D, d_obs, np.diag(d_std**2), normalize=True)
    np.testing.assert_allclose(L_diag, L_full, rtol=1e-9)


# ---------------------------------------------------------------------------
# End-to-end: integrate_rejection EV with diagonal d_std vs Cd = diag(std^2)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def diag_and_full(build_case, case_dir):
    """For gauss_diag / gauss_diag_nan: the same d_obs saved once with d_std
    (the case itself) and once with full Cd = diag(NOISE_STD^2)."""
    import integrate as ig

    out = {}
    for name in ("gauss_diag", "gauss_diag_nan"):
        c = build_case(name)
        DATA = ig.load_data(c.f_data_h5, id_arr=[1], showInfo=-1)
        d_obs = DATA["d_obs"][0]
        nd = d_obs.shape[1]
        f_full = str(case_dir / ("DATA_%s_fullcd.h5" % name))
        ig.save_data_gaussian(d_obs, id=1, f_data_h5=f_full, showInfo=-1,
                              Cd=np.eye(nd) * cases.NOISE_STD**2)
        full = cases.Case(name=name + "_as_fullcd", f_prior_h5=c.f_prior_h5,
                          f_data_h5=f_full, n_types=1)
        out[name] = (c, full)
    return out


@pytest.mark.parametrize("backend", ["numpy", pytest.param("jax", marks=pytest.mark.jax)])
@pytest.mark.parametrize("name", ["gauss_diag", "gauss_diag_nan"])
def test_rejection_ev_diagonal_equals_full(name, backend, diag_and_full, tmp_path):
    diag, full = diag_and_full[name]
    a = run_rejection(diag, backend, tmp_path)
    b = run_rejection(full, backend, tmp_path)
    # JAX keeps likelihoods in float32 -> float32 tolerance
    tol = dict(rtol=1e-6, atol=1e-5) if backend == "jax" else dict(rtol=1e-10)
    np.testing.assert_allclose(a["EV"], b["EV"], err_msg="EV", **tol)
    np.testing.assert_allclose(a["T"], b["T"], rtol=1e-5, err_msg="T")
