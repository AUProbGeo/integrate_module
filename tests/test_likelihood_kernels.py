"""Unit tests: JAX kernels vs their NumPy counterparts.

When a backend comparison in test_rejection_backends.py fails, these tell
whether the likelihood or the temperature estimate is the cause.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.jax


@pytest.fixture(scope="module")
def kernels():
    pytest.importorskip("jax")
    from integrate.integrate_rejection_jax import _get_jax_kernels
    return _get_jax_kernels()


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    N, nd = 500, 12
    D = rng.normal(size=(N, nd))
    d_obs = rng.normal(size=nd)
    d_std = rng.uniform(0.5, 2.0, size=nd)
    return D, d_obs, d_std


def _numpy_L(D, d_obs, d_std):
    from integrate.integrate_rejection import likelihood_gaussian_diagonal
    return likelihood_gaussian_diagonal(D, d_obs, d_std, normalize=False)


def test_gaussian_diagonal_kernel(kernels, data):
    single, _ = kernels
    D, d_obs, d_std = data
    np.testing.assert_allclose(np.asarray(single(D, d_obs, d_std)),
                               _numpy_L(D, d_obs, d_std), rtol=1e-5)


def test_gaussian_diagonal_kernel_nan_in_dobs(kernels, data):
    single, _ = kernels
    D, d_obs, d_std = data
    d_obs = d_obs.copy()
    d_obs[[1, 5]] = np.nan
    np.testing.assert_allclose(np.asarray(single(D, d_obs, d_std)),
                               _numpy_L(D, d_obs, d_std), rtol=1e-5)


def test_gaussian_diagonal_kernel_nan_in_D(kernels, data):
    """NaN in a prior realization where d_obs is valid -> logL = -inf (issue #47)."""
    single, _ = kernels
    D, d_obs, d_std = data
    D = D.copy()
    D[3, 2] = np.nan
    L_jax = np.asarray(single(D, d_obs, d_std))
    assert L_jax[3] == -np.inf
    L_np = _numpy_L(D, d_obs, d_std)
    assert not np.isfinite(L_np[3]) or L_np[3] == -np.inf
    ok = np.arange(D.shape[0]) != 3
    np.testing.assert_allclose(L_jax[ok], L_np[ok], rtol=1e-5)


def test_gaussian_diagonal_batch_kernel(kernels, data):
    single, batch = kernels
    D, _, _ = data
    rng = np.random.default_rng(1)
    d_obs_b = rng.normal(size=(4, D.shape[1]))
    d_std_b = rng.uniform(0.5, 2.0, size=(4, D.shape[1]))
    L_b = np.asarray(batch(D, d_obs_b, d_std_b))
    assert L_b.shape == (4, D.shape[0])
    for k in range(4):
        np.testing.assert_allclose(L_b[k], _numpy_L(D, d_obs_b[k], d_std_b[k]),
                                   rtol=1e-5)


@pytest.mark.parametrize("variant", ["plain", "with_nan"])
@pytest.mark.parametrize("N_above,P_acc", [(10, 0.2), (3, 0.5)])
def test_logl_T_est_jax(variant, N_above, P_acc):
    pytest.importorskip("jax")
    import integrate as ig
    from integrate.integrate_rejection_jax import _logl_T_est_jax
    rng = np.random.default_rng(7)
    L = -np.abs(rng.normal(scale=200.0, size=1000))
    if variant == "with_nan":
        L[rng.choice(L.size, 50, replace=False)] = np.nan
    T_np = ig.logl_T_est(L, N_above=N_above, P_acc_lev=P_acc)
    T_jax = float(_logl_T_est_jax(L, N_above, P_acc))
    np.testing.assert_allclose(T_jax, T_np, rtol=1e-5)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_logl_T_est_jax_all_nan():
    """All-NaN logL -> T = inf. np.nanmax warns on this input by design."""
    pytest.importorskip("jax")
    import integrate as ig
    from integrate.integrate_rejection_jax import _logl_T_est_jax
    L = np.full(100, np.nan)
    assert ig.logl_T_est(L) == np.inf
    assert float(_logl_T_est_jax(L, 10, 0.2)) == np.inf
