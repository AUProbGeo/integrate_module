"""Synthetic prior/data cases for the test suite.

Each case writes a small PRIOR.h5 and DATA.h5 using the module's own IO
functions (``save_prior_model``, ``save_prior_data``, ``save_data_gaussian``,
``save_data_multinomial``), with a *linear* forward model instead of an EM
forward code, so the cases build in well under a second and need no external
forward backend.

To add a case: write a ``build_<name>(f_prior_h5, f_data_h5, rng)`` function
and register it in ``CASES``.  Every test parametrized over ``CASES`` then
picks it up automatically.
"""
from dataclasses import dataclass, field

import numpy as np

import integrate as ig

# Default sizes: small enough to be fast, large enough that the posterior is
# not trivially one sample.
N_PRIOR = 2000   # prior realizations
NM = 10          # model parameters (layers)
ND = 15          # data channels per sounding
NDP = 25         # data points (soundings)
NOISE_STD = 0.5  # absolute Gaussian noise std


@dataclass
class Case:
    name: str
    f_prior_h5: str
    f_data_h5: str
    n_types: int
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _prior_model(rng, N=N_PRIOR, nm=NM):
    """Smooth-ish continuous prior: random walk in 'log-resistivity'."""
    steps = rng.normal(0.0, 0.3, size=(N, nm))
    return 2.0 + np.cumsum(steps, axis=1)


def _forward(M, G):
    return M @ G


def _observed(D, rng, ndp=NDP, noise_std=NOISE_STD):
    """Pick ndp 'true' prior realizations and add noise to their data."""
    i_true = rng.choice(D.shape[0], ndp, replace=False)
    d_obs = D[i_true] + rng.normal(0.0, noise_std, size=(ndp, D.shape[1]))
    return d_obs, i_true


def _save_prior(f_prior_h5, M_list, D_list):
    for im, M in enumerate(M_list, start=1):
        kw = {}
        if np.issubdtype(M.dtype, np.integer):
            cls = np.unique(M)
            kw = dict(is_discrete=1, class_id=cls,
                      class_name=['c%d' % c for c in cls])
        ig.save_prior_model(f_prior_h5, M, im=im, showInfo=-1, **kw)
    for id_, D in enumerate(D_list, start=1):
        ig.save_prior_data(f_prior_h5, D, id=id_, showInfo=-1)


def _save_gauss(f_data_h5, d_obs, id_, **kw):
    ig.save_data_gaussian(d_obs, id=id_, f_data_h5=f_data_h5, showInfo=-1, **kw)


# ---------------------------------------------------------------------------
# case builders
# ---------------------------------------------------------------------------

def build_gauss_diag(f_prior_h5, f_data_h5, rng):
    """One Gaussian data type with diagonal noise (JAX on-device kernel)."""
    M = _prior_model(rng)
    G = rng.normal(0.0, 1.0, size=(NM, ND)) / np.sqrt(NM)
    D = _forward(M, G)
    _save_prior(f_prior_h5, [M], [D])
    d_obs, i_true = _observed(D, rng)
    _save_gauss(f_data_h5, d_obs, 1, D_std=np.full_like(d_obs, NOISE_STD))
    return 1, {'i_true': i_true}


def build_gauss_diag_nan(f_prior_h5, f_data_h5, rng):
    """Diagonal Gaussian with NaN channels in d_obs (issue #47 handling)."""
    M = _prior_model(rng)
    G = rng.normal(0.0, 1.0, size=(NM, ND)) / np.sqrt(NM)
    D = _forward(M, G)
    _save_prior(f_prior_h5, [M], [D])
    d_obs, i_true = _observed(D, rng)
    d_obs[rng.random(d_obs.shape) < 0.2] = np.nan   # ~20% missing channels
    _save_gauss(f_data_h5, d_obs, 1, D_std=np.full_like(d_obs, NOISE_STD))
    return 1, {'i_true': i_true}


def build_gauss_fullcd(f_prior_h5, f_data_h5, rng):
    """Gaussian with a full (shared) covariance matrix (NumPy fallback in JAX)."""
    M = _prior_model(rng)
    G = rng.normal(0.0, 1.0, size=(NM, ND)) / np.sqrt(NM)
    D = _forward(M, G)
    _save_prior(f_prior_h5, [M], [D])
    d_obs, i_true = _observed(D, rng)
    # exponential correlation between neighbouring channels
    k = np.arange(ND)
    Cd = NOISE_STD**2 * np.exp(-np.abs(k[:, None] - k[None, :]) / 3.0)
    _save_gauss(f_data_h5, d_obs, 1, Cd=Cd)
    return 1, {'i_true': i_true}


def build_multinomial(f_prior_h5, f_data_h5, rng):
    """Discrete model observed through a multinomial data type."""
    nclass = 3
    M = rng.integers(1, nclass + 1, size=(N_PRIOR, NM)).astype(np.int16)
    D = M.copy()                       # data = class at each layer
    _save_prior(f_prior_h5, [M], [D])
    i_true = rng.choice(N_PRIOR, NDP, replace=False)
    # P_obs (ndp, nclass, nm): 0.7 on the true class, rest spread out
    P = np.full((NDP, nclass, NM), 0.15)
    for ip, i in enumerate(i_true):
        P[ip, M[i] - 1, np.arange(NM)] = 0.7
    ig.save_data_multinomial(P, id=1, f_data_h5=f_data_h5, showInfo=-1)
    return 1, {'i_true': i_true}


def build_multi_type_masked(f_prior_h5, f_data_h5, rng):
    """Two Gaussian types; type 2 is switched off (i_use=0) at half the points."""
    M = _prior_model(rng)
    G1 = rng.normal(0.0, 1.0, size=(NM, ND)) / np.sqrt(NM)
    G2 = rng.normal(0.0, 1.0, size=(NM, 5)) / np.sqrt(NM)
    D1, D2 = _forward(M, G1), _forward(M, G2)
    _save_prior(f_prior_h5, [M], [D1, D2])
    i_true = rng.choice(N_PRIOR, NDP, replace=False)
    d1 = D1[i_true] + rng.normal(0.0, NOISE_STD, size=(NDP, D1.shape[1]))
    d2 = D2[i_true] + rng.normal(0.0, NOISE_STD, size=(NDP, D2.shape[1]))
    i_use2 = (np.arange(NDP) % 2).reshape(-1, 1).astype(float)
    _save_gauss(f_data_h5, d1, 1, D_std=np.full_like(d1, NOISE_STD))
    _save_gauss(f_data_h5, d2, 2, D_std=np.full_like(d2, NOISE_STD), i_use=i_use2)
    return 2, {'i_true': i_true, 'i_use2': i_use2.ravel()}


def build_gauss_em_scaled(f_prior_h5, f_data_h5, rng):
    """EM-like magnitudes: data ~1e-8..1e-12, 3% relative noise, 40 channels.

    With normalize_likelihood=True the Gaussian normalization constant is
    large (EV ~ +1e3), which is where float32 rounding in the JAX backend
    is largest, so this case guards the EV tolerance.
    """
    nd = 40
    M = _prior_model(rng)
    G = rng.normal(0.0, 1.0, size=(NM, nd)) / np.sqrt(NM)
    D = 1e-8 * np.exp(-np.linspace(0, 8, nd)) * np.exp(0.3 * (M @ G))
    _save_prior(f_prior_h5, [M], [D])
    i_true = rng.choice(N_PRIOR, NDP, replace=False)
    d_std = 0.03 * np.abs(D[i_true])
    d_obs = D[i_true] + rng.normal(0.0, 1.0, size=d_std.shape) * d_std
    _save_gauss(f_data_h5, d_obs, 1, D_std=d_std)
    return 1, {'i_true': i_true}


CASES = {
    'gauss_diag': build_gauss_diag,
    'gauss_em_scaled': build_gauss_em_scaled,
    'gauss_diag_nan': build_gauss_diag_nan,
    'gauss_fullcd': build_gauss_fullcd,
    'multinomial': build_multinomial,
    'multi_type_masked': build_multi_type_masked,
}


def build_case(name, directory, seed=0):
    """Build case ``name`` into ``directory`` and return a :class:`Case`."""
    f_prior_h5 = str(directory / ('PRIOR_%s.h5' % name))
    f_data_h5 = str(directory / ('DATA_%s.h5' % name))
    rng = np.random.default_rng(seed)
    n_types, info = CASES[name](f_prior_h5, f_data_h5, rng)
    return Case(name=name, f_prior_h5=f_prior_h5, f_data_h5=f_data_h5,
                n_types=n_types, info=info)
