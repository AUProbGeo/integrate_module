#!/usr/bin/env python3
"""
Unit tests for the refactored integrate_posterior_stats.

Verifies:
1. Individual stat functions (Mean, Median, Mode, Entropy, ...) on small arrays
2. N_UNIQUE vectorized computation matches np.unique per row
3. End-to-end run on a synthetic POST.h5 + PRIOR.h5 matches a reference
   computed inline with the original sequential formulas
"""

import os
import sys
import tempfile
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrate.integrate import (
    _CONTINUOUS_STATS as CONTINUOUS_STATS,
    _DISCRETE_STATS as DISCRETE_STATS,
    _ps_compute_n_unique as _compute_n_unique,
    _ps_class_counts as _class_counts,
    _ps_mean as _mean,
    _ps_logmean as _logmean,
    _ps_std as _std,
    _ps_logstd as _logstd,
    _ps_median as _median,
    _ps_harmonic_mean as _harmonic_mean,
    _ps_mode as _mode,
    _ps_entropy as _entropy,
    _ps_prob as _prob,
    _ps_kl_discrete as _kl_discrete,
)


# ---------------------------------------------------------------------------
# 1. Stat functions
# ---------------------------------------------------------------------------

def test_continuous_stats():
    np.random.seed(0)
    m = np.random.rand(4, 100, 3) + 0.1   # (B, nr, nm)

    assert np.allclose(_mean(m), m.mean(axis=1))
    assert np.allclose(_std(m), m.std(axis=1))
    assert np.allclose(_median(np.sort(m, axis=1)), np.median(m, axis=1))
    # LogMean = exp(mean(log(x)))
    assert np.allclose(_logmean(m), np.exp(np.mean(np.log(m), axis=1)))
    # LogStd = std(log10(x))
    expected = np.std(np.log10(np.maximum(m, 1e-10)), axis=1)
    assert np.allclose(_logstd(m), expected)
    print("test_continuous_stats: OK")


def test_harmonic_mean():
    # Build a known small case: values 1..10, trim 10% -> trim 1 each side -> mean(2..9) in conductivity
    m = np.arange(1, 11, dtype=float).reshape(1, 10, 1)
    # conductivity = 1/[1..10], sort, trim 1 each side -> mean(1/2..1/9), invert
    c = np.sort(1.0 / m[0, :, 0])
    expected = 1.0 / np.mean(c[1:9])
    got = _harmonic_mean(np.sort(m, axis=1))[0, 0]
    assert np.isclose(got, expected), f"{got} vs {expected}"
    print("test_harmonic_mean: OK")


def test_discrete_stats():
    # (B=2, nr=5, nm=2) with class ids [1, 2, 3]
    class_id = np.array([1, 2, 3])
    m = np.array([
        [[1, 1], [1, 2], [2, 2], [3, 3], [1, 2]],   # b=0
        [[3, 3], [3, 3], [2, 2], [1, 1], [1, 1]],   # b=1
    ])
    counts = _class_counts(m, class_id)
    # b=0, j=0: 1 appears 3x, 2 appears 1x, 3 appears 1x
    assert counts.shape == (2, 3, 2)
    assert np.array_equal(counts[0, :, 0], [3, 1, 1])
    assert np.array_equal(counts[1, :, 1], [2, 1, 2])

    # Mode: most frequent class (ties resolve to lowest class id via argmax)
    mode = _mode(m, class_id)
    assert np.array_equal(mode[0], [1, 2])
    # b=1, j=0: [3,3,2,1,1] -> tie between class 1 and 3 -> argmax picks class 1
    assert np.array_equal(mode[1, 0], 1)

    # Entropy normalized by log(n_classes): uniform distribution -> 1.0
    m_uniform = np.array([[[1, 1], [2, 2], [3, 3], [1, 2], [2, 3], [3, 1]]])  # (1, 6, 2)
    ent_uniform = _entropy(m_uniform, class_id)
    assert np.allclose(ent_uniform, 1.0, atol=1e-6)

    # P sums to 1 over classes
    p = _prob(m, class_id)
    assert np.allclose(p.sum(axis=1), 1.0)
    print("test_discrete_stats: OK")


def test_n_unique():
    np.random.seed(1)
    i_use = np.random.randint(0, 50, (20, 100))
    fast = _compute_n_unique(i_use)
    slow = np.array([len(np.unique(i_use[i])) for i in range(20)])
    assert np.array_equal(fast, slow), f"\n{fast}\nvs\n{slow}"
    print("test_n_unique: OK")


def test_kl_discrete():
    # When posterior == prior, KL should be ~0
    class_id = np.array([1, 2, 3])
    np.random.seed(2)
    m = np.random.choice(class_id, (1, 1000, 2))
    prior_hist = _class_counts(m, class_id)[0] / 1000.0
    prior_hist = (prior_hist + 1e-10) / (prior_hist.sum(axis=0, keepdims=True) + 3 * 1e-10)
    kl = _kl_discrete(m, class_id, prior_hist=prior_hist)
    assert np.allclose(kl, 0.0, atol=1e-6), f"KL should be ~0, got {kl}"
    print("test_kl_discrete: OK")


# ---------------------------------------------------------------------------
# 2. Registry
# ---------------------------------------------------------------------------

def test_registry():
    # All the original stats must be registered
    for name in ["Mean", "LogMean", "Std", "LogStd", "Median", "HarmonicMean", "KL"]:
        assert name in CONTINUOUS_STATS, f"missing continuous stat: {name}"
    for name in ["Mode", "Entropy", "P", "KL"]:
        assert name in DISCRETE_STATS, f"missing discrete stat: {name}"
    print("test_registry: OK")


# ---------------------------------------------------------------------------
# 3. End-to-end on a synthetic POST/PRIOR pair
# ---------------------------------------------------------------------------

def _make_synthetic(tmpdir):
    """Build small PRIOR.h5 + POST.h5 (with i_use only) in tmpdir."""
    N, nm_cont, nm_disc, n_classes, nr, nsounding = 2000, 5, 3, 4, 50, 30
    rng = np.random.RandomState(42)

    f_prior = os.path.join(tmpdir, "PRIOR.h5")
    with h5py.File(f_prior, "w") as f:
        M1 = rng.rand(N, nm_cont) * 100 + 1
        m1 = f.create_dataset("M1", data=M1)
        m1.attrs["is_discrete"] = 0
        m1.attrs["name"] = "Resistivity"

        M2 = rng.randint(1, n_classes + 1, size=(N, nm_disc)).astype(float)
        m2 = f.create_dataset("M2", data=M2)
        m2.attrs["is_discrete"] = 1
        m2.attrs["class_id"] = np.arange(1, n_classes + 1, dtype=float)
        m2.attrs["class_name"] = np.array([f"c{i}" for i in range(1, n_classes + 1)], dtype=object)

    f_post = os.path.join(tmpdir, "POST.h5")
    i_use = rng.randint(0, N, (nsounding, nr))
    with h5py.File(f_post, "w") as f:
        f.create_dataset("i_use", data=i_use)
        f.attrs["f5_prior"] = f_prior
        f.attrs["f5_data"] = f_prior  # not used here (no geometry copy needed)
    return f_prior, f_post, i_use, M1, M2


def _reference_stats(i_use, M1, M2, class_id, computeKL=False):
    """Compute stats with the ORIGINAL sequential formulas — the baseline."""
    nsounding, nr = i_use.shape
    nm_cont = M1.shape[1]
    nm_disc = M2.shape[1]
    n_classes = len(class_id)

    out = {}
    out["M1/Mean"] = np.full((nsounding, nm_cont), np.nan)
    out["M1/LogMean"] = np.full((nsounding, nm_cont), np.nan)
    out["M1/Median"] = np.full((nsounding, nm_cont), np.nan)
    out["M1/Std"] = np.full((nsounding, nm_cont), np.nan)
    out["M1/LogStd"] = np.full((nsounding, nm_cont), np.nan)
    out["M1/HarmonicMean"] = np.full((nsounding, nm_cont), np.nan)
    out["M2/Mode"] = np.full((nsounding, nm_disc), np.nan)
    out["M2/Entropy"] = np.full((nsounding, nm_disc), np.nan)
    out["M2/P"] = np.full((nsounding, n_classes, nm_disc), np.nan)
    out["N_UNIQUE"] = np.full(nsounding, np.nan)

    for i in range(nsounding):
        m = M1[i_use[i], :]
        out["M1/Mean"][i] = m.mean(axis=0)
        out["M1/LogMean"][i] = np.exp(np.mean(np.log(m), axis=0))
        out["M1/Median"][i] = np.median(m, axis=0)
        out["M1/Std"][i] = m.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["M1/LogStd"][i] = np.std(np.log10(np.maximum(m, 1e-10)), axis=0)
        _c = 1.0 / np.maximum(m, 1e-10)
        _k = int(np.floor(0.10 * _c.shape[0]))
        _cs = np.sort(_c, axis=0)
        out["M1/HarmonicMean"][i] = 1.0 / np.mean(_cs[_k:_c.shape[0] - _k, :], axis=0)

        md = M2[i_use[i], :]
        n_count = np.zeros((n_classes, nm_disc))
        for ic in range(n_classes):
            n_count[ic, :] = np.sum(class_id[ic] == md, axis=0) / nr
        out["M2/P"][i] = n_count
        out["M2/Mode"][i] = class_id[np.argmax(n_count, axis=0)]
        # entropy: -sum(p*log(p))/log(n_classes)
        p_safe = np.clip(n_count, 1e-12, None)
        out["M2/Entropy"][i] = -np.sum(p_safe * np.log(p_safe), axis=0) / np.log(n_classes)

        out["N_UNIQUE"][i] = len(np.unique(i_use[i]))
    return out


def test_end_to_end():
    from integrate.integrate import integrate_posterior_stats
    with tempfile.TemporaryDirectory() as tmpdir:
        f_prior, f_post, i_use, M1, M2 = _make_synthetic(tmpdir)
        class_id = np.arange(1, 5, dtype=float)

        # Run new implementation
        integrate_posterior_stats(f_post, showInfo=-1, computeKL=False,
                                  seed=42, batch_size=8)

        # Reference
        ref = _reference_stats(i_use, M1, M2, class_id)

        # Compare
        with h5py.File(f_post, "r") as f:
            # float32 rounding + small numerical differences
            tol = 1e-4
            for key in ref:
                got = f[key][:]
                exp = ref[key]
                diff = np.abs(got - exp)
                # NaN-where-both-NaN
                both_nan = np.isnan(got) & np.isnan(exp)
                diff = np.where(both_nan, 0.0, diff)
                max_diff = np.nanmax(diff) if diff.size else 0.0
                assert max_diff < tol, f"{key}: max_diff={max_diff}"
        print("test_end_to_end: OK")


if __name__ == "__main__":
    test_continuous_stats()
    test_harmonic_mean()
    test_discrete_stats()
    test_n_unique()
    test_kl_discrete()
    test_registry()
    test_end_to_end()
    print("\nAll tests passed.")