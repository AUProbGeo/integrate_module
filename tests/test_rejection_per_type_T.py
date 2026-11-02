"""Annealing temperature per data type (issue #53).

``autoT`` and ``T_base`` can be given per data type (list in the order of
``id_use``, or dict ``{data_id: value}``). Then each data type i is tempered
with its own temperature T_i (estimated from L_i alone when autoT[i]=1, else
T_base[i]), and the posterior holds the per-type results ``/T_all``,
``/EV_all`` and ``/CHI2_all`` next to the combined ``/T``, ``/EV`` and
``/CHI2``.

Uses the ``multi_type_masked`` case: two Gaussian data types (15 and 5
channels), type 2 switched off at the even data points.
"""
import h5py
import numpy as np
import pytest

import cases
import integrate as ig
from conftest import run_rejection
from integrate.integrate_rejection import _autoT_tag, _resolve_temperature

# number of data channels per data type in multi_type_masked
N_CH = np.array([cases.ND, 5])

EV_TOL_F32 = dict(rtol=1e-6, atol=1e-5)
T_TOL_F32 = dict(rtol=1e-5)


@pytest.fixture
def mt(build_case):
    return build_case("multi_type_masked")


def _type2_active(mt):
    return mt.info["i_use2"] > 0


# ---------------------------------------------------------------------------
# option parsing
# ---------------------------------------------------------------------------

def test_resolve_scalar_is_legacy():
    per_type, a, b = _resolve_temperature(1, 1, [1, 2, 3])
    assert not per_type
    np.testing.assert_array_equal(a, [1, 1, 1])
    np.testing.assert_array_equal(b, [1, 1, 1])


def test_resolve_list_and_dict():
    per_type, a, b = _resolve_temperature([1, 0, 0], [4, 1, 1], [1, 2, 3])
    assert per_type
    np.testing.assert_array_equal(a, [1, 0, 0])
    np.testing.assert_array_equal(b, [4, 1, 1])
    # dict: ids not listed get the default 1
    per_type, a, b = _resolve_temperature({2: 0, 3: 0}, {1: 4}, [1, 2, 3])
    assert per_type
    np.testing.assert_array_equal(a, [1, 0, 0])
    np.testing.assert_array_equal(b, [4, 1, 1])
    # a per-type T_base alone also selects per-type mode
    assert _resolve_temperature(0, [4, 1, 1], [1, 2, 3])[0]


@pytest.mark.parametrize("autoT, T_base", [
    ([1, 0, 0], 1),         # wrong length
    ({3: 1}, 1),            # unknown data id
    (0, [0, 1]),            # T_base must be > 0
])
def test_resolve_errors(autoT, T_base):
    with pytest.raises(ValueError):
        _resolve_temperature(autoT, T_base, [1, 2])


def test_autoT_tag():
    assert _autoT_tag(1) == "aT1"
    assert _autoT_tag(True) == "aT1"
    assert _autoT_tag([1, 0, 0]) == "aT1-0-0"
    assert _autoT_tag({2: 0, 1: 1}) == "aTd1-0"


def test_wrong_length_raises_in_integrate_rejection(mt, tmp_path):
    with pytest.raises(ValueError):
        run_rejection(mt, "numpy", tmp_path, autoT=[1, 0, 0])


# ---------------------------------------------------------------------------
# legacy (scalar) mode: per-type outputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tkw", [dict(autoT=1), dict(autoT=0, T_base=5)])
def test_legacy_T_all(mt, tmp_path, tkw):
    post = run_rejection(mt, "numpy", tmp_path, **tkw)
    on2 = _type2_active(mt)
    assert post["T_all"].shape == (cases.NDP, 2)
    np.testing.assert_array_equal(post["T_all"][:, 0], post["T"])
    np.testing.assert_array_equal(post["T_all"][on2, 1], post["T"][on2])
    assert np.all(np.isnan(post["T_all"][~on2, 1]))
    if tkw["autoT"] == 0:
        assert np.all(post["T"] == 5)


def test_combined_chi2_is_weighted_mean(mt, tmp_path):
    post = run_rejection(mt, "numpy", tmp_path)
    chi2 = post["CHI2_all"]
    n = np.where(np.isnan(chi2), 0, N_CH[None, :])
    expected = np.nansum(chi2 * n, axis=1) / n.sum(axis=1)
    np.testing.assert_allclose(post["CHI2"], expected, rtol=1e-12)


def test_EV_all_is_evidence_of_each_type_alone(mt, tmp_path):
    """EV_all[:, k] equals EV of an inversion with data type k only."""
    post = run_rejection(mt, "numpy", tmp_path, tag="_both")
    on2 = _type2_active(mt)
    ev1 = run_rejection(mt, "numpy", tmp_path, tag="_D1", id_use=[1])["EV"]
    ev2 = run_rejection(mt, "numpy", tmp_path, tag="_D2", id_use=[2])["EV"]
    np.testing.assert_allclose(post["EV_all"][:, 0], ev1, rtol=1e-12)
    np.testing.assert_allclose(post["EV_all"][on2, 1], ev2[on2], rtol=1e-12)
    assert np.all(np.isnan(post["EV_all"][~on2, 1]))


# ---------------------------------------------------------------------------
# per-type mode
# ---------------------------------------------------------------------------

def test_per_type_T1_equals_legacy_T1(mt, tmp_path):
    """No annealing on any data type: same posterior as legacy T_base=1."""
    a = run_rejection(mt, "numpy", tmp_path, tag="_legacy", autoT=0, T_base=1)
    b = run_rejection(mt, "numpy", tmp_path, tag="_types", autoT=[0, 0], T_base=[1, 1])
    np.testing.assert_array_equal(b["T"], a["T"])
    np.testing.assert_array_equal(b["T_all"], a["T_all"])
    np.testing.assert_allclose(b["EV"], a["EV"], rtol=1e-12)
    frac = np.mean(a["i_use"] == b["i_use"])
    assert frac >= 0.999, "only %.4f of i_use identical" % frac


def test_auto_em_fixed_other(mt, tmp_path):
    """autoT=[1,0]: type 1 annealed (T from L_1 alone), type 2 not annealed."""
    post = run_rejection(mt, "numpy", tmp_path, tag="_types", autoT=[1, 0], T_base=[1, 1])
    on2 = _type2_active(mt)
    # T of data type 1 alone (legacy autoT=1 with only D1)
    T1 = run_rejection(mt, "numpy", tmp_path, tag="_D1", id_use=[1], autoT=1)["T"]
    np.testing.assert_allclose(post["T_all"][:, 0], T1, rtol=1e-12)
    assert np.all(post["T_all"][on2, 1] == 1)
    assert np.all(np.isnan(post["T_all"][~on2, 1]))
    np.testing.assert_array_equal(post["T"], np.nanmax(post["T_all"], axis=1))


def test_dict_equals_list(mt, tmp_path):
    a = run_rejection(mt, "numpy", tmp_path, tag="_list", autoT=[1, 0], T_base=[1, 3])
    b = run_rejection(mt, "numpy", tmp_path, tag="_dict", autoT={2: 0}, T_base={2: 3})
    for k in ("i_use", "T", "T_all", "EV", "EV_all", "CHI2", "CHI2_all"):
        np.testing.assert_array_equal(b[k], a[k], err_msg=k)


def test_attrs(mt, tmp_path):
    f_post = str(tmp_path / "POST_attrs.h5")
    ig.integrate_rejection(mt.f_prior_h5, mt.f_data_h5, f_post_h5=f_post,
                           autoT=[1, 0], T_base=[1, 2], nr=100, Ncpu=1,
                           parallel=False, updatePostStat=False, showInfo=-1)
    with h5py.File(f_post, "r") as f:
        np.testing.assert_array_equal(f.attrs["id_use"], [1, 2])
        np.testing.assert_array_equal(f.attrs["autoT"], [1, 0])
        np.testing.assert_array_equal(f.attrs["T_base"], [1, 2])
        assert f.attrs["per_type_T"] == 1


def test_default_file_name(mt, tmp_path):
    f_post = ig.integrate_rejection(mt.f_prior_h5, mt.f_data_h5, post_dir=str(tmp_path),
                                    autoT=[1, 0], nr=100, Ncpu=1, parallel=False,
                                    updatePostStat=False, showInfo=-1)
    assert f_post.endswith("_aT1-0.h5"), f_post


@pytest.mark.jax
def test_per_type_numpy_vs_jax(mt, tmp_path):
    kw = dict(autoT=[1, 0], T_base=[1, 3])
    a = run_rejection(mt, "numpy", tmp_path, **kw)
    b = run_rejection(mt, "jax", tmp_path, **kw)
    for k, tol in (("T", T_TOL_F32), ("T_all", T_TOL_F32),
                   ("EV", EV_TOL_F32), ("EV_all", EV_TOL_F32)):
        np.testing.assert_array_equal(np.isnan(a[k]), np.isnan(b[k]), err_msg=k)
        np.testing.assert_allclose(b[k], a[k], equal_nan=True, err_msg=k, **tol)
    frac = np.mean(a["i_use"] == b["i_use"])
    assert frac >= 0.999, "only %.4f of i_use identical" % frac
    np.testing.assert_allclose(b["CHI2_all"], a["CHI2_all"], rtol=1e-4, equal_nan=True)


@pytest.mark.gpu
@pytest.mark.parametrize("kw", [dict(autoT=1), dict(autoT=[1, 0], T_base=[1, 3])])
def test_per_type_gpu(mt, tmp_path, kw):
    """GPU kernel: deterministic T/EV per data type match the numpy backend."""
    a = run_rejection(mt, "numpy", tmp_path, **kw)
    b = run_rejection(mt, "jax", tmp_path, **kw)
    for k, tol in (("T", T_TOL_F32), ("T_all", T_TOL_F32),
                   ("EV", EV_TOL_F32), ("EV_all", EV_TOL_F32)):
        np.testing.assert_array_equal(np.isnan(a[k]), np.isnan(b[k]), err_msg=k)
        np.testing.assert_allclose(b[k], a[k], equal_nan=True, err_msg=k, **tol)


def test_per_type_parallel(mt, tmp_path):
    """Per-type results reach the multiprocessing path (T/EV are RNG-independent).

    Run in a fresh Python process, see test_numpy_dtype_parallel.
    """
    import subprocess
    import sys

    kw = dict(autoT=[1, 0], T_base=[1, 3])
    serial = run_rejection(mt, "numpy", tmp_path, **kw)
    f_post = str(tmp_path / "POST_parallel.h5")
    code = (
        "import integrate as ig\n"
        "ig.integrate_rejection(%r, %r, f_post_h5=%r, backend='numpy', nr=400,"
        " Ncpu=2, parallel=True, autoT=[1, 0], T_base=[1, 3], updatePostStat=False,"
        " showInfo=-1, console_progress=False)\n"
        % (mt.f_prior_h5, mt.f_data_h5, f_post))
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    with h5py.File(f_post, "r") as f:
        for k in ("T", "T_all", "EV", "EV_all"):
            np.testing.assert_allclose(f[k][:], serial[k], rtol=1e-12,
                                       equal_nan=True, err_msg=k)
        assert f["CHI2"].shape == (cases.NDP,)
        assert f["CHI2_all"].shape == (cases.NDP, 2)
