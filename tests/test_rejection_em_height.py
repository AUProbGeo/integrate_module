"""Tests of em_height_bins and integrate_rejection_em_height.

The EM forward model is replaced by a synthetic height-dependent linear
model, given as ``prior_data_fn``, so no EM forward code is needed:

    D1 = (M1 @ G) * exp(-height / 50)

with the flight height in ``M2`` of the prior and ``D2`` of the data.
"""
import h5py
import numpy as np
import pytest

import cases
import integrate as ig

N = 3000      # prior realizations
NDP = 300     # soundings
H_STD_REL = 0.05
EM_STD = 0.2
RUN = dict(nr=200, Ncpu=1, parallel=False, showInfo=-1, console_progress=False)


# ---------------------------------------------------------------------------
# em_height_bins
# ---------------------------------------------------------------------------

def _heights(n=5000, seed=0):
    """Mostly 25-45 m, with a long tail of high flight heights."""
    rng = np.random.default_rng(seed)
    h = rng.normal(35, 5, n)
    h[: n // 20] = rng.uniform(50, 150, n // 20)
    return h


def test_bins_single():
    h = _heights()
    edges, i_bin = ig.em_height_bins(h, n_bins=1)
    assert len(edges) == 2
    assert np.all(i_bin == 0)
    assert edges[0] <= h.min() and edges[1] >= h.max()


def test_bins_equal_counts():
    """Quantile intervals have equal counts, up to the rounding of the edges.

    With the default dh_round=0.5 m, narrow central intervals (~1.3 m here)
    can differ by ~20 %, so the property is tested with fine rounding.
    """
    h = np.random.default_rng(1).normal(35, 5, 5000)
    edges, i_bin = ig.em_height_bins(h, n_bins=10, dh_max=None, n_min_bin=1,
                                     dh_round=0.001)
    counts = np.bincount(i_bin, minlength=len(edges) - 1)
    assert len(counts) == 10
    assert counts.max() / counts.min() < 1.02, counts


def test_bins_dh_max_split():
    h = _heights()
    edges, _ = ig.em_height_bins(h, n_bins=10, dh_max=20, n_min_bin=1)
    assert np.all(np.diff(edges) <= 20 + 0.5), np.diff(edges)


def test_bins_n_min_bin():
    h = _heights()
    edges, i_bin = ig.em_height_bins(h, n_bins=10, dh_max=20, n_min_bin=200)
    counts = np.bincount(i_bin, minlength=len(edges) - 1)
    assert np.all(counts >= 200), counts


def test_bins_every_height_in_its_interval():
    h = _heights()
    edges, i_bin = ig.em_height_bins(h, n_bins=10, dh_max=20, n_min_bin=50)
    assert np.all(np.diff(edges) > 0)
    assert np.all(edges[i_bin] <= h) and np.all(h <= edges[i_bin + 1])


def test_bins_nan_raises():
    with pytest.raises(ValueError):
        ig.em_height_bins([30.0, np.nan, 40.0])


# ---------------------------------------------------------------------------
# integrate_rejection_em_height (synthetic height-dependent forward model)
# ---------------------------------------------------------------------------

class Synthetic:
    """Prior models (M1), DATA.h5 (D1 'EM', D2 flight height), forward fn."""

    def __init__(self, directory, seed=0):
        rng = np.random.default_rng(seed)
        self.G = rng.normal(0, 1, (cases.NM, cases.ND)) / np.sqrt(cases.NM)
        self.f_prior_h5 = str(directory / "PRIOR.h5")
        self.f_data_h5 = str(directory / "DATA.h5")
        self.n_forward = 0

        ig.save_prior_model(self.f_prior_h5, cases._prior_model(rng, N), im=1, showInfo=-1)

        M_true = cases._prior_model(rng, NDP)
        h_true = rng.normal(35, 5, NDP)
        h_true[: NDP // 10] = rng.uniform(55, 110, NDP // 10)
        self.h_true = h_true
        d1 = self.forward(M_true, h_true) + rng.normal(0, EM_STD, (NDP, cases.ND))
        h_std = H_STD_REL * h_true
        self.h_obs = h_true + rng.normal(0, 1, NDP) * h_std
        cases._save_gauss(self.f_data_h5, d1, 1, D_std=np.full_like(d1, EM_STD))
        cases._save_gauss(self.f_data_h5, self.h_obs[:, None], 2, D_std=h_std[:, None])

    def forward(self, M, h):
        return (M @ self.G) * np.exp(-np.asarray(h).ravel() / 50)[:, None]

    def prior_data_fn(self, f_prior_data_h5):
        """EM prior data D1 from M1 and the flight height in M2."""
        self.n_forward += 1
        with h5py.File(f_prior_data_h5, "r") as f:
            M, h = f["/M1"][:], f["/M2"][:]
        ig.save_prior_data(f_prior_data_h5, self.forward(M, h), id=1,
                           force_delete=True, showInfo=-1)


@pytest.fixture
def syn(tmp_path):
    return Synthetic(tmp_path)


def _post(f_post_h5):
    out = {}
    with h5py.File(f_post_h5, "r") as f:
        for k in ("i_use", "T", "EV"):
            out[k] = f[k][:]
        out["M2_Median"] = f["/M2/Median"][:].ravel()
        out["attrs"] = dict(f.attrs)
    return out


def test_requires_height(syn):
    with pytest.raises(ValueError):
        ig.integrate_rejection_em_height(syn.f_prior_h5, syn.f_data_h5,
                                         prior_data_fn=syn.prior_data_fn, **RUN)


def test_nbins1_matches_manual_workflow(syn, tmp_path):
    """Nbins=1 is the one-prior workflow of marokko_inversion.py."""
    seed = 3
    np.random.seed(11)
    f_post = ig.integrate_rejection_em_height(
        syn.f_prior_h5, syn.f_data_h5, f_post_h5=str(tmp_path / "POST_auto.h5"),
        id_height=2, Nbins=1, seed=seed, prior_data_fn=syn.prior_data_fn, **RUN)
    auto = _post(f_post)
    assert len(auto["attrs"]["f5_prior_data_bin"]) == 1

    # The same by hand: copy prior, flight height as M2, D1, D2 = identity(M2)
    f_pd = str(tmp_path / "PRIOR_DATA_manual.h5")
    np.random.seed(11)
    ig.copy_prior(syn.f_prior_h5, f_pd)
    rng = np.random.default_rng(seed)
    h_sim = np.clip(rng.choice(syn.h_obs, N) + rng.normal(0, 2, N), 5, None)
    ig.save_prior_model(f_pd, h_sim, im=2, force_replace=True, showInfo=-1)
    syn.prior_data_fn(f_pd)
    ig.prior_data_identity(f_pd, id=2, im=2, showInfo=-1)
    f_post_m = ig.integrate_rejection(f_pd, syn.f_data_h5, f_post_h5=str(tmp_path / "POST_manual.h5"),
                                      ip_range=np.arange(NDP), **RUN)
    manual = _post(f_post_m)

    np.testing.assert_array_equal(auto["i_use"], manual["i_use"])
    np.testing.assert_array_equal(auto["T"], manual["T"])
    np.testing.assert_array_equal(auto["EV"], manual["EV"])


def test_nbins_multiple(syn, tmp_path):
    f_post = ig.integrate_rejection_em_height(
        syn.f_prior_h5, syn.f_data_h5, f_post_h5=str(tmp_path / "POST_mh.h5"),
        id_height=2, Nbins=4, n_min_bin=30, prior_data_fn=syn.prior_data_fn, **RUN)
    post = _post(f_post)
    a = post["attrs"]
    n_bin = len(a["f5_post_bin"])
    assert n_bin >= 2
    assert a["height_edges"].shape == (n_bin, 2)
    assert a["im_height"] == 2 and a["id_height"] == 2 and a["id_em"] == 1

    # every sounding inverted, and in exactly one interval
    assert np.all(np.isfinite(post["T"]))
    edges = np.r_[a["height_edges"][:, 0], a["height_edges"][-1, 1]]
    _, i_bin = ig.em_height_bins(syn.h_obs, n_bins=4, n_min_bin=30)
    assert np.bincount(i_bin).sum() == NDP

    # statistics of each sounding come from the posterior of its interval
    for k, f_bin in enumerate(a["f5_post_bin"]):
        ii = np.where(i_bin == k)[0]
        assert np.all((syn.h_obs[ii] >= edges[k]) & (syn.h_obs[ii] <= edges[k + 1]))
        bin_post = _post(f_bin)
        np.testing.assert_array_equal(post["M2_Median"][ii], bin_post["M2_Median"][ii])

    # the inverted flight height is close to the true one
    rel = np.abs(post["M2_Median"] - syn.h_true) / syn.h_true
    assert np.median(rel) < 0.05, np.median(rel)


def test_height_as_array(syn, tmp_path):
    """Flight height as an array: it sets the priors, but is not used as data."""
    h = syn.h_obs.copy()
    h[:5] = np.nan                     # not inverted
    f_post = ig.integrate_rejection_em_height(
        syn.f_prior_h5, syn.f_data_h5, f_post_h5=str(tmp_path / "POST_arr.h5"),
        height=h, id_use=[1], Nbins=3, n_min_bin=30,
        prior_data_fn=syn.prior_data_fn, **RUN)
    post = _post(f_post)
    assert post["attrs"]["id_height"] == 0
    assert np.all(np.isnan(post["T"][:5]))
    assert np.all(np.isfinite(post["T"][5:]))
    for f_pd in post["attrs"]["f5_prior_data_bin"]:
        with h5py.File(f_pd, "r") as f:
            assert "/D1" in f and "/M2" in f and "/D2" not in f


def test_prior_data_reused(syn, tmp_path):
    kw = dict(id_height=2, Nbins=3, n_min_bin=30, prior_data_fn=syn.prior_data_fn, **RUN)
    ig.integrate_rejection_em_height(syn.f_prior_h5, syn.f_data_h5,
                                     f_post_h5=str(tmp_path / "POST_1.h5"), **kw)
    n_first = syn.n_forward
    assert n_first >= 2
    ig.integrate_rejection_em_height(syn.f_prior_h5, syn.f_data_h5,
                                     f_post_h5=str(tmp_path / "POST_2.h5"), **kw)
    assert syn.n_forward == n_first                    # reused
    ig.integrate_rejection_em_height(syn.f_prior_h5, syn.f_data_h5,
                                     f_post_h5=str(tmp_path / "POST_3.h5"),
                                     reuse_prior_data=False, **kw)
    assert syn.n_forward == 2 * n_first                # recomputed


def test_prior_data_base_name(syn, tmp_path):
    base = str(tmp_path / "PRIOR_DATA_custom")
    f_post = ig.integrate_rejection_em_height(
        syn.f_prior_h5, syn.f_data_h5, f_post_h5=str(tmp_path / "POST_b.h5"),
        id_height=2, Nbins=3, n_min_bin=30, f_prior_data_base=base,
        prior_data_fn=syn.prior_data_fn, **RUN)
    files = _post(f_post)["attrs"]["f5_prior_data_bin"]
    assert len(files) >= 2
    assert all(f.startswith(base + "_h") and f.endswith(".h5") for f in files)


def test_per_type_autoT(syn, tmp_path):
    """autoT per data type: EM data annealed, flight-height data not (T=1).

    Checks the default file name tag and that the per-type datasets and
    attributes survive the merge of the interval posteriors.
    """
    f_post = ig.integrate_rejection_em_height(
        syn.f_prior_h5, syn.f_data_h5, post_dir=str(tmp_path),
        id_height=2, Nbins=3, n_min_bin=30, prior_data_fn=syn.prior_data_fn,
        autoT=[1, 0], T_base=[1, 1], **RUN)
    assert "_aT1-0_EMh" in f_post, f_post
    with h5py.File(f_post, "r") as f:
        T_all = f["T_all"][:]
        assert T_all.shape == (NDP, 2)
        assert f["CHI2"].shape == (NDP,)
        assert f["CHI2_all"].shape == (NDP, 2)
        assert f["EV_all"].shape == (NDP, 2)
        np.testing.assert_array_equal(f.attrs["autoT"], [1, 0])
        np.testing.assert_array_equal(f.attrs["id_use"], [1, 2])
        T = f["T"][:]
    assert np.all(T_all[:, 1] == 1)
    assert np.all(T_all[:, 0] >= 1)
    np.testing.assert_array_equal(T, np.nanmax(T_all, axis=1))
