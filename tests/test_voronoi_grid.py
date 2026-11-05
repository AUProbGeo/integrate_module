"""post_to_voronoi_grid / plot_voronoi_3d / plot_voronoi_slices on a small synthetic DATA/PRIOR/POST set."""
import h5py
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pytest.importorskip("shapely")

import integrate as ig  # noqa: E402

N, NZ = 12, 8


@pytest.fixture
def post_h5(tmp_path):
    rng = np.random.default_rng(0)
    f_data = tmp_path / "DATA.h5"
    f_prior = tmp_path / "PRIOR.h5"
    f_post = tmp_path / "POST.h5"
    x = np.arange(N) * 50.0 + 500000.0
    y = 6200000.0 + 20.0 * np.sin(np.arange(N))
    with h5py.File(f_data, "w") as f:
        f["UTMX"] = x[:, None]
        f["UTMY"] = y[:, None]
        f["LINE"] = np.ones((N, 1))
        f["ELEVATION"] = (100.0 + np.arange(N))[:, None]
    with h5py.File(f_prior, "w") as f:
        d = f.create_dataset("M1", data=rng.uniform(10, 100, (5, NZ)))
        d.attrs["x"] = np.arange(NZ) * 5.0
        d.attrs["is_discrete"] = 0
        d.attrs["clim"] = np.array([1.0, 1000.0])
    med = rng.uniform(10, 100, (N, NZ))
    med[0, :] = np.nan          # sounding 0 not inverted
    med[1, NZ - 2:] = np.nan    # sounding 1: below the DOI
    with h5py.File(f_post, "w") as f:
        f.attrs["f5_data"] = "DATA.h5"      # relative: resolved next to the POST file
        f.attrs["f5_prior"] = "PRIOR.h5"
        f["/M1/Median"] = med
        f["/M1/Std"] = rng.uniform(1, 2, (N, NZ))
    return str(f_post), med


def test_grid_cells_soundings_and_nan(post_h5):
    f_post, med = post_h5
    grid = ig.post_to_voronoi_grid(f_post, r_max=100)
    assert isinstance(grid, pv.UnstructuredGrid)
    assert set(np.unique(grid.celltypes)) == {pv.CellType.WEDGE}
    s = np.asarray(grid.cell_data["sounding"])
    iz = np.asarray(grid.cell_data["iz"])
    assert 0 not in s                                   # all NaN: left out
    assert iz[s == 1].max() == NZ - 3                   # NaN below the DOI left out
    # each cell carries the value of its sounding and depth
    np.testing.assert_allclose(grid.cell_data["value"], med[s, iz])
    np.testing.assert_allclose(grid.cell_data["log10_value"], np.log10(med[s, iz]))
    assert list(grid.field_data["clim"]) == [1.0, 1000.0]
    assert int(grid.field_data["log"][0]) == 1
    assert str(grid.field_data["key"][0]) == "Median"
    # the prisms hang below the surface elevation, along the depth grid
    zmax = grid.bounds[5]
    assert zmax == pytest.approx(100.0 + N - 1)
    # every wedge has positive volume (correct orientation)
    vol = grid.compute_cell_sizes(length=False, area=False, volume=True).cell_data["Volume"]
    assert np.all(vol > 0)


def test_grid_options(post_h5):
    f_post, _ = post_h5
    full = ig.post_to_voronoi_grid(f_post)
    sub = ig.post_to_voronoi_grid(f_post, ii=np.arange(N)[::2], iz_step=2, z_max=20)
    assert sub.n_cells < full.n_cells
    assert set(np.unique(sub.cell_data["iz"])) <= {0, 2, 4}
    assert set(np.unique(sub.cell_data["sounding"])) <= set(range(0, N, 2))
    std = ig.post_to_voronoi_grid(f_post, key="Std")
    assert "log10_value" not in std.cell_data
    assert int(std.field_data["log"][0]) == 0


def test_single_sounding_and_errors(post_h5):
    f_post, _ = post_h5
    one = ig.post_to_voronoi_grid(f_post, ii=[3], r_max=10)
    assert one.n_cells == 4 * NZ                        # a square: 4 wedges per depth cell
    with pytest.raises(ValueError):
        ig.post_to_voronoi_grid(f_post, ii=[0])          # all NaN
    with pytest.raises(ValueError):
        ig.post_to_voronoi_grid(f_post, ii=[])


def test_roundtrip_vtu_and_plots(post_h5, tmp_path):
    f_post, _ = post_h5
    grid = ig.post_to_voronoi_grid(f_post)
    f_vtu = str(tmp_path / "g.vtu")
    grid.save(f_vtu)
    back = pv.read(f_vtu)
    assert back.n_cells == grid.n_cells
    assert str(back.field_data["key"][0]) == "Median"
    np.testing.assert_allclose(back.field_data["cmap_rgb"], grid.field_data["cmap_rgb"])
    pl = pv.Plotter(off_screen=True)
    ig.plot_voronoi_3d(back, threshold=[20, 50], plotter=pl)
    assert "opaque" in pl.actors
    pl.close()
    pl = pv.Plotter(off_screen=True)
    ig.plot_voronoi_slices(back, z=[90], plotter=pl)
    pl.close()


def test_columns_match_grid(post_h5):
    f_post, med = post_h5
    col = ig.post_to_voronoi_columns(f_post, r_max=100, iz_step=2)
    assert 0 not in col["sounding"]                       # all-NaN column left out
    nc = len(col["sounding"])
    assert col["values"].shape == (nc, len(col["iz"]))
    assert len(col["z"]) == len(col["iz"]) + 1
    assert col["poly_offset"][-1] == len(col["poly_xy"])
    np.testing.assert_allclose(col["values"], med[col["sounding"]][:, col["iz"]])
    # polygons are counter-clockwise (positive signed area) and contain their sounding
    for c in range(nc):
        p = col["poly_xy"][col["poly_offset"][c]:col["poly_offset"][c + 1]]
        area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
        assert area > 0
    grid = ig.voronoi_columns_to_grid(col)
    n_finite_layers = np.isfinite(col["values"]).sum()
    assert grid.n_cells == np.sum(np.isfinite(col["values"]) * np.diff(col["poly_offset"])[:, None])
    assert len(np.unique(np.c_[grid.cell_data["sounding"], grid.cell_data["iz"]], axis=0)) == n_finite_layers


def test_duplicate_xy_keeps_first(tmp_path, post_h5):
    f_post, _ = post_h5
    f_data = tmp_path / "DATA.h5"
    n = len(ig.post_to_voronoi_columns(f_post)["sounding"])
    with h5py.File(f_data, "r+") as f:
        for i in (3, 4, 5):
            f["UTMX"][i] = f["UTMX"][2]
            f["UTMY"][i] = f["UTMY"][2]
    col = ig.post_to_voronoi_columns(f_post)
    assert col["n_duplicate"] == 3
    assert len(col["sounding"]) == n - 3
    assert 2 in col["sounding"] and not set(col["sounding"]) & {3, 4, 5}
    grid = ig.voronoi_columns_to_grid(col)
    assert np.all(grid.compute_cell_sizes(length=False, area=False)["Volume"] > 0)
