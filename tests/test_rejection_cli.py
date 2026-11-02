"""integrate_rejection command-line tool: --auto-temp / --temp-base per data type."""
import sys

import h5py
import numpy as np
import pytest

from integrate.integrate_rejection_cli import main, parse_per_type


@pytest.mark.parametrize("s, cast, expected", [
    ("1", int, 1),
    ("0", int, 0),
    ("1.0", float, 1.0),
    ("1,0,0", int, [1, 0, 0]),
    ("4,1,1", float, [4.0, 1.0, 1.0]),
    ("2:0,3:0", int, {2: 0, 3: 0}),
    (" 1 : 4 ", float, {1: 4.0}),
])
def test_parse_per_type(s, cast, expected):
    assert parse_per_type(s, cast) == expected


@pytest.mark.parametrize("s", ["", "a", "1,a", "1:2:3", "1:0,1", ":1"])
def test_parse_per_type_errors(s):
    with pytest.raises(ValueError):
        parse_per_type(s, int)


def _run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["integrate_rejection", *argv])
    return main()


@pytest.mark.parametrize("extra, autoT, T_base, per_type", [
    ([], [0, 0], [1, 1], 0),                                # default: no auto T
    (["--auto-temp"], [1, 1], [1, 1], 0),                    # old flag form
    (["-T", "1,0", "--temp-base", "1,3"], [1, 0], [1, 3], 1),
    (["--auto-temp", "2:0", "--temp-base", "2:3"], [1, 0], [1, 3], 1),
])
def test_cli_temperature(monkeypatch, build_case, tmp_path, extra, autoT, T_base, per_type):
    mt = build_case("multi_type_masked")
    f_post = str(tmp_path / "POST_cli.h5")
    rc = _run_cli(monkeypatch, "--prior", mt.f_prior_h5, "--data", mt.f_data_h5,
                  "--output", f_post, "--no-parallel", "--backend", "numpy",
                  "--nr", "100", *extra)
    assert rc == 0
    with h5py.File(f_post, "r") as f:
        np.testing.assert_array_equal(f.attrs["autoT"], autoT)
        np.testing.assert_array_equal(f.attrs["T_base"], T_base)
        assert f.attrs["per_type_T"] == per_type
        assert f["T_all"].shape[1] == 2
        if autoT[1] == 0:
            on2 = ~np.isnan(f["T_all"][:, 1])
            assert np.all(f["T_all"][on2, 1] == T_base[1])


def test_cli_bad_temperature(monkeypatch, build_case, tmp_path):
    mt = build_case("multi_type_masked")
    rc = _run_cli(monkeypatch, "--prior", mt.f_prior_h5, "--data", mt.f_data_h5,
                  "--output", str(tmp_path / "POST_bad.h5"), "--no-parallel",
                  "--auto-temp", "1,0,0")      # 3 values, 2 data types
    assert rc == 1
