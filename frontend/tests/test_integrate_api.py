"""`integrate_api` against the real `examples/*.h5` (no jobs spawned here)."""

from pathlib import Path

import pytest

from frontend import config
from frontend.services import integrate_api as api

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture(autouse=True)
def _workspace():
    saved = config.get_workspace()
    config.set_workspace(str(EXAMPLES))
    yield
    config.set_workspace(str(saved))


def test_parse_helpers():
    assert api.parse_int_list("1, 2 ;3") == [1, 2, 3]
    assert api.parse_int_list("") == []
    assert api.parse_range("0:500") == [0, 500]
    assert api.parse_range("1,2,3") == [1, 2, 3]
    assert api.parse_range("") == []


@pytest.mark.skipif(not EXAMPLES.is_dir(), reason="examples/ not present")
def test_list_h5_and_xlsx():
    names = [f["name"] for f in api.list_h5()]
    assert any(n.endswith(".h5") for n in names)
    assert all(f["kind"] in {"DATA", "PRIOR", "POSTERIOR", "UNKNOWN", "UNREADABLE"}
               for f in api.list_h5())
    # daugaard_standard.xlsx ships in examples/
    assert "daugaard_standard.xlsx" in api.list_xlsx()


@pytest.mark.skipif(not (EXAMPLES / "DAUGAARD_AVG.h5").exists(), reason="no DATA example")
def test_file_detail_data():
    d = api.file_detail("DAUGAARD_AVG.h5")
    assert d["kind"] == "DATA"
    labels = [lbl for lbl, _ in d["stats"]]
    assert labels[:2] == ["Soundings", "Data types"]
    assert "Continuous" in labels and "Discrete" in labels
    assert d["types"]["columns"][0] == "#"
    assert d["types"]["rows"], "expected at least one /D{i} row"


@pytest.mark.skipif(not (EXAMPLES / "DAUGAARD_PRIOR_GENERIC.h5").exists(), reason="no PRIOR example")
def test_file_detail_prior():
    d = api.file_detail("DAUGAARD_PRIOR_GENERIC.h5")
    assert d["kind"] == "PRIOR"
    labels = [lbl for lbl, _ in d["stats"]]
    assert labels == ["Realizations", "Model types", "Prior-data types"]
    cols = d["types"]["columns"]
    assert "Dimension" in cols and "Type" in cols
    types_seen = {row[2] for row in d["types"]["rows"]}
    assert types_seen <= {"CONTINUOUS", "DISCRETE", "SCALAR"}


@pytest.mark.skipif(not (EXAMPLES / "daugaard_standard.xlsx").exists(), reason="no xlsx example")
def test_cond_resistivity_figure(tmp_path):
    xl = str(EXAMPLES / "daugaard_standard.xlsx")

    # analytic-only render -> a cached PNG under FIGURES_DIR
    url = api.cond_resistivity_figure(xl)
    assert url and url.endswith(".png")
    assert (config.FIGURES_DIR / Path(url).name).exists()

    # with an empirical overlay from a freshly generated scratch .h5
    h5 = tmp_path / "cond.h5"
    api.geoprior_preview_run(xl, str(h5), Nreals=30, dmax=90, dz=1)
    url2 = api.cond_resistivity_figure(xl, h5_path=str(h5), overlay=True)
    assert url2 and url2.endswith(".png")
    assert url2 != url  # overlay + new .h5 -> different cache key

    # malformed workbook -> None, no exception
    bad = tmp_path / "bad.xlsx"
    bad.write_bytes(b"not a workbook")
    assert api.cond_resistivity_figure(str(bad)) is None


@pytest.mark.skipif(not (EXAMPLES / "DAUGAARD_POSTERIOR.h5").exists(), reason="no POSTERIOR example")
def test_file_detail_posterior():
    d = api.file_detail("DAUGAARD_POSTERIOR.h5")
    assert d["kind"] == "POSTERIOR"
    labels = [lbl for lbl, _ in d["stats"]]
    assert labels[:3] == ["Soundings", "Realizations / sounding", "Model types"]
    meta_keys = {k for k, _ in d["meta"]}
    assert {"Data file", "Prior file"} <= meta_keys
