"""App boots and every nav route returns 200."""

from pathlib import Path

from starlette.testclient import TestClient

from frontend.app import app
from frontend.components import leaf_slugs


def _client():
    return TestClient(app)


def test_home_ok():
    r = _client().get("/")
    assert r.status_code == 200
    assert "INTEGRATE" in r.text
    assert "Data files" in r.text


def test_all_nav_routes_ok():
    c = _client()
    for slug in leaf_slugs():
        path = "/" if slug == "files" else f"/{slug}"
        assert c.get(path).status_code == 200, path


def test_files_list_partial():
    c = _client()
    assert c.get("/files/list?ffilter=all").status_code == 200
    assert c.get("/files/list?ffilter=data").status_code == 200


def test_static_assets():
    c = _client()
    assert c.get("/static/modernist.css").status_code == 200
    assert c.get("/static/htmx.min.js").status_code == 200


def test_geoprior_preview_controls():
    """Preview toggle + a no-workbook preview POST return partials, not 500s."""
    c = _client()
    hx = {"HX-Request": "true"}
    # auto-update on -> HX-Trigger fires gp-cond (analytic panel) + gp-changed
    r = c.post("/geoprior/preview/toggle", data={"autopreview": "on", "preview_n": "50"}, headers=hx)
    assert r.status_code == 200
    events = {e.strip() for e in r.headers.get("hx-trigger", "").split(",")}
    assert events == {"gp-cond", "gp-changed"}
    # auto-update off -> only gp-cond
    r = c.post("/geoprior/preview/toggle", data={"preview_n": "50"}, headers=hx)
    assert r.status_code == 200
    assert r.headers.get("hx-trigger") == "gp-cond"
    # a cell edit with no workbook loaded -> no crash
    r = c.post("/geoprior/cell", data={"sheet": "S", "r": "0", "c": "0", "v": "x"}, headers=hx)
    assert r.status_code == 200
    # no .xlsx loaded -> graceful messages, not errors
    r = c.post("/geoprior/preview", data={"preview_n": "50", "dmax": "90", "dz": "1"}, headers=hx)
    assert r.status_code == 200
    assert "Load an .xlsx" in r.text
    r = c.post("/geoprior/cond", headers=hx)
    assert r.status_code == 200
    assert "Load an .xlsx" in r.text
    r = c.post("/geoprior/cond/toggle", data={"cond_overlay": "on"}, headers=hx)
    assert r.status_code == 200
    assert r.headers.get("hx-trigger") == "gp-cond"


def test_workspace_controls():
    import tempfile

    from frontend import config

    c = _client()
    saved = config.get_workspace()
    try:
        assert "WORKING FOLDER" in c.get("/").text
        assert c.get("/workspace/edit").status_code == 200
        assert c.get("/workspace/cancel").status_code == 200
        # bad path -> re-rendered bar with an error, still 200
        r = c.post("/workspace", data={"path_text": "/no/such/folder/xyz"})
        assert r.status_code == 200
        assert "Not a folder" in r.text
        # good path -> 204 + HX-Redirect, and it takes effect
        with tempfile.TemporaryDirectory() as d:
            r = c.post("/workspace", data={"path_text": d}, follow_redirects=False)
            assert r.status_code == 204
            assert r.headers.get("hx-redirect") == "/"
            assert config.get_workspace() == Path(d).resolve()
    finally:
        config.set_workspace(str(saved))
