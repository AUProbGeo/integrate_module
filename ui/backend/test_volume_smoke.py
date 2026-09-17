"""Smoke test for the Query Volume router against examples/DAUGAARD_POSTERIOR.h5."""
import os
os.environ["INTEGRATE_WORKSPACE"] = "examples"

import numpy as np
from fastapi.testclient import TestClient
from ui.backend.main import app

client = TestClient(app)
F = "DAUGAARD_POSTERIOR.h5"

# M2 = Lithology; raw-material classes (sand/gravel) per the Daugaard example:
# 2=Meltwater sand, 5=Gravel, 6=Miocene sand. Fine (overburden): the rest.
RAW_CLASSES = [1, 2, 5, 6, 7, 8, 3]
geo = {"hull_ratio": 0.10, "edge_buffer": None, "cell_area_k": 6.0, "elong_max": 4.0}
prob_dict = {"constraints": [{"im": 2, "classes": RAW_CLASSES,
                              "thickness_mode": "cumulative",
                              "thickness_threshold": 10.0,
                              "depth_max": 30.0}]}
r = client.post("/api/volume/prob", json={"f": F, "query_dict": prob_dict, "geo": geo})
assert r.status_code == 200, r.json()
prob = r.json()
assert prob["n"] > 0 and len(prob["x"]) == len(prob["p"]) == prob["n"]
assert len(prob["boundary"]) >= 4
print("A ok: n=%d dropped=%d meanP=%.3f" % (prob["n"], prob["n_dropped"], prob["mean_probability"]))

# B: grow two areas (seeded at the two Daugaard centers from the example)
r = client.post("/api/volume/grow", json={
    "f": F, "p": prob["p"], "p_min": 0.2,
    "x_center": 543039.3, "y_center": 6175596.0, "geo": geo})
assert r.status_code == 200, r.json()
g1 = r.json()
assert g1["n_soundings"] > 0 and len(g1["polygon"]) >= 4
print("B1 ok: n=%d area=%.0f m^2" % (g1["n_soundings"], g1["area_m2"]))

r = client.post("/api/volume/grow", json={
    "f": F, "p": prob["p"], "p_min": 0.2,
    "x_center": 544500.0, "y_center": 6175800.0, "geo": geo})
assert r.status_code == 200, r.json()
g2 = r.json()
print("B2 ok: n=%d area=%.0f m^2" % (g2["n_soundings"], g2["area_m2"]))

# B': cache hit — same geo params, second file not needed; just ensure repeat works
r = client.post("/api/volume/grow", json={
    "f": F, "p": prob["p"], "p_min": 0.3,
    "x_center": 543039.3, "y_center": 6175596.0, "geo": geo})
assert r.status_code == 200
assert r.json()["n_soundings"] <= g1["n_soundings"]   # higher cutoff → no larger region
print("B' ok: p_min=0.3 shrinks/keeps region")

# C: volumes over both areas
vol_dict = {"metric": {"im": 2, "classes": RAW_CLASSES,
                       "thickness_mode": "cumulative", "depth_min": 0.0},
            "percentiles": [5, 50, 95]}
r = client.post("/api/volume/volumes", json={
    "f": F, "query_dict": vol_dict,
    "areas": [{"name": "Area 1", "indices": g1["indices"]},
              {"name": "Area 2", "indices": g2["indices"]}],
    "text": "cumulative thickness of raw-material classes", "geo": geo})
assert r.status_code == 200, r.json()
vol = r.json()
assert vol["percentiles"] == [5, 50, 95]
for a in vol["areas"]:
    v = a["volumes"]
    assert len(v) == 3 and v[0] <= v[1] <= v[2], (a["name"], v)
    print("C ok: %s P5/P50/P95 = %s m^3" % (a["name"], [round(x) for x in v]))
assert vol["figure"]                            # base64 PNG present

# Error paths
r = client.post("/api/volume/grow", json={"f": F, "p": prob["p"], "p_min": 0.2,
                                          "x_center": 543039.0, "geo": geo})
assert r.status_code == 400, r.json()           # x_center without y_center
r = client.post("/api/volume/prob", json={"f": F, "query_dict": vol_dict, "geo": geo})
assert r.status_code == 400, r.json()           # percentile dict rejected by /prob
r = client.post("/api/volume/volumes", json={"f": F, "query_dict": prob_dict,
                                             "areas": [{"name": "x", "indices": g1["indices"]}]})
assert r.status_code == 400, r.json()           # probability dict rejected by /volumes
print("error paths ok")
print("ALL SMOKE TESTS PASSED")