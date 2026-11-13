import json
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from buaro import static_fetch


class Reply:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self):
        self.calls = 0

    def headers(self, auth):
        return {"Authorization": "Bearer tok"} if auth else {}

    def post_request(self, datasets):
        return Reply({"TaskIds": [{"TaskID": "t-" + datasets[0]["FileID"][:4]}]})

    def status(self, task_id):
        self.calls += 1
        if self.calls % 2:
            return Reply({task_id: {"Status": "In_progress"}})
        return Reply({task_id: {"Status": "Finished_ok", "DownloadURL": f"https://example.org/{task_id}.zip", "UserEmail": "x@y"}})


def build_sources(tmp):
    regions = gpd.GeoDataFrame(
        {"NUTS_ID": static_fetch.REGIONS + ["EL411"], "NAME_LATN": ["a"] * 8, "LEVL_CODE": [3] * 8},
        geometry=[box(23.0 + 0.1 * i, 37.9, 23.1 + 0.1 * i, 38.0) for i in range(8)],
        crs="EPSG:4326",
    )
    regions.to_file(tmp / "nuts.geojson", driver="GeoJSON")
    area = regions[regions["NUTS_ID"].isin(static_fetch.REGIONS)].to_crs("EPSG:3035")
    minx, miny, maxx, maxy = area.total_bounds
    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cells.append(box(x, y, x + 1000, y + 1000))
            y += 1000
        x += 1000
    grid = gpd.GeoDataFrame({"T": [100] * len(cells), "Y_GE65": [20] * len(cells)}, geometry=cells, crs="EPSG:3035")
    (tmp / "census").mkdir()
    grid.to_file(tmp / "census" / "ESTAT_Census_2021_V3.gpkg", driver="GPKG")
    with zipfile.ZipFile(tmp / "census.zip", "w") as z:
        z.write(tmp / "census" / "ESTAT_Census_2021_V3.gpkg", "ESTAT_Census_2021_V3.gpkg")
    trees = gpd.GeoDataFrame({"code": ["T"] * 7}, geometry=[box(23.02 + 0.1 * i, 37.92, 23.06 + 0.1 * i, 37.96) for i in range(7)], crs="EPSG:4326").to_crs("EPSG:3035")
    trees.to_file(tmp / "stl.fgb", driver="FlatGeobuf")
    with zipfile.ZipFile(tmp / "inner.zip", "w") as z:
        z.write(tmp / "stl.fgb", "Data/stl.fgb")
    with zipfile.ZipFile(tmp / "stl.zip", "w") as z:
        z.write(tmp / "inner.zip", "inner.zip")
    ua = gpd.GeoDataFrame({"code_2021": ["11100", "14100"] * 7, "Pop2021": [500, 0] * 7},
                          geometry=[box(23.0 + 0.1 * (i // 2), 37.9 + 0.05 * (i % 2), 23.1 + 0.1 * (i // 2), 37.95 + 0.05 * (i % 2)) for i in range(14)], crs="EPSG:4326").to_crs("EPSG:3035")
    ua.to_file(tmp / "ua.fgb", driver="FlatGeobuf")
    with zipfile.ZipFile(tmp / "ua.zip", "w") as z:
        z.write(tmp / "ua.fgb", "ua.fgb")


def test_static_fetch_end_to_end(tmp_path, monkeypatch):
    build_sources(tmp_path)
    mapping = {
        static_fetch.NUTS_URL: tmp_path / "nuts.geojson",
        static_fetch.CENSUS_URL: tmp_path / "census.zip",
        "https://example.org/t-f91d.zip": tmp_path / "stl.zip",
        "https://example.org/t-acc2.zip": tmp_path / "ua.zip",
    }

    def fake_download(url, dest, headers=None, chunk=0):
        if url.startswith("https://example.org/") and not (headers or {}).get("Authorization"):
            response = static_fetch.requests.Response()
            response.status_code = 403
            raise static_fetch.requests.HTTPError(response=response)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(mapping[url], dest)
        return Path(dest)

    monkeypatch.setattr(static_fetch, "download", fake_download)
    monkeypatch.setattr(static_fetch, "probe_response", lambda url, headers: {"status": 403})
    monkeypatch.setattr(static_fetch.clms.Client, "__new__", lambda cls, *a, **k: FakeClient())
    monkeypatch.setattr(static_fetch.clms, "service_key", lambda: {"k": 1})
    monkeypatch.setattr(static_fetch.clms, "access_token", lambda key: "tok")
    out = tmp_path / "static"
    report = static_fetch.main(raw=str(tmp_path / "raw"), out=str(out), sleep=lambda s: None)
    assert report["nuts"]["regions"] == static_fetch.REGIONS
    assert report["census"]["cells"] > 0
    assert set(report["census"]["fields"]) >= {"T", "Y_GE65", "NUTS_ID"}
    stl = report["clms"]["street_tree_layer_2021"]
    assert stl["wait"]["state"] == "done" and stl["rows"] == 7
    assert stl["download"][-1] == {"attempt": 0, "auth": True, "status": "ok"}
    ua = report["clms"]["urban_atlas_2021"]
    assert ua["class_column"] == "code_2021" and ua["population_columns"] == ["Pop2021"]
    assert "UserEmail" not in json.dumps(report)
    for name in ["attica_nuts3.geojson", "census_grid_attica.gpkg", "street_tree_layer_2021_by_nuts3.xlsx", "urban_atlas_2021_by_nuts3.xlsx", "report.json"]:
        assert (out / name).exists()


def test_static_fetch_reuses_cached_layers(tmp_path, monkeypatch):
    test_static_fetch_end_to_end(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(static_fetch, "census_attica", lambda *a: calls.append("census"))
    monkeypatch.setattr(static_fetch, "attica_nuts", lambda *a: calls.append("nuts"))
    report = static_fetch.main(raw=str(tmp_path / "raw"), out=str(tmp_path / "static"), sleep=lambda s: None)
    assert calls == []
    assert report["census"]["cached"] is True and report["census"]["cells"] > 0


def test_existing_outputs_send_no_new_requests(tmp_path):
    from buaro import static_fetch
    out = tmp_path / "out"
    out.mkdir()
    (out / "urban_atlas_2021_by_nuts3.xlsx").write_bytes(b"x")

    class Silent:
        def post_request(self, *a, **k):
            raise AssertionError("request sent")

    entry = static_fetch.clms_layer(Silent(), "urban_atlas_2021", static_fetch.CLMS_FILES["urban_atlas_2021"], tmp_path / "raw", out, None, previous={"rows": 147})
    assert entry == {"rows": 147, "cached": True}


def test_cell_features_measure_known_shares():
    cells = gpd.GeoDataFrame({"GRD_ID": ["a", "b"], "NUTS_ID": ["EL301", "EL302"], "T": [1000, -9999], "Y_GE65": [200, 0]},
                             geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)], crs="EPSG:3035")
    ua = gpd.GeoDataFrame({"code_2021": ["11100", "14110", "50000", "11230"]},
                          geometry=[box(0, 0, 500, 1000), box(500, 0, 800, 1000), box(800, 0, 1000, 1000), box(1000, 0, 2000, 1000)], crs="EPSG:3035")
    trees = gpd.GeoDataFrame({"STL": [1]}, geometry=[box(0, 0, 400, 1000)], crs="EPSG:3035")
    table = static_fetch.cell_features(cells, ua, "code_2021", trees).set_index("GRD_ID")
    assert abs(table.loc["a", "land_m2"] - 800000) < 1
    assert abs(table.loc["a", "dense_fabric_share"] - 0.625) < 1e-9
    assert abs(table.loc["a", "green_urban_share"] - 0.375) < 1e-9
    assert abs(table.loc["a", "tree_share"] - 0.5) < 1e-9
    assert table.loc["b", "dense_fabric_share"] == 0 and table.loc["b", "artificial_share"] == 1
    assert static_fetch.pd.isna(table.loc["b", "T"])
    exposure = static_fetch.exposure_by_region(table).set_index("NUTS_ID")
    assert abs(exposure.loc["EL301", "tree_share_pop"] - 0.5) < 1e-9
    assert abs(exposure.loc["EL301", "share_65plus_census"] - 0.2) < 1e-9


def test_cells_built_once(tmp_path, monkeypatch):
    test_static_fetch_end_to_end(tmp_path, monkeypatch)
    out = tmp_path / "static"
    for name in ("cells_attica.xlsx", "exposure_by_nuts3.xlsx", "cells_landuse.xlsx"):
        assert (out / name).exists()
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["cells"]["cells"] > 0 and report["cells"]["from"] == "Copernicus layers"
    first = static_fetch.pd.read_excel(out / "exposure_by_nuts3.xlsx")
    assert (first["green_urban_share_pop"] > 0).any()
    monkeypatch.setattr(static_fetch.clms.Client, "__new__", lambda cls, *a, **k: (_ for _ in ()).throw(AssertionError("client created")))
    again = static_fetch.main(raw=str(tmp_path / "raw"), out=str(out), sleep=lambda s: None)
    assert again["cells"]["from"] == "stored land use" and again["cells"]["cells"] == report["cells"]["cells"]
    assert all(v.get("cached") for v in again["clms"].values())
    second = static_fetch.pd.read_excel(out / "exposure_by_nuts3.xlsx")
    static_fetch.pd.testing.assert_frame_equal(first, second)
