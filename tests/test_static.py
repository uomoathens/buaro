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
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(mapping[url], dest)
        return Path(dest)

    monkeypatch.setattr(static_fetch, "download", fake_download)
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
    ua = report["clms"]["urban_atlas_2021"]
    assert ua["class_column"] == "code_2021" and ua["population_columns"] == ["Pop2021"]
    assert "UserEmail" not in json.dumps(report)
    for name in ["attica_nuts3.geojson", "census_grid_attica.gpkg", "street_tree_layer_2021_by_nuts3.xlsx", "urban_atlas_2021_by_nuts3.xlsx", "report.json"]:
        assert (out / name).exists()
