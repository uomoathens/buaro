import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from buaro import grid100, lst, static_fetch


def small_grid(width=40, height=30, x0=5520000, y0=1760000):
    return {"transform": from_origin(x0, y0 + height * 100, 100, 100), "width": width, "height": height,
            "bounds": (x0, y0, x0 + width * 100, y0 + height * 100), "crs": "EPSG:3035"}


def write_tif(path, data, transform, crs):
    with rasterio.open(path, "w", driver="GTiff", width=data.shape[1], height=data.shape[0], count=1, dtype=data.dtype, crs=crs, transform=transform) as dst:
        dst.write(data, 1)


def test_scene_is_masked_scaled_and_averaged(tmp_path):
    g = small_grid()
    st = np.full((200, 200), int(round((35 + 273.15 - 149.0) / 0.00341802)), dtype="uint16")
    qa = np.full((200, 200), 1 << 6, dtype="uint16")
    qa[:, :100] = (1 << 3)
    transform = from_origin(5515000, 1768000, 30, 30)
    write_tif(tmp_path / "st.tif", st, transform, "EPSG:3035")
    write_tif(tmp_path / "qa.tif", qa, transform, "EPSG:3035")
    out = lst.scene_lst(str(tmp_path / "st.tif"), str(tmp_path / "qa.tif"), g)
    assert out.shape == (30, 40)
    ok = np.isfinite(out)
    assert ok.any() and (~ok).any()
    assert np.allclose(out[ok], 35, atol=0.01)
    x_edge = 5515000 + 100 * 30
    cols = (np.arange(40) + 0.5) * 100 + 5520000
    assert not ok[:, cols < x_edge - 100].any()


def test_search_follows_pages():
    class Session:
        def __init__(self):
            self.calls = []

        def request(self, method, url, json=None, timeout=None):
            self.calls.append((method, url, json))
            page = len(self.calls)
            body = {"features": [{"id": f"s{page}", "properties": {"datetime": f"2024-07-0{page}"}}], "links": []}
            if page == 1:
                body["links"] = [{"rel": "next", "href": "https://next", "method": "POST", "body": {"token": "t"}}]
            return type("R", (), {"raise_for_status": lambda self: None, "json": lambda self, b=body: b})()

    s = Session()
    items = lst.search(s, lst.LANDSAT, [0, 0, 1, 1], "2024-06-01/2024-08-31", {"eo:cloud_cover": {"lt": 20}})
    assert [i["id"] for i in items] == ["s1", "s2"]
    assert s.calls[1][2] == {"token": "t"}


def synthetic(seed=0, effect=-8.0, years=range(2016, 2026)):
    rng = np.random.default_rng(seed)
    g = small_grid(100, 100)
    land = grid100.frame(g, land_frac=np.ones(10000), tree_share=rng.uniform(0, 0.4, 10000), green_urban_share=rng.uniform(0, 0.3, 10000),
                         dense_fabric_share=rng.uniform(0, 1, 10000), artificial_share=rng.uniform(0.5, 1, 10000))
    dem = pd.DataFrame({"cell": land["cell"], "elevation": rng.uniform(0, 300, 10000)})
    km_effect = pd.Series(rng.normal(0, 2, land["km"].nunique()), index=sorted(land["km"].unique()))
    rows = []
    for y in years:
        anom = effect * land["tree_share"] + 1.5 * land["dense_fabric_share"] - 0.004 * dem["elevation"] + land["km"].map(km_effect).to_numpy() + rng.normal(0, 0.3, 10000)
        rows.append(pd.DataFrame({"cell": land["cell"], "n": 4, "anomaly": anom, "lst": 40 + anom, "year": y}))
    return land, dem, pd.concat(rows, ignore_index=True)


def test_analysis_recovers_cooling_and_validates_on_unseen_summers():
    land, dem, years = synthetic()
    eff, by_year, validation, curve = lst.analyse(land, dem, years)
    train = eff[eff["period"] == "training summers"].set_index("feature")
    assert abs(train.loc["tree_share", "effect_c"] + 0.8) < 0.05
    assert abs(train.loc["dense_fabric_share", "effect_c"] - 0.15) < 0.03
    assert abs(train.loc["elevation", "effect_c"] + 0.4) < 0.05
    assert validation["test_r2_within"] > 0.5
    assert set(by_year["summer"]) == {2023, 2024, 2025}
    assert curve["effect_c"].iloc[-1] < curve["effect_c"].iloc[1] < 0
    assert (eff["period"] == "unseen summers").any()


def test_landuse_100m_fractions():
    g = small_grid(2, 1)
    x0, y0 = g["bounds"][0], g["bounds"][1]
    ua = gpd.GeoDataFrame({"code_2021": ["11100", "14110", "50000", "31000"]},
                          geometry=[box(x0, y0, x0 + 50, y0 + 100), box(x0 + 50, y0, x0 + 80, y0 + 100), box(x0 + 80, y0, x0 + 100, y0 + 100), box(x0 + 100, y0, x0 + 200, y0 + 100)],
                          crs="EPSG:3035")
    trees = gpd.GeoDataFrame({"STL": [1]}, geometry=[box(x0, y0, x0 + 40, y0 + 100)], crs="EPSG:3035")
    table = static_fetch.landuse_100m(ua, "code_2021", trees, g).set_index("cell")
    assert abs(table.loc[0, "land_frac"] - 0.8) < 1e-9
    assert abs(table.loc[0, "dense_fabric_share"] - 0.625) < 1e-9
    assert abs(table.loc[0, "green_urban_share"] - 0.375) < 1e-9
    assert abs(table.loc[0, "tree_share"] - 0.5) < 1e-9
    assert table.loc[1, "artificial_share"] == 0 and table.loc[1, "land_frac"] == 1


def test_main_caches_years_and_writes_results(tmp_path, monkeypatch):
    static = tmp_path / "static"
    static.mkdir()
    land, dem, years = synthetic(years=range(2016, 2026))
    g = small_grid(100, 100)
    monkeypatch.setattr(lst.grid100, "spec", lambda d: g)
    land.to_parquet(static / "landuse_100m.parquet", index=False)
    calls = []

    def fake_year(session, g_, bbox, year, urban, sleep=None):
        calls.append(year)
        t = years[years["year"] == year].copy()
        return t, [{"id": f"x{year}", "used": True}]

    monkeypatch.setattr(lst, "year_summary", fake_year)
    monkeypatch.setattr(lst, "dem_grid", lambda s, g_, b: dem["elevation"].to_numpy().reshape(100, 100))
    out, res = tmp_path / "lst", tmp_path / "results"
    lst.main(["--static", str(static), "--out", str(out), "--results", str(res), "--last-year", "2025"])
    assert calls[0] == 2014 and calls[-1] == 2025
    report = json.loads((out / "report.json").read_text())
    assert report["analysis"]["test_r2_within"] > 0.5
    eff = pd.read_excel(res / "buaro_attica_lst.xlsx", sheet_name="Cooling effects")
    assert (eff["feature"] == "tree_share").any()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["curve"] and summary["unseen"] and summary["scenes_used"] > 0
    from buaro import report
    text = "\n".join(report.surface_markdown(summary))
    assert "Verdict: street trees cool the city surface" in text
    calls.clear()
    lst.main(["--static", str(static), "--out", str(out), "--results", str(res), "--last-year", "2025"])
    assert calls == []


def test_scene_in_utm_is_reprojected(tmp_path):
    from pyproj import Transformer
    g = small_grid()
    t = Transformer.from_crs("EPSG:3035", "EPSG:32634", always_xy=True)
    xs, ys = t.transform([g["bounds"][0], g["bounds"][2]], [g["bounds"][1], g["bounds"][3]])
    left, top = min(xs) - 1000, max(ys) + 1000
    st = np.full((300, 300), int(round((30 + 273.15 - 149.0) / 0.00341802)), dtype="uint16")
    qa = np.full((300, 300), 1 << 6, dtype="uint16")
    transform = from_origin(left, top, 30, 30)
    write_tif(tmp_path / "st.tif", st, transform, "EPSG:32634")
    write_tif(tmp_path / "qa.tif", qa, transform, "EPSG:32634")
    out = lst.scene_lst(str(tmp_path / "st.tif"), str(tmp_path / "qa.tif"), g)
    assert np.isfinite(out).mean() > 0.95
    assert np.allclose(out[np.isfinite(out)], 30, atol=0.01)


def test_years_without_scene_log_are_rebuilt(tmp_path, monkeypatch):
    static = tmp_path / "static"
    static.mkdir()
    land, dem, years = synthetic(years=range(2016, 2026))
    g = small_grid(100, 100)
    monkeypatch.setattr(lst.grid100, "spec", lambda d: g)
    land.to_parquet(static / "landuse_100m.parquet", index=False)
    out = tmp_path / "lst"
    out.mkdir()
    for y in range(2016, 2026):
        years[years["year"] == y].to_parquet(out / f"lst_{y}.parquet", index=False)
    calls = []

    def fake_year(session, g_, bbox, year, urban, sleep=None):
        calls.append(year)
        return years[years["year"] == year].copy(), [{"id": f"x{year}", "used": True}, {"id": f"y{year}", "used": False}]

    monkeypatch.setattr(lst, "year_summary", fake_year)
    monkeypatch.setattr(lst, "dem_grid", lambda s, g_, b: dem["elevation"].to_numpy().reshape(100, 100))
    lst.main(["--static", str(static), "--out", str(out), "--results", str(tmp_path / "r"), "--last-year", "2025"])
    assert calls == list(range(2014, 2026))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["scenes_used"] == 12 and summary["scenes_found"] == 24
