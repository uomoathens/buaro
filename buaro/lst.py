import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from pyproj import Transformer

from . import grid100

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}"
LANDSAT = "landsat-c2-l2"
DEM = "cop-dem-glo-30"
SCALE, OFFSET = 0.00341802, 149.0
MAX_CLOUD = 20
MIN_VALID = 0.5
FIRST_YEAR = 2014
FIRST_TEST_YEAR = 2023
MIN_URBAN = 0.5
DOSE_BINS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 1.0]
FEATURES = ["tree_share", "green_urban_share", "dense_fabric_share", "artificial_share", "elevation"]
LABELS = {
    "tree_share": "Street tree cover",
    "green_urban_share": "Green urban areas",
    "dense_fabric_share": "Dense, highly sealed urban fabric",
    "artificial_share": "Built-up land",
    "elevation": "Elevation",
}
UNIT = {"tree_share": 0.1, "green_urban_share": 0.1, "dense_fabric_share": 0.1, "artificial_share": 0.1, "elevation": 100.0}
MIN_LAND = 0.9
BLOCK = 5000


def wgs84_bounds(g):
    t = Transformer.from_crs(g["crs"], "EPSG:4326", always_xy=True)
    xmin, ymin, xmax, ymax = g["bounds"]
    xs, ys = t.transform([xmin, xmin, xmax, xmax], [ymin, ymax, ymin, ymax])
    return [min(xs), min(ys), max(xs), max(ys)]


def search(session, collection, bbox, datetime=None, query=None):
    body = {"collections": [collection], "bbox": bbox, "limit": 100}
    if datetime:
        body["datetime"] = datetime
    if query:
        body["query"] = query
    items, url, method = [], STAC, "POST"
    for _ in range(50):
        r = session.request(method, url, json=body if method == "POST" else None, timeout=120)
        r.raise_for_status()
        page = r.json()
        items.extend(page.get("features", []))
        nxt = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        if not nxt:
            break
        url, method = nxt["href"], nxt.get("method", "GET")
        body = nxt.get("body", body) if method == "POST" else None
    return items


def summer_scenes(session, bbox, year):
    query = {"eo:cloud_cover": {"lt": MAX_CLOUD}, "platform": {"in": ["landsat-8", "landsat-9"]}}
    items = search(session, LANDSAT, bbox, f"{year}-06-01T00:00:00Z/{year}-08-31T23:59:59Z", query)
    return sorted(items, key=lambda i: i["properties"].get("datetime", ""))


def token(session, collection):
    r = session.get(TOKEN.format(collection=collection), timeout=60)
    r.raise_for_status()
    return r.json()["token"]


def signed(href, sas):
    return f"{href}{'&' if '?' in href else '?'}{sas}"


def window_read(path, g):
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    with rasterio.open(path) as src:
        left, bottom, right, top = transform_bounds(g["crs"], src.crs, *g["bounds"], densify_pts=21)
        window = from_bounds(left, bottom, right, top, src.transform).round_offsets().round_lengths()
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(1, window=window)
        return data, src.window_transform(window), src.crs


GDAL = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF", "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "5"}


def scene_lst(st_path, qa_path, g):
    import rasterio
    with rasterio.Env(**GDAL):
        return scene_lst_inner(st_path, qa_path, g)


def scene_lst_inner(st_path, qa_path, g):
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    st, transform, crs = window_read(st_path, g)
    qa, _, _ = window_read(qa_path, g)
    qa = qa.astype(np.int64)
    clear = (st > 0) & ((qa & 1) == 0) & (((qa >> 6) & 1) == 1) & (((qa >> 7) & 1) == 0)
    celsius = np.where(clear, st.astype("float64") * SCALE + OFFSET - 273.15, np.nan).astype("float32")
    out = np.full((g["height"], g["width"]), np.nan, dtype="float32")
    reproject(celsius, out, src_transform=transform, src_crs=crs, src_nodata=np.nan,
              dst_transform=g["transform"], dst_crs=g["crs"], dst_nodata=np.nan, resampling=Resampling.average)
    return out


def dem_grid(session, g, bbox):
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    import rasterio
    items = search(session, DEM, bbox)
    if not items:
        raise RuntimeError("no DEM tiles")
    sas = token(session, DEM)
    env = rasterio.Env(**GDAL)
    env.__enter__()
    sources = [rasterio.open(signed(i["assets"]["data"]["href"], sas)) for i in items]
    try:
        mosaic, transform = merge(sources, bounds=wgs84_bounds(g))
        crs = sources[0].crs
    finally:
        for s in sources:
            s.close()
        env.__exit__(None, None, None)
    from rasterio.warp import reproject
    out = np.full((g["height"], g["width"]), np.nan, dtype="float32")
    reproject(mosaic[0].astype("float32"), out, src_transform=transform, src_crs=crs, src_nodata=np.nan,
              dst_transform=g["transform"], dst_crs=g["crs"], dst_nodata=np.nan, resampling=Resampling.average)
    return out


def urban_mask(static_dir, g):
    land = pd.read_parquet(Path(static_dir) / "landuse_100m.parquet") if (Path(static_dir) / "landuse_100m.parquet").exists() else None
    mask = np.zeros(g["height"] * g["width"], dtype=bool)
    if land is not None:
        mask[land.loc[land["artificial_share"].fillna(0) >= 0.5, "cell"].to_numpy()] = True
    else:
        cells = grid100.land_cells(static_dir)
        table = pd.read_excel(Path(static_dir) / "cells_attica.xlsx", dtype={"GRD_ID": str})
        urban = set(table.loc[table["artificial_share"] >= 0.5, "GRD_ID"])
        km = grid100.parent(g).ravel()
        keys = set()
        for geom in cells[cells["GRD_ID"].astype(str).isin(urban)].geometry:
            x, y = geom.centroid.x, geom.centroid.y
            keys.add(int(np.floor(x / 1000)) * 100000 + int(np.floor(y / 1000)))
        mask = np.isin(km, list(keys))
    return mask.reshape(g["height"], g["width"])


def year_summary(session, g, bbox, year, urban, sleep=time.sleep):
    items = summer_scenes(session, bbox, year)
    sas = token(session, LANDSAT) if items else None
    total = np.zeros((g["height"], g["width"]))
    level = np.zeros((g["height"], g["width"]))
    count = np.zeros((g["height"], g["width"]))
    scenes = []
    for item in items:
        entry = {"id": item["id"], "date": item["properties"].get("datetime"), "cloud": item["properties"].get("eo:cloud_cover")}
        try:
            assets = item["assets"]
            lst = None
            for attempt in range(3):
                try:
                    lst = scene_lst(signed(assets["lwir11"]["href"], sas), signed(assets["qa_pixel"]["href"], sas), g)
                    break
                except Exception as error:
                    if attempt == 2:
                        raise
                    sleep(20)
                    sas = token(session, LANDSAT)
            valid = np.isfinite(lst) & urban
            entry["urban_valid"] = float(valid.sum() / max(urban.sum(), 1))
            if entry["urban_valid"] < MIN_VALID:
                entry["used"] = False
            else:
                ref = float(np.nanmedian(lst[valid]))
                ok = np.isfinite(lst)
                total[ok] += lst[ok] - ref
                level[ok] += lst[ok]
                count[ok] += 1
                entry.update({"used": True, "urban_median_c": ref})
        except Exception as error:
            entry.update({"used": False, "error": str(error)[:300]})
        scenes.append(entry)
    with np.errstate(invalid="ignore", divide="ignore"):
        table = grid100.frame(g, n=count, anomaly=total / count, lst=level / count)
    table = table[table["n"] > 0][["cell", "n", "anomaly", "lst"]].reset_index(drop=True)
    table["year"] = year
    return table, scenes


def within(frame, columns, key="km"):
    out = frame.copy()
    means = out.groupby(key)[columns].transform("mean")
    out[columns] = out[columns] - means
    return out


def urban(land):
    return land[land["artificial_share"].fillna(0) >= MIN_URBAN]


def assemble(land, dem, years, period):
    land = urban(land)
    sub = years[years["year"].isin(period)]
    if sub.empty:
        return pd.DataFrame()
    w = sub.assign(wa=sub["anomaly"] * sub["n"])
    agg = w.groupby("cell").agg(n=("n", "sum"), wa=("wa", "sum"))
    agg["anomaly"] = agg["wa"] / agg["n"]
    data = land.merge(agg[["n", "anomaly"]], left_on="cell", right_index=True, how="inner")
    if dem is not None:
        data = data.merge(dem, on="cell", how="left")
    data = data[(data["land_frac"] >= MIN_LAND) & (data["n"] >= 3)].dropna(subset=["anomaly"] + [f for f in FEATURES if f in data.columns])
    sizes = data.groupby("km")["cell"].transform("size")
    return data[sizes >= 5].reset_index(drop=True)


def fit(data, features):
    w = within(data, ["anomaly"] + features)
    X = w[features].to_numpy()
    y = w["anomaly"].to_numpy()
    groups = (np.floor(data["x"] / BLOCK).astype(np.int64) * 100000 + np.floor(data["y"] / BLOCK).astype(np.int64)).to_numpy()
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return model, w


def effects(model, features, period, n):
    rows = []
    ci = np.asarray(model.conf_int())
    for i, f in enumerate(features):
        u = UNIT[f]
        rows.append({"period": period, "feature": f, "label": LABELS[f], "unit": "+10 percentage points" if u == 0.1 else "+100 m",
                     "effect_c": float(model.params[i] * u), "low_c": float(ci[i, 0] * u), "high_c": float(ci[i, 1] * u),
                     "p_value": float(model.pvalues[i]), "cells": int(n)})
    return rows


def dose(data, features):
    others = [f for f in features if f != "tree_share"]
    labels = [f"{int(lo * 100)}-{int(hi * 100)}%" for lo, hi in zip(DOSE_BINS[:-1], DOSE_BINS[1:])]
    bins = pd.cut(data["tree_share"], [-1e-9] + DOSE_BINS[1:], labels=labels)
    dummies = pd.get_dummies(bins, drop_first=True).astype(float)
    dummies.columns = [str(c) for c in dummies.columns]
    dummies = dummies.loc[:, dummies.sum() >= 30]
    frame = pd.concat([data[["x", "y", "km", "anomaly"] + others], dummies], axis=1)
    model, _ = fit(frame, others + list(dummies.columns))
    ci = np.asarray(model.conf_int())
    counts = bins.value_counts()
    rows = [{"tree_cover": labels[0], "effect_c": 0.0, "low_c": 0.0, "high_c": 0.0, "cells": int(counts.get(labels[0], 0)), "reference": True}]
    for j, name in enumerate(dummies.columns):
        i = len(others) + j
        rows.append({"tree_cover": name, "effect_c": float(model.params[i]), "low_c": float(ci[i, 0]), "high_c": float(ci[i, 1]),
                     "cells": int(counts.get(name, 0)), "reference": False})
    return pd.DataFrame(rows)


def analyse(land, dem, years):
    train_years = sorted(int(y) for y in years["year"].unique() if y < FIRST_TEST_YEAR)
    test_years = sorted(int(y) for y in years["year"].unique() if y >= FIRST_TEST_YEAR)
    train = assemble(land, dem, years, train_years)
    test = assemble(land, dem, years, test_years)
    features = [f for f in FEATURES if f in train.columns and train[f].std() > 0]
    if len(train) < 500 or not features:
        return pd.DataFrame(), pd.DataFrame(), {"error": "too few cells", "train_cells": int(len(train))}, pd.DataFrame()
    model, _ = fit(train, features)
    beta = np.asarray(model.params)
    rows = effects(model, features, "training summers", len(train))
    validation = {"train_years": train_years, "test_years": test_years, "train_cells": int(len(train)),
                  "train_r2_within": float(model.rsquared), "min_built_up": MIN_URBAN}
    by_year = []
    if len(test) >= 500:
        test_model, wt = fit(test, features)
        rows += effects(test_model, features, "unseen summers", len(test))
        resid = wt["anomaly"].to_numpy() - wt[features].to_numpy() @ beta
        validation.update({"test_cells": int(len(test)), "test_r2_within": float(1 - (resid ** 2).sum() / (wt["anomaly"] ** 2).sum()),
                           "test_rmse_c": float(np.sqrt((resid ** 2).mean())), "test_sd_c": float(wt["anomaly"].std())})
        for y in test_years:
            one = assemble(land, dem, years, [y])
            if len(one) < 500:
                continue
            m1, w1 = fit(one, features)
            r1 = w1["anomaly"].to_numpy() - w1[features].to_numpy() @ beta
            i = features.index("tree_share") if "tree_share" in features else None
            entry = {"summer": int(y), "cells": int(len(one)), "r2_within": float(1 - (r1 ** 2).sum() / (w1["anomaly"] ** 2).sum()),
                     "rmse_c": float(np.sqrt((r1 ** 2).mean()))}
            if i is not None:
                c1 = np.asarray(m1.conf_int())
                entry.update({"tree_effect_c": float(m1.params[i] * UNIT["tree_share"]), "tree_low_c": float(c1[i, 0] * UNIT["tree_share"]),
                              "tree_high_c": float(c1[i, 1] * UNIT["tree_share"])})
            by_year.append(entry)
    curve = dose(train, features) if "tree_share" in features else pd.DataFrame()
    return pd.DataFrame(rows), pd.DataFrame(by_year), validation, curve


def anomaly_map(g, land, years, nuts_path, path):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = years.assign(wa=years["anomaly"] * years["n"]).groupby("cell").agg(n=("n", "sum"), wa=("wa", "sum"))
    mean = np.full(g["width"] * g["height"], np.nan)
    ok = land.loc[land["land_frac"] >= MIN_LAND, "cell"]
    w = w.loc[w.index.intersection(ok)]
    mean[w.index.to_numpy()] = (w["wa"] / w["n"]).to_numpy()
    grid = mean.reshape(g["height"], g["width"])
    xmin, ymin, xmax, ymax = g["bounds"]
    fig, ax = plt.subplots(figsize=(7, 9.6), dpi=130)
    image = ax.imshow(grid, extent=(xmin, xmax, ymin, ymax), cmap="RdBu_r", vmin=-6, vmax=6, interpolation="nearest")
    if Path(nuts_path).exists():
        gpd.read_file(nuts_path).to_crs(g["crs"]).boundary.plot(ax=ax, color="#3a3a3a", linewidth=0.6)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_axis_off()
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("Summer surface temperature compared with the city median (°C)")
    bar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def scene_log(out, report):
    path = out / "scenes.xlsx"
    old = pd.read_excel(path) if path.exists() else pd.DataFrame()
    new = pd.DataFrame([dict(s, summer=int(y)) for y, v in report["years"].items() if isinstance(v, dict) for s in v.get("detail", [])])
    log = pd.concat([old, new], ignore_index=True)
    if not log.empty and "id" in log.columns:
        log = log.drop_duplicates("id", keep="last").sort_values(["summer", "date"] if "date" in log.columns else ["summer"]).reset_index(drop=True)
    log.to_excel(path, index=False)
    return log


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", default="data/static")
    parser.add_argument("--out", default="data/lst")
    parser.add_argument("--results", default="results")
    parser.add_argument("--last-year", type=int, default=None)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now("UTC")
    last = args.last_year or (today.year if today.month >= 10 else today.year - 1)
    report = {"run": today.strftime("%Y-%m-%dT%H:%M:%SZ"), "years": {}}
    g = grid100.spec(args.static)
    bbox = wgs84_bounds(g)
    report["grid"] = {"width": g["width"], "height": g["height"], "bounds": g["bounds"], "bbox_wgs84": bbox}
    session = requests.Session()
    urban = urban_mask(args.static, g)
    report["urban_cells"] = int(urban.sum())
    for year in range(FIRST_YEAR, last + 1):
        path = out / f"lst_{year}.parquet"
        if path.exists():
            report["years"][year] = {"cached": True}
            continue
        try:
            table, scenes = year_summary(session, g, bbox, year, urban)
            used = sum(1 for s in scenes if s.get("used"))
            if used:
                table.to_parquet(path, index=False)
            report["years"][year] = {"scenes": len(scenes), "used": used, "cells": int(len(table)), "detail": scenes}
        except Exception as error:
            report["years"][year] = {"error": str(error)[:500]}
    dem_path = out / "elevation_100m.parquet"
    if not dem_path.exists():
        try:
            dem = dem_grid(session, g, bbox)
            grid100.frame(g, elevation=dem)[["cell", "elevation"]].dropna().to_parquet(dem_path, index=False)
            report["dem"] = {"cells": int(np.isfinite(dem).sum())}
        except Exception as error:
            report["dem"] = {"error": str(error)[:500]}
    else:
        report["dem"] = {"cached": True}
    land_path = Path(args.static) / "landuse_100m.parquet"
    files = sorted(out.glob("lst_*.parquet"))
    if land_path.exists() and files:
        land = pd.read_parquet(land_path)
        dem = pd.read_parquet(dem_path) if dem_path.exists() else None
        years = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        eff, by_year, validation, curve = analyse(land, dem, years)
        report["analysis"] = validation
        scenes = scene_log(out, report)
        summary = years.groupby("year").agg(cells=("cell", "size"), scenes=("n", "max"), mean_lst_c=("lst", "mean")).reset_index()
        Path(args.results).mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(Path(args.results) / "buaro_attica_lst.xlsx") as xl:
            eff.to_excel(xl, sheet_name="Cooling effects", index=False)
            curve.to_excel(xl, sheet_name="Tree cover curve", index=False)
            pd.DataFrame([validation]).astype(str).to_excel(xl, sheet_name="Validation", index=False)
            by_year.to_excel(xl, sheet_name="Unseen summers", index=False)
            summary.to_excel(xl, sheet_name="Summers", index=False)
            scenes.to_excel(xl, sheet_name="Scenes", index=False)
        try:
            anomaly_map(g, land, years, Path(args.static) / "attica_nuts3.geojson", out / "lst_map.png")
            report["map"] = "lst_map.png"
        except Exception as error:
            report["map_error"] = str(error)[:300]
        (out / "summary.json").write_text(json.dumps({
            "effects": eff.to_dict("records"), "curve": curve.to_dict("records"), "unseen": by_year.to_dict("records"),
            "validation": validation, "summers": summary.to_dict("records"), "scenes_used": int(summary["scenes"].sum()),
            "map": report.get("map"),
        }, indent=2, default=float), encoding="utf-8")
    else:
        report["analysis"] = {"waiting_for": [p for p, ok in (("landuse_100m.parquet", land_path.exists()), ("lst years", bool(files))) if not ok]}
    slim = {k: v for k, v in report.items() if k != "years"}
    slim["years"] = {y: {k: v for k, v in d.items() if k != "detail"} for y, d in report["years"].items()}
    (out / "report.json").write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
