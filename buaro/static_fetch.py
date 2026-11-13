import json
import re
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import requests

from . import clms

NUTS_URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2021_4326_LEVL_3.geojson"
CENSUS_URL = "https://gisco-services.ec.europa.eu/census/2021/Eurostat_Census-GRID_2021_V3.zip"
CLMS_FILES = {
    "street_tree_layer_2021": {"dataset": "efc14569ae0548148a1eba237ea2df14", "file": "f91d4546-ca91-4f09-8e53-02157ee4aa79"},
    "urban_atlas_2021": {"dataset": "52717fed21404857a943c1ef92e505ab", "file": "acc22b3c-ef21-4a98-9d49-0b39257ebe3c"},
}
REGIONS = ["EL301", "EL302", "EL303", "EL304", "EL305", "EL306", "EL307"]
EQUAL_AREA = "EPSG:3035"


def find_values(obj, pattern):
    rx = re.compile(pattern, re.I)
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if rx.fullmatch(str(k)) and isinstance(v, (str, int, float)):
                out.append(v)
            out.extend(find_values(v, pattern))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(find_values(v, pattern))
    return out


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items() if not re.search(r"(mail|user|token|key)", str(k), re.I)}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def classify(status):
    s = str(status).lower()
    if "nok" in s or "reject" in s or "cancel" in s or "error" in s or "fail" in s:
        return "failed"
    if "finished" in s or "complete" in s or s in {"ok", "done", "ready"}:
        return "done"
    return "pending"


def download(url, dest, headers=None, chunk=1 << 20):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600, headers=headers or {}) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for part in r.iter_content(chunk):
                f.write(part)
    return dest


BROWSER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36",
    "Accept": "application/zip,application/octet-stream,*/*;q=0.8",
    "Referer": "https://land.copernicus.eu/",
}


def probe_response(url, headers):
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True)
        info = {"status": r.status_code, "final_url": r.url, "server": r.headers.get("Server"), "type": r.headers.get("Content-Type"),
                "length": r.headers.get("Content-Length"), "body": r.raw.read(400, decode_content=True).decode("utf-8", "replace") if r.status_code >= 400 else ""}
        r.close()
        return info
    except Exception as error:
        return {"error": str(error)[:300]}


def download_clms(client, url, dest, attempts=8, wait=60, sleep=time.sleep):
    tried = [{"diagnostic_plain": probe_response(url, {}), "diagnostic_browser": probe_response(url, BROWSER)}]
    for attempt in range(attempts):
        for auth in (False, True):
            headers = dict(BROWSER)
            if auth:
                try:
                    headers["Authorization"] = client.headers(True)["Authorization"]
                except Exception as error:
                    tried.append({"attempt": attempt, "auth": auth, "error": str(error)[:200]})
                    continue
            try:
                download(url, dest, headers=headers)
                tried.append({"attempt": attempt, "auth": auth, "status": "ok"})
                return dest, tried
            except requests.HTTPError as error:
                code = error.response.status_code if error.response is not None else None
                tried.append({"attempt": attempt, "auth": auth, "status": code})
        sleep(wait)
    return None, tried


def request_clms(client, dataset, file_id):
    response = client.post_request([{"DatasetID": dataset, "FileID": file_id}])
    try:
        body = response.json()
    except ValueError:
        body = {"text": response.text[:1000]}
    tasks = [str(t) for t in find_values(body, r"task_?id")]
    return {"status": response.status_code, "body": sanitize(body), "tasks": tasks}


def wait_clms(client, task_id, minutes=150, every=60, sleep=time.sleep):
    history = []
    deadline = time.time() + minutes * 60
    while True:
        response = client.status(task_id)
        try:
            body = response.json()
        except ValueError:
            body = {"text": response.text[:1000]}
        states = find_values(body, r"status")
        state = classify(states[0]) if states else "pending"
        history.append({"http": response.status_code, "status": states[:1]})
        if state != "pending":
            urls = [u for u in find_values(body, r"download_?url") if str(u).startswith("http")]
            urls = urls or [u for u in find_values(body, r".*url.*") if str(u).startswith("http")]
            return {"state": state, "body": body, "urls": urls, "history": history[-5:]}
        if time.time() > deadline:
            return {"state": "timeout", "body": body, "urls": [], "history": history[-5:]}
        sleep(every)


def extract_layers(archive, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    found = []
    with zipfile.ZipFile(archive) as z:
        z.extractall(folder)
    for path in folder.rglob("*"):
        if path.suffix.lower() == ".zip":
            found.extend(extract_layers(path, path.with_suffix("")))
        elif path.suffix.lower() in {".fgb", ".gpkg", ".shp", ".geojson"}:
            found.append(path)
    return sorted(set(found))


def describe(path):
    info = pyogrio.read_info(path)
    return {
        "path": str(path),
        "features": int(info.get("features", -1)),
        "crs": str(info.get("crs")),
        "geometry": str(info.get("geometry_type")),
        "fields": [str(f) for f in info.get("fields", [])],
        "bounds": [float(b) for b in info.get("total_bounds", [])] if info.get("total_bounds") is not None else None,
    }


def attica_nuts(raw, out):
    path = download(NUTS_URL, Path(raw) / "nuts3_2021.geojson")
    nuts = gpd.read_file(path)
    nuts = nuts[nuts["NUTS_ID"].isin(REGIONS)][["NUTS_ID", "NAME_LATN", "geometry"]].reset_index(drop=True)
    nuts.to_file(Path(out) / "attica_nuts3.geojson", driver="GeoJSON")
    return nuts


def class_column(columns):
    for c in columns:
        if re.fullmatch(r"(code|class)_?20\d\d", c, re.I):
            return c
    for c in columns:
        if re.search(r"code", c, re.I):
            return c
    return None


def population_columns(columns):
    return [c for c in columns if re.match(r"pop", c, re.I)]


def by_region(layer, nuts, group=None):
    layer = layer.to_crs(EQUAL_AREA)
    regions = nuts.to_crs(EQUAL_AREA)[["NUTS_ID", "geometry"]]
    keep = ["geometry"] + ([group] if group else [])
    parts = gpd.overlay(layer[keep], regions, how="intersection", keep_geom_type=True)
    parts["area_m2"] = parts.geometry.area
    keys = ["NUTS_ID"] + ([group] if group else [])
    table = parts.groupby(keys, as_index=False)["area_m2"].sum()
    totals = regions.assign(region_m2=regions.geometry.area)[["NUTS_ID", "region_m2"]]
    return table.merge(totals, on="NUTS_ID", how="left")


def census_attica(raw, out, nuts):
    archive = download(CENSUS_URL, Path(raw) / "census_grid_2021.zip")
    layers = extract_layers(archive, Path(raw) / "census_grid_2021")
    described = []
    for path in layers:
        try:
            described.append(describe(path))
        except Exception as error:
            described.append({"path": str(path), "error": str(error)[:300]})
    target = next((d for d in described if "error" not in d and d["features"] > 0), None)
    if target is None:
        return {"layers": described, "cells": 0}
    crs = target["crs"]
    box = nuts.to_crs(crs).total_bounds
    cells = gpd.read_file(target["path"], bbox=tuple(box))
    cells = cells.to_crs(EQUAL_AREA)
    points = cells.copy()
    points["geometry"] = cells.geometry.representative_point()
    joined = gpd.sjoin(points, nuts.to_crs(EQUAL_AREA)[["NUTS_ID", "geometry"]], how="inner", predicate="within")
    cells = cells.loc[joined.index].assign(NUTS_ID=joined["NUTS_ID"].values)
    cells.to_file(Path(out) / "census_grid_attica.gpkg", driver="GPKG")
    return {"layers": described, "used": target["path"], "cells": int(len(cells)), "fields": [c for c in cells.columns if c != "geometry"]}


def clms_layer(client, name, spec, raw, out, nuts, sleep=time.sleep, previous=None):
    if (Path(out) / f"{name}_by_nuts3.xlsx").exists():
        entry = dict(previous or {})
        entry["cached"] = True
        return entry
    entry = {"request": request_clms(client, spec["dataset"], spec["file"])}
    if not entry["request"]["tasks"]:
        return entry
    result = wait_clms(client, entry["request"]["tasks"][0], sleep=sleep)
    entry["wait"] = {k: v for k, v in result.items() if k != "body"}
    entry["wait"]["body"] = sanitize(json.loads(json.dumps(result["body"], default=str)))
    if result["state"] != "done" or not result["urls"]:
        return entry
    archive, tried = download_clms(client, result["urls"][0], Path(raw) / f"{name}.zip", sleep=sleep)
    entry["download"] = tried[:1] + tried[-4:]
    if archive is None:
        return entry
    layers = extract_layers(archive, Path(raw) / name)
    entry["layers"] = [describe(p) for p in layers]
    if not layers:
        return entry
    layer = gpd.read_file(layers[0])
    group = class_column(layer.columns) if name == "urban_atlas_2021" else None
    table = by_region(layer, nuts, group)
    table.to_excel(Path(out) / f"{name}_by_nuts3.xlsx", index=False)
    entry["class_column"] = group
    entry["population_columns"] = population_columns(layer.columns)
    if entry["population_columns"] and group:
        pop = layer.groupby(group)[entry["population_columns"]].sum(numeric_only=True).reset_index()
        pop.to_excel(Path(out) / f"{name}_population_by_class.xlsx", index=False)
    entry["rows"] = int(len(table))
    return entry


def main(raw="data/raw/static", out="data/static", sleep=time.sleep):
    Path(raw).mkdir(parents=True, exist_ok=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    old = Path(out) / "report.json"
    try:
        previous = json.loads(old.read_text(encoding="utf-8")).get("clms", {}) if old.exists() else {}
    except Exception:
        previous = {}
    report = {"run": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        cached = Path(out) / "attica_nuts3.geojson"
        nuts = gpd.read_file(cached) if cached.exists() else attica_nuts(raw, out)
        report["nuts"] = {"regions": nuts["NUTS_ID"].tolist(), "cached": cached.exists()}
    except Exception as error:
        report["nuts"] = {"error": str(error)[:500]}
        nuts = None
    if nuts is not None:
        try:
            grid = Path(out) / "census_grid_attica.gpkg"
            if grid.exists():
                cells = gpd.read_file(grid)
                report["census"] = {"cached": True, "cells": int(len(cells)), "fields": [c for c in cells.columns if c != "geometry"]}
            else:
                report["census"] = census_attica(raw, out, nuts)
        except Exception as error:
            report["census"] = {"error": str(error)[:500]}
        client = clms.Client()
        try:
            client.key = clms.service_key()
            report["clms_auth"] = "ok" if clms.access_token(client.key) else "failed"
        except Exception as error:
            report["clms_auth"] = f"failed: {str(error)[:200]}"
        if report["clms_auth"] == "ok":
            report["clms"] = {}
            for name, spec in CLMS_FILES.items():
                try:
                    report["clms"][name] = clms_layer(client, name, spec, raw, out, nuts, sleep, previous.get(name))
                except Exception as error:
                    report["clms"][name] = {"error": str(error)[:500]}
    (Path(out) / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else list(v)) for k, v in report.items()}, indent=2, default=str))
    return report


if __name__ == "__main__":
    main()
