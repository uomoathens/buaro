import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


def fetch_jsonstat(dataset, filters, cache_dir=None, api=API, retries=4, timeout=120):
    params = [("format", "JSON"), ("lang", "EN")]
    for key, values in filters.items():
        if isinstance(values, (list, tuple)):
            params.extend((key, v) for v in values)
        else:
            params.append((key, values))
    cache_file = None
    if cache_dir is not None:
        tag = re.sub(r"[^A-Za-z0-9]+", "_", "_".join(f"{k}{v}" for k, v in params[2:]))[:120]
        cache_file = Path(cache_dir) / f"{dataset}_{tag}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(f"{api}/{dataset}", params=params, timeout=timeout)
            if response.status_code == 200:
                payload = response.json()
                if cache_file is not None:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(payload), encoding="utf-8")
                return payload
            last_error = RuntimeError(f"{dataset}: HTTP {response.status_code} {response.text[:300]}")
        except requests.RequestException as error:
            last_error = error
        time.sleep(5 * (attempt + 1))
    raise last_error


def jsonstat_to_frame(payload):
    ids = payload["id"]
    sizes = payload["size"]
    dims = payload["dimension"]
    codes = []
    for dim in ids:
        index = dims[dim]["category"]["index"]
        if isinstance(index, list):
            ordered = index
        else:
            ordered = [code for code, _ in sorted(index.items(), key=lambda item: item[1])]
        codes.append(ordered)
    values = payload.get("value", {})
    if isinstance(values, list):
        items = [(i, v) for i, v in enumerate(values) if v is not None]
    else:
        items = [(int(k), v) for k, v in values.items() if v is not None]
    if not items:
        return pd.DataFrame(columns=ids + ["value"])
    flat = np.array([i for i, _ in items], dtype=np.int64)
    vals = np.array([float(v) for _, v in items])
    positions = np.unravel_index(flat, sizes)
    data = {dim: np.asarray(codes[d], dtype=object)[positions[d]] for d, dim in enumerate(ids)}
    data["value"] = vals
    return pd.DataFrame(data)


def labels(payload, dimension):
    return dict(payload["dimension"][dimension]["category"].get("label", {}))


def weekly_deaths(regions, datasets, cache_dir=None, api=API):
    errors = []
    for dataset in datasets:
        filters = {"geo": regions, "unit": "NR"}
        if dataset.endswith("_ts"):
            filters["sex"] = "T"
        frame = None
        for attempt_filters in (filters, {"geo": regions}):
            try:
                frame = jsonstat_to_frame(fetch_jsonstat(dataset, attempt_filters, cache_dir, api))
                break
            except Exception as error:
                errors.append(f"{dataset} {sorted(attempt_filters)}: {error}")
        if frame is None:
            continue
        if "unit" in frame.columns:
            frame = frame[frame["unit"] == "NR"]
        if frame.empty:
            errors.append(f"{dataset}: empty response")
            continue
        if "sex" in frame.columns:
            frame = frame[frame["sex"] == "T"]
        frame = frame.assign(time=frame["time"].astype(str).map(normalise_week)).dropna(subset=["time"])
        out = frame.rename(columns={"time": "week", "value": "deaths"})[["geo", "week", "deaths"]]
        out = out.groupby(["geo", "week"], as_index=False)["deaths"].sum()
        out.attrs["dataset"] = dataset
        return out
    raise RuntimeError("No weekly deaths dataset could be read: " + " | ".join(errors))


def normalise_week(label):
    m = re.fullmatch(r"(\d{4})-?W(\d{1,2})", str(label).strip())
    if not m:
        return None
    week = int(m.group(2))
    if week < 1 or week > 53:
        return None
    return f"{m.group(1)}-W{week:02d}"


def parse_age(code):
    code = str(code)
    m = re.fullmatch(r"Y(\d+)-(\d+)", code)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"Y_GE(\d+)", code)
    if m:
        return int(m.group(1)), None
    m = re.fullmatch(r"Y_LT(\d+)", code)
    if m:
        return 0, int(m.group(1)) - 1
    return None


def partition_from(codes, start):
    parsed = {c: parse_age(c) for c in codes}
    parsed = {c: p for c, p in parsed.items() if p is not None}
    chosen = []
    lo = start
    while True:
        open_code = next((c for c, (a, b) in parsed.items() if a == lo and b is None), None)
        if open_code is not None:
            chosen.append(open_code)
            return chosen
        closed = sorted(((b - a, c, b) for c, (a, b) in parsed.items() if a == lo and b is not None))
        if not closed:
            return None
        _, code, hi = closed[0]
        chosen.append(code)
        lo = hi + 1


def population(regions, dataset, cache_dir=None, api=API):
    try:
        payload = fetch_jsonstat(dataset, {"geo": regions, "unit": "NR", "sex": "T"}, cache_dir, api)
    except Exception:
        payload = fetch_jsonstat(dataset, {"geo": regions}, cache_dir, api)
    frame = jsonstat_to_frame(payload)
    if "unit" in frame.columns:
        frame = frame[frame["unit"] == "NR"]
    if "sex" in frame.columns:
        frame = frame[frame["sex"] == "T"]
    frame["year"] = frame["time"].astype(str).str[:4].astype(int)
    rows = []
    for (geo, year), group in frame.groupby(["geo", "year"]):
        by_age = dict(zip(group["age"], group["value"]))
        total = by_age.get("TOTAL")
        if total is None:
            base = partition_from(by_age.keys(), 0)
            total = sum(by_age[c] for c in base) if base else np.nan
        older = partition_from(by_age.keys(), 65)
        pop65 = sum(by_age[c] for c in older) if older else np.nan
        rows.append({"geo": geo, "year": year, "population": total, "population_65plus": pop65})
    return pd.DataFrame(rows)
