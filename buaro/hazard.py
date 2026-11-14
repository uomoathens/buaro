import argparse
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATASET = "multi-origin-c3s-atlas"
CDS = "https://cds.climate.copernicus.eu/api"
PROCESS = f"{CDS}/retrieve/v1/processes/{DATASET}"
GWL_URL = "https://raw.githubusercontent.com/IPCC-WG1/Atlas/main/warming-levels/CMIP6_Atlas_WarmingLevels.csv"
EXPERIMENTS = {"historical": "1850-2014", "ssp1_2_6": "2015-2100", "ssp2_4_5": "2015-2100", "ssp3_7_0": "2015-2100", "ssp5_8_5": "2015-2100"}
SSP_CODE = {"ssp1_2_6": "ssp126", "ssp2_4_5": "ssp245", "ssp3_7_0": "ssp370", "ssp5_8_5": "ssp585"}
AREA = [39.0, 23.0, 37.0, 25.0]
POINT = (37.98, 23.73)
LEVELS = (1.5, 2.0, 3.0)
PRESENT = (2011, 2030)
MONTHS = (6, 7, 8, 9)
SERIES = "cmip6_attica_tx.xlsx"
DELTAS = "warming_level_deltas.xlsx"


def enum_values(schema):
    out = []
    if isinstance(schema, dict):
        for key in ("enum",):
            if key in schema:
                out.extend(schema[key])
        for value in schema.values():
            out.extend(enum_values(value))
    elif isinstance(schema, list):
        for value in schema:
            out.extend(enum_values(value))
    return out


def describe(session):
    r = session.get(PROCESS, timeout=60)
    r.raise_for_status()
    inputs = r.json().get("inputs", {})
    return {name: sorted(set(str(v) for v in enum_values(spec)))[:200] for name, spec in inputs.items()}


def pick_variable(options):
    values = options.get("variable", [])
    for v in values:
        low = v.lower()
        if "daily_maximum_temperature" in low and "extreme" not in low and "hot" not in low and "frost" not in low:
            return v
    return "monthly_mean_of_daily_maximum_temperature"


def gwl_table(session):
    r = session.get(GWL_URL, timeout=60)
    r.raise_for_status()
    from io import StringIO
    table = pd.read_csv(StringIO(r.text), na_values=["NA"])
    rows = []
    for rec in table.to_dict("records"):
        for column, year in rec.items():
            if column == "model_run" or pd.isna(year):
                continue
            level, ssp = column.split("_", 1)
            rows.append({"model_run": rec["model_run"], "level": float(level), "ssp": ssp, "central_year": int(year)})
    return pd.DataFrame(rows)


def member_key(member):
    parts = str(member).split("_")
    return "_".join(parts[-2:]) if len(parts) >= 3 else str(member)


def read_point(paths, variable_hint="tx"):
    import xarray as xr
    frames = []
    for path in paths:
        ds = xr.open_dataset(path)
        name = variable_hint if variable_hint in ds.data_vars else [v for v in ds.data_vars if ds[v].ndim >= 3][0]
        da = ds[name]
        lat = "lat" if "lat" in da.dims else "latitude"
        lon = "lon" if "lon" in da.dims else "longitude"
        point = da.interp({lat: POINT[0], lon: POINT[1]}, method="linear")
        if bool(np.isnan(point).all()):
            point = da.sel({lat: POINT[0], lon: POINT[1]}, method="nearest")
        if "member_id" in ds.coords:
            members = [str(m) for m in ds["member_id"].values]
        elif "member" in ds.coords and ds["member"].dtype.kind in "OUS":
            members = [str(m) for m in ds["member"].values]
        else:
            members = [f"member{i}" for i in range(point.sizes.get("member", 1))]
        values = point.transpose("member", "time").values if "member" in point.dims else point.values[None, :]
        times = pd.to_datetime(point["time"].values)
        for i, m in enumerate(members):
            frames.append(pd.DataFrame({"member": m, "model_run": member_key(m), "year": times.year, "month": times.month, "tx": values[i]}))
        ds.close()
    data = pd.concat(frames, ignore_index=True)
    if data["tx"].dropna().mean() > 200:
        data["tx"] = data["tx"] - 273.15
    return data


def retrieve(experiment, variable, raw, key, client=None):
    target = Path(raw) / f"tx_{experiment}.zip"
    if not target.exists():
        if client is None:
            import cdsapi
            client = cdsapi.Client(url=CDS, key=key, quiet=True, progress=False)
        request = {"origin": "cmip6", "experiment": experiment, "period": EXPERIMENTS[experiment], "variable": variable,
                   "bias_adjustment": "no_bias_adjustment", "area": AREA}
        client.retrieve(DATASET, request).download(str(target))
    paths = []
    if zipfile.is_zipfile(target):
        with zipfile.ZipFile(target) as z:
            for name in z.namelist():
                if name.endswith(".nc"):
                    z.extract(name, Path(raw) / experiment)
                    paths.append(Path(raw) / experiment / name)
    else:
        paths.append(target)
    return paths


def deltas(series, gwl):
    rows = []
    summer = series[series["month"].isin(MONTHS)]
    hist = summer[summer["experiment"] == "historical"]
    for ssp_name, code in SSP_CODE.items():
        fut = summer[summer["experiment"] == ssp_name]
        if fut.empty:
            continue
        both = pd.concat([hist, fut], ignore_index=True)
        for run, g in both.groupby("model_run"):
            base = g[g["year"].between(*PRESENT)].groupby("month")["tx"].mean()
            if base.empty:
                continue
            for level in LEVELS:
                hit = gwl[(gwl["model_run"] == run) & (gwl["ssp"] == code) & (gwl["level"] == level)]
                if hit.empty:
                    continue
                c = int(hit["central_year"].iloc[0])
                win = g[g["year"].between(c - 9, c + 10)].groupby("month")["tx"].mean()
                if win.empty or g["year"].max() < c + 10:
                    continue
                for month in MONTHS:
                    if month in win.index and month in base.index:
                        rows.append({"model_run": run, "ssp": code, "level": level, "central_year": c, "month": month,
                                     "delta_c": float(win[month] - base[month])})
    per_run = pd.DataFrame(rows)
    if per_run.empty:
        return per_run, pd.DataFrame()
    by_model = per_run.groupby(["model_run", "level", "month"], as_index=False)["delta_c"].mean()
    summary = by_model.groupby(["level", "month"])["delta_c"].agg(
        delta_median=lambda s: float(np.median(s)), delta_p10=lambda s: float(np.percentile(s, 10)),
        delta_p90=lambda s: float(np.percentile(s, 90)), models="size").reset_index()
    return per_run, summary


def main(argv=None, session=None, client=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/climate")
    parser.add_argument("--out", default="data/climate")
    args = parser.parse_args(argv)
    raw, out = Path(args.raw), Path(args.out)
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    report = {"run": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        gwl = gwl_table(session)
        gwl.to_excel(out / "ipcc_warming_levels_cmip6.xlsx", index=False)
        report["warming_levels"] = {"runs": int(gwl["model_run"].nunique())}
    except Exception as error:
        report["warming_levels"] = {"error": str(error)[:400]}
        gwl = None
    try:
        options = describe(session)
        report["form"] = {k: v[:40] for k, v in options.items()}
    except Exception as error:
        options = {}
        report["form"] = {"error": str(error)[:400]}
    variable = pick_variable(options)
    report["variable"] = variable
    series_path = out / SERIES
    if series_path.exists():
        series = pd.read_excel(series_path)
        report["series"] = {"cached": True, "members": int(series["model_run"].nunique())}
    else:
        key = os.environ.get("CDSAPI_KEY", "").strip()
        if not key and client is None:
            report["series"] = {"waiting_for": "CDSAPI_KEY secret"}
            series = None
        else:
            parts, status = [], {}
            for experiment in EXPERIMENTS:
                try:
                    data = read_point(retrieve(experiment, variable, raw, key, client))
                    data["experiment"] = experiment
                    parts.append(data)
                    status[experiment] = {"members": int(data["model_run"].nunique()), "years": [int(data["year"].min()), int(data["year"].max())]}
                except Exception as error:
                    status[experiment] = {"error": str(error)[:400]}
            report["series"] = status
            series = pd.concat(parts, ignore_index=True) if parts else None
            if series is not None and "historical" in status and "members" in status["historical"]:
                series.to_excel(series_path, index=False)
    if series is not None and gwl is not None:
        per_run, summary = deltas(series, gwl)
        if not summary.empty:
            with pd.ExcelWriter(out / DELTAS) as xl:
                summary.to_excel(xl, sheet_name="Ensemble", index=False)
                per_run.to_excel(xl, sheet_name="Per model run", index=False)
            report["deltas"] = summary.round(3).to_dict("records")
        else:
            report["deltas"] = {"error": "no model run matched the IPCC warming level table"}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
