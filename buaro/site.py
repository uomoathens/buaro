import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE = Path(__file__).with_name("site_template.html")

SOURCES = [
    {"name": "Eurostat, demo_r_mwk3_t", "url": "https://ec.europa.eu/eurostat/databrowser/view/demo_r_mwk3_t/default/table?lang=en", "what": "deaths by week and NUTS 3 region"},
    {"name": "Eurostat, demo_r_pjangrp3", "url": "https://ec.europa.eu/eurostat/databrowser/view/demo_r_pjangrp3/default/table?lang=en", "what": "population by age group and NUTS 3 region"},
    {"name": "Copernicus Climate Change Service, ERA5-Land", "url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land", "what": "daily temperature reanalysis, retrieved through the Open-Meteo archive API"},
    {"name": "Eurostat, Census 2021 population grid", "url": "https://ec.europa.eu/eurostat/en/web/gisco/geodata/population-distribution/population-grids", "what": "residents and residents aged 65+ per 1 km cell"},
    {"name": "Copernicus Land Monitoring Service, Urban Atlas 2021 and Street Tree Layer 2021", "url": "https://land.copernicus.eu/en/products/urban-atlas", "what": "land use, sealing classes and street tree cover for the Athens functional urban area"},
    {"name": "EuroMOMO, methods", "url": "https://www.euromomo.eu/how-it-works/methods", "what": "calm spring and autumn weeks as the basis of expected mortality"},
    {"name": "Janoš et al. (2025), Nature Medicine", "url": "https://doi.org/10.1038/s41591-025-03954-7", "what": "heat-related mortality in Europe during 2024, used as a cross-check"},
    {"name": "European Commission, COM(2026) 720", "url": "https://climate.ec.europa.eu/document/download/476ef20a-0c66-4be2-a104-70ba96622477_en?filename=com_2026_720_en.pdf", "what": "Communication on the EU Framework for Climate Resilience"},
]


def clean(value):
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def records(frame, columns):
    if frame is None or frame.empty:
        return []
    return [clean(dict(zip(columns, row))) for row in frame[columns].itertuples(index=False)]


def panel_payload(panel):
    out = {}
    for geo, group in panel.sort_values("date").groupby("geo"):
        weeks = group["week"].tolist()
        out[geo] = {
            "index": {w: i for i, w in enumerate(weeks)},
            "deaths": [clean(v) for v in group["deaths"].to_numpy(dtype=float)],
            "tmax": [None if not np.isfinite(v) else round(float(v), 2) for v in group["tmax"].to_numpy(dtype=float)],
        }
    return out


def backtest_payload(weekly, winner):
    out = {}
    if weekly is None or weekly.empty:
        return out
    best = weekly[weekly["baseline"] == winner]
    for row in best.itertuples(index=False):
        out.setdefault(row.geo, {})[row.week] = [round(float(row.predicted_with_heat), 2), round(float(row.predicted_without_heat), 2)]
    return out


def build(docs_dir, meta, vuln, summary, weekly, response, robustness, panel, regions, mods=None, areas=None, files=()):
    docs = Path(docs_dir)
    data_dir = docs / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        if Path(f).exists():
            shutil.copy2(f, data_dir / Path(f).name)
    winner = meta["baseline_selected"]
    best = summary[summary["baseline"] == winner] if summary is not None and not summary.empty else pd.DataFrame()
    best2024 = best.loc[best["year"] == 2024, "heat_attributable_deaths"] if not best.empty else pd.Series(dtype=float)
    years = sorted(int(y) for y, g in panel[~panel["pandemic"]].groupby("year") if g["isoweek"].between(14, 44).sum() > 0)
    payload = {
        "meta": clean(meta),
        "areas": [{"code": r["code"], "name": r["name_en"]} for r in regions],
        "thresholds": {r.geo: clean(r.threshold_c) for r in vuln.itertuples()},
        "vuln": records(vuln, ["geo", "name_en", "rr_per_degree", "rr_low", "rr_high", "threshold_c", "share_65plus"]),
        "summary": records(summary, ["baseline", "year", "observed_deaths", "predicted_with_heat", "predicted_without_heat", "heat_attributable_deaths", "rmse_with_heat", "rmse_without_heat", "skill_score"]),
        "response": records(response, ["year", "summer_excess_deaths", "degree_weeks_above_threshold", "excess_per_degree_week", "hottest_week_tmax"]),
        "robustness": records(robustness, ["left_out_summer", "delta", "delta_low", "delta_high", "verdict"]),
        "panel": panel_payload(panel),
        "backtest": backtest_payload(weekly, winner),
        "years": years,
        "backtestYears": sorted(int(y) for y in best["year"].unique()) if not best.empty else [],
        "best2024": clean(float(best2024.iloc[0])) if not best2024.empty else None,
        "improved": int((best["skill_score"] > 0).sum()) if not best.empty else 0,
        "tested": int(len(best)),
        "mods": records(mods, ["covariate", "label", "min", "max", "change_pp_per_sd", "change_low", "change_high", "p_value", "explained_heterogeneity", "verdict"]),
        "areaFeatures": records(areas, [c for c in (areas.columns if areas is not None else []) if c not in ("name_el",)]),
        "sources": SOURCES,
    }
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", text)
    (docs / "index.html").write_text(html, encoding="utf-8")
    (docs / ".nojekyll").write_text("", encoding="utf-8")
    return docs / "index.html"
