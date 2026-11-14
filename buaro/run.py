import argparse
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import backtest, climate, data, eurostat, model, modifiers, report, site


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def fetch_all(config, cache_dir, end_date):
    study = config["study"]
    sources = config["sources"]
    regions = config["regions"]
    codes = [r["code"] for r in regions]
    try:
        deaths = eurostat.weekly_deaths(codes, sources["deaths_datasets"], cache_dir, sources["eurostat_api"])
        dataset = deaths.attrs.get("dataset", "")
    except RuntimeError:
        deaths = pd.DataFrame(columns=["geo", "week", "deaths"])
        dataset = ""
    available = sorted(deaths["geo"].unique())
    used_regions = [r for r in regions if r["code"] in available]
    level = "NUTS3"
    if not used_regions:
        nuts2 = study["nuts2"]
        deaths = eurostat.weekly_deaths([nuts2], sources["deaths_datasets"], cache_dir, sources["eurostat_api"])
        dataset = deaths.attrs.get("dataset", "")
        points = [p for r in regions for p in r["points"]]
        used_regions = [{"code": nuts2, "name_en": "Attica", "name_el": "Αττική", "points": points}]
        level = "NUTS2"
    used_codes = [r["code"] for r in used_regions]
    try:
        pop = eurostat.population(used_codes, sources["population_dataset"], cache_dir, sources["eurostat_api"])
    except Exception:
        pop = pd.DataFrame(columns=["geo", "year", "population", "population_65plus"])
    start = f"{int(study['start_year']) - 1}-12-01"
    temps = climate.regional_weekly(used_regions, start, end_date, cache_dir, sources["climate_api"], sources["climate_model"])
    return deaths, pop, temps, used_regions, dataset, level


def vulnerability(panel, regions, percentile, study, kind):
    names = {r["code"]: r for r in regions}
    fits = {}
    adapt = {}
    for geo, group in panel.groupby("geo"):
        train = group[~group["pandemic"]]
        fits[geo] = model.fit_region(train, percentile, study, kind, True)
        adapt[geo] = model.fit_region(train, percentile, study, kind, True, adaptation=True)
    geos = list(fits)
    pooled = model.pool([fits[g]["beta"] for g in geos], [fits[g]["var"] for g in geos])
    pooled_delta = model.pool([adapt[g]["delta"] for g in geos], [adapt[g]["delta_var"] for g in geos])
    pooled_delta["beta_2019"] = model.pool([adapt[g]["beta"] for g in geos], [adapt[g]["var"] for g in geos])["pooled"]
    rows = []
    for i, geo in enumerate(geos):
        b = float(pooled["blup"][i])
        se = float(np.sqrt(pooled["blup_var"][i]))
        share = panel.loc[panel["geo"] == geo, "share_65plus"].dropna()
        rows.append(
            {
                "geo": geo,
                "name_en": names[geo]["name_en"],
                "name_el": names[geo]["name_el"],
                "threshold_c": fits[geo]["thr"]["hot"],
                "beta_region": fits[geo]["beta"],
                "se_region": float(np.sqrt(fits[geo]["var"])),
                "beta_shrunk": b,
                "se_shrunk": se,
                "shrinkage_weight": float(pooled["shrink"][i]),
                "rr_per_degree": float(np.exp(b)),
                "rr_low": float(np.exp(b - 1.96 * se)),
                "rr_high": float(np.exp(b + 1.96 * se)),
                "delta_per_year": adapt[geo]["delta"],
                "delta_se": float(np.sqrt(adapt[geo]["delta_var"])),
                "share_65plus": float(share.iloc[-1]) if not share.empty else np.nan,
                "dispersion": fits[geo]["dispersion"],
                "model_weeks": fits[geo]["n"],
                "population_offset": fits[geo]["uses_offset"],
            }
        )
    return pd.DataFrame(rows), pooled, pooled_delta


def adaptation_summary(pooled_delta):
    d = pooled_delta["pooled"]
    se = float(np.sqrt(pooled_delta["pooled_var"]))
    b = pooled_delta["beta_2019"]
    lo, hi = d - 1.96 * se, d + 1.96 * se
    if hi < 0:
        verdict = "decreasing"
    elif lo > 0:
        verdict = "increasing"
    else:
        verdict = "no clear change"
    return {
        "delta": d,
        "delta_se": se,
        "delta_low": lo,
        "delta_high": hi,
        "rr_2015": float(np.exp(b + d * (2015 - model.ADAPTATION_YEAR))),
        "rr_2025": float(np.exp(b + d * (2025 - model.ADAPTATION_YEAR))),
        "verdict": verdict,
    }


def adaptation_robustness(panel, regions, percentile, study, kind, adaptation):
    summer = panel[~panel["pandemic"] & panel["isoweek"].between(26, 36)]
    counts = summer.groupby("year")["isoweek"].nunique()
    years = sorted(int(y) for y, n in counts.items() if n >= 8)
    rows = []
    for year in years:
        _, _, pooled_delta = vulnerability(panel[panel["year"] != year], regions, percentile, study, kind)
        a = adaptation_summary(pooled_delta)
        rows.append({"left_out_summer": year, "delta": a["delta"], "delta_low": a["delta_low"], "delta_high": a["delta_high"], "verdict": a["verdict"]})
    table = pd.DataFrame(rows)
    flips = [int(r.left_out_summer) for r in table.itertuples() if r.verdict != adaptation["verdict"]]
    return table, flips


def summer_response(panel, vuln):
    thr = vuln.set_index("geo")["threshold_c"]
    frame = panel[~panel["pandemic"]].copy()
    frame["degrees"] = np.clip(frame["tmax"] - frame["geo"].map(thr), 0, None)
    rows = []
    for year, group in frame.groupby("year"):
        calm = group[group["isoweek"].between(16, 25) | group["isoweek"].between(37, 44)]
        summer = group[group["isoweek"].between(26, 36)]
        if calm.empty or summer["isoweek"].nunique() < 11:
            continue
        normal = calm["deaths"].sum() / calm["isoweek"].nunique()
        weekly = summer.groupby("isoweek").agg(deaths=("deaths", "sum"), degrees=("degrees", "mean"), tmax=("tmax", "mean"))
        excess = float((weekly["deaths"] - normal).sum())
        degree_weeks = float(weekly["degrees"].sum())
        rows.append({
            "year": int(year),
            "normal_weekly_deaths": normal,
            "summer_excess_deaths": excess,
            "degree_weeks_above_threshold": degree_weeks,
            "hottest_week_tmax": float(weekly["tmax"].max()),
            "excess_per_degree_week": excess / degree_weeks if degree_weeks > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/attica.yaml")
    parser.add_argument("--out", default="results")
    parser.add_argument("--cache", default="data/raw")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--docs", default="docs")
    parser.add_argument("--static", default="data/static")
    parser.add_argument("--lst", default="data/lst")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    study = config["study"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    end_date = args.end_date or (dt.date.today() - dt.timedelta(days=7)).isoformat()

    deaths, pop, temps, regions, dataset, level = fetch_all(config, args.cache, end_date)
    panel = data.build_panel(deaths, temps, pop, study)
    if panel.empty:
        raise SystemExit("No overlapping deaths and temperature data.")
    first_test = min(int(y) for y in study["backtest_years"])
    cutoff = pd.Timestamp(year=first_test, month=6, day=1)
    selection_train = {
        geo: backtest.training_slice(g, cutoff) for geo, g in panel.groupby("geo")
        if len(backtest.training_slice(g, cutoff)) >= int(study["min_training_weeks"])
    }
    if not selection_train:
        raise SystemExit("Not enough training data before the first backtest year.")

    percentiles = {}
    scores = {}
    weekly = []
    for kind in model.BASELINES:
        percentiles[kind], scores[kind] = model.select_percentile(selection_train, study["threshold_percentiles"], study, kind)
        for year in study["backtest_years"]:
            result = backtest.run_year(panel, int(year), percentiles[kind], study, kind)
            if not result.empty:
                weekly.append(result)
    weekly = pd.concat(weekly, ignore_index=True) if weekly else pd.DataFrame()
    total, summary = backtest.summarise(weekly, len(regions))

    if summary.empty:
        winner = "serfling"
        comparison = pd.DataFrame()
    else:
        comparison = (
            summary.groupby("baseline")
            .agg(summers=("year", "nunique"), mean_rmse_with_heat=("rmse_with_heat", "mean"), mean_skill=("skill_score", "mean"),
                 summers_improved=("skill_score", lambda x: int((x > 0).sum())))
            .reset_index()
        )
        complete = comparison[comparison["summers"] == comparison["summers"].max()]
        winner = str(complete.sort_values("mean_rmse_with_heat").iloc[0]["baseline"])

    vuln, pooled, pooled_delta = vulnerability(panel, regions, percentiles[winner], study, winner)
    adaptation = adaptation_summary(pooled_delta)
    robustness, flips = adaptation_robustness(panel, regions, percentiles[winner], study, winner, adaptation)
    adaptation["robust"] = not flips
    adaptation["depends_on"] = flips
    response = summer_response(panel, vuln)
    mods, areas = modifiers.analyse(vuln, modifiers.load(args.static))
    se = np.sqrt(pooled["pooled_var"])
    ref = config["reference"]
    ref_attica = ref["greece_heat_deaths_2024"] * ref["attica_population_2021"] / ref["greece_population_2021"]

    meta = {
        "run_date": dt.date.today().isoformat(),
        "end_date": end_date,
        "deaths_dataset": dataset,
        "spatial_level": level,
        "regions": [r["code"] for r in regions],
        "first_week": panel["week"].iloc[panel["date"].argmin()],
        "last_week": panel["week"].iloc[panel["date"].argmax()],
        "baseline_selected": winner,
        "selection_rule": "lowest mean RMSE of the heat model over all tested summers",
        "percentile": int(percentiles[winner]),
        "percentiles": {k: int(v) for k, v in percentiles.items()},
        "percentile_scores": {k: {str(p): v for p, v in sc.items()} for k, sc in scores.items()},
        "pooled_beta": pooled["pooled"],
        "pooled_rr": float(np.exp(pooled["pooled"])),
        "pooled_rr_low": float(np.exp(pooled["pooled"] - 1.96 * se)),
        "pooled_rr_high": float(np.exp(pooled["pooled"] + 1.96 * se)),
        "tau2": pooled["tau2"],
        "i2": pooled["i2"],
        "adaptation": adaptation,
        "reference_attica_2024": ref_attica,
        "modifiers_available": bool(not mods.empty),
    }

    report.chart_vulnerability(vuln, out / "vulnerability.png")
    best = summary[summary["baseline"] == winner] if not summary.empty else summary
    best_total = total[total["baseline"] == winner] if not total.empty else total
    for stale in out.glob("backtest_*.png"):
        stale.unlink()
    if not best.empty:
        for year in best["year"]:
            report.chart_backtest(best_total, int(year), out / f"backtest_{int(year)}.png")
        report.chart_skill(best, out / "skill.png")
        report.chart_compare(summary, out / "baseline_comparison.png")

    cov = data.coverage(panel)
    sources = pd.DataFrame(
        [
            {"item": "Weekly deaths", "source": f"Eurostat {dataset}", "link": f"{config['sources']['eurostat_api']}/{dataset}"},
            {"item": "Population by age", "source": f"Eurostat {config['sources']['population_dataset']}", "link": f"{config['sources']['eurostat_api']}/{config['sources']['population_dataset']}"},
            {"item": "Daily temperature", "source": "ERA5-Land reanalysis via Open-Meteo archive API", "link": config["sources"]["climate_api"]},
            {"item": "Residents and residents aged 65+ per 1 km cell", "source": "Eurostat, Census 2021 population grid", "link": "https://ec.europa.eu/eurostat/en/web/gisco/geodata/population-distribution/population-grids"},
            {"item": "Land use and sealing classes", "source": "Copernicus Land Monitoring Service, Urban Atlas 2021", "link": "https://land.copernicus.eu/en/products/urban-atlas"},
            {"item": "Street tree cover", "source": "Copernicus Land Monitoring Service, Urban Atlas Street Tree Layer 2021", "link": "https://land.copernicus.eu/en/products/urban-atlas"},
            {"item": "Cross-check", "source": ref["label"], "link": "https://doi.org/10.1038/s41591-025-03954-7"},
        ]
    )
    meta_frame = pd.DataFrame([{"key": k, "value": str(v)} for k, v in meta.items()])
    report.workbook(
        out / "buaro_attica_step1.xlsx",
        [
            ("Summary", meta_frame, ["BUARO step 1: heat vulnerability and backtest for Attica."]),
            ("Vulnerability", vuln, ["rr_per_degree: relative risk of weekly death per 1°C of weekly mean Tmax above the area threshold, summed over lag 0 and lag 1, after empirical Bayes shrinkage."]),
            ("Backtest summary", summary, ["Each summer forecast uses only data up to 31 May of that year. skill_score = 1 - MSE(with heat) / MSE(without heat)."]),
            ("Baseline comparison", comparison, [f"Selected baseline: {winner}. Rule: lowest mean RMSE of the heat model over all tested summers."]),
            ("Backtest weekly Attica", total, None),
            ("Backtest weekly by area", weekly, None),
            ("Data coverage", cov, None),
            ("Adaptation test", pd.DataFrame([{k: (", ".join(map(str, v)) if isinstance(v, list) else v) for k, v in adaptation.items()}]), ["delta: change per year in the log relative risk per 1°C (reference year 2019). Negative means heat is becoming less deadly per degree."]),
            ("Adaptation robustness", robustness, ["The adaptation test repeated with one summer left out each time. The verdict counts as robust only if no single summer changes it."]),
            ("Summer response", response, ["Model-free check: summer excess deaths (weeks 26 to 36) above the calm-week level of the same year, per degree-week above the area thresholds."]),
            ("Heat modifiers", mods, ["Random-effects meta-regression of each area's heat effect on one area feature at a time. change_pp_per_sd: change in extra deaths per +1°C, in percentage points, for one standard deviation of the feature across areas, with 95% range (t distribution). With seven areas this is an association between areas, not proof of cause."]),
            ("Area exposure", areas, ["Area features weighted by the residents of each 1 km census cell (Eurostat Census 2021 grid) and measured on Copernicus Urban Atlas 2021 and its Street Tree Layer."]),
            ("Sources", sources, None),
        ],
    )
    panel_cols = ["geo", "week", "date", "year", "isoweek", "deaths", "tmax", "tmean", "tmax_lag1", "tmean_lag1", "population", "share_65plus", "pandemic"]
    report.workbook(out / "data_panel.xlsx", [("Panel", panel[panel_cols], ["Weekly deaths (Eurostat), ERA5-Land temperature and population used by the model."])])
    lst_path = Path(args.lst) / "summary.json"
    surface = json.loads(lst_path.read_text(encoding="utf-8")) if lst_path.exists() else None
    text = report.summary_markdown(meta, vuln, summary, comparison, ref, response, mods, surface)
    (out / "SUMMARY.md").write_text(text, encoding="utf-8")
    report.write_meta(out / "run_metadata.json", meta)
    site.build(args.docs, meta, vuln, summary, weekly, response, robustness, panel, regions, mods=mods, areas=areas, surface=surface,
               files=[out / "buaro_attica_step1.xlsx", out / "data_panel.xlsx", Path(args.static) / "cells_attica.xlsx", out / "buaro_attica_lst.xlsx"],
               images=[Path(args.lst) / "lst_map.png"])
    print(text)
    return meta


if __name__ == "__main__":
    main()
