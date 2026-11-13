import numpy as np
import pandas as pd

from .model import fit_region, predict


def training_slice(panel, cutoff):
    return panel[(~panel["pandemic"]) & (panel["date"] < cutoff)]


def summer_mask(frame, summer):
    return frame["isoweek"].between(summer[0], summer[1])


def run_year(panel, year, percentile, study, kind="serfling"):
    summer = study["summer_weeks"]
    cutoff = pd.Timestamp(year=year, month=6, day=1)
    rows = []
    for geo, group in panel.groupby("geo"):
        train = training_slice(group, cutoff)
        test = group[(group["year"] == year) & summer_mask(group, summer)]
        if len(train) < int(study["min_training_weeks"]) or test.empty:
            continue
        with_heat = fit_region(train, percentile, study, kind, True)
        without_heat = fit_region(train, percentile, study, kind, False, thr=with_heat["thr"])
        p1 = predict(with_heat, test)
        p0 = predict(without_heat, test)
        out = test[["geo", "week", "date", "deaths", "tmax"]].copy()
        out["year"] = year
        out["baseline"] = kind
        out["threshold"] = with_heat["thr"]["hot"]
        out["predicted_with_heat"] = p1["expected"].to_numpy()
        out["predicted_without_heat"] = p0["expected"].to_numpy()
        out["heat_attributable"] = p1["attributable"].to_numpy()
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def skill(observed, with_heat, without_heat):
    e1 = observed - with_heat
    e0 = observed - without_heat
    mse1 = float(np.mean(e1 ** 2))
    mse0 = float(np.mean(e0 ** 2))
    return {
        "rmse_with_heat": float(np.sqrt(mse1)),
        "rmse_without_heat": float(np.sqrt(mse0)),
        "mae_with_heat": float(np.mean(np.abs(e1))),
        "mae_without_heat": float(np.mean(np.abs(e0))),
        "skill_score": float(1 - mse1 / mse0) if mse0 > 0 else np.nan,
    }


def summarise(weekly, n_regions_expected):
    if weekly.empty:
        return pd.DataFrame(), pd.DataFrame()
    total = (
        weekly.groupby(["baseline", "year", "week", "date"], as_index=False)
        .agg(
            regions=("geo", "nunique"),
            deaths=("deaths", "sum"),
            predicted_with_heat=("predicted_with_heat", "sum"),
            predicted_without_heat=("predicted_without_heat", "sum"),
            heat_attributable=("heat_attributable", "sum"),
            tmax=("tmax", "mean"),
        )
        .sort_values("date")
    )
    total = total[total["regions"] == n_regions_expected]
    rows = []
    for (kind, year), group in total.groupby(["baseline", "year"]):
        s = skill(group["deaths"].to_numpy(), group["predicted_with_heat"].to_numpy(), group["predicted_without_heat"].to_numpy())
        excess = group["deaths"] - group["predicted_without_heat"]
        corr = float(np.corrcoef(excess, group["heat_attributable"])[0, 1]) if group["heat_attributable"].std() > 0 else np.nan
        rows.append(
            {
                "baseline": kind,
                "year": int(year),
                "weeks": int(len(group)),
                "observed_deaths": float(group["deaths"].sum()),
                "predicted_with_heat": float(group["predicted_with_heat"].sum()),
                "predicted_without_heat": float(group["predicted_without_heat"].sum()),
                "heat_attributable_deaths": float(group["heat_attributable"].sum()),
                "excess_vs_no_heat_baseline": float(excess.sum()),
                "correlation_excess_vs_attributable": corr,
                **s,
            }
        )
    return total.reset_index(drop=True), pd.DataFrame(rows)
