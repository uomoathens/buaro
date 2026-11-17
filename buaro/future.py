from pathlib import Path

import numpy as np
import pandas as pd

DELTAS = "warming_level_deltas.xlsx"
SUMMER_WEEKS = (23, 39)
MONTHS = (6, 7, 8, 9)
LEVELS = (1.5, 2.0, 3.0)


def load(climate_dir):
    path = Path(climate_dir) / DELTAS
    if not path.exists():
        return None
    return pd.read_excel(path, sheet_name="Ensemble")


def month_of(dates):
    return pd.to_datetime(dates).dt.month.clip(lower=MONTHS[0], upper=MONTHS[-1])


def year_ranges(years):
    years = sorted(int(y) for y in years)
    if not years:
        return ""
    spans = []
    start = prev = years[0]
    for y in years[1:]:
        if y == prev + 1:
            prev = y
            continue
        spans.append((start, prev))
        start = prev = y
    spans.append((start, prev))
    parts = [str(a) if a == b else f"{a} to {b}" for a, b in spans]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def full_summers(panel, weeks=None):
    weeks = weeks or SUMMER_WEEKS
    frame = panel[(~panel["pandemic"]) & panel["isoweek"].between(*weeks) & panel["deaths"].notna()]
    counts = frame.groupby(["geo", "year"])["isoweek"].nunique()
    full = counts[counts == weeks[1] - weeks[0] + 1].reset_index()
    per_year = full.groupby("year")["geo"].nunique()
    return sorted(int(y) for y, n in per_year.items() if n == panel["geo"].nunique())


def heat(tmax, lag, thr):
    return (np.clip(tmax - thr, 0, None) + np.clip(lag - thr, 0, None)) / 2


def per_summer(panel, vuln, weeks):
    frame = panel[(~panel["pandemic"]) & panel["isoweek"].between(*weeks) & panel["deaths"].notna()].copy()
    full = frame.groupby(["geo", "year"])["isoweek"].transform("nunique") == weeks[1] - weeks[0] + 1
    frame = frame[full]
    parts = []
    for v in vuln.itertuples():
        g = frame[frame["geo"] == v.geo]
        if g.empty:
            continue
        h = heat(g["tmax"].to_numpy(), g["tmax_lag1"].to_numpy(), v.threshold_c)
        deaths = g["deaths"].to_numpy() * (1 - np.exp(-v.beta_shrunk * h))
        parts.append(pd.DataFrame({"geo": v.geo, "year": g["year"].to_numpy(), "deaths": deaths}))
    if not parts:
        return pd.Series(dtype=float)
    out = pd.concat(parts, ignore_index=True)
    areas = out.groupby("year")["geo"].nunique()
    totals = out.groupby("year")["deaths"].sum()
    return totals[areas == len(vuln)]


def project(panel, vuln, ensemble):
    if ensemble is None or ensemble.empty:
        return pd.DataFrame(), pd.DataFrame()
    weeks = panel[(~panel["pandemic"]) & panel["isoweek"].between(*SUMMER_WEEKS) & panel["deaths"].notna()].copy()
    full = weeks.groupby(["geo", "year"])["isoweek"].transform("nunique") == SUMMER_WEEKS[1] - SUMMER_WEEKS[0] + 1
    weeks = weeks[full].copy()
    weeks["month"] = month_of(weeks["date"])
    weeks["lag_month"] = month_of(pd.to_datetime(weeks["date"]) - pd.Timedelta(days=7))
    summers = sorted(weeks["year"].unique())
    rows = []
    for v in vuln.itertuples():
        g = weeks[weeks["geo"] == v.geo]
        if g.empty:
            continue
        beta = {"central": v.beta_shrunk, "low": np.log(v.rr_low), "high": np.log(v.rr_high)}
        h_now = heat(g["tmax"].to_numpy(), g["tmax_lag1"].to_numpy(), v.threshold_c)
        base = g["deaths"].to_numpy() / np.exp(beta["central"] * h_now)
        for level in (0.0,) + LEVELS:
            for band, dcol, bkey in (("central", "delta_median", "central"), ("low", "delta_p10", "low"), ("high", "delta_p90", "high")):
                if level == 0.0:
                    shift = np.zeros(len(g))
                    lag_shift = shift
                    models = None
                else:
                    e = ensemble[ensemble["level"] == level].set_index("month")
                    if e.empty:
                        continue
                    shift = g["month"].map(e[dcol]).fillna(e[dcol].mean()).to_numpy()
                    lag_shift = g["lag_month"].map(e[dcol]).fillna(e[dcol].mean()).to_numpy()
                    models = int(e["models"].min())
                h = heat(g["tmax"].to_numpy() + shift, g["tmax_lag1"].to_numpy() + lag_shift, v.threshold_c)
                deaths = base * (np.exp(beta[bkey] * h) - 1)
                per_year = pd.DataFrame({"year": g["year"].to_numpy(), "deaths": deaths, "hot": (g["tmax"].to_numpy() + shift) > v.threshold_c})
                agg = per_year.groupby("year").agg(deaths=("deaths", "sum"), hot=("hot", "sum"))
                rows.append({"geo": v.geo, "name_en": v.name_en, "level": level, "band": band,
                             "heat_deaths_per_summer": float(agg["deaths"].mean()), "hot_weeks_per_summer": float(agg["hot"].mean()),
                             "mean_shift_c": float(np.mean(shift)), "models": models, "summers": len(summers)})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    wide = detail.pivot_table(index=["geo", "name_en", "level"], columns="band", values=["heat_deaths_per_summer", "hot_weeks_per_summer", "mean_shift_c"]).reset_index()
    wide.columns = ["_".join(c for c in col if c) for col in wide.columns]
    total = detail.groupby(["level", "band"], as_index=False)["heat_deaths_per_summer"].sum()
    total = total.pivot(index="level", columns="band", values="heat_deaths_per_summer").reset_index()
    hot = detail[detail["band"] == "central"].groupby("level", as_index=False)["hot_weeks_per_summer"].mean()
    shift = detail[detail["band"] == "central"].groupby("level", as_index=False)["mean_shift_c"].mean()
    models = detail.groupby("level", as_index=False)["models"].max()
    attica = total.merge(hot, on="level").merge(shift, on="level").merge(models, on="level")
    attica = attica.rename(columns={"central": "heat_deaths_central", "low": "heat_deaths_low", "high": "heat_deaths_high",
                                    "hot_weeks_per_summer": "hot_weeks_per_area", "mean_shift_c": "summer_warming_c"})
    now = float(attica.loc[attica["level"] == 0.0, "heat_deaths_central"].iloc[0])
    attica["times_today"] = attica["heat_deaths_central"] / now if now > 0 else np.nan
    return attica, wide
