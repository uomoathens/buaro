import numpy as np
import pandas as pd


def week_key(label):
    year, week = str(label).split("-W")
    return int(year), int(week)


def week_start(label):
    year, week = week_key(label)
    return pd.Timestamp.fromisocalendar(year, week, 1)


def in_window(label, start, end):
    return week_key(start) <= week_key(label) <= week_key(end)


def build_panel(deaths, climate, population, study):
    panel = deaths.merge(climate, on=["geo", "week"], how="inner")
    keys = panel["week"].map(week_key)
    panel["year"] = keys.map(lambda k: k[0]).astype(int)
    panel["isoweek"] = keys.map(lambda k: k[1]).astype(int)
    panel = panel[panel["year"] >= int(study["start_year"])].copy()
    panel["date"] = panel["week"].map(week_start)
    panel = panel.sort_values(["geo", "date"]).reset_index(drop=True)
    panel["tmax_lag1"] = panel.groupby("geo")["tmax"].shift(1)
    panel["tmean_lag1"] = panel.groupby("geo")["tmean"].shift(1)
    contiguous = panel.groupby("geo")["date"].diff().dt.days.eq(7)
    panel.loc[~contiguous, ["tmax_lag1", "tmean_lag1"]] = np.nan
    panel = panel.dropna(subset=["tmax_lag1", "tmean_lag1", "deaths"]).copy()
    excl = study["pandemic_exclusion"]
    panel["pandemic"] = panel["week"].map(lambda w: in_window(w, excl["start"], excl["end"]))
    panel["population"] = np.nan
    panel["share_65plus"] = np.nan
    if population is not None and not population.empty:
        pieces = []
        for geo, group in panel.groupby("geo"):
            pop = population[population["geo"] == geo].sort_values("year")
            pop = pop.dropna(subset=["population"])
            if pop.empty:
                pieces.append(group)
                continue
            years = pop["year"].to_numpy(dtype=float)
            frac = group["year"].to_numpy(dtype=float) + (group["isoweek"].to_numpy(dtype=float) - 1) / 52.0
            group = group.copy()
            group["population"] = np.interp(frac, years, pop["population"].to_numpy(dtype=float))
            share = (pop["population_65plus"] / pop["population"]).to_numpy(dtype=float)
            ok = np.isfinite(share)
            if ok.any():
                group["share_65plus"] = np.interp(frac, years[ok], share[ok])
            pieces.append(group)
        panel = pd.concat(pieces, ignore_index=True)
    return panel.sort_values(["geo", "date"]).reset_index(drop=True)


def coverage(panel):
    return (
        panel.groupby("geo")
        .agg(first_week=("week", "first"), last_week=("week", "last"), weeks=("week", "size"), deaths=("deaths", "sum"))
        .reset_index()
    )
