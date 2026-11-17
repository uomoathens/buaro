from pathlib import Path

import numpy as np
import pandas as pd

from . import future

AIR_TO_SURFACE = 0.006 / 0.075
PHYSICAL_STEPS = (0.05, 0.10)
BEST_OF_ATTICA = (("tree_share_pop", "Street trees"), ("green_urban_share_pop", "Green urban areas"))
TREES_TO_BEST_AIR = "Street trees up to the best area of Attica, cooler air only"


def summer_weeks(panel):
    weeks = panel[(~panel["pandemic"]) & panel["isoweek"].between(*future.SUMMER_WEEKS) & panel["deaths"].notna()].copy()
    full = weeks.groupby(["geo", "year"])["isoweek"].transform("nunique") == future.SUMMER_WEEKS[1] - future.SUMMER_WEEKS[0] + 1
    weeks = weeks[full].copy()
    weeks["month"] = future.month_of(weeks["date"])
    weeks["lag_month"] = future.month_of(pd.to_datetime(weeks["date"]) - pd.Timedelta(days=7))
    return weeks


def shifts(g, ensemble, level):
    if level == 0.0 or ensemble is None:
        z = np.zeros(len(g))
        return z, z
    e = ensemble[ensemble["level"] == level].set_index("month")
    return (g["month"].map(e["delta_median"]).fillna(e["delta_median"].mean()).to_numpy(),
            g["lag_month"].map(e["delta_median"]).fillna(e["delta_median"].mean()).to_numpy())


def deaths_per_summer(g, base, beta, thr, shift, lag_shift, cooling=0.0):
    h = future.heat(g["tmax"].to_numpy() + shift - cooling, g["tmax_lag1"].to_numpy() + lag_shift - cooling, thr)
    d = base * (np.exp(max(beta, 0.0) * h) - 1)
    return float(pd.Series(d).groupby(g["year"].to_numpy()).sum().mean())


def built_land_ha(static_dir):
    path = Path(static_dir) / "cells_attica.xlsx"
    if not path.exists():
        return {}
    cells = pd.read_excel(path)
    lived = cells[(cells["T"].fillna(0) > 0) & cells["land_m2"].notna()].copy()
    lived["built"] = lived["land_m2"] * lived["artificial_share"].fillna(0)
    return (lived.groupby("NUTS_ID")["built"].sum() / 1e4).to_dict()


def tree_surface_effect(surface):
    if not surface or not surface.get("effects"):
        return None
    for e in surface["effects"]:
        if e.get("feature") == "tree_share" and e.get("period") == "training summers":
            return {"central": -e["effect_c"] / 10, "low": -e["high_c"] / 10, "high": -e["low_c"] / 10}
    return None


def evaluate(panel, vuln, mods, areas, surface=None, ensemble=None, static_dir="data/static"):
    weeks = summer_weeks(panel)
    land = built_land_ha(static_dir)
    levels = [0.0] + [l for l in future.LEVELS if ensemble is not None and not ensemble.empty and (ensemble["level"] == l).any()]
    surface_per_point = tree_surface_effect(surface)
    slopes = mods.set_index("covariate") if mods is not None and not mods.empty else pd.DataFrame()
    rows = []
    for v in vuln.itertuples():
        g = weeks[weeks["geo"] == v.geo]
        if g.empty:
            continue
        h_now = future.heat(g["tmax"].to_numpy(), g["tmax_lag1"].to_numpy(), v.threshold_c)
        base = g["deaths"].to_numpy() / np.exp(v.beta_shrunk * h_now)
        a = areas[areas["geo"] == v.geo] if areas is not None and not areas.empty else pd.DataFrame()
        built = land.get(v.geo, np.nan)
        for level in levels:
            shift, lag_shift = shifts(g, ensemble, level)
            now = deaths_per_summer(g, base, v.beta_shrunk, v.threshold_c, shift, lag_shift)
            common = {"geo": v.geo, "name_en": v.name_en, "level": level, "heat_deaths_without_measure": now, "built_land_ha": built}
            if surface_per_point:
                steps = [(step, f"Street trees +{step * 100:.0f} points") for step in PHYSICAL_STEPS]
                if not a.empty and "tree_share_pop" in a.columns and not areas["tree_share_pop"].isna().all():
                    gap = float(areas["tree_share_pop"].max()) - float(a["tree_share_pop"].iloc[0])
                    if np.isfinite(gap):
                        steps.append((max(gap, 0.0), TREES_TO_BEST_AIR))
                for step, label in steps:
                    points = step * 100
                    cool = {k: surface_per_point[k] * points for k in surface_per_point}
                    central = now - deaths_per_summer(g, base, v.beta_shrunk, v.threshold_c, shift, lag_shift, cool["central"] * AIR_TO_SURFACE)
                    low = now - deaths_per_summer(g, base, v.beta_shrunk, v.threshold_c, shift, lag_shift, cool["low"] * AIR_TO_SURFACE)
                    high = now - deaths_per_summer(g, base, v.beta_shrunk, v.threshold_c, shift, lag_shift, cool["central"])
                    rows.append(dict(common, measure=label, evidence="cooling measured from space", kind="physical",
                                     increase_points=points, surface_cooling_c=cool["central"], air_cooling_c=cool["central"] * AIR_TO_SURFACE,
                                     deaths_avoided=central, deaths_avoided_low=low, deaths_avoided_high=high,
                                     new_cover_ha=built * step))
            for covariate, name in BEST_OF_ATTICA:
                if covariate not in slopes.index or a.empty or covariate not in a.columns or areas[covariate].isna().all():
                    continue
                s = slopes.loc[covariate]
                best = float(areas[covariate].max())
                current = float(a[covariate].iloc[0])
                gap = max(best - current, 0.0)
                per_unit = {k: s[c] / 100 / s["sd"] for k, c in (("central", "change_pp_per_sd"), ("low", "change_low"), ("high", "change_high"))}
                after = {k: deaths_per_summer(g, base, v.beta_shrunk + per_unit[k] * gap, v.threshold_c, shift, lag_shift) for k in per_unit}
                rows.append(dict(common, measure=f"{name} up to the best area of Attica", evidence="difference between areas", kind="association",
                                 increase_points=gap * 100, surface_cooling_c=np.nan, air_cooling_c=np.nan,
                                 deaths_avoided=now - after["central"], deaths_avoided_low=now - after["high"], deaths_avoided_high=now - after["low"],
                                 new_cover_ha=built * gap, evidence_p=float(s["p_value"]), evidence_verdict=s["verdict"]))
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    detail["share_avoided"] = detail["deaths_avoided"] / detail["heat_deaths_without_measure"]
    detail["deaths_avoided_per_100ha"] = np.where(detail["new_cover_ha"] > 0, detail["deaths_avoided"] / detail["new_cover_ha"] * 100, np.nan)
    attica = detail.groupby(["measure", "evidence", "kind", "level"], as_index=False).agg(
        heat_deaths_without_measure=("heat_deaths_without_measure", "sum"), deaths_avoided=("deaths_avoided", "sum"),
        deaths_avoided_low=("deaths_avoided_low", "sum"), deaths_avoided_high=("deaths_avoided_high", "sum"),
        new_cover_ha=("new_cover_ha", "sum"))
    if "evidence_verdict" in detail.columns:
        verdicts = detail.groupby(["measure", "level"])["evidence_verdict"].first().rename("evidence_verdict").reset_index()
        attica = attica.merge(verdicts, on=["measure", "level"], how="left")
        attica["supported"] = (attica["kind"] == "physical") | (attica["evidence_verdict"] == "lower heat risk where higher")
    else:
        attica["supported"] = attica["kind"] == "physical"
    attica["share_avoided"] = attica["deaths_avoided"] / attica["heat_deaths_without_measure"]
    attica["deaths_avoided_per_100ha"] = attica["deaths_avoided"] / attica["new_cover_ha"] * 100
    return attica, detail
