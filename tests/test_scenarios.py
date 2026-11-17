import numpy as np
import pandas as pd

from buaro import scenarios


def world():
    rows = []
    geos = ["EL301", "EL302"]
    for geo in geos:
        for year in (2016, 2017, 2018):
            for week in range(1, 53):
                date = pd.Timestamp.fromisocalendar(year, week, 1)
                t = 25 + 9 * np.exp(-((week - 30) / 6) ** 2)
                rows.append({"geo": geo, "week": f"{year}-W{week:02d}", "date": date, "year": year, "isoweek": week,
                             "deaths": 100.0, "tmax": t, "tmax_lag1": t, "pandemic": False})
    panel = pd.DataFrame(rows)
    vuln = pd.DataFrame([{"geo": g, "name_en": g, "threshold_c": 31.0, "beta_shrunk": b, "rr_low": np.exp(b - 0.01), "rr_high": np.exp(b + 0.01)}
                         for g, b in (("EL301", 0.04), ("EL302", 0.06))])
    mods = pd.DataFrame([{"covariate": "tree_share_pop", "sd": 0.03, "change_pp_per_sd": -1.0, "change_low": -1.8, "change_high": -0.2, "p_value": 0.01, "verdict": "lower heat risk where higher"},
                         {"covariate": "green_urban_share_pop", "sd": 0.03, "change_pp_per_sd": -1.1, "change_low": -1.6, "change_high": -0.6, "p_value": 0.002, "verdict": "lower heat risk where higher"}])
    areas = pd.DataFrame([{"geo": "EL301", "tree_share_pop": 0.08, "green_urban_share_pop": 0.07}, {"geo": "EL302", "tree_share_pop": 0.03, "green_urban_share_pop": 0.03}])
    surface = {"effects": [{"feature": "tree_share", "period": "training summers", "effect_c": -0.27, "low_c": -0.30, "high_c": -0.23}]}
    return panel, vuln, mods, areas, surface


def test_measures_are_ordered_and_bounded(tmp_path):
    panel, vuln, mods, areas, surface = world()
    attica, detail = scenarios.evaluate(panel, vuln, mods, areas, surface, None, str(tmp_path))
    t = attica.set_index("measure")
    five, ten = t.loc["Street trees +5 points"], t.loc["Street trees +10 points"]
    assert 0 < five["deaths_avoided"] < ten["deaths_avoided"]
    assert ten["deaths_avoided_low"] <= ten["deaths_avoided"] <= ten["deaths_avoided_high"]
    assert ten["deaths_avoided"] < ten["heat_deaths_without_measure"]
    best = detail[(detail["geo"] == "EL301") & (detail["measure"] == "Street trees up to the best area of Attica")].iloc[0]
    assert best["deaths_avoided"] == 0 and best["increase_points"] == 0
    other = detail[(detail["geo"] == "EL302") & (detail["measure"] == "Street trees up to the best area of Attica")].iloc[0]
    assert abs(other["increase_points"] - 5) < 1e-9 and other["deaths_avoided"] > 0
    assert other["deaths_avoided_low"] <= other["deaths_avoided"] <= other["deaths_avoided_high"]


def test_air_cooling_uses_published_ratio(tmp_path):
    panel, vuln, mods, areas, surface = world()
    _, detail = scenarios.evaluate(panel, vuln, mods, areas, surface, None, str(tmp_path))
    row = detail[detail["measure"] == "Street trees +10 points"].iloc[0]
    assert abs(row["surface_cooling_c"] - 0.27) < 1e-9
    assert abs(row["air_cooling_c"] - 0.27 * 0.08) < 1e-9


def test_measures_follow_warming_levels(tmp_path):
    panel, vuln, mods, areas, surface = world()
    ens = pd.DataFrame([{"level": l, "month": m, "delta_median": d, "delta_p10": d, "delta_p90": d, "models": 5}
                        for l, d in ((1.5, 0.5), (2.0, 1.2), (3.0, 2.6)) for m in (6, 7, 8, 9)])
    attica, _ = scenarios.evaluate(panel, vuln, mods, areas, surface, ens, str(tmp_path))
    ten = attica[attica["measure"] == "Street trees +10 points"].set_index("level")
    assert list(ten.index) == [0.0, 1.5, 2.0, 3.0]
    assert ten.loc[3.0, "heat_deaths_without_measure"] > ten.loc[0.0, "heat_deaths_without_measure"]


def test_trees_to_best_cooler_air_matches_association_gap(tmp_path):
    panel, vuln, mods, areas, surface = world()
    _, detail = scenarios.evaluate(panel, vuln, mods, areas, surface, None, str(tmp_path))
    air = detail[(detail["measure"] == scenarios.TREES_TO_BEST_AIR)].set_index("geo")
    assoc = detail[(detail["measure"] == "Street trees up to the best area of Attica")].set_index("geo")
    assert air.loc["EL301", "deaths_avoided"] == 0
    assert abs(air.loc["EL302", "increase_points"] - assoc.loc["EL302", "increase_points"]) < 1e-9
    assert abs(air.loc["EL302", "air_cooling_c"] - 0.027 * 5 * 0.08) < 1e-9
    assert 0 < air.loc["EL302", "deaths_avoided"] <= air.loc["EL302", "deaths_avoided_high"]
