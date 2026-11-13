import datetime as dt
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pytest
import yaml

from buaro import climate, eurostat, run

ROOT = Path(__file__).resolve().parents[1]
TRUE_BETA = {"EL301": 0.030, "EL302": 0.050, "EL303": 0.045, "EL304": 0.035, "EL305": 0.040, "EL306": 0.055, "EL307": 0.040}
POPULATION = {"EL301": 590000, "EL302": 490000, "EL303": 1030000, "EL304": 530000, "EL305": 520000, "EL306": 160000, "EL307": 500000}
SHOCK_2025 = {"winter": 1.0, "rest": 1.0}
DRIFT = {"per_year": 0.0}
ODD_SUMMER = {}
SHARE_65 = {"EL301": 0.24, "EL302": 0.23, "EL303": 0.25, "EL304": 0.26, "EL305": 0.19, "EL306": 0.18, "EL307": 0.24}


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)[:200] if payload is not None else "not found"

    def json(self):
        return self._payload


def jsonstat(dims, values):
    ids = list(dims)
    sizes = [len(dims[d]) for d in ids]
    dimension = {d: {"category": {"index": {c: i for i, c in enumerate(dims[d])}, "label": {c: c for c in dims[d]}}} for d in ids}
    flat = {}
    for key, value in values.items():
        idx = np.ravel_multi_index([dims[d].index(k) for d, k in zip(ids, key)], sizes)
        flat[str(int(idx))] = value
    return {"version": "2.0", "class": "dataset", "id": ids, "size": sizes, "dimension": dimension, "value": flat}


def point_series(code, start, end):
    dates = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(sum(ord(ch) for ch in code))
    doy = dates.dayofyear.to_numpy()
    seasonal = 19 + 10 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    anomaly = np.zeros(len(dates))
    noise = rng.normal(0, 1.2, len(dates))
    for i in range(1, len(dates)):
        anomaly[i] = 0.85 * anomaly[i - 1] + noise[i]
    tmean = seasonal + anomaly
    tmax = tmean + 6 + rng.normal(0, 0.5, len(dates))
    return dates, tmax, tmean


@pytest.fixture()
def fake_world(monkeypatch, tmp_path):
    config = yaml.safe_load((ROOT / "config" / "attica.yaml").read_text(encoding="utf-8"))
    regions = config["regions"]
    start, end = "2014-12-01", "2025-10-31"
    daily = {}
    for r in regions:
        dates, tmax, tmean = point_series(r["code"], start, end)
        for lat, lon in r["points"]:
            daily[(f"{lat:.4f}", f"{lon:.4f}")] = (dates, tmax, tmean)
    weeks = {}
    rng = np.random.default_rng(7)
    for r in regions:
        dates, tmax, tmean = daily[(f"{r['points'][0][0]:.4f}", f"{r['points'][0][1]:.4f}")]
        frame = pd.DataFrame({"date": dates, "tmax": tmax, "tmean": tmean})
        weekly = climate.weekly_from_daily(frame)
        weekly["date"] = weekly["week"].map(lambda w: pd.Timestamp.fromisocalendar(int(w[:4]), int(w[6:]), 1))
        weekly = weekly[weekly["date"] >= "2015-01-05"].reset_index(drop=True)
        thr = np.percentile(weekly["tmax"], 80)
        cold = np.percentile(weekly["tmean"], 20)
        h0 = np.clip(weekly["tmax"] - thr, 0, None)
        h1 = np.clip(weekly["tmax"].shift(1).fillna(weekly["tmax"].iloc[0]) - thr, 0, None)
        c0 = np.clip(cold - weekly["tmean"], 0, None)
        doy = weekly["date"].dt.dayofyear
        base = POPULATION[r["code"]] * 0.011 / 52.18
        season = np.exp(0.12 * np.cos(2 * np.pi * (doy - 15) / 365.25))
        yrs = weekly["week"].str[:4].astype(int).to_numpy()
        beta = TRUE_BETA[r["code"]] * np.exp(DRIFT["per_year"] * (yrs - 2019))
        beta = beta * np.array([ODD_SUMMER.get(str(y), 1.0) for y in yrs])
        mu = base * season * np.exp(0.6 * beta * h0 + 0.4 * beta * h1 + 0.02 * c0)
        pandemic = weekly["week"].between("2020-W10", "2022-W26")
        mu = np.where(pandemic, mu * 1.35, mu)
        y2025 = weekly["week"].str.startswith("2025-")
        wk = weekly["week"].str[-2:].astype(int)
        mu = np.where(y2025 & (wk <= 12), mu * SHOCK_2025["winter"], mu)
        mu = np.where(y2025 & (wk > 12), mu * SHOCK_2025["rest"], mu)
        deaths = rng.poisson(mu)
        last = pd.Timestamp("2025-09-29")
        for w, d, dd in zip(weekly["week"], deaths, weekly["date"]):
            if dd <= last:
                weeks[(r["code"], w)] = int(d)
    all_weeks = sorted({w for _, w in weeks})
    codes = [r["code"] for r in regions]
    deaths_ts = jsonstat(
        {"freq": ["W"], "sex": ["F", "M", "T"], "unit": ["NR"], "geo": codes, "time": all_weeks},
        {("W", "T", "NR", g, w): v for (g, w), v in weeks.items()},
    )
    ages = ["TOTAL", "Y_LT5"] + [f"Y{a}-{a + 4}" for a in range(5, 85, 5)] + ["Y_GE85", "Y85-89", "Y_GE90"]
    pop_values = {}
    for g in codes:
        for year in range(2014, 2026):
            total = POPULATION[g]
            pop_values[("A", "NR", "T", "TOTAL", g, str(year))] = total
            older = total * SHARE_65[g]
            younger = total - older
            for a in ages[1:]:
                lo, hi = eurostat.parse_age(a)
                if a in ("Y_GE85",):
                    v = older * 0.18
                elif a == "Y85-89":
                    v = older * 0.12
                elif a == "Y_GE90":
                    v = older * 0.06
                elif lo >= 65:
                    v = older * 0.82 / 4
                else:
                    v = younger / 13
                pop_values[("A", "NR", "T", a, g, str(year))] = v
    pop = jsonstat(
        {"freq": ["A"], "unit": ["NR"], "sex": ["T"], "age": ages, "geo": codes, "time": [str(y) for y in range(2014, 2026)]},
        pop_values,
    )

    def fake_get(url, params=None, timeout=None):
        path = urlparse(url).path
        if "open-meteo" in url:
            p = dict(params)
            dates, tmax, tmean = daily[(p["latitude"], p["longitude"])]
            mask = (dates >= p["start_date"]) & (dates <= p["end_date"])
            return FakeResponse(200, {"daily": {"time": [d.strftime("%Y-%m-%d") for d in dates[mask]], "temperature_2m_max": list(np.round(tmax[mask], 1)), "temperature_2m_mean": list(np.round(tmean[mask], 1))}})
        if path.endswith("/demo_r_mwk3_t"):
            return FakeResponse(404, None)
        if path.endswith("/demo_r_mwk3_ts"):
            return FakeResponse(200, deaths_ts)
        if path.endswith("/demo_r_pjangrp3"):
            return FakeResponse(200, pop)
        return FakeResponse(404, None)

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return tmp_path


def test_jsonstat_roundtrip():
    payload = jsonstat({"geo": ["A", "B"], "time": ["2024-W01", "2024-W02"]}, {("A", "2024-W02"): 5, ("B", "2024-W01"): 7})
    frame = eurostat.jsonstat_to_frame(payload).sort_values(["geo", "time"])
    assert frame[["geo", "time", "value"]].values.tolist() == [["A", "2024-W02", 5.0], ["B", "2024-W01", 7.0]]


def test_age_partition_avoids_double_counting():
    codes = ["TOTAL", "Y60-64", "Y65-69", "Y70-74", "Y75-79", "Y80-84", "Y_GE85", "Y85-89", "Y_GE90"]
    assert eurostat.partition_from(codes, 65) == ["Y65-69", "Y70-74", "Y75-79", "Y80-84", "Y_GE85"]
    assert eurostat.partition_from(["Y65-69", "Y70-74", "Y75-79", "Y80-84", "Y85-89", "Y_GE90"], 65)[-1] == "Y_GE90"


def test_iso_weeks_complete_only():
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", "2024-01-10"), "tmax": 10.0, "tmean": 5.0})
    weekly = climate.weekly_from_daily(frame)
    assert weekly["week"].tolist() == ["2024-W01"]


def test_full_pipeline_recovers_known_effect(fake_world):
    out = fake_world / "results"
    docs = fake_world / "docs"
    meta = run.main(["--config", str(ROOT / "config" / "attica.yaml"), "--out", str(out), "--cache", str(fake_world / "cache"), "--end-date", "2025-10-31", "--docs", str(docs)])
    html = (docs / "index.html").read_text(encoding="utf-8")
    assert "__DATA__" not in html
    payload = json.loads(html.split('<script id="payload" type="application/json">')[1].split("</script>")[0])
    assert set(payload["panel"]) == set(TRUE_BETA)
    assert payload["backtestYears"] == [2023, 2024, 2025]
    assert (docs / "data" / "data_panel.xlsx").exists()
    assert meta["deaths_dataset"] == "demo_r_mwk3_ts"
    assert meta["spatial_level"] == "NUTS3"
    truth = np.mean(list(TRUE_BETA.values()))
    assert abs(meta["pooled_beta"] - truth) / truth < 0.35
    vuln = pd.read_excel(out / "buaro_attica_step1.xlsx", sheet_name="Vulnerability", skiprows=2)
    assert set(vuln["geo"]) == set(TRUE_BETA)
    share = vuln.set_index("geo")["share_65plus"]
    for g, s in SHARE_65.items():
        assert abs(share[g] - s) < 1e-6
    summary = pd.read_excel(out / "buaro_attica_step1.xlsx", sheet_name="Backtest summary", skiprows=2)
    assert set(summary["year"]) == {2023, 2024, 2025}
    assert set(summary["baseline"]) == {"serfling", "euromomo"}
    assert (summary[summary["baseline"] == meta["baseline_selected"]]["skill_score"] > 0).all()
    assert meta["adaptation"]["verdict"] == "no clear change"
    panel = pd.read_excel(out / "data_panel.xlsx", sheet_name="Panel", skiprows=2)
    assert set(panel["geo"]) == set(TRUE_BETA)
    for name in ["SUMMARY.md", "vulnerability.png", "skill.png", "backtest_2024.png", "baseline_comparison.png", "run_metadata.json", "data_panel.xlsx"]:
        assert (out / name).exists()


def test_week_labels_normalised():
    assert eurostat.normalise_week("2024W5") == "2024-W05"
    assert eurostat.normalise_week("2024-W30") == "2024-W30"
    assert eurostat.normalise_week("2024-W99") is None
    assert eurostat.normalise_week("2024") is None


def test_falls_back_to_nuts2_when_nuts3_missing(monkeypatch, tmp_path):
    rng = np.random.default_rng(3)
    weeks = [f"{y}-W{w:02d}" for y in range(2015, 2026) for w in range(1, 53)]
    payload = jsonstat({"freq": ["W"], "unit": ["NR"], "geo": ["EL30"], "time": weeks}, {("W", "NR", "EL30", w): int(rng.poisson(700)) for w in weeks})
    dates = pd.date_range("2014-12-01", "2025-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    tmean = 19 + 10 * np.sin(2 * np.pi * (doy - 110) / 365.25) + rng.normal(0, 1.5, len(dates))

    def fake_get(url, params=None, timeout=None):
        if "open-meteo" in url:
            p = dict(params)
            mask = (dates >= p["start_date"]) & (dates <= p["end_date"])
            return FakeResponse(200, {"daily": {"time": [d.strftime("%Y-%m-%d") for d in dates[mask]], "temperature_2m_max": list(tmean[mask] + 6), "temperature_2m_mean": list(tmean[mask])}})
        geos = [v for k, v in params if k == "geo"]
        if geos == ["EL30"] and urlparse(url).path.endswith("/demo_r_mwk3_t"):
            return FakeResponse(200, payload)
        return FakeResponse(404, None)

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("time.sleep", lambda s: None)
    meta = run.main(["--config", str(ROOT / "config" / "attica.yaml"), "--out", str(tmp_path / "r"), "--cache", str(tmp_path / "c"), "--end-date", "2025-10-31", "--docs", str(tmp_path / "docs")])
    assert meta["spatial_level"] == "NUTS2"
    assert meta["regions"] == ["EL30"]
    assert (tmp_path / "r" / "buaro_attica_step1.xlsx").exists()


def test_euromomo_baseline_survives_level_shift(monkeypatch, request):
    monkeypatch.setitem(SHOCK_2025, "winter", 1.30)
    monkeypatch.setitem(SHOCK_2025, "rest", 0.90)
    world = request.getfixturevalue("fake_world")
    out = world / "results"
    meta = run.main(["--config", str(ROOT / "config" / "attica.yaml"), "--out", str(out), "--cache", str(world / "cache"), "--end-date", "2025-10-31", "--docs", str(world / "docs")])
    summary = pd.read_excel(out / "buaro_attica_step1.xlsx", sheet_name="Backtest summary", skiprows=2)
    s25 = summary[summary["year"] == 2025].set_index("baseline")
    assert s25.loc["euromomo", "rmse_with_heat"] < s25.loc["serfling", "rmse_with_heat"]
    assert s25.loc["euromomo", "skill_score"] > 0


def test_adaptation_is_detected(monkeypatch, request):
    monkeypatch.setitem(DRIFT, "per_year", -0.12)
    world = request.getfixturevalue("fake_world")
    meta = run.main(["--config", str(ROOT / "config" / "attica.yaml"), "--out", str(world / "results"), "--cache", str(world / "cache"), "--end-date", "2025-10-31", "--docs", str(world / "docs")])
    assert meta["adaptation"]["verdict"] == "decreasing"
    assert meta["adaptation"]["robust"] is True
    assert meta["adaptation"]["rr_2025"] < meta["adaptation"]["rr_2015"]


def test_one_odd_summer_is_not_called_adaptation(monkeypatch, request):
    monkeypatch.setitem(SHOCK_2025, "rest", 0.85)
    monkeypatch.setitem(ODD_SUMMER, "2025", 0.15)
    world = request.getfixturevalue("fake_world")
    out = world / "results"
    meta = run.main(["--config", str(ROOT / "config" / "attica.yaml"), "--out", str(out), "--cache", str(world / "cache"), "--end-date", "2025-10-31", "--docs", str(world / "docs")])
    a = meta["adaptation"]
    assert a["verdict"] == "no clear change" or (a["robust"] is False and 2025 in a["depends_on"])
    text = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert "measurable evidence of adaptation" not in text
    response = pd.read_excel(out / "buaro_attica_step1.xlsx", sheet_name="Summer response", skiprows=2)
    assert 2025 in set(response["year"])
