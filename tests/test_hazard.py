import json
import zipfile

import numpy as np
import pandas as pd
import xarray as xr

from buaro import future, hazard

GWL_CSV = "model_run,1.5_ssp126,2_ssp126,3_ssp126,4_ssp126,1.5_ssp245,2_ssp245,3_ssp245,4_ssp245,1.5_ssp370,2_ssp370,3_ssp370,4_ssp370,1.5_ssp585,2_ssp585,3_ssp585,4_ssp585\n" \
          "MOD-A_r1i1p1f1,2030,2050,NA,NA,2030,2050,2080,NA,2030,2050,2070,NA,2030,2045,2065,NA\n" \
          "MOD-B_r1i1p1f1,2035,NA,NA,NA,2035,2060,NA,NA,2035,2055,2075,NA,2032,2048,2068,NA\n"


class Session:
    def get(self, url, timeout=None):
        if url == hazard.GWL_URL:
            body = GWL_CSV
            return type("R", (), {"raise_for_status": lambda self: None, "text": body})()
        form = {"inputs": {"variable": {"schema": {"enum": ["monthly_daily_maximum_temperature", "monthly_extreme_hot_days"]}},
                           "experiment": {"schema": {"enum": list(hazard.EXPERIMENTS)}},
                           "domain": {"schema": {"enum": ["europe", "global"]}}}}
        return type("R", (), {"raise_for_status": lambda self: None, "json": lambda self: form})()


def synthetic_tx(years, offset):
    times = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-01", freq="MS")
    lat = np.array([37.0, 38.0, 39.0])
    lon = np.array([23.0, 24.0, 25.0])
    members = ["INST_MOD-A_r1i1p1f1", "INST_MOD-B_r1i1p1f1"]
    yrs = times.year.to_numpy()
    summer = np.isin(times.month, [6, 7, 8, 9])
    data = np.zeros((2, len(times), 3, 3))
    for i, slope in enumerate((0.04, 0.03)):
        series = 273.15 + 20 + 10 * summer + slope * (yrs - 2000) + offset
        data[i] = series[:, None, None]
    return xr.Dataset({"tx": (("member", "time", "lat", "lon"), data)},
                      coords={"member": np.arange(2), "member_id": ("member", members), "time": times, "lat": lat, "lon": lon})


class Client:
    def __init__(self, tmp):
        self.tmp = tmp
        self.requests = []

    def retrieve(self, dataset, request):
        self.requests.append(request)
        exp = request["experiment"]
        years = (1990, 2014) if exp == "historical" else (2015, 2100)
        nc = self.tmp / f"{exp}.nc"
        synthetic_tx(years, 0.0).to_netcdf(nc)

        class Result:
            def download(self, target):
                with zipfile.ZipFile(target, "w") as z:
                    z.write(nc, arcname=f"tx_CMIP6_{exp}.nc")
        return Result()


def test_hazard_builds_warming_level_deltas(tmp_path):
    client = Client(tmp_path)
    report = hazard.main(["--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "out")], session=Session(), client=client)
    assert report["variable"] == "monthly_daily_maximum_temperature"
    assert {r["experiment"] for r in client.requests} == set(hazard.EXPERIMENTS)
    assert all(r["bias_adjustment"] == "no_bias_adjustment" and r["domain"] == "global" for r in client.requests)
    ens = pd.read_excel(tmp_path / "out" / hazard.DELTAS, sheet_name="Ensemble")
    july = ens[ens["month"] == 7].set_index("level")
    assert july.loc[1.5, "delta_median"] > 0
    assert july.loc[2.0, "delta_median"] < july.loc[3.0, "delta_median"]
    assert int(july.loc[1.5, "models"]) == 2
    again = hazard.main(["--raw", str(tmp_path / "raw2"), "--out", str(tmp_path / "out")], session=Session(), client=None)
    assert again["series"]["cached"] is True


def test_hazard_waits_for_key(tmp_path, monkeypatch):
    monkeypatch.delenv("CDSAPI_KEY", raising=False)
    report = hazard.main(["--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "out")], session=Session())
    assert report["series"] == {"waiting_for": "CDSAPI_KEY secret"}
    assert (tmp_path / "out" / "ipcc_warming_levels_cmip6.xlsx").exists()


def panel_and_vuln():
    rows = []
    for year in (2016, 2017, 2018):
        for week in range(1, 53):
            date = pd.Timestamp.fromisocalendar(year, week, 1)
            t = 25 + 9 * np.exp(-((week - 30) / 6) ** 2)
            rows.append({"geo": "EL301", "week": f"{year}-W{week:02d}", "date": date, "year": year, "isoweek": week,
                         "deaths": 100.0, "tmax": t, "tmax_lag1": t, "pandemic": False})
    panel = pd.DataFrame(rows)
    vuln = pd.DataFrame([{"geo": "EL301", "name_en": "Northern Athens", "threshold_c": 31.0, "beta_shrunk": 0.05,
                          "rr_low": np.exp(0.04), "rr_high": np.exp(0.06)}])
    return panel, vuln


def test_future_projection_grows_with_warming():
    panel, vuln = panel_and_vuln()
    ens = pd.DataFrame([{"level": l, "month": m, "delta_median": d, "delta_p10": d * 0.5, "delta_p90": d * 1.5, "models": 10}
                        for l, d in ((1.5, 0.5), (2.0, 1.2), (3.0, 2.6)) for m in (6, 7, 8, 9)])
    attica, areas = future.project(panel, vuln, ens)
    deaths = attica.set_index("level")["heat_deaths_central"]
    assert deaths.loc[0.0] > 0
    assert deaths.loc[0.0] < deaths.loc[1.5] < deaths.loc[2.0] < deaths.loc[3.0]
    row = attica.set_index("level").loc[3.0]
    assert row["heat_deaths_low"] < row["heat_deaths_central"] < row["heat_deaths_high"]
    assert abs(attica.set_index("level").loc[0.0, "times_today"] - 1) < 1e-12


def test_future_without_projections_is_empty():
    panel, vuln = panel_and_vuln()
    attica, areas = future.project(panel, vuln, None)
    assert attica.empty and areas.empty
