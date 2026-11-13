import json
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://archive-api.open-meteo.com/v1/archive"


def fetch_daily(lat, lon, start, end, cache_dir=None, api=API, model="era5_land", retries=4, timeout=120):
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_max,temperature_2m_mean",
        "timezone": "Europe/Athens",
        "models": model,
    }
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"climate_{lat:.4f}_{lon:.4f}_{start}_{end}_{model}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(api, params=params, timeout=timeout)
            if response.status_code == 200:
                payload = response.json()
                if cache_file is not None:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(payload), encoding="utf-8")
                return payload
            last_error = RuntimeError(f"Open-Meteo HTTP {response.status_code}: {response.text[:300]}")
            if response.status_code == 429:
                time.sleep(65 * (attempt + 1))
                continue
        except requests.RequestException as error:
            last_error = error
        time.sleep(10 * (attempt + 1))
    raise last_error


def daily_frame(payload):
    daily = payload["daily"]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "tmax": pd.to_numeric(pd.Series(daily["temperature_2m_max"]), errors="coerce"),
            "tmean": pd.to_numeric(pd.Series(daily["temperature_2m_mean"]), errors="coerce"),
        }
    )
    return frame


def iso_week_label(dates):
    iso = dates.dt.isocalendar()
    return iso["year"].astype(int).astype(str) + "-W" + iso["week"].astype(int).astype(str).str.zfill(2)


def weekly_from_daily(frame):
    frame = frame.dropna(subset=["tmax", "tmean"]).copy()
    frame["week"] = iso_week_label(frame["date"])
    grouped = frame.groupby("week").agg(tmax=("tmax", "mean"), tmean=("tmean", "mean"), days=("tmax", "size"))
    grouped = grouped[grouped["days"] == 7].drop(columns="days")
    return grouped.reset_index()


def regional_weekly(regions, start, end, cache_dir=None, api=API, model="era5_land", pause=8.0):
    rows = []
    for region in regions:
        point_frames = []
        for lat, lon in region["points"]:
            cached = cache_dir is not None and (Path(cache_dir) / f"climate_{lat:.4f}_{lon:.4f}_{start}_{end}_{model}.json").exists()
            payload = fetch_daily(lat, lon, start, end, cache_dir, api, model)
            point_frames.append(daily_frame(payload))
            if not cached:
                time.sleep(pause)
        combined = pd.concat(point_frames).groupby("date", as_index=False)[["tmax", "tmean"]].mean()
        weekly = weekly_from_daily(combined)
        weekly["geo"] = region["code"]
        rows.append(weekly)
    return pd.concat(rows, ignore_index=True)[["geo", "week", "tmax", "tmean"]]
