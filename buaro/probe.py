import datetime as dt
import json
import re
from pathlib import Path

import requests

from . import clms

SEARCHES = [
    "Street Tree Layer 2021",
    "Urban Atlas 2021",
    "Imperviousness Density 2021",
    "Tree Cover Density 2021",
]

URLS = {
    "gisco_nuts3_2021_geojson": "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2021_4326_LEVL_3.geojson",
    "gisco_nuts_2021_files": "https://gisco-services.ec.europa.eu/distribution/v2/nuts/nuts-2021-files.json",
    "worldcover_2021_N36E021": "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N36E021_Map.tif",
    "worldcover_2021_N36E024": "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N36E024_Map.tif",
}

PAGES = {
    "eurostat_population_grids": "https://ec.europa.eu/eurostat/web/gisco/geodata/population-distribution/population-grids",
}

ATHENS = re.compile(r"(EL001|ATHIN|ATHEN|Αθήν)", re.I)


def short(value, limit=400):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text[:limit]


def head(url):
    try:
        r = requests.head(url, allow_redirects=True, timeout=60)
        if r.status_code >= 400 or "content-length" not in {k.lower() for k in r.headers}:
            r = requests.get(url, stream=True, allow_redirects=True, timeout=60)
            r.close()
        return {"url": url, "status": r.status_code, "final_url": r.url, "bytes": r.headers.get("Content-Length"), "type": r.headers.get("Content-Type")}
    except Exception as error:
        return {"url": url, "error": short(str(error))}


def page_links(url):
    try:
        r = requests.get(url, timeout=60)
        links = sorted(set(re.findall(r'href="([^"]+)"', r.text)))
        keep = [l for l in links if re.search(r"(census|grid|geostat|\.zip|\.gpkg|\.csv)", l, re.I)]
        return {"url": url, "status": r.status_code, "links": keep[:80]}
    except Exception as error:
        return {"url": url, "error": short(str(error))}


def describe(item):
    files = item.get("downloadable_files") or {}
    rows = files.get("items", []) if isinstance(files, dict) else files if isinstance(files, list) else []
    athens = [r for r in rows if ATHENS.search(json.dumps(r, ensure_ascii=False))]
    return {
        "title": item.get("title"),
        "id": item.get("@id"),
        "UID": item.get("UID"),
        "dataset_full_format": item.get("dataset_full_format"),
        "dataset_download_information": item.get("dataset_download_information"),
        "downloadable_files_count": len(rows),
        "downloadable_files_keys": sorted({k for r in rows if isinstance(r, dict) for k in r}),
        "downloadable_files_athens": athens[:20],
        "downloadable_files_sample": rows[:3],
    }


def main():
    report = {"run": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"}
    client = clms.Client()
    try:
        client.key = clms.service_key()
        report["key_fields"] = sorted(client.key.keys())
        client.token = clms.access_token(client.key)
        report["auth"] = "ok"
        r = client.my_requests()
        report["datarequest_search"] = {"status": r.status_code, "body": short(r.text, 1500)}
    except Exception as error:
        report["auth"] = "failed"
        report["auth_error"] = short(str(error))
    catalogue = {}
    for text in SEARCHES:
        try:
            catalogue[text] = [describe(i) for i in client.search(text)][:15]
        except Exception as error:
            catalogue[text] = {"error": short(str(error))}
    report["catalogue"] = catalogue
    report["urls"] = {k: head(v) for k, v in URLS.items()}
    report["pages"] = {k: page_links(v) for k, v in PAGES.items()}
    links = report["pages"]["eurostat_population_grids"].get("links", [])
    census = [l for l in links if re.search(r"census", l, re.I) and re.search(r"\.(zip|gpkg|csv)", l, re.I)]
    report["census_grid_files"] = [head(l if l.startswith("http") else "https://ec.europa.eu" + l) for l in census[:10]]
    out = Path("data/probe")
    out.mkdir(parents=True, exist_ok=True)
    (out / "step2_probe.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"auth": report.get("auth"), "catalogue_hits": {k: (len(v) if isinstance(v, list) else v) for k, v in catalogue.items()}}, indent=2))


if __name__ == "__main__":
    main()
