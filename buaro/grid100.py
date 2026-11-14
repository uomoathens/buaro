from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.transform import from_origin

CRS = "EPSG:3035"
RES = 100


def land_cells(static_dir):
    static_dir = Path(static_dir)
    grid = gpd.read_file(static_dir / "census_grid_attica.gpkg")
    table = pd.read_excel(static_dir / "cells_attica.xlsx", dtype={"GRD_ID": str})
    keep = table.loc[table["land_m2"].notna(), "GRD_ID"]
    if "GRD_ID" not in grid.columns:
        grid = grid.assign(GRD_ID=[f"cell{i}" for i in range(len(grid))])
    return grid[grid["GRD_ID"].astype(str).isin(set(keep))].to_crs(CRS)


def spec(static_dir):
    xmin, ymin, xmax, ymax = land_cells(static_dir).total_bounds
    xmin, ymin = np.floor(xmin / 1000) * 1000, np.floor(ymin / 1000) * 1000
    xmax, ymax = np.ceil(xmax / 1000) * 1000, np.ceil(ymax / 1000) * 1000
    width, height = int(round((xmax - xmin) / RES)), int(round((ymax - ymin) / RES))
    return {"transform": from_origin(xmin, ymax, RES, RES), "width": width, "height": height,
            "bounds": (float(xmin), float(ymin), float(xmax), float(ymax)), "crs": CRS}


def centres(g):
    t = g["transform"]
    cols, rows = np.meshgrid(np.arange(g["width"]), np.arange(g["height"]))
    x = t.c + (cols + 0.5) * t.a
    y = t.f + (rows + 0.5) * t.e
    return x, y


def parent(g):
    x, y = centres(g)
    return (np.floor(x / 1000).astype(np.int64) * 100000 + np.floor(y / 1000).astype(np.int64))


def frame(g, **layers):
    x, y = centres(g)
    data = {"cell": np.arange(g["width"] * g["height"]), "x": x.ravel(), "y": y.ravel(), "km": parent(g).ravel()}
    for name, values in layers.items():
        data[name] = np.asarray(values).ravel()
    return pd.DataFrame(data)
