from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import model

EXPOSURE = "exposure_by_nuts3.xlsx"
COVARIATES = [
    ("tree_share_pop", "Street tree cover where people live"),
    ("green_urban_share_pop", "Green urban areas where people live"),
    ("dense_fabric_share_pop", "Dense, highly sealed urban fabric where people live"),
    ("artificial_share_pop", "Built-up land where people live"),
    ("share_65plus_census", "Share of residents aged 65+ (census 2021)"),
]


def load(static_dir):
    path = Path(static_dir) / EXPOSURE
    if not path.exists():
        return None
    frame = pd.read_excel(path)
    return frame.rename(columns={"NUTS_ID": "geo"})


def meta_regression(y, v, x):
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    x = np.asarray(x, dtype=float)
    k = len(y)
    X = np.column_stack([np.ones(k), x])
    W = np.diag(1.0 / v)
    XtWX = X.T @ W @ X
    b_fixed = np.linalg.solve(XtWX, X.T @ W @ y)
    q = float(((y - X @ b_fixed) ** 2 / v).sum())
    P = W - W @ X @ np.linalg.solve(XtWX, X.T @ W)
    tau2 = max(0.0, (q - (k - 2)) / float(np.trace(P)))
    Ws = np.diag(1.0 / (v + tau2))
    cov = np.linalg.inv(X.T @ Ws @ X)
    b = cov @ X.T @ Ws @ y
    return {"intercept": float(b[0]), "slope": float(b[1]), "se": float(np.sqrt(cov[1, 1])), "tau2": tau2, "q": q, "k": k}


MIN_COVERAGE = 0.8


def fit_one(merged, column, total):
    values = merged[column].astype(float)
    ok = values.notna() & merged["se_region"].gt(0)
    if ok.sum() < 4 or values[ok].std(ddof=1) <= 0:
        return None
    sd = float(values[ok].std(ddof=1))
    z = (values[ok] - values[ok].mean()) / sd
    fit = meta_regression(merged.loc[ok, "beta_region"], merged.loc[ok, "se_region"] ** 2, z)
    df = fit["k"] - 2
    t = stats.t.ppf(0.975, df)
    low, high = fit["slope"] - t * fit["se"], fit["slope"] + t * fit["se"]
    return {
        "areas": int(ok.sum()),
        "mean": float(values[ok].mean()),
        "sd": sd,
        "min": float(values[ok].min()),
        "max": float(values[ok].max()),
        "change_pp_per_sd": 100 * fit["slope"],
        "change_low": 100 * low,
        "change_high": 100 * high,
        "p_value": float(2 * stats.t.sf(abs(fit["slope"] / fit["se"]), df)),
        "explained_heterogeneity": float(max(0.0, 1 - fit["tau2"] / total)) if total > 0 else float("nan"),
        "verdict": "lower heat risk where higher" if high < 0 else ("higher heat risk where higher" if low > 0 else "no clear link"),
    }


def analyse(vuln, exposure):
    if exposure is None or exposure.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = vuln.merge(exposure, on="geo", how="inner")
    keep = ["geo", "name_en", "name_el", "beta_region", "se_region", "rr_per_degree", "rr_low", "rr_high"]
    extra = [c for c, _ in COVARIATES if c in merged.columns] + [c for c in ("population_census", "population_65plus_census", "population_covered") if c in merged.columns]
    areas = merged[keep + extra].copy()
    if len(merged) < 4:
        return pd.DataFrame(), areas
    total = model.pool(merged["beta_region"], merged["se_region"] ** 2)["tau2"]
    covered = merged["population_covered"].fillna(1.0) if "population_covered" in merged.columns else pd.Series(1.0, index=merged.index)
    low_cover = merged.loc[covered < MIN_COVERAGE]
    subset = merged.loc[covered >= MIN_COVERAGE].reset_index(drop=True)
    total_subset = model.pool(subset["beta_region"], subset["se_region"] ** 2)["tau2"] if len(subset) >= 4 else 0.0
    rows = []
    for column, label in COVARIATES:
        if column not in merged.columns:
            continue
        main = fit_one(merged, column, total)
        if main is None:
            continue
        row = {"covariate": column, "label": label, **main}
        if not low_cover.empty and column.endswith("_pop") and len(subset) >= 5:
            check = fit_one(subset, column, total_subset)
            if check is not None:
                row["check_without"] = ", ".join(low_cover["name_en"])
                row["check_change_pp_per_sd"] = check["change_pp_per_sd"]
                row["check_low"] = check["change_low"]
                row["check_high"] = check["change_high"]
                row["check_p_value"] = check["p_value"]
                row["check_verdict"] = check["verdict"]
        rows.append(row)
    return pd.DataFrame(rows), areas
