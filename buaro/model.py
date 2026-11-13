import numpy as np
import pandas as pd
import statsmodels.api as sm

COLD_PERCENTILE = 20
ORIGIN = pd.Timestamp("2015-01-05")
ADAPTATION_YEAR = 2019
BASELINES = ("serfling", "euromomo")


def thresholds(train, percentile):
    return {
        "hot": float(np.nanpercentile(train["tmax"], percentile)),
        "cold": float(np.nanpercentile(train["tmean"], COLD_PERCENTILE)),
        "percentile": percentile,
    }


def offset(frame):
    pop = frame["population"].to_numpy(dtype=float)
    if np.all(np.isfinite(pop)) and np.all(pop > 0):
        return np.log(pop)
    return np.zeros(len(frame))


def fourier(frame, harmonics):
    doy = frame["date"].dt.dayofyear.to_numpy(dtype=float) + 3.0
    cols = {}
    for k in range(1, harmonics + 1):
        cols[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        cols[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    return cols


def heat_columns(frame, thr, adaptation):
    h0 = np.clip(frame["tmax"].to_numpy(dtype=float) - thr["hot"], 0, None)
    h1 = np.clip(frame["tmax_lag1"].to_numpy(dtype=float) - thr["hot"], 0, None)
    cols = {"heat0": h0, "heat1": h1}
    if adaptation:
        t = frame["year"].to_numpy(dtype=float) - ADAPTATION_YEAR
        cols["heat0_t"] = h0 * t
        cols["heat1_t"] = h1 * t
    return cols


def heat_summary(result, adaptation):
    names = ["heat0", "heat1"]
    b = result.params[names].to_numpy()
    v = result.cov_params().loc[names, names].to_numpy()
    out = {"beta": float(b.sum()), "var": float(v.sum())}
    if adaptation:
        tn = ["heat0_t", "heat1_t"]
        out["delta"] = float(result.params[tn].sum())
        out["delta_var"] = float(result.cov_params().loc[tn, tn].to_numpy().sum())
    return out


def heat_part(X, params):
    part = np.zeros(len(X))
    for name in ("heat0", "heat1", "heat0_t", "heat1_t"):
        if name in X.columns:
            part = part + X[name].to_numpy() * params[name]
    return part


def serfling_design(frame, thr, harmonics, with_heat, adaptation):
    years = (frame["date"] - ORIGIN).dt.days.to_numpy(dtype=float) / 365.25
    cols = {"const": np.ones(len(frame)), "trend": years}
    cols.update(fourier(frame, harmonics))
    cols["cold0"] = np.clip(thr["cold"] - frame["tmean"].to_numpy(dtype=float), 0, None)
    cols["cold1"] = np.clip(thr["cold"] - frame["tmean_lag1"].to_numpy(dtype=float), 0, None)
    if with_heat:
        cols.update(heat_columns(frame, thr, adaptation))
    return pd.DataFrame(cols, index=frame.index)


def fit_serfling(train, thr, study, with_heat, adaptation):
    harmonics = int(study["fourier_harmonics"])
    X = serfling_design(train, thr, harmonics, with_heat, adaptation)
    off = offset(train)
    result = sm.GLM(train["deaths"].to_numpy(dtype=float), X, family=sm.families.Poisson(), offset=off).fit(scale="X2")
    fit = {"kind": "serfling", "thr": thr, "harmonics": harmonics, "with_heat": with_heat, "adaptation": adaptation,
           "params": result.params, "deviance": float(result.deviance), "dispersion": float(result.scale),
           "uses_offset": bool(np.any(off != 0)), "n": int(len(train))}
    if with_heat:
        fit.update(heat_summary(result, adaptation))
    return fit


def predict_serfling(fit, frame):
    X = serfling_design(frame, fit["thr"], fit["harmonics"], fit["with_heat"], fit["adaptation"])
    eta = X.to_numpy() @ fit["params"].reindex(X.columns).to_numpy()
    if fit["uses_offset"]:
        eta = eta + offset(frame)
    mu = np.exp(eta)
    return mu, heat_part(X, fit["params"])


def in_weeks(frame, ranges):
    mask = np.zeros(len(frame), dtype=bool)
    for lo, hi in ranges:
        mask |= frame["isoweek"].between(lo, hi).to_numpy()
    return mask


def baseline_design(frame, years, harmonics):
    cols = {}
    for y in years:
        cols[f"y{y}"] = (frame["year"].to_numpy() == y).astype(float)
    cols.update(fourier(frame, harmonics))
    return pd.DataFrame(cols, index=frame.index)


def level_years(frame, years):
    known = np.array(sorted(years))
    mapped = frame.copy()
    target = []
    for y in mapped["year"].to_numpy():
        if y in years:
            target.append(y)
        else:
            earlier = known[known <= y]
            target.append(int(earlier[-1]) if len(earlier) else int(known[0]))
    mapped["year"] = target
    return mapped


def fit_em_baseline(train, study):
    em = study["euromomo"]
    stable = train[in_weeks(train, em["stable_weeks"])]
    years = sorted(int(y) for y in stable["year"].unique())
    harmonics = int(em["harmonics"])
    X = baseline_design(stable, years, harmonics)
    off = offset(stable)
    result = sm.GLM(stable["deaths"].to_numpy(dtype=float), X, family=sm.families.Poisson(), offset=off).fit(scale="X2")
    return {"years": years, "harmonics": harmonics, "params": result.params, "uses_offset": bool(np.any(off != 0))}


def predict_em_baseline(base, frame):
    mapped = level_years(frame, base["years"])
    X = baseline_design(mapped, base["years"], base["harmonics"])
    eta = X.to_numpy() @ base["params"].reindex(X.columns).to_numpy()
    if base["uses_offset"]:
        eta = eta + offset(frame)
    return np.exp(eta)


def fit_euromomo(train, thr, study, with_heat, adaptation):
    base = fit_em_baseline(train, study)
    window = train[in_weeks(train, [study["euromomo"]["model_weeks"]])]
    cols = {"const": np.ones(len(window))}
    if with_heat:
        cols.update(heat_columns(window, thr, adaptation))
    X = pd.DataFrame(cols, index=window.index)
    off = np.log(predict_em_baseline(base, window))
    result = sm.GLM(window["deaths"].to_numpy(dtype=float), X, family=sm.families.Poisson(), offset=off).fit(scale="X2")
    fit = {"kind": "euromomo", "thr": thr, "with_heat": with_heat, "adaptation": adaptation, "base": base,
           "params": result.params, "deviance": float(result.deviance), "dispersion": float(result.scale),
           "uses_offset": base["uses_offset"], "n": int(len(window))}
    if with_heat:
        fit.update(heat_summary(result, adaptation))
    return fit


def predict_euromomo(fit, frame):
    cols = {"const": np.ones(len(frame))}
    if fit["with_heat"]:
        cols.update(heat_columns(frame, fit["thr"], fit["adaptation"]))
    X = pd.DataFrame(cols, index=frame.index)
    eta = np.log(predict_em_baseline(fit["base"], frame)) + X.to_numpy() @ fit["params"].reindex(X.columns).to_numpy()
    return np.exp(eta), heat_part(X, fit["params"])


def fit_region(train, percentile, study, kind="serfling", with_heat=True, thr=None, adaptation=False):
    thr = thr or thresholds(train, percentile)
    if kind == "euromomo":
        return fit_euromomo(train, thr, study, with_heat, adaptation)
    return fit_serfling(train, thr, study, with_heat, adaptation)


def predict(fit, frame):
    if fit["kind"] == "euromomo":
        mu, part = predict_euromomo(fit, frame)
    else:
        mu, part = predict_serfling(fit, frame)
    return pd.DataFrame({"expected": mu, "attributable": mu * (1 - np.exp(-part))}, index=frame.index)


def select_percentile(train_by_region, grid, study, kind="serfling"):
    scores = {}
    for p in grid:
        scores[p] = float(sum(fit_region(t, p, study, kind)["deviance"] for t in train_by_region.values()))
    return min(scores, key=scores.get), scores


def pool(betas, variances):
    b = np.asarray(betas, dtype=float)
    v = np.asarray(variances, dtype=float)
    k = len(b)
    w = 1.0 / v
    fixed = np.sum(w * b) / np.sum(w)
    q = float(np.sum(w * (b - fixed) ** 2))
    denom = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / denom) if k > 1 and denom > 0 else 0.0
    w_star = 1.0 / (v + tau2)
    pooled = float(np.sum(w_star * b) / np.sum(w_star))
    pooled_var = float(1.0 / np.sum(w_star))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    shrink = tau2 / (tau2 + v) if tau2 > 0 else np.zeros(k)
    blup = pooled + shrink * (b - pooled)
    blup_var = shrink * v + (1 - shrink) ** 2 * pooled_var
    return {"pooled": pooled, "pooled_var": pooled_var, "tau2": tau2, "q": q, "i2": i2,
            "blup": blup, "blup_var": blup_var, "shrink": shrink}
