import numpy as np
import pandas as pd

from buaro import modifiers


def frame(slope, noise, seed=1):
    rng = np.random.default_rng(seed)
    trees = np.array([0.02, 0.03, 0.05, 0.06, 0.08, 0.09, 0.11])
    z = (trees - trees.mean()) / trees.std(ddof=1)
    se = np.full(7, 0.004)
    beta = 0.05 + slope * z + rng.normal(0, noise, 7)
    geo = [f"EL30{i + 1}" for i in range(7)]
    vuln = pd.DataFrame({"geo": geo, "name_en": geo, "name_el": geo, "beta_region": beta, "se_region": se,
                         "rr_per_degree": np.exp(beta), "rr_low": np.exp(beta - 0.008), "rr_high": np.exp(beta + 0.008)})
    exposure = pd.DataFrame({"geo": geo, "tree_share_pop": trees, "share_65plus_census": 0.2 + rng.normal(0, 0.01, 7)})
    return vuln, exposure


def test_meta_regression_finds_protective_trees():
    mods, areas = modifiers.analyse(*frame(-0.01, 0.0005))
    row = mods.set_index("covariate").loc["tree_share_pop"]
    assert row["verdict"] == "lower heat risk where higher"
    assert abs(row["change_pp_per_sd"] + 1.0) < 0.2
    assert row["explained_heterogeneity"] > 0.8
    assert len(areas) == 7


def test_meta_regression_does_not_invent_links():
    mods, _ = modifiers.analyse(*frame(0.0, 0.0005))
    assert mods.set_index("covariate").loc["tree_share_pop", "verdict"] == "no clear link"


def test_missing_exposure_is_skipped(tmp_path):
    assert modifiers.load(tmp_path) is None
    mods, areas = modifiers.analyse(frame(0.0, 0.0)[0], None)
    assert mods.empty and areas.empty


def test_low_coverage_area_is_checked_and_signal_is_reported():
    from buaro import report
    vuln, exposure = frame(-0.006, 0.003, seed=4)
    exposure["population_covered"] = [1, 1, 1, 1, 1, 0.3, 1]
    mods, _ = modifiers.analyse(vuln, exposure)
    row = mods.set_index("covariate").loc["tree_share_pop"]
    assert row["check_without"] == "EL306" and np.isfinite(row["check_p_value"])
    near = mods.assign(verdict="no clear link", p_value=0.05)
    text = "\n".join(report.modifiers_markdown(near))
    assert "comes closest" in text and "signal, not a finding" in text and "tested again without it" in text
