import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
OBSERVED = "#2a78d6"
WITH_HEAT = "#eb6834"
WITHOUT_HEAT = "#1baf7a"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_backtest(total, year, path):
    data = total[total["year"] == year].sort_values("date")
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.plot(data["date"], data["deaths"], color=OBSERVED, linewidth=2, marker="o", markersize=4, label="Recorded deaths")
    ax.plot(data["date"], data["predicted_with_heat"], color=WITH_HEAT, linewidth=2, label="Model with heat")
    ax.plot(data["date"], data["predicted_without_heat"], color=WITHOUT_HEAT, linewidth=2, linestyle=(0, (4, 3)), label="Model without heat")
    ax.set_ylabel("Deaths per week, all of Attica", color=INK_2, fontsize=9)
    ax.set_title(f"Summer {year}: forecast made with data up to 31 May {year}", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def chart_vulnerability(vuln, path):
    data = vuln.sort_values("rr_per_degree")
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(data) + 1.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(data))
    pct = (data["rr_per_degree"] - 1) * 100
    lo = (data["rr_low"] - 1) * 100
    hi = (data["rr_high"] - 1) * 100
    ax.hlines(y, lo, hi, color=OBSERVED, linewidth=2)
    ax.plot(pct, y, "o", color=OBSERVED, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(data["name_en"], color=INK, fontsize=9)
    for yi, value in zip(y, pct):
        ax.annotate(f"{value:+.1f}%", (value, yi), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Extra deaths per 1°C above the local heat threshold (weekly, % change)", color=INK_2, fontsize=9)
    ax.set_title("How strongly heat raises mortality in each part of Attica", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def chart_skill(summary, path):
    if summary.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    years = summary["year"].astype(str).tolist()
    x = np.arange(len(years))
    width = 0.36
    ax.bar(x - width / 2 - 0.01, summary["rmse_without_heat"], width, color=WITHOUT_HEAT, label="Model without heat")
    ax.bar(x + width / 2 + 0.01, summary["rmse_with_heat"], width, color=WITH_HEAT, label="Model with heat")
    for xi, a, b in zip(x, summary["rmse_without_heat"], summary["rmse_with_heat"]):
        ax.annotate(f"{a:.0f}", (xi - width / 2, a), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
        ax.annotate(f"{b:.0f}", (xi + width / 2, b), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels(years, color=INK)
    ax.set_ylabel("Typical weekly error (deaths, RMSE)", color=INK_2, fontsize=9)
    ax.set_title("Lower is better: forecast error in summer weeks", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


BASELINE_LABEL = {"serfling": "All-weeks baseline", "euromomo": "EuroMOMO-style baseline"}


def chart_compare(summary, path):
    if summary.empty:
        return None
    kinds = [k for k in ("serfling", "euromomo") if k in set(summary["baseline"])]
    years = sorted(summary["year"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    x = np.arange(len(years))
    width = 0.36
    colours = {"serfling": WITHOUT_HEAT, "euromomo": OBSERVED}
    for i, kind in enumerate(kinds):
        rows = summary[summary["baseline"] == kind].set_index("year").reindex(years)
        pos = x + (i - (len(kinds) - 1) / 2) * (width + 0.02)
        ax.bar(pos, rows["rmse_with_heat"], width, color=colours[kind], label=BASELINE_LABEL[kind])
        for xi, v in zip(pos, rows["rmse_with_heat"]):
            if np.isfinite(v):
                ax.annotate(f"{v:.0f}", (xi, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], color=INK)
    ax.set_ylabel("Typical weekly error (deaths, RMSE)", color=INK_2, fontsize=9)
    ax.set_title("Which baseline forecasts summers better (lower is better)", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def write_sheet(wb, title, frame, notes=None):
    ws = wb.create_sheet(title=title[:31])
    row = 1
    if notes:
        for note in notes:
            ws.cell(row=row, column=1, value=note).font = Font(italic=True, color="52514E")
            row += 1
        row += 1
    header_fill = PatternFill("solid", fgColor="E8F0FB")
    for j, col in enumerate(frame.columns, start=1):
        cell = ws.cell(row=row, column=j, value=str(col))
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for i, record in enumerate(frame.itertuples(index=False), start=row + 1):
        for j, value in enumerate(record, start=1):
            if isinstance(value, (pd.Timestamp,)):
                value = value.to_pydatetime().date()
            elif isinstance(value, (np.floating,)):
                value = None if np.isnan(value) else float(value)
            elif isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.bool_,)):
                value = bool(value)
            elif isinstance(value, float) and np.isnan(value):
                value = None
            ws.cell(row=i, column=j, value=value)
    for j, col in enumerate(frame.columns, start=1):
        width = max(12, min(48, len(str(col)) + 2))
        ws.column_dimensions[get_column_letter(j)].width = width
    ws.freeze_panes = ws.cell(row=row + 1, column=1)
    return ws


def workbook(path, sheets):
    wb = Workbook()
    wb.remove(wb.active)
    for title, frame, notes in sheets:
        write_sheet(wb, title, frame, notes)
    wb.save(path)
    return path


def fmt(value, digits=0):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


def modifiers_markdown(mods):
    lines = ["## What explains the differences between areas", ""]
    if mods is None or mods.empty:
        lines.append("The area features (census grid, Urban Atlas, Street Tree Layer) are not available yet, so this test has not run.")
        lines.append("")
        return lines
    lines.append("Each area's heat effect is compared with one feature of the places where its residents live, measured on the Eurostat 1 km census grid and the Copernicus Urban Atlas 2021 and Street Tree Layer 2021.")
    lines.append("The test is a random-effects meta-regression across the seven areas, one feature at a time. It shows associations between areas, not proof of cause.")
    lines.append("")
    lines.append("| Feature | Range across areas | Change in extra deaths per +1°C, per 1 SD | 95% range | p | Share of area differences explained | Verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in mods.itertuples():
        scale = 100 if r.covariate.endswith(("_pop", "_census")) else 1
        lines.append(f"| {r.label} | {fmt(r.min * scale, 1)}% to {fmt(r.max * scale, 1)}% | {r.change_pp_per_sd:+.2f} pp | {r.change_low:+.2f} to {r.change_high:+.2f} | {fmt(r.p_value, 3)} | {fmt(r.explained_heterogeneity * 100, 0)}% | {r.verdict} |")
    lines.append("")
    if "check_without" in mods.columns and mods["check_without"].notna().any():
        left = mods["check_without"].dropna().iloc[0]
        lines.append(f"Check: the Urban Atlas covers less than 80% of the residents of {left}, so each land feature is tested again without it.")
        lines.append("")
        lines.append("| Feature | Change per 1 SD, without the low-coverage area | 95% range | p | Verdict |")
        lines.append("|---|---|---|---|---|")
        for r in mods[mods["check_without"].notna()].itertuples():
            lines.append(f"| {r.label} | {r.check_change_pp_per_sd:+.2f} pp | {r.check_low:+.2f} to {r.check_high:+.2f} | {fmt(r.check_p_value, 3)} | {r.check_verdict} |")
        lines.append("")
    clear = mods[mods["verdict"] != "no clear link"]
    near = mods[(mods["verdict"] == "no clear link") & (mods["p_value"] < 0.1)]
    if clear.empty and not near.empty:
        r = near.sort_values("p_value").iloc[0]
        direction = "lower" if r["change_pp_per_sd"] < 0 else "higher"
        lines.append(f"Verdict: {r['label'].lower()} comes closest. Areas where it is one standard deviation higher show about {abs(r['change_pp_per_sd']):.1f} percentage points {direction} extra deaths per +1°C, and it accounts for {fmt(r['explained_heterogeneity'] * 100, 0)}% of the differences between areas, but with p = {fmt(r['p_value'], 3)} and seven areas this is a signal, not a finding. Step 3 tests it at 100 m with satellite surface temperature.")
    elif clear.empty:
        lines.append("Verdict: no single feature separates the areas clearly with seven areas. The differences in heat risk remain unexplained at this level, which is why step 3 moves to 100 m surface temperature.")
    else:
        names = ", ".join(clear["label"].str.lower())
        lines.append(f"Verdict: {names} {'is' if len(clear) == 1 else 'are'} clearly linked to the differences in heat risk between areas.")
    lines.append("")
    return lines


def summary_markdown(meta, vuln, summary, comparison, reference, response=None, mods=None):
    winner = meta["baseline_selected"]
    best = summary[summary["baseline"] == winner] if not summary.empty else summary
    lines = []
    lines.append("# BUARO results: Attica, step 2")
    lines.append("")
    lines.append(f"Run date: {meta['run_date']}. Data up to week {meta['last_week']}. Deaths dataset: `{meta['deaths_dataset']}`.")
    lines.append("")
    lines.append("## What this step answers")
    lines.append("")
    lines.append("How much does heat raise the number of deaths in each part of Attica, can a model that knows this forecast a summer better than a model that ignores heat, and is Attica becoming less vulnerable over time?")
    lines.append("")
    lines.append("## Two ways to set the normal level of deaths")
    lines.append("")
    lines.append("Heat deaths are measured against a normal level of deaths. The pipeline tests two ways to set it, and keeps the one that forecasts unseen summers better:")
    lines.append("")
    lines.append("- **All-weeks baseline**: learns the normal level from every week of the year, with a long-term trend.")
    lines.append("- **EuroMOMO-style baseline**: learns the normal level only from calm weeks in spring (weeks 16 to 25) and autumn (weeks 37 to 44), with a separate level for each year, as the European mortality monitoring network EuroMOMO does. A hard winter or a hot summer cannot distort it.")
    lines.append("")
    if not comparison.empty:
        lines.append("| Baseline | Summers tested | Mean weekly error with heat | Summers where heat improved the forecast |")
        lines.append("|---|---|---|---|")
        for r in comparison.itertuples():
            lines.append(f"| {BASELINE_LABEL.get(r.baseline, r.baseline)} | {r.summers} | {fmt(r.mean_rmse_with_heat, 1)} | {r.summers_improved} of {r.summers} |")
        lines.append("")
        lines.append(f"Selected: **{BASELINE_LABEL.get(winner, winner)}** (rule fixed in advance: lowest mean weekly error over all tested summers).")
        lines.append("")
        lines.append("![Baseline comparison](baseline_comparison.png)")
        lines.append("")
    lines.append("## How vulnerable each area is")
    lines.append("")
    lines.append(f"The heat threshold was learned from the data: weeks hotter than the {meta['percentile']}th percentile of each area's weekly maximum temperature.")
    lines.append(f"Across Attica, each extra 1°C above that threshold raises weekly deaths by about {fmt((meta['pooled_rr'] - 1) * 100, 1)}% (95% range {fmt((meta['pooled_rr_low'] - 1) * 100, 1)}% to {fmt((meta['pooled_rr_high'] - 1) * 100, 1)}%).")
    lines.append(f"Differences between areas beyond chance (I²): {fmt(meta['i2'] * 100, 0)}%.")
    if meta["pooled_rr_low"] > 1:
        lines.append("Verdict: the data show a clear heat effect on mortality.")
    else:
        lines.append("Verdict: with the data available, the heat effect cannot be told apart from zero.")
    lines.append("")
    lines.append("| Area | Extra deaths per +1°C | 95% range | Share aged 65+ | Threshold (°C) |")
    lines.append("|---|---|---|---|---|")
    for r in vuln.sort_values("rr_per_degree", ascending=False).itertuples():
        share = fmt(r.share_65plus * 100, 1) + "%" if np.isfinite(r.share_65plus) else "n/a"
        lines.append(f"| {r.name_en} | {fmt((r.rr_per_degree - 1) * 100, 1)}% | {fmt((r.rr_low - 1) * 100, 1)}% to {fmt((r.rr_high - 1) * 100, 1)}% | {share} | {fmt(r.threshold_c, 1)} |")
    lines.append("")
    lines.append("![Vulnerability](vulnerability.png)")
    lines.append("")
    lines.extend(modifiers_markdown(mods))
    a = meta["adaptation"]
    lines.append("## Is Attica adapting?")
    lines.append("")
    lines.append("The model lets the heat effect change linearly from year to year. A negative trend means each degree of heat is becoming less deadly.")
    lines.append(f"Estimated extra deaths per +1°C: {fmt((a['rr_2015'] - 1) * 100, 1)}% in 2015 and {fmt((a['rr_2025'] - 1) * 100, 1)}% in 2025.")
    lines.append(f"Yearly change in the log relative risk: {a['delta']:+.4f} (95% range {a['delta_low']:+.4f} to {a['delta_high']:+.4f}).")
    if a["verdict"] == "no clear change":
        lines.append("Verdict: no clear change over time can be shown yet with the data available.")
    elif a.get("robust", True):
        if a["verdict"] == "decreasing":
            lines.append("Verdict: heat is becoming less deadly per degree, and this holds whichever single summer is left out. This is measurable evidence of adaptation.")
        else:
            lines.append("Verdict: heat is becoming more deadly per degree, and this holds whichever single summer is left out. Vulnerability is rising.")
    else:
        years = ", ".join(str(y) for y in a.get("depends_on", []))
        direction = "less" if a["verdict"] == "decreasing" else "more"
        lines.append(f"Verdict: not robust. The trend points to heat becoming {direction} deadly per degree, but it disappears when the summer of {years} is left out. One unusual summer is not evidence of adaptation, so no claim is made yet.")
    lines.append("")
    if response is not None and not response.empty:
        lines.append("A check that needs no model: how many extra summer deaths each degree of heat brought, year by year.")
        lines.append("")
        lines.append("| Summer | Extra deaths, weeks 26 to 36 | Degree-weeks above threshold | Extra deaths per degree-week |")
        lines.append("|---|---|---|---|")
        for r in response.itertuples():
            lines.append(f"| {r.year} | {fmt(r.summer_excess_deaths)} | {fmt(r.degree_weeks_above_threshold, 1)} | {fmt(r.excess_per_degree_week, 1)} |")
        lines.append("")
    lines.append("## The test: forecasting summers the model has never seen")
    lines.append("")
    lines.append("For each summer, the model is trained only on data available up to 31 May of that year, then asked to predict June to September.")
    lines.append("A skill score above 0 means that knowing about heat made the forecast better.")
    lines.append("")
    if best.empty:
        lines.append("No summer could be tested yet with the data available.")
    else:
        lines.append("| Summer | Recorded deaths | Forecast with heat | Forecast without heat | Heat-attributable deaths | Skill score |")
        lines.append("|---|---|---|---|---|---|")
        for r in best.itertuples():
            lines.append(f"| {r.year} | {fmt(r.observed_deaths)} | {fmt(r.predicted_with_heat)} | {fmt(r.predicted_without_heat)} | {fmt(r.heat_attributable_deaths)} | {fmt(r.skill_score, 2)} |")
        lines.append("")
        better = int((best["skill_score"] > 0).sum())
        lines.append(f"Knowing about heat improved the forecast in {better} of {len(best)} summers tested.")
        lines.append("")
        for year in best["year"]:
            lines.append(f"![Summer {year}](backtest_{year}.png)")
            lines.append("")
        lines.append("![Skill](skill.png)")
    lines.append("")
    lines.append("## Cross-check against published research")
    lines.append("")
    lines.append(f"{reference['label']}: {fmt(reference['greece_heat_deaths_2024'])} heat-related deaths in Greece. Scaled to Attica's population share, that is about {fmt(meta['reference_attica_2024'])}.")
    row = best[best["year"] == 2024] if not best.empty else best
    if not row.empty:
        lines.append(f"This model estimates {fmt(float(row['heat_attributable_deaths'].iloc[0]))} for Attica in summer 2024.")
    lines.append(reference["note"])
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append("Every weekly number used by the model is in `data_panel.xlsx`, so anyone can check the results.")
    lines.append("")
    lines.append("## Honest limits")
    lines.append("")
    lines.append("- Deaths are weekly and by area (NUTS 3), not daily or by neighbourhood. The area features are summarised from the 1 km census grid; step 3 adds 100 m satellite surface temperature.")
    lines.append("- Temperature is ERA5-Land reanalysis (about 11 km), averaged over two points per area.")
    lines.append("- Pandemic weeks (2020-W10 to 2022-W26) are left out of training, because COVID-19 deaths would distort the heat signal.")
    lines.append("- The adaptation test assumes a straight-line change over time and has few post-pandemic summers. It is repeated with each summer left out, and a trend is only reported as evidence if no single summer changes the verdict.")
    lines.append("- With seven areas, the comparison of heat risk with tree cover, sealing and age can show associations between areas only. Cause and effect is tested in step 3 at street scale.")
    lines.append("- The pooled estimates are empirical Bayes (random-effects shrinkage), a first step towards the full Bayesian layer of the framework.")
    lines.append("")
    return "\n".join(lines)


def write_meta(path, meta):
    Path(path).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
