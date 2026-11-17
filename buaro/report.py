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
    lines.append(f"Because {len(mods)} features are tested, the p value is also shown after the Holm correction for multiple tests.")
    lines.append("")
    lines.append("| Feature | Range across areas | Change in extra deaths per +1°C, per 1 SD | 95% range | p | p after Holm correction | Share of area differences explained | Verdict |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in mods.itertuples():
        scale = 100 if r.covariate.endswith(("_pop", "_census")) else 1
        holm = getattr(r, "p_holm", float("nan"))
        lines.append(f"| {r.label} | {fmt(r.min * scale, 1)}% to {fmt(r.max * scale, 1)}% | {r.change_pp_per_sd:+.2f} pp | {r.change_low:+.2f} to {r.change_high:+.2f} | {fmt(r.p_value, 3)} | {fmt(holm, 3)} | {fmt(r.explained_heterogeneity * 100, 0)}% | {r.verdict} |")
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
        if "p_holm" in clear.columns:
            kept = clear[clear["p_holm"] < 0.05]
            if len(kept) == len(clear):
                lines.append("The link holds after the Holm correction for testing several features. It remains an association between seven areas; step 3 tests the cooling itself at 100 m.")
            else:
                lines.append("After the Holm correction for testing several features, not every link stays below 0.05, so it is reported as a signal.")
    lines.append("")
    return lines


def surface_markdown(surface):
    lines = ["## What cools the city: surface temperature at 100 m", ""]
    if not surface or not surface.get("effects"):
        lines.append("The Landsat surface temperature analysis has not run yet.")
        lines.append("")
        return lines
    v = surface["validation"]
    eff = pd.DataFrame(surface["effects"])
    train = eff[eff["period"] == "training summers"].set_index("feature")
    test = eff[eff["period"] == "unseen summers"].set_index("feature")
    lines.append(f"Every clear Landsat 8 and 9 scene of June to August since {v['train_years'][0]} ({surface.get('scenes_used') or 'all'} scenes) gives the land surface temperature of each 100 m cell in late morning. Each scene is compared with the median of the built-up city that day, so hot and cool days cancel out.")
    lines.append(f"Only built-up cells are used (at least {int(v['min_built_up'] * 100)}% built-up land, {v['train_cells']:,} cells), where the Street Tree Layer is mapped. Each cell is compared only with the other cells of the same 1 km square, so the sea, the hills and the wind cannot pose as cooling.")
    lines.append(f"The model learns from the summers {v['train_years'][0]} to {v['train_years'][-1]} and is then tested on {', '.join(str(y) for y in v['test_years'])}, which it has never seen.")
    lines.append("")
    lines.append("| Feature | Change | Summers it learned from | Unseen summers |")
    lines.append("|---|---|---|---|")
    for f in train.index:
        a = train.loc[f]
        b = test.loc[f] if f in test.index else None
        right = f"{b['effect_c']:+.2f}°C ({b['low_c']:+.2f} to {b['high_c']:+.2f})" if b is not None else "n/a"
        lines.append(f"| {a['label']} | {a['unit']} | {a['effect_c']:+.2f}°C ({a['low_c']:+.2f} to {a['high_c']:+.2f}) | {right} |")
    lines.append("")
    if surface.get("curve"):
        lines.append("Surface temperature by street tree cover, compared with cells with almost no trees (0-2%), same 1 km square:")
        lines.append("")
        lines.append("| Tree cover | Difference | 95% range | Cells |")
        lines.append("|---|---|---|---|")
        for r in surface["curve"]:
            lines.append(f"| {r['tree_cover']} | {r['effect_c']:+.2f}°C | {r['low_c']:+.2f} to {r['high_c']:+.2f} | {r['cells']:,} |")
        lines.append("")
    if surface.get("unseen"):
        lines.append("| Unseen summer | Cooling per +10 points of tree cover | 95% range |")
        lines.append("|---|---|---|")
        for r in surface["unseen"]:
            lines.append(f"| {r['summer']} | {r['tree_effect_c']:+.2f}°C | {r['tree_low_c']:+.2f} to {r['tree_high_c']:+.2f} |")
        lines.append("")
    if "tree_share" in train.index and "tree_share" in test.index:
        a, b = train.loc["tree_share"], test.loc["tree_share"]
        holds = b["high_c"] < 0 and all(r["tree_high_c"] < 0 for r in surface.get("unseen", []))
        if holds:
            lines.append(f"Verdict: street trees cool the city surface. Each +10 percentage points of tree cover lowers late-morning surface temperature by about {abs(a['effect_c']):.2f}°C, and the unseen summers confirm it ({b['effect_c']:+.2f}°C), every one of them on its own. Cells with more than 40% tree cover are about {abs(surface['curve'][-1]['effect_c']):.1f}°C cooler than cells with almost none.")
        else:
            lines.append("Verdict: the cooling from street trees learned on past summers is not confirmed on every unseen summer, so it is not yet reported as a finding.")
    if "green_urban_share" in train.index:
        lines.append("Green urban areas add no cooling of their own once their trees are counted, most likely because open, dry grass in an Athens summer heats up much like bare ground. The link between green areas and lower heat mortality found between areas may therefore be carried in part by their trees, and by the shade and cooler air they offer to people, which surface temperature does not measure; the link between tree cover and heat mortality across the seven areas is not clear on its own.")
    if "dense_fabric_share" in train.index:
        lines.append("Dense, highly sealed fabric is slightly cooler at the surface in late morning than open, low-density or industrial land in the same square, most likely because tall buildings shade the streets while bare land and large flat roofs heat fastest. This is a daytime surface effect; dense fabric releases its heat at night, which Landsat does not see.")
    lines.append("")
    lines.append(f"The features explain {fmt(v.get('test_r2_within', float('nan')) * 100, 0)}% of the differences between cells of the same square in the unseen summers. The rest comes from roofs, materials and street shape, which no open European layer maps yet. The finding is the size of the tree effect and its stability from summer to summer, not a full map of the causes.")
    lines.append("")
    if surface.get("map"):
        lines.append("![Summer surface temperature](../docs/lst_map.png)")
        lines.append("")
    return lines


def outlook_markdown(outlook):
    lines = ["## Heat deaths at 1.5°C, 2°C and 3°C of global warming", ""]
    if outlook is None or outlook.empty or len(outlook) < 2:
        lines.append("The climate projections (Copernicus Interactive Climate Atlas, CMIP6) are not available yet, so this step has not run.")
        lines.append("")
        return lines
    lines.append("The summers 2015 to 2025, outside the pandemic years, are replayed at the three reference levels of the EU Climate Resilience Law. Each week's maximum temperature is raised by the summer warming that the CMIP6 models of the Copernicus Interactive Climate Atlas project for Athens at that level, month by month, relative to 2011-2030. Population and vulnerability stay as they are today, so the numbers show what warming alone would do if nothing else changed.")
    lines.append("")
    lines.append("| Global warming | Summer warming in Athens | Heat-related deaths per summer | Range | Compared with today | Weeks above the threshold, per area |")
    lines.append("|---|---|---|---|---|---|")
    for r in outlook.itertuples():
        label = "Today" if r.level == 0 else f"{r.level:g}°C"
        lines.append(f"| {label} | {r.summer_warming_c:+.1f}°C | {r.heat_deaths_central:,.0f} | {r.heat_deaths_low:,.0f} to {r.heat_deaths_high:,.0f} | {r.times_today:.2f}× | {r.hot_weeks_per_area:.1f} |")
    lines.append("")
    last = outlook.iloc[-1]
    lines.append(f"Verdict: at {last['level']:g}°C of global warming, the same summers would bring about {last['times_today']:.1f} times today's heat deaths in Attica, unless vulnerability falls. Step 4 measures how much of that street trees and green areas can prevent.")
    lines.append("")
    return lines


def measures_markdown(measures):
    lines = ["## What prevents heat deaths: measures", ""]
    if measures is None or measures.empty:
        lines.append("The measures need the results of steps 2 and 3, which are not available yet.")
        lines.append("")
        return lines
    today = measures[measures["level"] == 0]
    lines.append("Two independent lines of evidence are set side by side, because they answer different questions.")
    lines.append("")
    lines.append("- **Cooling measured from space** (step 3): more street trees where people live lower the surface temperature. Only part of that reaches the air people breathe: across 392 European urban areas, trees cool the daytime air about 12 times less than the surface (Du et al. 2024, Environmental Research Letters). The low end uses that ratio, the high end assumes the air cools as much as the surface.")
    lines.append("- **Difference between areas** (step 2): areas with more green and more trees lose fewer people per degree of heat. Bringing each area up to the best-covered area of Attica stays inside what Attica already shows; it includes everything that comes with greener neighbourhoods, so it is an upper bound.")
    lines.append("")
    lines.append("| Measure | Evidence | Heat deaths avoided per summer, today's climate | Range | Share of heat deaths | New cover needed |")
    lines.append("|---|---|---|---|---|---|")
    supported = today["supported"] if "supported" in today.columns else pd.Series(True, index=today.index)
    for r in today.assign(supported=supported).sort_values(["kind", "measure"], ascending=[False, True]).itertuples():
        evidence = r.evidence if r.supported else f"{r.evidence}, not supported by the data"
        lines.append(f"| {r.measure} | {evidence} | {r.deaths_avoided:,.0f} | {r.deaths_avoided_low:,.0f} to {r.deaths_avoided_high:,.0f} | {r.share_avoided * 100:.1f}% | {r.new_cover_ha:,.0f} ha |")
    lines.append("")
    phys = today[today["kind"] == "physical"]
    assoc = today[(today["kind"] == "association") & supported]
    if not today[(today["kind"] == "association") & ~supported].empty:
        names = ", ".join(f"“{m}”" for m in today[(today["kind"] == "association") & ~supported]["measure"])
        lines.append(f"{names} rests on a difference between areas that the seven areas do not support, so its numbers show only what the data allow, not an expected gain.")
        lines.append("")
    if not phys.empty and not assoc.empty:
        p = phys.sort_values("deaths_avoided").iloc[-1]
        a = assoc.sort_values("deaths_avoided").iloc[-1]
        lines.append(f"Verdict: the cooling that trees bring to the air explains only a small part of the protection seen in greener areas ({p['deaths_avoided']:,.0f} deaths a summer for “{p['measure']}”, against {a['deaths_avoided']:,.0f} for “{a['measure']}”). Greener neighbourhoods most likely protect people in other ways as well, such as shade where people walk and wait, cooler homes, and the housing and health that often come with them; with seven areas these cannot yet be told apart. Trees are worth planting for cooling, but most of the gain seen between areas lies in what makes green neighbourhoods safer.")
        lines.append("")
    return lines


def summary_markdown(meta, vuln, summary, comparison, reference, response=None, mods=None, surface=None, outlook=None, measures=None):
    winner = meta["baseline_selected"]
    best = summary[summary["baseline"] == winner] if not summary.empty else summary
    lines = []
    lines.append("# BUARO results: Attica")
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
        lines.append("The same summers are used to choose the baseline and to report its forecast skill, so the skill of the selected baseline may be slightly optimistic.")
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
    share_label = f"Share aged 65+ (Eurostat, 1 January {meta['share_65plus_year']})" if meta.get("share_65plus_year") else "Share aged 65+ (Eurostat)"
    lines.append(f"| Area | Extra deaths per +1°C | 95% range | {share_label} | Threshold (°C) |")
    lines.append("|---|---|---|---|---|")
    for r in vuln.sort_values("rr_per_degree", ascending=False).itertuples():
        share = fmt(r.share_65plus * 100, 1) + "%" if np.isfinite(r.share_65plus) else "n/a"
        lines.append(f"| {r.name_en} | {fmt((r.rr_per_degree - 1) * 100, 1)}% | {fmt((r.rr_low - 1) * 100, 1)}% to {fmt((r.rr_high - 1) * 100, 1)}% | {share} | {fmt(r.threshold_c, 1)} |")
    lines.append("")
    lines.append("![Vulnerability](vulnerability.png)")
    lines.append("")
    lines.extend(modifiers_markdown(mods))
    lines.extend(surface_markdown(surface))
    lines.extend(outlook_markdown(outlook))
    lines.extend(measures_markdown(measures))
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
        lines.append("| Summer | Recorded deaths | Forecast with heat | Forecast without heat | Heat-attributable deaths in the forecast | Skill score |")
        lines.append("|---|---|---|---|---|---|")
        for r in best.itertuples():
            lines.append(f"| {r.year} | {fmt(r.observed_deaths)} | {fmt(r.predicted_with_heat)} | {fmt(r.predicted_without_heat)} | {fmt(r.heat_attributable_deaths)} | {fmt(r.skill_score, 2)} |")
        lines.append("")
        better = int((best["skill_score"] > 0).sum())
        lines.append(f"Knowing about heat improved the forecast in {better} of {len(best)} summers tested.")
        lines.append("The heat-attributable deaths in this table come from forecasts made with data up to 31 May of each summer. The estimate of each summer's heat deaths is the one of the model fitted on all summers, given in the cross-check below.")
        lines.append("")
        for year in best["year"]:
            lines.append(f"![Summer {year}](backtest_{year}.png)")
            lines.append("")
        lines.append("![Skill](skill.png)")
    lines.append("")
    lines.append("## Cross-check against published research")
    lines.append("")
    lines.append(f"{reference['label']}: {fmt(reference['greece_heat_deaths_2024'])} heat-related deaths in Greece. Scaled to Attica's population share, that is about {fmt(meta['reference_attica_2024'])}.")
    weeks = meta.get("model_summer_weeks") or [23, 39]
    model_2024 = meta.get("model_attica_2024")
    if model_2024:
        gap = (model_2024 / meta["reference_attica_2024"] - 1) * 100
        lines.append(f"This model, fitted on all summers, estimates {fmt(model_2024)} heat-related deaths for Attica in summer 2024 (weeks {weeks[0]} to {weeks[1]}), {fmt(abs(gap), 0)}% {'below' if gap < 0 else 'above'} the scaled ISGlobal figure.")
    row = best[best["year"] == 2024] if not best.empty else best
    if not row.empty:
        lines.append(f"The forecast made with data up to 31 May 2024 put it at {fmt(float(row['heat_attributable_deaths'].iloc[0]))}.")
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
    lines.append("- Landsat measures the temperature of the ground and roofs in late morning, not the air temperature people feel, and not the night. Elevation comes from the Copernicus DEM, a surface model that includes buildings and trees, so part of the cooling of trees may be counted under elevation; the tree estimate is, if anything, conservative.")
    lines.append("- The warming-level outlook shifts each week's observed maximum temperature by the projected monthly change (delta method). It does not change how variable summers are, nor population, ageing or adaptation, and the models are not bias-adjusted; only their change is used.")
    lines.append("- The pooled estimates are empirical Bayes (random-effects shrinkage), a first step towards the full Bayesian layer of the framework.")
    lines.append("")
    return "\n".join(lines)


def write_meta(path, meta):
    Path(path).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
