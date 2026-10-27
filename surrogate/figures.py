"""
Figures and tables of the paper (step 8).

Figures (png and pdf) and LaTeX tables of the paper, written to P['figures']:
  fig_learning_curves     Zambia: WMAPE, R2 and pair accuracy vs number of training runs
  fig_heatmap             Zambia: WMAPE and share of the output total per technology group (1,000 runs)
  fig_uncertainty         Zambia: coverage and half-width of the 90% intervals
  fig_ghana_wmape / _pairs / _r2   Ghana: with and without transfer, per output category
  fig_demonstration       the four steps of the demonstration
  supp_*.tex              supplementary tables (zero detection, intervals and pair accuracy per group)
The tables of the levels of use (table_minimum_runs_*.tex) are written by minimum_runs.
One colour and marker per output category (Okabe-Ito, colour-blind safe), the same in every figure.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .definitions import (ACTIVE_THRESHOLD, ALPHA, CALIBRATION_RUNS, LAND_CLUSTERS, LATEX_LABEL,
    TRANSFER_OUTPUTS, ZAMBIA_OUTPUTS)
from .utils import log
from .scenarios import load_scenarios
from .transfer import ARMS
from .demonstration import BEST_SHARE


STYLE = {"CAPACITY": ("#0072B2", "o"), "ELC001": ("#E69F00", "s"), "EMICO2": ("#009E73", "^"),
         "LANDUSE": ("#CC79A7", "D"), "CROPPROD": ("#56B4E9", "v"), "AGRWATER": ("#D55E00", "P"),
         "FIXEDCOST": ("#F0E442", "X"), "VARCOST": ("#000000", "*"), "TDC": ("#882255", "h")}
TICK, FONT, LEGEND, LINE, MARKER = 7, 8, 7, 1.0, 3
INPUT_LABELS = {"demand_multiplier": "Energy demand", "discount_rate": "Discount rate",
                "fuel_cost_multiplier": "Fuel cost", "capital_cost_multiplier": "Capital cost",
                "capacity_factor_multiplier": "Capacity factor", "iar_multiplier": "Input ratio (IAR)",
                "oar_multiplier": "Output ratio (OAR)", "water_accumulated_multiplier": "Water demand",
                "crop_accumulated_multiplier": "Crop demand"}
GROUP_LABELS = {"ALL": "All groups", "BIO": "Biomass", "COA": "Coal", "DIST": "Distribution", "GEO": "Geothermal",
                "HYD": "Hydropower", "OHC": "Oil-fired", "SOL": "Solar", "TRN": "Transmission", "TRNIMP": "Imports",
                "WND": "Wind", "AGRDSL": "Agricultural diesel", "CAS": "Cassava", "MAI": "Maize",
                "OTC": "Other crops", "SOY": "Soybean", "WHE": "Wheat", "TOTAL": "Total"}


def save_figure(P, fig, name, **kwargs):
    import matplotlib.pyplot as plt
    for ext in ("png", "pdf"):
        fig.savefig(Path(P["figures"]) / f"{name}.{ext}", dpi=300, **kwargs)
    plt.close(fig)
    log(f"figure {name}")


def size_axis(ax, sizes, labelled):
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{t:,}" if t in labelled else "" for t in sizes])
    ax.minorticks_off()
    ax.grid(True, which="major", color="0.9", linewidth=0.6)


def category_rows(P, study, category, file):
    t = pd.read_csv(Path(P["results"]) / study / category / file)
    return t[t["GROUP" if file == "ranking.csv" else "group"] == "ALL"].sort_values("size")


def zambia_curves(P, panels, figsize):
    """Output categories of the Zambia surrogate vs number of training runs, one panel per measure.
    panels: (file, column, sd column, factor, lower and upper limit of the +/- 1 sd band)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(panels), 1, figsize=figsize, sharex=True)
    sizes = set()
    for category in ZAMBIA_OUTPUTS:
        colour, marker = STYLE[category]
        style = dict(color=colour, marker=marker, markersize=MARKER, linewidth=LINE, markeredgewidth=0.6)
        for i, (ax, (file, column, sd_column, factor, lower, upper)) in enumerate(zip(axes, panels)):
            try:
                t = category_rows(P, "zambia", category, file)
            except FileNotFoundError:
                continue
            x, y, sd = t["size"], factor * t[column], factor * t[sd_column].fillna(0)
            sizes |= set(x)
            ax.plot(x, y, label=ZAMBIA_OUTPUTS[category]["label"] if i == 0 else None, **style)
            ax.fill_between(x, (y - sd).clip(lower=lower), (y + sd).clip(upper=upper), color=colour, alpha=0.15,
                            linewidth=0)
    for ax in axes:
        size_axis(ax, sorted(sizes), {100, 250, 500, 1000, 2000, 5000})
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Number of training runs")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=LEGEND)
    return fig, axes, sizes


def fig_learning_curves(P):
    """(a) WMAPE, (b) R2, (c) pair accuracy of the output categories vs number of training runs."""
    panels = [("summary.csv", "wmape_mean", "wmape_sd", 100, 1e-3, None),
              ("summary.csv", "r2_mean", "r2_sd", 1, 1e-3, 1.0),
              ("ranking.csv", "pair_accuracy_clear_mean", "pair_accuracy_clear_std", 1, None, 1.0)]
    fig, (ax1, ax2, ax3), _ = zambia_curves(P, panels, (5.5, 8.0))
    ax1.set_yscale("log")
    ax1.set_yticks([1, 2, 5, 10, 20, 50])
    ax1.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    for ax, label, title in [(ax1, "WMAPE (%)", "(a) Error of the predicted values"),
                             (ax2, "R$^2$", "(b) Variation captured"),
                             (ax3, "Clear pair accuracy", "(c) Ordering of scenarios")]:
        ax.set_ylabel(label)
        ax.set_title(title, fontsize=FONT)
    ax2.set_ylim(top=1.0)
    ax3.set_ylim(top=1.0)
    fig.tight_layout(rect=(0, 0.10, 1, 1), h_pad=1.2)
    save_figure(P, fig, "fig_learning_curves")


def fig_uncertainty(P):
    """(a) coverage of the non-zero values and (b) half-width of the 90% intervals (250 calibration runs)."""
    cov, wid = f"coverage_nonzero_cal{CALIBRATION_RUNS}", f"width_nonzero_cal{CALIBRATION_RUNS}"
    panels = [("summary.csv", f"{cov}_mean", f"{cov}_sd", 100, None, None),
              ("summary.csv", f"{wid}_mean", f"{wid}_sd", 50, 0.1, None)]          # 50 x width = half-width in %
    fig, (ax1, ax2), sizes = zambia_curves(P, panels, (5.5, 6.0))
    target = 100 * (1 - ALPHA)
    ax1.axhline(target, color="0.4", linestyle="--", linewidth=0.9)
    ax1.text(min(sizes), target, f" target {target:.0f}%", va="bottom", ha="left", fontsize=TICK, color="0.3")
    ax1.set_ylim(80, 95)
    ax1.set_ylabel("Coverage of non-zero values (%)")
    ax1.set_title("(a) Share of true values inside the 90% interval", fontsize=FONT)
    ax2.set_yscale("log")
    ax2.minorticks_off()
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200) if t <= ax2.get_ylim()[1] * 1.5]
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([str(t) for t in ticks])
    ax2.set_ylabel("Interval half-width ($\\pm$%)")
    ax2.set_title("(b) Width of the 90% interval", fontsize=FONT)
    fig.tight_layout(rect=(0, 0.12, 1, 1), h_pad=1.2)
    save_figure(P, fig, "fig_uncertainty")


def fig_heatmap(P, size=1000, colour_range=(1, 100)):
    """WMAPE (colour, upper number) and share of the output total (in brackets) per technology group."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    data = pd.read_csv(Path(P["results"]) / "zambia" / "heatmap_numbers.csv")
    data = data[data["size"] == size]
    power_cols = {"ELC001": "Electricity\ngeneration", "CAPACITY": "Installed\ncapacity", "EMICO2": "CO$_2$\nemissions"}
    power_rows = {"HYD": "Hydropower", "TRNIMP": "Imports", "TRN": "Transmission", "DIST": "Distribution",
                  "OHC": "Other (OHC)", "BIO": "Biomass", "GEO": "Geothermal", "SOL": "Solar", "COA": "Coal",
                  "WND": "Wind", "AGRDSL": "Agricultural diesel"}
    land_cols = {"LANDUSE": "Land\nuse", "CROPPROD": "Crop\nproduction", "AGRWATER": "Agricultural\nwater",
                 "FIXEDCOST": "Fixed\ncost", "VARCOST": "Variable\ncost"}
    crops = {"MAI": "Maize", "CAS": "Cassava", "SOY": "Soybean", "OTC": "Other crops", "WHE": "Wheat"}
    land_rows = [f"{c}_{k}" for c in crops for k in LAND_CLUSTERS]
    land_labels = [f"{crops[c]} {k}" for c in crops for k in LAND_CLUSTERS]
    norm, cmap = LogNorm(*colour_range), plt.get_cmap("cividis_r")

    def draw(ax, rows, cols, row_labels):
        w = 100 * data.pivot(index="group", columns="output", values="wmape_mean").reindex(index=rows, columns=cols)
        s = 100 * data.pivot(index="group", columns="output", values="share_of_output").reindex(index=rows, columns=cols)
        image = ax.imshow(np.ma.masked_invalid(w.to_numpy()), cmap=cmap, norm=norm, aspect="auto")
        ax.set_facecolor("0.88")                                     # grey = group not in the output
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                v = w.iat[i, j]
                if np.isnan(v):
                    continue
                colour = "white" if norm(min(max(v, colour_range[0]), colour_range[1])) > 0.5 else "black"
                text = f">{colour_range[1]:.0f}" if v > colour_range[1] else (f"{v:.0f}" if v >= 10 else f"{v:.1f}")
                share = s.iat[i, j]
                ax.text(j, i - 0.13, text, ha="center", va="center", fontsize=6.5, color=colour)
                ax.text(j, i + 0.22, "(<0.1%)" if share < 0.1 else f"({share:.1f}%)", ha="center", va="center",
                        fontsize=5, color=colour)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(list(cols.values()), fontsize=TICK)
        ax.xaxis.tick_top()
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(row_labels, fontsize=TICK)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return image

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 6.2),
                                   gridspec_kw={"width_ratios": [len(power_cols), len(land_cols)], "wspace": 0.55})
    draw(ax1, list(power_rows), power_cols, list(power_rows.values()))
    image = draw(ax2, land_rows, land_cols, land_labels)
    for i in (3, 6, 9, 12):                                          # lines between the crops
        ax2.axhline(i - 0.5, color="white", linewidth=1.2)
    ax1.set_title("(a) Power sector and emissions", fontsize=FONT, pad=28)
    ax2.set_title("(b) Land outputs", fontsize=FONT, pad=28)
    bar = fig.colorbar(image, ax=[ax1, ax2], orientation="horizontal", fraction=0.03, pad=0.03, aspect=50)
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100) if colour_range[0] <= t <= colour_range[1]]
    bar.set_ticks(ticks)
    bar.set_ticklabels([str(t) for t in ticks])
    bar.ax.tick_params(labelsize=TICK)
    bar.set_label(f"WMAPE (%) with {size:,} training runs", fontsize=FONT)
    save_figure(P, fig, "fig_heatmap", bbox_inches="tight")


GHANA_FIGURES = {   # column, sd column, file, scale, axis label, log axis, threshold of the levels of use
    "wmape": ("wmape_mean", "wmape_sd", "summary.csv", 100, "WMAPE (%)", True, 5),
    "pairs": ("pair_accuracy_clear_mean", "pair_accuracy_clear_std", "ranking.csv", 1, "Pair accuracy", False, 0.90),
    "r2": ("r2_mean", "r2_sd", "summary.csv", 1, "R$^2$", False, 0.95),
}


def fig_ghana(P, metric):
    """One panel per output category: with transfer (solid) and without transfer (dashed)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    column, sd_column, file, factor, ylabel, log_axis, limit = GHANA_FIGURES[metric]
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.9), sharex=True)
    for ax, category in zip(axes.ravel(), TRANSFER_OUTPUTS):
        try:
            t = category_rows(P, "ghana", category, file)
        except FileNotFoundError:
            ax.set_visible(False)
            continue
        colour = STYLE[category][0]
        for arm, line in [("transfer", "-"), ("plain", "--")]:
            a = t[t["arm"] == arm].sort_values("size")
            x, y, sd = a["size"], factor * a[column], factor * a[sd_column].fillna(0)
            ax.plot(x, y, linestyle=line, color=colour, marker="o" if arm == "transfer" else "s", markersize=2.5,
                    linewidth=LINE, markerfacecolor=colour if arm == "transfer" else "white", markeredgewidth=0.6)
            ax.fill_between(x, (y - sd).clip(lower=1e-3), y + sd, color=colour,
                            alpha=0.15 if arm == "transfer" else 0.07, linewidth=0)
        ax.axhline(limit, color="0.5", linestyle=":", linewidth=0.8)
        ax.set_title(ZAMBIA_OUTPUTS[category]["label"], fontsize=FONT)
        size_axis(ax, sorted(t["size"].unique()), {50, 100, 250, 500, 1000})
        if log_axis:
            ax.set_yscale("log")
            lo, hi = ax.get_ylim()
            ticks = [v for v in (1, 2, 5, 10, 20, 50, 100, 200) if lo <= v <= hi]
            if ticks:
                ax.set_yticks(ticks)
                ax.set_yticklabels([str(v) for v in ticks])
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_ylabel(ylabel, fontsize=FONT)
    handles = [Line2D([], [], linestyle="-", marker="o", color="0.2", markersize=2.5, linewidth=LINE,
                      label="With transfer from Zambia"),
               Line2D([], [], linestyle="--", marker="s", color="0.2", markersize=2.5, linewidth=LINE,
                      markerfacecolor="white", label="Without transfer"),
               Line2D([], [], linestyle=":", color="0.5", linewidth=0.8, label="Threshold, levels of use")]
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=0.8, h_pad=1.0)
    fig.text(0.5, 0.055, "Number of Ghana training scenarios", ha="center", va="center", fontsize=FONT)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=LEGEND)
    save_figure(P, fig, f"fig_ghana_{metric}")


def fig_demonstration(P, year=2050):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    demo = Path(P["results"]) / "demonstration"
    s1 = pd.read_csv(demo / "step1_scenarios.csv", index_col="run_id")
    s2 = pd.read_csv(demo / "step2_lowest_co2.csv", index_col="run_id")
    s3 = pd.read_csv(demo / "step3_details.csv")
    inputs = load_scenarios(P, "Zambia")
    colours = {"correct": "#0072B2", "wrong": "#E69F00", "missed": "#333333", "other": "#C8C8C8", "clews": "#555555"}
    fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(7.2, 6.8))

    # (a) step 1
    kept, truly = s1["kept"] == 1, s1["truly_passing"] == 1
    for name, mask in {"other": ~kept & ~truly, "missed": ~kept & truly, "wrong": kept & ~truly,
                       "correct": kept & truly}.items():
        a.scatter(s1.loc[mask, "pred_cost"] / 1e3, s1.loc[mask, "pred_co2"] / 1e3, s=4 if name == "other" else 3.5,
                  color=colours[name], linewidths=0, alpha=0.6 if name == "other" else 0.9)
    a.axvline(s1["pred_cost"].quantile(BEST_SHARE) / 1e3, color="0.4", linestyle=":", linewidth=0.8)
    a.axhline(s1["pred_co2"].quantile(BEST_SHARE) / 1e3, color="0.4", linestyle=":", linewidth=0.8)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("Predicted total discounted cost (thousands)")
    a.set_ylabel("Predicted CO$_2$ emissions (thousands)")
    a.set_title(f"(a) Step 1, filter: {len(s1):,} scenarios $\\rightarrow$ {kept.sum()} kept "
                f"({100 * (kept & truly).sum() / kept.sum():.0f}% correct)", fontsize=TICK, loc="left")
    a.legend(handles=[Line2D([], [], marker="o", linestyle="", color=colours[k], markersize=4, label=l)
                      for k, l in [("correct", "Kept, correct"), ("wrong", "Kept, wrong"), ("missed", "Missed"),
                                   ("other", "Not kept")]], fontsize=LEGEND, frameon=False, loc="upper left")

    # (b) step 2
    k = s1[kept]
    top = k.index.isin(s2.index)
    b.scatter(k.loc[~top, "true_co2"] / 1e3, k.loc[~top, "pred_co2"] / 1e3, s=7, color=colours["other"], linewidths=0)
    b.scatter(k.loc[top, "true_co2"] / 1e3, k.loc[top, "pred_co2"] / 1e3, s=12, color=colours["correct"],
              edgecolors="white", linewidths=0.4)
    lim = [min(k["true_co2"].min(), k["pred_co2"].min()) / 1e3, max(k["true_co2"].max(), k["pred_co2"].max()) / 1e3]
    b.plot(lim, lim, color="0.4", linewidth=0.8, linestyle="--")
    b.set_xlabel("CLEWs CO$_2$ emissions (thousands)")
    b.set_ylabel("Predicted CO$_2$ emissions (thousands)")
    rho = k["pred_co2"].corr(k["true_co2"], method="spearman")
    b.set_title(f"(b) Step 2, rank: lowest {len(s2)} of {kept.sum()} (rank correlation {rho:.2f})",
                fontsize=TICK, loc="left")
    b.legend(handles=[Line2D([], [], marker="o", linestyle="", color=colours["correct"], markersize=4,
                             label=f"Lowest {len(s2)} by prediction"),
                      Line2D([], [], color="0.4", linestyle="--", label="Prediction = CLEWs")],
             fontsize=LEGEND, frameon=False, loc="upper left")

    # (c) step 3: difference from CLEWs in one year, with the 90% intervals
    y = s3[(s3["YEAR"] == year) & (s3["true"] > ACTIVE_THRESHOLD)].copy()
    for col in ["pred", "lo", "hi"]:
        y[col + "_rel"] = 100 * (y[col] / y["true"] - 1)
    quantities = list(dict.fromkeys(s3["quantity"]))
    for i, q in enumerate(quantities):
        part = y[y["quantity"] == q]
        x = i + np.linspace(-0.3, 0.3, len(part))
        c.vlines(x, part["lo_rel"], part["hi_rel"], color=colours["correct"], alpha=0.35, linewidth=1.2)
        c.scatter(x, part["pred_rel"], s=7, color=colours["correct"], linewidths=0, zorder=3)
    c.axhline(0, color="0.3", linewidth=0.8)
    c.set_xticks(range(len(quantities)))
    c.set_xticklabels([q.replace(" ", "\n") for q in quantities])
    c.set_ylabel(f"Surrogate vs CLEWs in {year} (%)")
    c.set_title(f"(c) Step 3, detail: {len(s2)} scenarios, 90% intervals", fontsize=TICK, loc="left")
    c.legend(handles=[Line2D([], [], marker="o", linestyle="", color=colours["correct"], markersize=4, label="Prediction"),
                      Line2D([], [], color=colours["correct"], alpha=0.35, linewidth=3, label="90% interval")],
             fontsize=LEGEND, frameon=False, loc="upper left")

    # (d) step 4: inputs of the low-emission scenarios, as position within the candidates' range
    s1.index, s2.index = s1.index.astype(str), s2.index.astype(str)
    clews_top = s1[s1["truly_passing"] == 1].nsmallest(len(s2), "true_co2").index
    names = list(INPUT_LABELS)
    x_all = inputs.loc[s1.index, names]
    scaled = (inputs[names] - x_all.min()) / (x_all.max() - x_all.min())
    sets = [("All candidates", s1.index, colours["other"], 0.22),
            (f"Surrogate, lowest {len(s2)}", s2.index, colours["correct"], 0.0),
            (f"CLEWs, lowest {len(s2)}", clews_top, colours["clews"], -0.22)]
    for _, idx, colour, off in sets:
        v = scaled.loc[idx]
        for i, col in enumerate(names):
            q5, q25, q50, q75, q95 = v[col].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            yy = len(names) - 1 - i + off
            d.hlines(yy, q5, q95, color=colour, linewidth=0.8)
            d.add_patch(plt.Rectangle((q25, yy - 0.08), q75 - q25, 0.16, color=colour, alpha=0.9, linewidth=0))
            d.plot(q50, yy, marker="o", markersize=3, color="white", markeredgecolor=colour, markeredgewidth=0.8)
    d.set_yticks(range(len(names)))
    d.set_yticklabels([INPUT_LABELS[n] for n in reversed(names)])
    d.set_xlim(-0.02, 1.02)
    d.set_ylim(-0.5, len(names) - 0.5)
    d.set_xlabel("Position in sampled range (0 = min, 1 = max)")
    d.set_title("(d) Step 4, conditions: inputs of the low-emission scenarios", fontsize=TICK, loc="left")
    d.legend(handles=[Patch(color=col, label=n) for n, _, col, _ in sets], fontsize=LEGEND - 1, frameon=False,
             loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    fig.tight_layout(h_pad=1.6, w_pad=1.4)
    save_figure(P, fig, "fig_demonstration")


# ---- supplementary tables
def group_label(code):
    crop, _, cluster = str(code).partition("_")
    name = GROUP_LABELS.get(crop, crop)
    return f"{name} {cluster}" if cluster else name


def longtable(spec, head, body, caption, label):
    head = "\n".join(head)
    return "\n".join([f"\\begin{{longtable}}{{{spec}}}", f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
                      "\\toprule", head, "\\midrule", "\\endfirsthead", "\\toprule", head, "\\midrule",
                      "\\endhead", "\\bottomrule", "\\endlastfoot", *body, "\\end{longtable}"])


def table_body(blocks, fmt):
    lines = []
    for n, (label, t) in enumerate(blocks):
        if n:
            lines.append("\\midrule")
        groups = list(t.index)
        order = (["ALL"] if "ALL" in groups else []) + sorted([g for g in groups if g != "ALL"], key=group_label)
        for i, g in enumerate(order):
            lines.append(f"{label if i == 0 else ''} & {group_label(g)} & " + " & ".join(fmt(v) for v in t.loc[g]) + " \\\\")
    return lines


def two_arm_header(sizes):
    n, s = len(sizes), " & ".join(f"{x:,}" for x in sizes)
    return [f" & & \\multicolumn{{{n}}}{{c}}{{With transfer}} & \\multicolumn{{{n}}}{{c}}{{Without transfer}} \\\\",
            f"\\cmidrule(lr){{3-{2 + n}}}\\cmidrule(lr){{{3 + n}-{2 + 2 * n}}}",
            f"Output category & Technology group & {s} & {s} \\\\"]


def ghana_table_blocks(P, file, value, factor, sizes):
    """Per transfer category: groups x (arm, size) of one measure."""
    blocks = []
    for category in TRANSFER_OUTPUTS:
        path = Path(P["results"]) / "ghana" / category / file
        if not path.exists():
            continue
        t = pd.read_csv(path)
        t = t[t["size"].isin(sizes)]
        cols = {}
        for arm in ARMS:
            if file == "ranking.csv":
                p = t[t["arm"] == arm].pivot(index="GROUP", columns="size", values=value)
            else:
                p = t.pivot(index="group", columns="size", values=f"{value}_{arm}")
            p = p.reindex(columns=list(sizes))
            for s in sizes:
                cols[(arm, s)] = factor * p[s]
        blocks.append((LATEX_LABEL[category], pd.DataFrame(cols)))
    return blocks


def supplementary_tables(P, zambia_sizes=(250, 1000, 5000), ghana_sizes=(100, 250, 1000), onoff_size=1000):
    out = {}
    fmt2 = lambda v: "--" if pd.isna(v) else f"{v:.2f}"
    fmt1 = lambda v: "--" if pd.isna(v) else f"{v:.1f}"
    cal = CALIBRATION_RUNS

    # zero detection of the Zambia surrogates (groups with zero values only)
    h = pd.read_csv(Path(P["results"]) / "zambia" / "heatmap_numbers.csv")
    h = h[(h["size"] == onoff_size) & (h["share_nonzero_mean"] < 1)]
    for name, cats, label in [("supp_onoff_power.tex", ["ELC001", "CAPACITY", "EMICO2"], "tab:onoff-power"),
                              ("supp_onoff_land.tex", ["LANDUSE", "CROPPROD", "AGRWATER", "FIXEDCOST", "VARCOST"],
                               "tab:onoff-land")]:
        lines = []
        for n, cat in enumerate(cats):
            t = h[h["output"] == cat].sort_values("share_nonzero_mean")
            if n and len(t):
                lines.append("\\midrule")
            for i, (_, r) in enumerate(t.iterrows()):
                share = 100 * r["share_of_output"]
                lines.append(f"{LATEX_LABEL[cat] if i == 0 else ''} & {group_label(r['group'])} & "
                             f"{100 * r['share_nonzero_mean']:.1f} & {fmt2(r['zero_bal_acc_mean'])} & "
                             f"{100 * r['wmape_mean']:.1f} & {'$<$0.1' if share < 0.1 else f'{share:.1f}'} \\\\")
        out[name] = longtable("llrrrr", ["Output category & Technology group & Non-zero (\\%) & Zero detection & "
                                         "WMAPE (\\%) & Share of total (\\%) \\\\"], lines,
                              f"Occurrence of non-zero values and zero detection of the technology groups with zero "
                              f"values, for surrogates trained on {onoff_size:,} scenarios (mean over ten repetitions, "
                              f"1,000 test scenarios). Zero detection is the balanced accuracy of the classifier.", label)

    # pair accuracy per group, Zambia
    blocks = []
    for cat in ZAMBIA_OUTPUTS:
        path = Path(P["results"]) / "zambia" / cat / "ranking.csv"
        if path.exists():
            r = pd.read_csv(path)
            blocks.append((LATEX_LABEL[cat], r[r["size"].isin(zambia_sizes)]
                           .pivot(index="GROUP", columns="size", values="pair_accuracy_clear_mean")
                           .reindex(columns=list(zambia_sizes))))
    sizes = " & ".join(f"{s:,}" for s in zambia_sizes)
    out["supp_zambia_pairs.tex"] = longtable(
        "ll" + "r" * len(zambia_sizes),
        [f" & & \\multicolumn{{{len(zambia_sizes)}}}{{c}}{{Training scenarios}} \\\\",
         f"\\cmidrule(lr){{3-{2 + len(zambia_sizes)}}}", f"Output category & Technology group & {sizes} \\\\"],
        table_body(blocks, fmt2),
        "Clear pair accuracy of the Zambia surrogate per technology group for different numbers of training "
        "scenarios (columns): the share of clearly different pairs of test scenarios that were put in the correct "
        "order, as the median over the years. Mean over ten repetitions, 1,000 test scenarios. ``All groups'' is "
        "the median over the technology groups of the output category, each group weighted by its share of the "
        "total value of the category, as used in the screening criterion.", "tab:zambia-pairs")

    # Ghana: coverage, half-width and pair accuracy per group, with and without transfer
    spec = "ll" + "r" * (2 * len(ghana_sizes))
    common_text = ("with and without transfer from the Zambia surrogate, for selected numbers of Ghana training "
                   "scenarios (columns). Mean over ten repetitions, 400 test scenarios")
    out["supp_ghana_coverage.tex"] = longtable(
        spec, two_arm_header(ghana_sizes),
        table_body(ghana_table_blocks(P, "comparison.csv", f"coverage_nonzero_cal{cal}", 100, ghana_sizes), fmt1),
        f"Coverage (\\%) of the non-zero values by the 90\\% prediction intervals of the Ghana surrogate, {common_text}, "
        f"intervals constructed with {cal} calibration scenarios. ``All groups'' refers to all technology groups of "
        f"the output category together.", "tab:ghana-coverage")
    out["supp_ghana_halfwidth.tex"] = longtable(
        spec, two_arm_header(ghana_sizes),
        table_body(ghana_table_blocks(P, "comparison.csv", f"width_nonzero_cal{cal}", 50, ghana_sizes), fmt1),
        f"Half-width ($\\pm$\\%) of the 90\\% prediction intervals of the Ghana surrogate for the non-zero values, "
        f"{common_text}, intervals constructed with {cal} calibration scenarios.", "tab:ghana-halfwidth")
    out["supp_ghana_pairs.tex"] = longtable(
        spec, two_arm_header(ghana_sizes),
        table_body(ghana_table_blocks(P, "ranking.csv", "pair_accuracy_clear_mean", 1, ghana_sizes), fmt2),
        f"Clear pair accuracy of the Ghana surrogate per technology group, {common_text}. Defined as in "
        f"Table~\\ref{{tab:zambia-pairs}}.", "tab:ghana-pairs")
    for name, text in out.items():
        (Path(P["figures"]) / name).write_text(text, encoding="utf-8")


def figures_and_tables(P, heatmap_size=1000):
    """All figures and supplementary tables; a part whose results are missing is skipped."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(P["figures"]).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": FONT, "axes.labelsize": FONT, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
                         "axes.spines.top": False, "axes.spines.right": False})
    parts = [("learning curves", lambda: fig_learning_curves(P)), ("heatmap", lambda: fig_heatmap(P, heatmap_size)),
             ("uncertainty", lambda: fig_uncertainty(P))] + \
            [(f"Ghana {m}", lambda m=m: fig_ghana(P, m)) for m in GHANA_FIGURES] + \
            [("demonstration", lambda: fig_demonstration(P)), ("tables", lambda: supplementary_tables(P))]
    for name, make in parts:
        try:
            make()
        except FileNotFoundError as error:
            log(f"[skip] {name}: {error.filename} missing")
