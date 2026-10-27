"""
Evaluation: pair accuracy and the number of CLEWs runs needed for each level of use (step 6).

Reads only saved predictions; nothing is trained.
1. Pair accuracy (results/<study>/<CATEGORY>/ranking.csv, ranking_by_year.csv). For every technology
   group and year, all pairs of test scenarios whose CLEWs values differ by more than CLEAR_SHARE of the
   5-95% range are formed; the pair accuracy is the share of these pairs that the surrogate orders
   correctly (a tie in the prediction counts as half). Group-years in which all test scenarios have the
   same value are skipped. Per technology group: the median over its years. Per output category (GROUP =
   ALL): the median over the groups, each weighted by its total CLEWs value over all years and test
   scenarios (as in the WMAPE), so that large groups determine the value of the category.
2. Technology-group numbers of the Zambia surrogates (results/zambia/heatmap_numbers.csv).
3. Smallest number of training runs for the two levels of use (results/<study>/minimum_runs.csv and
   figures/table_minimum_runs_<study>.tex). A sample size counts only if the criteria also hold at every
   larger size. CLEWs time = runs x solution time; detailed use adds the calibration runs.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .definitions import (CALIBRATION_RUNS, CLEAR_SHARE, COUNTRIES, DETAILED_CRITERIA, LATEX_LABEL,
    SCREENING_CRITERIA, TRANSFER_OUTPUTS, ZAMBIA_OUTPUTS)
from .utils import ghana_split, log
from .measures import clear_pair_accuracy, weighted_median
from .transfer import ARMS, rows_of


PAIR_METRICS = ["spearman", "pair_accuracy", "pair_accuracy_clear", "share_clear_pairs", "mae_to_spread"]
STUDIES = {"zambia": list(ZAMBIA_OUTPUTS), "ghana": list(TRANSFER_OUTPUTS)}


def group_year_scores(frame):
    """frame: GROUP, YEAR, true, pred of one surrogate -> one row of ordering measures per group and year."""
    from scipy.stats import kendalltau, spearmanr
    rows = []
    for (group, year), d in frame.groupby(["GROUP", "YEAR"], sort=False):
        t, p = d["true"].to_numpy(), d["pred"].to_numpy()
        if np.ptp(t) <= 1e-9:                      # nothing to order
            continue
        if np.ptp(p) <= 1e-9:                      # predictions all equal: no order
            rho, tau = 0.0, 0.0
        else:
            rho, tau = float(spearmanr(t, p).correlation), float(kendalltau(t, p).correlation)
        spread = np.quantile(t, 0.95) - np.quantile(t, 0.05)
        clear, share = clear_pair_accuracy(t, p, CLEAR_SHARE * (spread if spread > 1e-9 else np.ptp(t)))
        rows.append({"GROUP": group, "YEAR": year, "spearman": rho, "pair_accuracy": (1 + tau) / 2,
                     "pair_accuracy_clear": clear, "share_clear_pairs": share,
                     "mae_to_spread": float(np.abs(p - t).mean() / spread) if spread > 1e-9 else np.nan})
    return pd.DataFrame(rows)


def saved_predictions(P, study, category):
    """(arm / size / seed, test values and predictions) of every saved surrogate of a category."""
    folder = Path(P["results"]) / study / category
    if study == "zambia":
        rows = np.load(folder / "test_rows.npz", allow_pickle=True)
        base = pd.DataFrame({"GROUP": rows["group"], "YEAR": rows["year"], "true": rows["value"]})
        files = sorted((folder / "predictions").glob("n*_s*.npz"))                 # not the cal_* files
    else:
        data = pd.read_pickle(Path(P["data"]) / "transfer" / f"Ghana_{category}.pkl")
        test = rows_of(data, ghana_split(P)["test"])                                # order of ghana_job
        base = pd.DataFrame({"GROUP": test["GROUP"].to_numpy(), "YEAR": test["YEAR"].to_numpy(),
                             "true": test["VALUE"].to_numpy()})
        files = sorted((folder / "predictions").glob("*_n*_s*.npz"))
    for path in files:
        parts = path.stem.split("_")
        tag = {"size": int(parts[-2][1:]), "seed": int(parts[-1][1:])}
        if study == "ghana":
            tag = {"arm": parts[0], **tag}
        pred = np.load(path)["pred"]
        if len(pred) != len(base):
            raise ValueError(f"{path.name}: {len(pred)} predictions but {len(base)} test rows")
        yield tag, base.assign(pred=pred)


def category_pair_accuracy(per_year, group_weight, keys):
    """Per group: median over the years. ALL: median over the groups weighted by their total value."""
    by_group = per_year.groupby(keys + ["GROUP"])[PAIR_METRICS].median().reset_index()
    years = per_year.groupby(keys + ["GROUP"]).size().rename("years_used").reset_index()
    groups = by_group.merge(group_weight, on=keys + ["GROUP"], how="left")
    rows = []
    for key, d in groups.groupby(keys):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)), GROUP="ALL")
        for m in PAIR_METRICS:
            ok = d[m].notna() & (d["weight"] > 0)
            row[m] = weighted_median(d.loc[ok, m], d.loc[ok, "weight"]) if ok.any() else np.nan
        rows.append(row)
    return pd.concat([by_group.merge(years, on=keys + ["GROUP"]), pd.DataFrame(rows)], ignore_index=True)


def ranking(P, study, category):
    keys = ["size"] if study == "zambia" else ["arm", "size"]
    parts, weights = [], []
    for tag, frame in saved_predictions(P, study, category):
        parts.append(group_year_scores(frame).assign(**tag))
        w = frame["true"].abs().groupby(frame["GROUP"]).sum().rename("weight").reset_index()
        weights.append(w.assign(**tag))                          # the group's total over all years
    if not parts:
        log(f"[{study} {category}] no predictions found")
        return
    per_year = pd.concat(parts, ignore_index=True)
    per_seed = category_pair_accuracy(per_year, pd.concat(weights, ignore_index=True), keys + ["seed"])
    folder = Path(P["results"]) / study / category
    table = per_seed.groupby(keys + ["GROUP"])[PAIR_METRICS].agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table = table.reset_index().merge(
        per_seed.groupby(keys + ["GROUP"])["seed"].nunique().rename("seeds").reset_index(), on=keys + ["GROUP"])
    table.to_csv(folder / "ranking.csv", index=False)
    per_year.groupby(keys + ["GROUP", "YEAR"])[PAIR_METRICS].mean().reset_index() \
        .to_csv(folder / "ranking_by_year.csv", index=False)
    print(f"\n[{study} {category}] pair accuracy of the category (mean over seeds):")
    print(table[table["GROUP"] == "ALL"][keys + ["pair_accuracy_clear_mean", "spearman_mean"]].round(3).to_string(index=False))


def heatmap_numbers(P):
    """WMAPE, R2, non-zero share, zero detection, share of the output total and pair accuracy per group."""
    parts = []
    for category, spec in ZAMBIA_OUTPUTS.items():
        if len(spec["groups"]) < 2:                              # one group (TDC): nothing to break down
            continue
        folder = Path(P["results"]) / "zambia" / category
        s = pd.read_csv(folder / "summary.csv")
        s = s[s["group"] != "ALL"].rename(columns={"gate_bal_acc_mean": "zero_bal_acc_mean"})
        cols = ["size", "group", "wmape_mean", "wmape_sd", "r2_mean", "share_nonzero_mean", "zero_bal_acc_mean", "mae_mean"]
        s = s[[c for c in cols if c in s.columns]].copy()
        # the total CLEWs value of a group is proportional to MAE / WMAPE (same number of rows per group)
        total = (s["mae_mean"] / s["wmape_mean"]).where(s["wmape_mean"] > 0)
        s["share_of_output"] = total / total.groupby(s["size"]).transform("sum")
        r = pd.read_csv(folder / "ranking.csv")
        r = r[r["GROUP"] != "ALL"][["size", "GROUP", "pair_accuracy_clear_mean"]].rename(columns={"GROUP": "group"})
        parts.append(s.merge(r, on=["size", "group"], how="left").assign(output=category))
    table = pd.concat(parts, ignore_index=True)
    first = ["output", "size", "group"]
    table[first + [c for c in table.columns if c not in first]] \
        .to_csv(Path(P["results"]) / "zambia" / "heatmap_numbers.csv", index=False)


def smallest_size(table, criteria):
    """Smallest size from which all criteria hold at that and every larger size."""
    ok = np.ones(len(table), bool)
    for column, (op, limit) in criteria.items():
        v = table[column].to_numpy()
        ok &= (v >= limit) if op == ">=" else (v <= limit)
    sizes = table["size"].to_numpy()
    for i in range(len(sizes)):
        if ok[i:].all():
            return int(sizes[i])
    return None


def level_table(P, study, category, arm=None):
    """Category measures (ALL groups) per sample size, with the pair accuracy and the half-width."""
    folder = Path(P["results"]) / study / category
    s, r = pd.read_csv(folder / "summary.csv"), pd.read_csv(folder / "ranking.csv")
    s, r = s[s["group"] == "ALL"], r[r["GROUP"] == "ALL"]
    if arm is not None:
        s, r = s[s["arm"] == arm], r[r["arm"] == arm]
    t = s.merge(r[["size", "pair_accuracy_clear_mean"]], on="size").sort_values("size").reset_index(drop=True)
    t["half_width"] = t[f"width_nonzero_cal{CALIBRATION_RUNS}_mean"] / 2
    return t


def minimum_runs(P, study):
    country = "Zambia" if study == "zambia" else "Ghana"
    minutes = COUNTRIES[country]["solve_time_min"]
    arms = [None] if study == "zambia" else ARMS
    levels = {"screening": SCREENING_CRITERIA, "detailed": {**SCREENING_CRITERIA, **DETAILED_CRITERIA}}
    rows = []
    for category in STUDIES[study]:
        try:
            row = {"output": category, "label": LATEX_LABEL[category]}
            for arm in arms:
                t = level_table(P, study, category, arm)
                prefix = "" if arm is None else f"{arm}_"
                for level, criteria in levels.items():
                    n = smallest_size(t, criteria)
                    runs = None if n is None else n + (CALIBRATION_RUNS if level == "detailed" else 0)
                    row[f"{prefix}{level}_runs"] = n
                    row[f"{prefix}{level}_hours"] = None if runs is None else runs * minutes / 60
                last = t.iloc[-1]                               # largest size: which criterion is missing?
                row.update({f"{prefix}pairs_max": last["pair_accuracy_clear_mean"], f"{prefix}r2_max": last["r2_mean"],
                            f"{prefix}wmape_max": last["wmape_mean"], f"{prefix}half_width_max": last["half_width"]})
            rows.append(row)
        except FileNotFoundError as error:
            log(f"[skip] {category}: {error.filename} missing")
    out = pd.DataFrame(rows)
    out.to_csv(Path(P["results"]) / study / "minimum_runs.csv", index=False)

    runs = lambda x: "not reached" if pd.isna(x) else f"{int(x):,}"
    hours = lambda x: "--" if pd.isna(x) else f"{x:.1f}"
    if study == "zambia":
        head = r"Output category & Training scenarios & CLEWs time (h) & Training scenarios & CLEWs time (h) \\"
        body = [f"{r['label']} & {runs(r['screening_runs'])} & {hours(r['screening_hours'])} & "
                f"{runs(r['detailed_runs'])} & {hours(r['detailed_hours'])} \\\\" for _, r in out.iterrows()]
        caption = (r"Smallest number of training scenarios for which the surrogate meets the example criteria of "
                   r"each level of use (Section~\ref{sec:evaluation}), and the corresponding CLEWs computing time. "
                   r"Screening: a pair accuracy of at least 0.90. Detailed use: the screening criterion together "
                   r"with R\textsuperscript{2} $\geq$ 0.95, WMAPE $\leq$ 5\% and a half-width of the 90\% prediction "
                   r"intervals of at most $\pm$10\%. A sample size is counted only if the criteria also hold for "
                   r"all larger sample sizes. For screening, the CLEWs time includes only the training scenarios, "
                   f"while for detailed use it also includes the {CALIBRATION_RUNS} calibration scenarios needed "
                   f"for the prediction intervals. Times assume {minutes} minutes per scenario on one core.")
        label = "tab:minimum-runs"
    else:
        head = r"Output category & With transfer & Without transfer & With transfer & Without transfer \\"
        cell = lambda r, a, l: runs(r[f"{a}_{l}_runs"]) if pd.isna(r[f"{a}_{l}_runs"]) else \
            f"{runs(r[f'{a}_{l}_runs'])} ({hours(r[f'{a}_{l}_hours'])} h)"
        body = [f"{r['label']} & " + " & ".join(cell(r, a, l) for l in levels for a in arms) + r" \\"
                for _, r in out.iterrows()]
        caption = (r"Smallest number of Ghana training scenarios for which the surrogate meets the example criteria "
                   r"of each level of use, with and without transfer from the Zambia surrogate, and the corresponding "
                   r"CLEWs computing time in brackets. Criteria as in Table~\ref{tab:minimum-runs}; a sample size is "
                   r"counted only if the criteria also hold for all larger sample sizes, up to 1,000 scenarios. For "
                   f"detailed use, the computing time includes the {CALIBRATION_RUNS} calibration scenarios needed "
                   r"for the prediction intervals, while screening needs no calibration scenarios. Times assume "
                   f"{minutes} minutes per Ghana scenario on one core. The 1,000 Zambia scenarios used to train the "
                   r"Zambia surrogate are not included.")
        label = "tab:minimum-runs-ghana"
    lines = [r"\begin{table}[H]", r"\centering", r"\footnotesize", r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Screening} & \multicolumn{2}{c}{Detailed use} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", head, r"\midrule", *body, r"\bottomrule", r"\end{tabular}",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\end{table}"]
    Path(P["figures"]).mkdir(parents=True, exist_ok=True)
    (Path(P["figures"]) / f"table_minimum_runs_{study}.tex").write_text("\n".join(lines), encoding="utf-8")
    show = [c for c in out.columns if c.endswith("_runs") or c.endswith("_hours")]
    print(f"\n[{study}] smallest number of training runs per level of use:")
    print(out[["output"] + show].round(1).to_string(index=False))


def evaluate(P, studies=("zambia", "ghana")):
    for study in studies:
        for category in STUDIES[study]:
            try:
                ranking(P, study, category)
            except FileNotFoundError as error:
                log(f"[skip] {study} {category}: {error.filename} missing")
        if study == "zambia":
            heatmap_numbers(P)
        minimum_runs(P, study)
