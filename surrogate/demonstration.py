"""
Demonstration of scenario exploration with the 1,000-run Zambia surrogate (step 7).

The Zambia surrogates trained on 1,000 scenarios (first repetition) are applied to 6,500 scenarios that
were not used for their training or calibration. These play the role of new scenarios that the modeller
has not solved with CLEWs; their CLEWs results are used only to check each step.
  Step 1, filter  The surrogate predicts for every candidate, over the whole period, the total discounted
                  cost, the CO2 emissions and the irrigation water per unit of crop production. A scenario
                  is kept if it is among the 30% lowest PREDICTED values in all three. The same limits are
                  applied to the CLEWs values to see which scenarios truly pass.
  Step 2, rank    The kept scenarios are ranked by predicted CO2 emissions and the 30 lowest are selected.
  Step 3, detail  For the selected scenarios, five quantities are predicted with 90% intervals. Each
                  quantity is the sum of its technology groups; its intervals are built as in the paper, on
                  the totals of the 250 calibration scenarios, separately for the cases the gate predicts as
                  used and as not used (a total counts as used if the gate predicts any of its groups as
                  non-zero).
  Step 4          The inputs of the selected scenarios are compared in the demonstration figure.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .definitions import ACTIVE_THRESHOLD, CALIBRATION_RUNS
from .utils import sample_runs_nested, zambia_split
from .scenarios import load_scenarios
from .measures import interval, mondrian_q, wmape
from .zambia import load_zambia_model, predict_zambia


DEMO_SIZE, DEMO_SEED = 1000, 1       # the surrogate used
N_CANDIDATES, CANDIDATE_SEED = 6500, 2026
BEST_SHARE = 0.30                    # step 1: the 30% lowest in all three outputs
TOP_N = 30                           # step 2: the 30 lowest emissions
QUANTITIES = {                       # step 3: name -> (output category, group prefixes)
    "Maize land": ("LANDUSE", ["MAI_"]),
    "Cassava land": ("LANDUSE", ["CAS_"]),
    "Soy production": ("CROPPROD", ["SOY_"]),
    "Solar power": ("ELC001", ["SOL"]),
    "Hydropower": ("ELC001", ["HYD"]),
}


def demonstration_candidates(P, size, seed):
    """N_CANDIDATES scenarios not used for training (the size-run sample) or calibration (first 250)."""
    train_pool, calibration, _ = zambia_split(P)
    tr, va = sample_runs_nested(train_pool, seed, size)
    used = set(tr) | set(va) | set(calibration[:CALIBRATION_RUNS])
    free = np.array(sorted(set(load_scenarios(P, "Zambia").index) - used))
    return np.sort(np.random.default_rng(CANDIDATE_SEED).choice(free, N_CANDIDATES, replace=False))


def predicted_totals(model, scenarios, chunk=500):
    """Predicted sum over all technology groups and years, per scenario."""
    parts = [predict_zambia(model, scenarios.iloc[a:a + chunk]).groupby("run_id")["pred"].sum()
             for a in range(0, len(scenarios), chunk)]
    return pd.concat(parts).reindex(scenarios.index)


def step1_outputs(totals):
    return pd.DataFrame({"cost": totals["TDC"], "co2": totals["EMICO2"], "water": totals["AGRWATER"] / totals["CROPPROD"]})


def demonstration_details(P, scenarios, top, size, seed):
    """Step 3: values with 90% intervals for the selected scenarios, every year."""
    cal = zambia_split(P)[1][:CALIBRATION_RUNS]
    runs = list(cal) + list(top)
    tables, rows = [], []
    for category in sorted({c for c, _ in QUANTITIES.values()}):
        pred = predict_zambia(load_zambia_model(P, category, size, seed), scenarios.loc[runs])
        true = pd.read_pickle(Path(P["data"]) / "zambia" / f"{category}_long.pkl")
        true = true[true["run_id"].isin(runs)][["run_id", "GROUP", "YEAR", "VALUE"]]
        both = pred.merge(true, on=["run_id", "GROUP", "YEAR"], how="left")
        both["true"] = both["VALUE"].fillna(0.0)
        for name, (c, prefixes) in QUANTITIES.items():
            if c != category:
                continue
            sel = both[both["GROUP"].str.startswith(tuple(prefixes))]
            q_tab = sel.groupby(["run_id", "YEAR"]).agg(pred=("pred", "sum"), true=("true", "sum"),
                                                        on=("on", "max")).reset_index()
            k = q_tab[q_tab["run_id"].isin(cal)]
            q = mondrian_q(k["true"].to_numpy(), k["pred"].to_numpy(), np.zeros(len(k), int), k["on"].to_numpy(),
                           np.zeros(len(k), int), 1, [CALIBRATION_RUNS])[CALIBRATION_RUNS][0]
            t = q_tab[q_tab["run_id"].isin(top)].copy()
            t["lo"], t["hi"] = interval(t["pred"], q[t["on"].to_numpy()])
            t.insert(0, "quantity", name)
            tables.append(t)
            nz = t["true"] > ACTIVE_THRESHOLD
            inside = (t["true"] >= t["lo"] - 1e-9) & (t["true"] <= t["hi"] + 1e-9)
            rows.append({"quantity": name, "WMAPE %": round(100 * wmape(t["true"].to_numpy(), t["pred"].to_numpy()), 1),
                         "coverage %": round(100 * inside[nz].mean()),
                         "half-width +/-%": round(100 * (t["hi"] - t["lo"])[nz].sum() / t["true"][nz].sum() / 2, 1)})
    pd.concat(tables).to_csv(Path(P["results"]) / "demonstration" / "step3_details.csv", index=False)
    print(f"\nStep 3 (detail): the {len(top)} selected scenarios, every year:")
    print(pd.DataFrame(rows).set_index("quantity").loc[list(QUANTITIES)].to_string())


def demonstration(P, size=DEMO_SIZE, seed=DEMO_SEED):
    """Steps 1-3 (results/demonstration/step1_scenarios.csv, step2_lowest_co2.csv, step3_details.csv)."""
    out = Path(P["results"]) / "demonstration"
    out.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenarios(P, "Zambia")
    cand = demonstration_candidates(P, size, seed)
    categories = ["TDC", "EMICO2", "AGRWATER", "CROPPROD"]
    p = step1_outputs({c: predicted_totals(load_zambia_model(P, c, size, seed), scenarios.loc[cand]) for c in categories})
    t = step1_outputs({c: pd.read_pickle(Path(P["data"]) / "zambia" / f"{c}_long.pkl").groupby("run_id")["VALUE"].sum()
                       for c in categories}).loc[cand]

    # step 1: filter with limits taken from the predictions
    limits = p.quantile(BEST_SHARE)
    kept, truly = (p <= limits).all(axis=1), (t <= limits).all(axis=1)
    s1 = p.add_prefix("pred_").join(t.add_prefix("true_"))
    s1["kept"], s1["truly_passing"] = kept.astype(int), truly.astype(int)
    s1.to_csv(out / "step1_scenarios.csv", index_label="run_id")
    both = int((kept & truly).sum())
    print(f"Step 1 (filter): {len(cand):,} candidates, {int(kept.sum())} kept by the surrogate, {int(truly.sum())} "
          f"truly passing, {both} in both ({100 * both / kept.sum():.0f}% of the kept are right, "
          f"{100 * both / truly.sum():.0f}% of the truly passing are found)")

    # step 2: the TOP_N lowest predicted emissions among the kept scenarios
    k = pd.DataFrame({"pred_co2": p["co2"][kept], "true_co2": t["co2"][kept], "truly_passing": truly[kept]})
    k["true_rank"] = k["true_co2"].rank(method="first")
    top = k.nsmallest(TOP_N, "pred_co2")
    top.to_csv(out / "step2_lowest_co2.csv", index_label="run_id")
    print(f"Step 2 (rank): of the {TOP_N} lowest predicted emissions, {int(top['truly_passing'].sum())} truly pass "
          f"step 1 and {int((top['true_rank'] <= TOP_N).sum())} are among the {TOP_N} lowest according to CLEWs "
          f"(worst true rank {int(top['true_rank'].max())}); rank correlation of the emissions of the kept "
          f"scenarios {k['pred_co2'].corr(k['true_co2'], method='spearman'):.2f}")

    # step 3: detailed values with intervals
    demonstration_details(P, scenarios, list(top.index), size, seed)
