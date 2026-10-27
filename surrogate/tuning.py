"""
Hyperparameter search of the Zambia regressor (step 3).
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import log, run_parallel, sample_runs, zambia_development_runs
from .measures import wmape
from .networks import set_seed
from .zambia import (MAX_EPOCHS, cache_features, fit_gate, fit_regressor, gate_decision, load_cache,
    make_cache, regressor_predict, rows_mask)


TUNING_CATEGORIES = ["EMICO2", "LANDUSE", "CAPACITY"]          # hard, moderate, easy
TUNING_SIZES, TUNING_STAGE1_SIZES = [100, 1000, 5000], [100, 1000]
N_CONFIGS, CONFIG_SEED, TOP_K, STAGE2_SEEDS = 30, 42, 3, [1, 2, 3]
SEARCH_SPACE = {"n_layers": [2, 3, 4], "units": [64, 128, 256, 512], "activation": ["relu", "gelu", "silu"],
                "dropout": (0.0, 0.25), "lr": (1e-4, 5e-3), "weight_decay": (1e-7, 1e-3),
                "batch_size": [256, 512, 1024, 2048]}


def tuning_configs():
    """The 30 random configurations of the search (uniform dropout, log-uniform learning rate and weight decay)."""
    rng, s = np.random.default_rng(CONFIG_SEED), SEARCH_SPACE
    configs = []
    for i in range(1, N_CONFIGS + 1):
        n_layers = int(rng.choice(s["n_layers"]))
        configs.append({"config": f"c{i:02d}", "hidden": [int(rng.choice(s["units"])) for _ in range(n_layers)],
                        "activation": str(rng.choice(s["activation"])), "dropout": float(rng.uniform(*s["dropout"])),
                        "lr": float(10 ** rng.uniform(*np.log10(s["lr"]))),
                        "weight_decay": float(10 ** rng.uniform(*np.log10(s["weight_decay"]))),
                        "batch_size": int(rng.choice(s["batch_size"]))})
    return configs


def n_parameters(hidden, n_inputs):
    sizes = [n_inputs] + list(hidden)
    return sum(a * b + b + 2 * b for a, b in zip(sizes[:-1], sizes[1:])) + sizes[-1] + 1    # linear + LayerNorm


def tuning_rows(c, seed, size):
    return [rows_mask(c, ids) for ids in sample_runs(c["runs"].tolist(), seed, size)]


def tuning_gate_job(job):
    """Gate of one tuning cell (fixed settings, shared by all configurations): saves its validation decisions."""
    os.environ["OMP_NUM_THREADS"] = str(job["threads"])
    c = load_cache(job["cache"])
    tr, va = tuning_rows(c, job["seed"], job["size"])
    gate, constant = fit_gate(c, tr, job["seed"], job["threads"])
    np.save(job["file"], gate_decision(gate, constant, cache_features(c, va)))
    return f"gate {job['category']} n{job['size']} s{job['seed']}"


def tuning_job(job):
    """Train one configuration in one cell; validation WMAPE on the original scale, gate included."""
    os.environ["OMP_NUM_THREADS"] = str(job["threads"])
    start = time.time()
    set_seed(job["seed"], job["threads"])
    c = load_cache(job["cache"])
    tr, va = tuning_rows(c, job["seed"], job["size"])
    reg = fit_regressor(c, tr, va, job["cfg"], job["seed"], job["max_epochs"])
    pred = regressor_predict(reg, cache_features(c, va), c["group"][va], np.load(job["gate_file"]))
    result = {"category": job["category"], "size": job["size"], "config": job["cfg"]["config"], "seed": job["seed"],
              "val_wmape": wmape(c["value"][va], pred), "val_mse": reg["val_mse"], "best_epoch": reg["best_epoch"],
              "batch_used": reg["batch"], "train_rows_nonzero": reg["rows_nonzero"],
              "minutes": (time.time() - start) / 60}
    Path(job["file"]).with_suffix(".tmp").write_text(json.dumps(result))
    Path(job["file"]).with_suffix(".tmp").replace(job["file"])
    return f"{job['category']:8s} n{job['size']:<5d} {job['cfg']['config']} s{job['seed']} WMAPE={result['val_wmape']:.4f}"


def tune(P, workers=12, threads=2, quick=False):
    """
    Hyperparameter search of the Zambia regressor on validation data only (samples from the 7,000
    development runs, 80% training / 20% validation; the test runs are never loaded).
      1. 30 random configurations, seed 1, on the three categories with 100 and 1,000 runs
         (5,000 runs is too slow for 30 configurations).
      2. Rank them in every cell (category x size) by validation WMAPE, keep the TOP_K best average ranks.
      3. Train these with seeds 1-3 and 100, 1,000 and 5,000 runs.
      4. A configuration is "as good as the best" in a cell if its mean WMAPE <= best mean + best SD.
         Among those as good as the best in ALL cells, the one with the fewest parameters is chosen;
         if none qualifies, the one with the best average rank. (In the study none qualified.)
    Writes tuning/chosen_config.json and the result tables.
    """
    out = Path(P["tuning"])
    sizes, stage1, n_configs, max_epochs = (([100, 300], [100], 4, 5) if quick else
                                            (TUNING_SIZES, TUNING_STAGE1_SIZES, N_CONFIGS, MAX_EPOCHS))
    for folder in ["jobs", "gates", "cache"]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    pool = zambia_development_runs(P)
    caches = {}
    for cat in TUNING_CATEGORIES:
        caches[cat] = out / "cache" / f"{cat}.npz"
        groups = sorted(pd.read_pickle(Path(P["data"]) / "zambia" / f"{cat}_long.pkl")["GROUP"].unique())
        make_cache(Path(P["data"]) / "zambia" / f"{cat}_long.pkl", pool, caches[cat], groups)
    configs = tuning_configs()[:n_configs]
    by_name = {cfg["config"]: cfg for cfg in configs}
    pd.DataFrame([{**cfg, "hidden": "-".join(map(str, cfg["hidden"]))} for cfg in configs]).to_csv(out / "configs.csv", index=False)
    gate_file = lambda cat, n, s: out / "gates" / f"{cat}_n{n}_s{s}.npy"
    job_file = lambda cat, n, name, s: out / "jobs" / f"{cat}_n{n}_{name}_s{s}.json"
    base = {"threads": threads, "max_epochs": max_epochs}

    gates = [{**base, "category": cat, "size": n, "seed": s, "cache": str(caches[cat]), "file": str(gate_file(cat, n, s))}
             for cat in TUNING_CATEGORIES for n in sizes for s in STAGE2_SEEDS if not gate_file(cat, n, s).exists()]
    run_parallel(tuning_gate_job, gates, "tuning gates", workers)

    def jobs(names, seeds, cell_sizes):
        return [{**base, "category": cat, "size": n, "seed": s, "cfg": by_name[name], "cache": str(caches[cat]),
                 "gate_file": str(gate_file(cat, n, s)), "file": str(job_file(cat, n, name, s))}
                for cat in TUNING_CATEGORIES for n in cell_sizes for name in names for s in seeds]

    def collect(js):
        return pd.DataFrame([json.loads(Path(j["file"]).read_text()) for j in js])

    jobs1 = jobs(list(by_name), [1], stage1)
    run_parallel(tuning_job, [j for j in jobs1 if not Path(j["file"]).exists()], "tuning stage 1", workers)
    s1 = collect(jobs1)
    s1["rank"] = s1.groupby(["category", "size"])["val_wmape"].rank(method="average")
    s1.to_csv(out / "stage1_results.csv", index=False)
    top = s1.groupby("config")["rank"].mean().nsmallest(TOP_K).index.tolist()
    log(f"[tuning] best {TOP_K} after stage 1: {top}")

    jobs2 = jobs(top, STAGE2_SEEDS, sizes)
    run_parallel(tuning_job, [j for j in jobs2 if not Path(j["file"]).exists()], "tuning stage 2", workers)
    s2 = collect(jobs2)
    s2.to_csv(out / "stage2_results.csv", index=False)

    cell = s2.groupby(["category", "size", "config"], as_index=False)["val_wmape"].agg(["mean", "std"])
    cell["rank"] = cell.groupby(["category", "size"])["mean"].rank(method="average")
    best = cell.loc[cell.groupby(["category", "size"])["mean"].idxmin(), ["category", "size", "mean", "std"]]
    cell = cell.merge(best.rename(columns={"mean": "best_mean", "std": "best_sd"}), on=["category", "size"])
    cell["as_good_as_best"] = cell["mean"] <= cell["best_mean"] + cell["best_sd"].fillna(0)
    n_inputs = max(10 + len(load_cache(caches[cat])["groups"]) for cat in TUNING_CATEGORIES)
    selection = cell.groupby("config").agg(mean_rank=("rank", "mean"),
                                           good_in_all_cells=("as_good_as_best", "all")).reset_index()
    selection["n_parameters"] = [n_parameters(by_name[c]["hidden"], n_inputs) for c in selection["config"]]
    qualified = selection[selection["good_in_all_cells"]]
    if len(qualified):
        chosen, rule = qualified.sort_values(["n_parameters", "mean_rank"]).iloc[0]["config"], \
            "as good as the best in all cells, fewest parameters"
    else:
        chosen, rule = selection.sort_values("mean_rank").iloc[0]["config"], "best average rank"
    selection["chosen"] = selection["config"].eq(chosen)
    wide = cell.pivot_table(index="config", columns=["category", "size"], values=["mean", "std"])
    wide.columns = [f"{cat}_n{n}_{stat}" for stat, cat, n in wide.columns]
    selection.merge(wide.reset_index(), on="config").sort_values("mean_rank").to_csv(out / "selection.csv", index=False)
    (out / "chosen_config.json").write_text(json.dumps({**by_name[chosen], "rule": rule}, indent=2))
    log(f"[tuning] chosen {chosen} ({rule}): {by_name[chosen]}")
