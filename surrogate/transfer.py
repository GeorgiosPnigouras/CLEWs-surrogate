"""
Transfer Zambia -> Ghana: Ghana surrogates with and without transfer (step 5).

The two-step Zambia surrogate cannot be transferred as a whole, as its XGBoost gate has no weights that
can serve as the starting point of another model. The transfer surrogate consists of two networks:
  front end  classifier with 11 classes per row: 0 = zero, 1-10 = size bins of the non-zero values per
             technology group and third of the period (fitted on the training runs)
  back end   regressor; inputs = the input row + the 11 class probabilities; target = log(1 + value) - mu_g,
             trained on ALL rows including zeros, mu_g = mean of log(1 + value) of the group over these rows;
             its estimate is used directly as the prediction
With transfer, both networks start from the networks pretrained on the 1,000 Zambia scenarios of the
Zambia surrogate (same seed, same runs); without transfer from random weights. The starting weights are
the only difference. Bin limits, mu_g and input scaling are always computed from the Ghana training runs.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .definitions import ACTIVE_THRESHOLD, ALPHA, COUNTRIES, INPUT_COLUMNS, TRANSFER_OUTPUTS
from .utils import (check_settings, ghana_split, run_parallel, sample_runs_nested, summarise_jobs, write_job,
    zambia_split)
from .measures import interval, mondrian_q, test_measures
from .networks import fit_early_stopping, set_seed
from .zambia import SEEDS


GHANA_SIZES = [50, 100, 150, 200, 250, 300, 500, 750, 1000]
ZAMBIA_PRETRAIN_SIZE = 1000
GHANA_CALIBRATION_SIZES = [100, 250]
NUM_CLASSES, HIDDEN, DROPOUT, LR, WEIGHT_DECAY = 11, 160, 0.10, 8e-4, 1e-5
PRETRAIN = {"batch": 4096, "epochs": 100, "patience": 12}     # training on the Zambia scenarios
FINETUNE = {"batch": 1024, "epochs": 150, "patience": 15}     # training on the Ghana scenarios (both arms)
YEAR_BANDS = ["early", "middle", "late"]
ARMS = ["transfer", "plain"]
_TABLES = {}


def transfer_group_names(category):
    names = list(TRANSFER_OUTPUTS[category]["groups"]["Zambia"])
    if names != list(TRANSFER_OUTPUTS[category]["groups"]["Ghana"]):
        raise ValueError(f"{category}: group names differ between the countries")
    return names


def transfer_table(data_dir, category, country):
    key = (str(data_dir), category, country)
    if key not in _TABLES:
        if len(_TABLES) >= 2:
            _TABLES.clear()
        df = pd.read_pickle(Path(data_dir) / "transfer" / f"{country}_{category}.pkl")
        years = COUNTRIES[country]["years"]
        df["normalized_year"] = (df["YEAR"] - years[0]) / (years[-1] - years[0])
        df["year_band"] = pd.cut(df["normalized_year"], [-1, 1 / 3, 2 / 3, 2], labels=YEAR_BANDS, right=False).astype(str)
        _TABLES[key] = df
    return _TABLES[key]


def rows_of(df, ids):
    return df[df["run_id"].isin(set(ids))].sort_values(["run_id", "GROUP", "YEAR"]).reset_index(drop=True)


def transfer_features(df, groups):
    """9 inputs, position of the year in the period (0-1), one-hot technology group, one-hot third of the period."""
    numeric = df[INPUT_COLUMNS + ["normalized_year"]].to_numpy(dtype="float32")
    tech = (df["GROUP"].to_numpy()[:, None] == np.array(groups)[None, :]).astype("float32")
    band = (df["year_band"].to_numpy()[:, None] == np.array(YEAR_BANDS)[None, :]).astype("float32")
    return np.hstack([numeric, tech, band])


def fit_bin_edges(df):
    quantiles = np.linspace(0, 1, NUM_CLASSES)[1:-1]
    edges = {}
    for key, g in df.groupby(["GROUP", "year_band"]):
        positive = g["VALUE"].to_numpy(float)
        positive = positive[positive > ACTIVE_THRESHOLD]
        edges[key] = np.unique(np.quantile(positive, quantiles)) if len(positive) else np.array([])
    return edges


def assign_bins(df, edges):
    values = df["VALUE"].to_numpy(float)
    labels = np.zeros(len(df), dtype=np.int64)
    for key, idx in df.groupby(["GROUP", "year_band"]).indices.items():
        v = values[idx]
        bins = 1 + np.searchsorted(edges.get(key, np.array([])), v, side="right")
        labels[idx] = np.where(v > ACTIVE_THRESHOLD, np.minimum(bins, NUM_CLASSES - 1), 0)
    return labels


def transfer_networks():
    import torch
    from torch import nn

    class Classifier(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.ReLU(), nn.LayerNorm(HIDDEN), nn.Dropout(DROPOUT),
                                         nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.LayerNorm(HIDDEN), nn.Dropout(DROPOUT))
            self.head = nn.Linear(HIDDEN, NUM_CLASSES)

        def forward(self, x):
            return self.head(self.encoder(x))

    class Regressor(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.ReLU(), nn.LayerNorm(HIDDEN), nn.Dropout(DROPOUT),
                                     nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.LayerNorm(HIDDEN), nn.Dropout(DROPOUT),
                                     nn.Linear(HIDDEN, HIDDEN // 2), nn.ReLU(), nn.Linear(HIDDEN // 2, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return torch, nn, Classifier, Regressor


def scale(train_x, *others):
    mu, sd = train_x.mean(0), train_x.std(0)
    sd[sd < 1e-8] = 1.0
    return [((x - mu) / sd).astype("float32") for x in (train_x,) + others]


def train_transfer_pipeline(train, val, groups, seed, threads, settings, init=None):
    """Front end + back end on one country's training rows (init = pretrained weights, or None).
    Returns a predictor (value and zero / non-zero decision), training information and the weights."""
    torch, nn, Classifier, Regressor = transfer_networks()
    edges = fit_bin_edges(train)
    f_tr, f_va = transfer_features(train, groups), transfer_features(val, groups)
    xc_tr, xc_va = scale(f_tr, f_va)
    set_seed(seed, threads)
    clf = Classifier(f_tr.shape[1])                     # built before loading: the same random stream in both arms
    if init is not None:
        clf.load_state_dict(init["classifier"], strict=True)
    clf_loss, clf_epoch, _ = fit_early_stopping(clf, xc_tr, assign_bins(train, edges), xc_va, assign_bins(val, edges),
                                                nn.CrossEntropyLoss(), LR, WEIGHT_DECAY, settings["batch"],
                                                settings["epochs"], settings["patience"], seed)
    mu_c, sd_c = f_tr.mean(0), f_tr.std(0)
    sd_c[sd_c < 1e-8] = 1.0

    def probs(df):
        x = torch.from_numpy(((transfer_features(df, groups) - mu_c) / sd_c).astype("float32"))
        with torch.no_grad():
            return torch.softmax(clf(x), dim=1).numpy().astype("float32")

    means = np.log1p(train["VALUE"]).groupby(train["GROUP"]).mean().to_dict()     # mu_g over all training rows
    r_tr, r_va = np.hstack([f_tr, probs(train)]), np.hstack([f_va, probs(val)])
    xr_tr, xr_va = scale(r_tr, r_va)
    mu_r, sd_r = r_tr.mean(0), r_tr.std(0)
    sd_r[sd_r < 1e-8] = 1.0
    target = lambda df: (np.log1p(df["VALUE"]) - df["GROUP"].map(means)).to_numpy(dtype="float32")
    set_seed(seed, threads)
    reg = Regressor(r_tr.shape[1])
    if init is not None:
        reg.load_state_dict(init["regressor"], strict=True)
    reg_loss, reg_epoch, _ = fit_early_stopping(reg, xr_tr, target(train), xr_va, target(val), nn.MSELoss(), LR,
                                                WEIGHT_DECAY, settings["batch"], settings["epochs"],
                                                settings["patience"], seed)

    def predict(df):
        p = probs(df)
        x = torch.from_numpy(((np.hstack([transfer_features(df, groups), p]) - mu_r) / sd_r).astype("float32"))
        with torch.no_grad():
            rel = reg(x).numpy()
        return np.maximum(np.expm1(rel + df["GROUP"].map(means).to_numpy()), 0.0), (p.argmax(axis=1) != 0).astype(np.uint8)

    info = {"clf_val_loss": clf_loss, "clf_best_epoch": clf_epoch, "reg_val_loss": reg_loss, "reg_best_epoch": reg_epoch}
    states = {"classifier": {k: v.detach().clone() for k, v in clf.state_dict().items()},
              "regressor": {k: v.detach().clone() for k, v in reg.state_dict().items()}}
    return predict, info, states


def pretrain_job(job):
    """Networks pretrained on the 1,000 Zambia scenarios of the Zambia surrogate (same seed, same runs)."""
    os.environ["OMP_NUM_THREADS"] = str(job["threads"])
    import torch
    start = time.time()
    zambia = transfer_table(job["data_dir"], job["category"], "Zambia")
    train_ids, val_ids = sample_runs_nested(job["zambia_pool"], job["seed"], job["zambia_size"])
    _, info, states = train_transfer_pipeline(rows_of(zambia, train_ids), rows_of(zambia, val_ids), job["groups"],
                                              job["seed"], job["threads"], job["pretrain"])
    path = Path(job["file"])
    torch.save({**states, "info": info}, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)
    return f"Zambia s{job['seed']}: epochs {info['clf_best_epoch']} / {info['reg_best_epoch']} " \
           f"({(time.time() - start) / 60:.1f} min)"


def ghana_job(job):
    """One Ghana surrogate (arm, sample size, seed): training, intervals from the calibration runs, test."""
    os.environ["OMP_NUM_THREADS"] = str(job["threads"])
    import torch
    start = time.time()
    out, arm, size, seed, groups = Path(job["out_dir"]), job["arm"], job["size"], job["seed"], job["groups"]
    ghana = transfer_table(job["data_dir"], job["category"], "Ghana")
    train_ids, val_ids = sample_runs_nested(job["ghana_pool"], seed, size)
    init = torch.load(job["pretrained"], map_location="cpu", weights_only=False) if arm == "transfer" else None
    t0 = time.time()
    predict, info, _ = train_transfer_pipeline(rows_of(ghana, train_ids), rows_of(ghana, val_ids), groups, seed,
                                               job["threads"], job["finetune"], init)
    train_minutes = (time.time() - t0) / 60
    index = {g: i for i, g in enumerate(groups)}

    cal = rows_of(ghana, job["calibration"])
    pred_cal, on_cal = predict(cal)
    rank = {r: i for i, r in enumerate(job["calibration"])}
    q = mondrian_q(cal["VALUE"].to_numpy(), pred_cal, cal["GROUP"].map(index).to_numpy(), on_cal,
                   cal["run_id"].map(rank).to_numpy(), len(groups), job["calibration_sizes"])

    test = rows_of(ghana, job["test"])
    t0 = time.time()
    pred, on = predict(test)
    predict_seconds = time.time() - t0
    true, g_te = test["VALUE"].to_numpy(), test["GROUP"].map(index).to_numpy()
    bounds = {n: interval(pred, qn[g_te, on]) for n, qn in q.items()}
    rows = test_measures(true, pred, on, g_te, groups, bounds, {"arm": arm, "size": size, "seed": seed})
    for row in rows:
        row.update(info, train_minutes=train_minutes, predict_seconds=predict_seconds,
                   total_minutes=(time.time() - start) / 60)
    lo, hi = bounds[max(bounds)]
    np.savez_compressed(out / "predictions" / f"{arm}_n{size}_s{seed}.npz", pred=pred.astype(np.float32),
                        lo=lo.astype(np.float32), hi=hi.astype(np.float32), on=on,
                        cal_pred=pred_cal.astype(np.float32), cal_on=on_cal)
    write_job(out / "jobs" / f"{arm}_n{size}_s{seed}.csv", rows)
    a = rows[-1]
    return (f"{arm:8s} n{size:<5d} s{seed:<2d} WMAPE={a['wmape']:.4f} R2={a['r2']:.4f} "
            f"nonzero coverage={a[f'coverage_nonzero_cal{max(bounds)}']:.3f} ({a['total_minutes']:.1f} min)")


def transfer(P, categories=None, workers=12, threads=2, quick=False):
    """Ghana surrogates with and without transfer, for every category, sample size and seed."""
    zambia_pool = zambia_split(P)[0]
    split = ghana_split(P)
    sizes, seeds, zambia_size = ([50, 100], [1, 2], 300) if quick else (GHANA_SIZES, SEEDS, ZAMBIA_PRETRAIN_SIZE)
    pretrain, finetune = (dict(PRETRAIN, epochs=3), dict(FINETUNE, epochs=3)) if quick else (PRETRAIN, FINETUNE)
    for category in categories or list(TRANSFER_OUTPUTS):
        groups = transfer_group_names(category)
        for country, needed in [("Zambia", zambia_pool), ("Ghana", split["test"] + split["calibration"] + split["pool"])]:
            table = pd.read_pickle(Path(P["data"]) / "transfer" / f"{country}_{category}.pkl")
            if sorted(table["GROUP"].unique()) != sorted(groups) or set(needed) - set(table["run_id"]):
                raise SystemExit(f"{category}: the {country} table does not match the study definitions or misses "
                                 f"runs; build the data tables again")
        out = Path(P["results"]) / "ghana" / category
        check_settings(out, {"category": category, "groups": groups, "sizes": sizes, "seeds": seeds,
                             "zambia_size": zambia_size, "pretrain": pretrain, "finetune": finetune, "hidden": HIDDEN,
                             "dropout": DROPOUT, "lr": LR, "weight_decay": WEIGHT_DECAY, "num_classes": NUM_CLASSES,
                             "ghana_test": len(split["test"]), "ghana_calibration": len(split["calibration"]),
                             "calibration_sizes": GHANA_CALIBRATION_SIZES, "alpha": ALPHA,
                             "technologies": {c: TRANSFER_OUTPUTS[category]["groups"][c] for c in COUNTRIES}})
        for folder in ["jobs", "models", "predictions"]:
            (out / folder).mkdir(parents=True, exist_ok=True)
        common = {"data_dir": str(P["data"]), "out_dir": str(out), "category": category, "groups": groups,
                  "threads": threads}
        model = lambda s: out / "models" / f"zambia_s{s}.pt"
        pre = [{**common, "seed": s, "zambia_pool": zambia_pool, "zambia_size": zambia_size, "pretrain": pretrain,
                "file": str(model(s))} for s in seeds if not model(s).exists()]
        run_parallel(pretrain_job, pre, f"{category} Zambia pretraining", workers, lambda j: f"s{j['seed']}")
        jobs = [{**common, "arm": arm, "size": n, "seed": s, "ghana_pool": split["pool"],
                 "calibration": split["calibration"], "test": split["test"], "finetune": finetune,
                 "calibration_sizes": GHANA_CALIBRATION_SIZES, "pretrained": str(model(s))}
                for n in sizes for s in seeds for arm in ARMS
                if not (out / "jobs" / f"{arm}_n{n}_s{s}.csv").exists()]
        failed = run_parallel(ghana_job, jobs, f"{category} Ghana", workers,
                              lambda j: f"{j['arm']} n{j['size']} s{j['seed']}")
        if failed:
            (out / "failed_jobs.txt").write_text("\n".join(failed))
        if any((out / "jobs").glob("*.csv")):
            transfer_comparison(out)


def transfer_comparison(out):
    """summary.csv and comparison.csv (transfer vs plain per size and group, seeds in which transfer wins)."""
    by_seed, _ = summarise_jobs(out, ["arm", "size"])
    key = ["size", "group", "seed"]
    t, p = by_seed[by_seed["arm"] == "transfer"].set_index(key), by_seed[by_seed["arm"] == "plain"].set_index(key)
    n = max(GHANA_CALIBRATION_SIZES)
    cols = ["wmape", "r2", f"coverage_nonzero_cal{n}", f"width_nonzero_cal{n}"]
    j = t[cols].join(p[cols], lsuffix="_transfer", rsuffix="_plain", how="inner").reset_index()
    j["transfer_better"] = (j["wmape_transfer"] < j["wmape_plain"]).astype(int)
    comparison = j.groupby(["size", "group"]).agg(
        **{f"{c}_{a}": (f"{c}_{a}", "mean") for c in cols for a in ARMS},
        transfer_wins=("transfer_better", "sum"), seeds=("seed", "nunique")).reset_index()
    comparison["wmape_difference"] = comparison["wmape_transfer"] - comparison["wmape_plain"]
    comparison.to_csv(Path(out) / "comparison.csv", index=False)
    print(comparison[comparison["group"] == "ALL"][["size", "wmape_transfer", "wmape_plain", "transfer_wins"]]
          .round(4).to_string(index=False))
