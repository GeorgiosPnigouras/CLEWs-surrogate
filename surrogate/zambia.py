"""
Zambia surrogate: gate and regressor, training of all sample sizes and repetitions, saved models (step 4).

The surrogate has two steps. A gate (XGBoost classifier with fixed settings, not tuned) decides whether
a value is zero. A regressor (neural network) estimates the value, trained on the non-zero rows with the
target log(1 + value) - mu_g, where mu_g is the mean of log(1 + value) of the technology group over these
rows. Prediction = regressor estimate x gate decision. One row per scenario, group and year; inputs =
the 9 scenario inputs, the year scaled to 0-1 and the technology group (one-hot).
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .definitions import ACTIVE_THRESHOLD, ALPHA, ZAMBIA_OUTPUTS
from .utils import check_settings, run_parallel, sample_runs_nested, summarise_jobs, write_job, zambia_split
from .measures import interval, mondrian_q, test_measures
from .networks import fit_early_stopping, set_seed


GATE_SETTINGS = dict(objective="binary:logistic", eval_metric="logloss", n_estimators=250, max_depth=4,
                     learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, tree_method="hist")
ZAMBIA_SIZES = [100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000]
SEEDS = list(range(1, 11))
MAX_EPOCHS, PATIENCE, MIN_STEPS = 300, 20, 10   # the batch is capped so every epoch has >= MIN_STEPS updates
ZAMBIA_CALIBRATION_SIZES = [100, 250, 500]      # first n runs of the calibration order (the paper uses 250)

_CACHE = {}


def make_cache(data_file, runs, path, groups):
    """Compact arrays of the given runs of a data table, saved once (path) and shared by the jobs."""
    if Path(path).exists():
        return
    data = pd.read_pickle(data_file)
    missing = sorted(set(runs) - set(data["run_id"]))
    if missing:
        raise ValueError(f"{data_file}: {len(missing)} runs missing, e.g. {missing[:5]}")
    if sorted(data["GROUP"].unique()) != sorted(groups):
        raise ValueError(f"{data_file}: the groups differ from the study definitions; build the data tables again")
    data = data[data["run_id"].isin(set(runs))].sort_values(["run_id", "GROUP", "YEAR"])
    run_ids = np.array(sorted(set(runs)))
    inputs = [c for c in data.columns if c not in ["run_id", "GROUP", "YEAR", "VALUE", "active"]]
    years = data["YEAR"].to_numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, run=np.searchsorted(run_ids, data["run_id"].to_numpy()).astype(np.int32),
             group=pd.Categorical(data["GROUP"], categories=groups).codes.astype(np.int16),
             year_int=years.astype(np.int16),
             year=((years - years.min()) / max(1, years.max() - years.min())).astype(np.float32),
             x=data[inputs].to_numpy(dtype=np.float32), value=data["VALUE"].to_numpy(dtype=np.float64),
             active=(data["VALUE"].to_numpy() > ACTIVE_THRESHOLD).astype(np.uint8),
             runs=run_ids, groups=np.array(groups), inputs=np.array(inputs))


def load_cache(path):
    if _CACHE.get("path") != str(path):                 # one table per worker process
        _CACHE.clear()
        with np.load(path, allow_pickle=False) as f:
            _CACHE.update({k: f[k] for k in f.files}, path=str(path))
    return _CACHE


def rows_mask(c, ids):
    lookup = np.zeros(len(c["runs"]), bool)
    lookup[np.searchsorted(c["runs"], ids)] = True
    return lookup[c["run"]]


def zambia_features(x, year_scaled, group_idx, n_groups):
    """Input rows of the Zambia surrogate: 9 inputs, scaled year, one-hot technology group."""
    return np.hstack([x, year_scaled[:, None], np.eye(n_groups, dtype=np.float32)[group_idx]])


def cache_features(c, mask):
    return zambia_features(c["x"][mask], c["year"][mask], c["group"][mask], len(c["groups"]))


def fit_gate(c, tr, seed, threads):
    """XGBoost gate on the training rows; None (with the constant label) if the training values are all zero
    or all non-zero."""
    labels = c["active"][tr]
    if labels.min() == labels.max():
        return None, int(labels[0])
    from xgboost import XGBClassifier
    gate = XGBClassifier(**GATE_SETTINGS, n_jobs=threads, random_state=seed)
    gate.fit(cache_features(c, tr), labels)
    return gate, None


def gate_decision(gate, constant, features):
    return gate.predict(features).astype(np.uint8) if gate is not None else np.full(len(features), constant, np.uint8)


def fit_regressor(c, tr, va, cfg, seed, max_epochs):
    """Regressor of the Zambia surrogate on the non-zero training rows (see the description above)."""
    import torch
    from torch import nn
    n_groups = len(c["groups"])
    tr_act, va_act = tr & (c["active"] == 1), va & (c["active"] == 1)
    log_tr = np.log1p(c["value"][tr_act])
    sums = np.bincount(c["group"][tr_act], weights=log_tr, minlength=n_groups)
    counts = np.bincount(c["group"][tr_act], minlength=n_groups)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)      # 0 for groups never non-zero
    x_raw = cache_features(c, tr_act)
    mu, sd = x_raw.mean(0), x_raw.std(0)
    sd[sd < 1e-8] = 1.0
    scaled = lambda m: torch.from_numpy(((cache_features(c, m) - mu) / sd).astype(np.float32))
    target = lambda m: torch.from_numpy((np.log1p(c["value"][m]) - means[c["group"][m]]).astype(np.float32))
    x_tr, y_tr = scaled(tr_act), target(tr_act)
    model = zambia_mlp(x_tr.shape[1], cfg)
    batch = int(max(32, min(cfg["batch_size"], len(y_tr) // MIN_STEPS)))
    best, best_epoch, best_state = fit_early_stopping(model, x_tr, y_tr, scaled(va_act), target(va_act), nn.MSELoss(),
                                                      cfg["lr"], cfg["weight_decay"], batch, max_epochs, PATIENCE,
                                                      seed, squeeze=True)
    return {"model": model, "state": best_state, "means": means, "mu": mu, "sd": sd, "val_mse": best,
            "best_epoch": best_epoch, "batch": batch, "rows_nonzero": int(len(y_tr))}


def regressor_predict(reg, features, group_idx, decision):
    import torch
    with torch.no_grad():
        rel = reg["model"](torch.from_numpy(((features - reg["mu"]) / reg["sd"]).astype(np.float32))).squeeze(-1).numpy()
    return np.maximum(np.expm1(rel + reg["means"][group_idx]), 0.0) * decision


def zambia_mlp(n_in, cfg):
    """Regressor of the Zambia surrogate: hidden layers with activation, layer normalisation and dropout."""
    from torch import nn
    act = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[cfg["activation"]]
    layers, width_in = [], n_in
    for width in cfg["hidden"]:
        layers += [nn.Linear(width_in, width), act(), nn.LayerNorm(width), nn.Dropout(cfg["dropout"])]
        width_in = width
    return nn.Sequential(*layers, nn.Linear(width_in, 1))


# ---- final Zambia surrogates --------------------------------------------------------------------
def zambia_model_file(P, category, size, seed):
    return Path(P["results"]) / "zambia" / category / "models" / f"n{size}_s{seed}.pt"


def zambia_job(job):
    """
    One Zambia surrogate (category, sample size, seed):
      1. sample  the runs from the 6,500-run train pool (nested; every 5th run validation)
      2. train   the gate and the regressor (tuned configuration)
      3. save    the trained surrogate (models/n<size>_s<seed>.pt and .gate.json)
      4. calibrate the 90% intervals on the 500 calibration runs (first 100, 250, 500), per group and gate decision
      5. test    on the 1,000 test runs: measures, predictions and intervals saved for later analysis
    """
    os.environ["OMP_NUM_THREADS"] = str(job["threads"])
    import torch
    out, size, seed = Path(job["out_dir"]), job["size"], job["seed"]
    start = time.time()
    set_seed(seed, job["threads"])
    c = load_cache(out / "cache.npz")
    train_ids, val_ids = sample_runs_nested(job["train_pool"], seed, size)
    tr, va = rows_mask(c, train_ids), rows_mask(c, val_ids)
    cal, te = rows_mask(c, job["calibration"]), rows_mask(c, job["test"])
    n_groups = len(c["groups"])

    t0 = time.time()
    gate, constant = fit_gate(c, tr, seed, job["threads"])
    gate_minutes = (time.time() - t0) / 60
    t0 = time.time()
    reg = fit_regressor(c, tr, va, job["cfg"], seed, job["max_epochs"])
    train_minutes = (time.time() - t0) / 60

    model_file = out / "models" / f"n{size}_s{seed}.pt"
    if gate is not None:
        gate.save_model(str(model_file.with_suffix(".gate.json")))
    torch.save({"state_dict": reg["state"], "hidden": job["cfg"]["hidden"], "activation": job["cfg"]["activation"],
                "dropout": job["cfg"]["dropout"], "means": reg["means"], "mu": reg["mu"], "sd": reg["sd"],
                "groups": [str(g) for g in c["groups"]], "inputs": [str(i) for i in c["inputs"]],
                "year_min": int(c["year_int"].min()), "year_max": int(c["year_int"].max()),
                "gate_constant": constant, "category": job["category"], "size": size, "seed": seed}, model_file)

    def predict(mask):
        feats = cache_features(c, mask)
        decision = gate_decision(gate, constant, feats)
        return regressor_predict(reg, feats, c["group"][mask], decision), decision

    pred_cal, gate_cal = predict(cal)
    rank = {r: i for i, r in enumerate(np.searchsorted(c["runs"], job["calibration"]))}
    position = np.array([rank[r] for r in c["run"][cal]])
    q = mondrian_q(c["value"][cal], pred_cal, c["group"][cal], gate_cal, position, n_groups, job["calibration_sizes"])

    t0 = time.time()
    pred, gate_te = predict(te)
    predict_seconds = time.time() - t0
    true, group_te = c["value"][te], c["group"][te]
    bounds = {n: interval(pred, qn[group_te, gate_te]) for n, qn in q.items()}
    rows = test_measures(true, pred, gate_te, group_te, c["groups"], bounds, {"size": size, "seed": seed})
    for row in rows:
        row.update(gate_bal_acc=row.pop("zero_bal_acc"), best_epoch=reg["best_epoch"], batch_used=reg["batch"],
                   gate_minutes=gate_minutes, train_minutes=train_minutes, predict_seconds_1000_runs=predict_seconds,
                   total_minutes=(time.time() - start) / 60)
    lo, hi = bounds[max(bounds)]
    np.savez_compressed(out / "predictions" / f"n{size}_s{seed}.npz", pred=pred.astype(np.float32),
                        lo=lo.astype(np.float32), hi=hi.astype(np.float32), gate=gate_te)
    np.savez_compressed(out / "predictions" / f"cal_n{size}_s{seed}.npz", pred=pred_cal.astype(np.float32), gate=gate_cal)
    write_job(out / "jobs" / f"n{size}_s{seed}.csv", rows)
    a = rows[-1]
    return (f"n{size:<5d} s{seed:<2d} WMAPE={a['wmape']:.4f} R2={a['r2']:.4f} "
            f"nonzero coverage={a[f'coverage_nonzero_cal{max(bounds)}']:.3f} ({a['total_minutes']:.1f} min)")


def train_zambia(P, categories=None, workers=24, threads=1, quick=False):
    """Zambia surrogates of every category, sample size and seed (finished jobs are skipped)."""
    cfg = json.loads((Path(P["tuning"]) / "chosen_config.json").read_text())
    train_pool, calibration, test = zambia_split(P)
    sizes, seeds, max_epochs = ([100, 250], [1, 2], 5) if quick else (ZAMBIA_SIZES, SEEDS, MAX_EPOCHS)
    for category in categories or list(ZAMBIA_OUTPUTS):
        out = Path(P["results"]) / "zambia" / category
        check_settings(out, {"category": category, "sizes": sizes, "seeds": seeds, "config": cfg,
                             "max_epochs": max_epochs, "patience": PATIENCE, "min_steps": MIN_STEPS, "alpha": ALPHA,
                             "calibration_sizes": ZAMBIA_CALIBRATION_SIZES,
                             "groups": ZAMBIA_OUTPUTS[category]["groups"]})
        for folder in ["jobs", "predictions", "models"]:
            (out / folder).mkdir(parents=True, exist_ok=True)
        groups = list(ZAMBIA_OUTPUTS[category]["groups"])
        data_file = Path(P["data"]) / "zambia" / f"{category}_long.pkl"
        make_cache(data_file, set(train_pool) | set(calibration) | set(test), out / "cache.npz", groups)
        if not (out / "test_rows.npz").exists():                  # the true test values, in prediction order
            c = load_cache(out / "cache.npz")
            m = rows_mask(c, test)
            np.savez_compressed(out / "test_rows.npz", run_id=c["runs"][c["run"][m]], group=c["groups"][c["group"][m]],
                                year=c["year_int"][m], value=c["value"][m])
        base = {"out_dir": str(out), "category": category, "cfg": cfg, "threads": threads, "train_pool": train_pool,
                "calibration": calibration, "test": test, "max_epochs": max_epochs,
                "calibration_sizes": ZAMBIA_CALIBRATION_SIZES}
        jobs = [{**base, "size": n, "seed": s} for n in sizes for s in seeds
                if not (out / "jobs" / f"n{n}_s{s}.csv").exists()]
        failed = run_parallel(zambia_job, jobs, f"Zambia {category}", workers, lambda j: f"n{j['size']} s{j['seed']}")
        if failed:
            (out / "failed_jobs.txt").write_text("\n".join(failed))
        if any((out / "jobs").glob("*.csv")):
            _, summary = summarise_jobs(out, ["size"])
            show = summary[summary["group"] == "ALL"][["size", "wmape_mean", "r2_mean", "coverage_nonzero_cal250_mean",
                                                       "width_nonzero_cal250_mean"]]
            print(show.round(4).to_string(index=False))


def load_zambia_model(P, category, size, seed):
    """A saved Zambia surrogate: settings, network and gate."""
    import torch
    path = zambia_model_file(P, category, size, seed)
    m = torch.load(path, map_location="cpu", weights_only=False)
    net = zambia_mlp(len(m["mu"]), {"hidden": m["hidden"], "activation": m["activation"], "dropout": m["dropout"]})
    net.load_state_dict(m["state_dict"])
    net.eval()
    gate = None
    if m["gate_constant"] is None:
        from xgboost import XGBClassifier
        gate = XGBClassifier()
        gate.load_model(str(path.with_suffix(".gate.json")))
    return {"settings": m, "reg": {"model": net, "means": m["means"], "mu": m["mu"], "sd": m["sd"]}, "gate": gate}


def predict_zambia(model, scenarios, chunk=500):
    """Predictions of a saved surrogate for any scenarios (DataFrame indexed by run_id with the 9 inputs):
    one row per scenario, technology group and year, with the gate decision ('on')."""
    m = model["settings"]
    groups, years = m["groups"], np.arange(m["year_min"], m["year_max"] + 1)
    n_g, n_y = len(groups), len(years)
    parts = []
    for a in range(0, len(scenarios), chunk):                    # in pieces to limit memory
        s = scenarios.iloc[a:a + chunk]
        x = np.repeat(s[m["inputs"]].to_numpy(np.float32), n_g * n_y, axis=0)
        g_idx = np.tile(np.repeat(np.arange(n_g), n_y), len(s))
        year = np.tile(years, len(s) * n_g)
        year_scaled = ((year - m["year_min"]) / max(1, m["year_max"] - m["year_min"])).astype(np.float32)
        feats = zambia_features(x, year_scaled, g_idx, n_g)
        on = gate_decision(model["gate"], m["gate_constant"], feats)
        parts.append(pd.DataFrame({"run_id": np.repeat(s.index.to_numpy(), n_g * n_y), "GROUP": np.array(groups)[g_idx],
                                   "YEAR": year, "pred": regressor_predict(model["reg"], feats, g_idx, on), "on": on}))
    return pd.concat(parts, ignore_index=True)
