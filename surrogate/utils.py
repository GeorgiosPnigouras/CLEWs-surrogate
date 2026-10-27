"""
Utilities: run ids, parallel jobs, saved settings, splits and samples.
"""

import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def log(message):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {message}", flush=True)


def normalize_run_id(x):
    """1 -> run_001; 'run_001' stays as it is."""
    s = str(x).strip()
    if s.startswith("run_"):
        return s
    try:
        return f"run_{int(float(s)):03d}"
    except ValueError:
        return s


def versions():
    import importlib
    out = {"python": platform.python_version()}
    for name in ["numpy", "pandas", "torch", "xgboost"]:
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:
            out[name] = "not installed"
    return out


def run_parallel(func, jobs, label, workers, describe=lambda j: ""):
    """Run jobs in parallel processes, longest (largest size) first; a failed job is reported
    and the others go on. Returns the list of failures."""
    todo = sorted(jobs, key=lambda j: -j.get("size", 0))
    log(f"[{label}] {len(todo)} jobs to run ({workers} at the same time)")
    failed, start = [], time.time()
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(todo) or 1))) as ex:
        futures = {ex.submit(func, j): j for j in todo}
        for i, f in enumerate(as_completed(futures), start=1):
            j = futures[f]
            try:
                message = f.result()
            except Exception as error:
                failed.append(f"{describe(j)}: {type(error).__name__}: {error}")
                message = f"FAILED {describe(j)}: {error}"
            print(f"[{label}] {i}/{len(todo)} {message} | {(time.time() - start) / 60:.1f} min", flush=True)
    return failed


def check_settings(out_dir, info):
    """Stop if out_dir holds results made with other settings; otherwise record the settings."""
    path = Path(out_dir) / "run_info.json"
    if path.exists():
        old = json.loads(path.read_text())
        changed = [k for k in info if old.get(k) != info[k]]
        if changed:
            raise SystemExit(f"{out_dir} holds results made with other settings ({changed}); rename or delete it")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**info, "versions": versions()}, indent=2, default=str))


def summarise_jobs(out_dir, keys):
    """All finished job files -> metrics_by_seed.csv and summary.csv (mean, sd, median, 25/75% over the seeds)."""
    by_seed = pd.concat([pd.read_csv(p) for p in sorted((Path(out_dir) / "jobs").glob("*.csv"))], ignore_index=True)
    by_seed = by_seed.sort_values(keys + ["group", "seed"])
    by_seed.to_csv(Path(out_dir) / "metrics_by_seed.csv", index=False)
    metrics = [c for c in by_seed.columns if c not in keys + ["seed", "group", "rows"] and
               pd.api.types.is_numeric_dtype(by_seed[c])]
    g = by_seed.groupby(keys + ["group"])[metrics]
    summary = pd.concat({"mean": g.mean(), "sd": g.std(), "median": g.median(),
                         "q25": g.quantile(0.25), "q75": g.quantile(0.75)}, axis=1)
    summary.columns = [f"{m}_{stat}" for stat, m in summary.columns]
    summary = summary.reset_index()
    summary["n_seeds"] = g.size().to_numpy()
    summary.to_csv(Path(out_dir) / "summary.csv", index=False)
    return by_seed, summary


def write_job(path, rows):
    """Write a job file at the very end of a job (its existence means 'done')."""
    path = Path(path)
    pd.DataFrame(rows).to_csv(path.with_suffix(".tmp"), index=False)
    path.with_suffix(".tmp").replace(path)


# ---- splits and samples
def write_ids(path, ids, ordered=False):
    df = pd.DataFrame({"run_id": list(ids)})
    if ordered:
        df["order"] = np.arange(1, len(df) + 1)
    df.to_csv(path, index=False)


def read_ids(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "order" in df.columns:
        df = df.sort_values("order")
    return df["run_id"].astype(str).tolist()


def zambia_split(P):
    """(train pool, calibration runs in their fixed order, test runs) of the Zambia study."""
    d = Path(P["splits"]) / "zambia"
    return read_ids(d / "train_pool_ids.csv"), read_ids(d / "calibration_ids.csv"), read_ids(d / "test_ids.csv")


def zambia_development_runs(P):
    """The 7,000 development runs (train pool + calibration set); the tuning draws its samples from these."""
    return read_ids(Path(P["splits"]) / "zambia" / "pool_ids.csv")


def ghana_split(P):
    """{'test', 'calibration' (fixed order), 'pool'} of the Ghana study."""
    d = Path(P["splits"]) / "ghana"
    return {n: read_ids(d / f"{n}.csv") for n in ["test", "calibration", "pool"]}


def sample_runs(pool_ids, seed, size, validation_share=0.20):
    """Tuning sample: `size` runs, a random 20% of them for validation."""
    order = np.array(sorted(pool_ids))
    np.random.default_rng(seed).shuffle(order)
    selected = list(order[:size])
    np.random.default_rng(seed + size).shuffle(selected)
    n_val = max(1, round(validation_share * size))
    return sorted(selected[n_val:]), sorted(selected[:n_val])


def sample_runs_nested(pool_ids, seed, size, validation_every=5):
    """Training sample of the surrogates: the first `size` runs of the seed's random order, every 5th
    of them for validation (20%). For one seed, the samples and their validation runs are nested,
    e.g. the 500-run sample lies inside the 750-run sample."""
    order = np.array(sorted(pool_ids))
    np.random.default_rng(seed).shuffle(order)
    selected = order[:size]
    is_val = (np.arange(size) % validation_every) == validation_every - 1
    return sorted(selected[~is_val]), sorted(selected[is_val])
