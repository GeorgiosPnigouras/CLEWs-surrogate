"""
Data tables: reading the CLEWs results, total discounted cost, splits (step 2).
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .definitions import (ACTIVE_THRESHOLD, COUNTRIES, EPS, REGION, TDC_EXCLUDE_PREFIXES, TRANSFER_OUTPUTS,
    ZAMBIA_OUTPUTS)
from .utils import ghana_split, log, write_ids, zambia_split
from .scenarios import load_scenarios


def result_path(P, country, run_id, name):
    """A result file of a solved scenario, from the first run folder of the country that has it."""
    folders = P["runs"][country]
    for folder in folders:
        path = Path(folder) / run_id / "results" / name
        if path.exists():
            return path
    return Path(folders[0]) / run_id / "results" / name


def read_result(path):
    """A CLEWs result CSV with upper-case columns and numeric YEAR and VALUE."""
    with open(path, encoding="utf-8-sig") as handle:                 # some files use ";"
        sep = ";" if ";" in handle.readline() else ","
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
    df.columns = [str(c).strip().upper() for c in df.columns]
    for col in ("TECHNOLOGY", "REGION", "COMMODITY", "EMISSION"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if df.empty:
        return df
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    return df.dropna(subset=["YEAR"])


def read_output(P, country, run_id, file, filter_, groups):
    """One output category of one scenario: one row per technology group and year. The technologies of a
    group are summed; missing technologies or years count as 0; small negative solver values are set to 0."""
    years = COUNTRIES[country]["years"]
    df = read_result(result_path(P, country, run_id, file))
    keep = df["REGION"].eq(REGION) & df["YEAR"].isin(years)
    if filter_ is not None:
        column, values = filter_
        keep &= df[column].astype(str).str.strip().isin(values)
    df = df[keep]
    tech_to_group = {t: g for g, techs in groups.items() for t in techs}
    df = df[df["TECHNOLOGY"].isin(tech_to_group)].assign(GROUP=lambda d: d["TECHNOLOGY"].map(tech_to_group))
    index = pd.MultiIndex.from_product([list(groups), years], names=["GROUP", "YEAR"])
    out = df.groupby(["GROUP", "YEAR"])["VALUE"].sum().reindex(index, fill_value=0.0).reset_index()
    out["VALUE"] = out["VALUE"].clip(lower=0.0)
    return out


def total_discounted_cost(P, country, run_id, discount_rate):
    """
    Total discounted cost per year without forest and grassland (TDC_EXCLUDE_PREFIXES): the
    TotalDiscountedCost reported by the model minus the discounted cost of the excluded technologies,
    discounted as in OSeMOSYS:
        capital investment / (1 + r)^(y - y0) + (fixed + variable operating cost) / (1 + r)^(y - y0 + 0.5)
        + discounted emission penalty - discounted salvage value
    with y0 the first model year. A missing part counts as 0 (in the Ghana model, forest and grassland
    have only operating costs).
    """
    reported = read_result(result_path(P, country, run_id, "TotalDiscountedCost.csv")).groupby("YEAR")["VALUE"].sum()
    y0 = int(reported.index.min())

    def excluded(name):
        path = result_path(P, country, run_id, name)
        df = read_result(path) if path.exists() else pd.DataFrame()
        if df.empty:
            return pd.Series(dtype=float)
        return df[df["TECHNOLOGY"].str.startswith(TDC_EXCLUDE_PREFIXES)].groupby("YEAR")["VALUE"].sum()

    parts = pd.DataFrame({n: excluded(f) for n, f in [
        ("capital", "CapitalInvestment.csv"), ("fixed", "AnnualFixedOperatingCost.csv"),
        ("variable", "AnnualVariableOperatingCost.csv"), ("penalty", "DiscountedTechnologyEmissionsPenalty.csv"),
        ("salvage", "DiscountedSalvageValue.csv")]}).reindex(reported.index).fillna(0.0)
    y, r = parts.index.to_numpy(dtype=float), float(discount_rate)
    land = (parts["capital"] / (1 + r) ** (y - y0) + (parts["fixed"] + parts["variable"]) / (1 + r) ** (y - y0 + 0.5)
            + parts["penalty"] - parts["salvage"])
    tdc = (reported - land).rename("VALUE")
    tdc = tdc[tdc.index.isin(COUNTRIES[country]["years"])]
    return tdc.reset_index().assign(GROUP="TOTAL")[["GROUP", "YEAR", "VALUE"]]


def build_table(P, country, spec, filter_, groups, out_file, workers=24):
    """Data table of one output category: run_id, GROUP, YEAR, VALUE, the 9 inputs, active (1 = non-zero).
    Also writes <name>_zero_share.csv and, if any, <name>_missing_runs.csv."""
    inputs = load_scenarios(P, country)
    name = Path(out_file).stem.replace("_long", "")

    def one(run_id):
        try:
            if spec["file"] is None:
                out = total_discounted_cost(P, country, run_id, inputs.loc[run_id, "discount_rate"])
            else:
                out = read_output(P, country, run_id, spec["file"], filter_, groups)
            out.insert(0, "run_id", run_id)
            return out, None
        except Exception as error:
            return None, {"run_id": run_id, "error": f"{type(error).__name__}: {error}"}

    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, sorted(inputs.index)))
    parts, missing = [t for t, _ in results if t is not None], [m for _, m in results if m is not None]
    if missing:
        pd.DataFrame(missing).to_csv(Path(out_file).with_name(f"{name}_missing_runs.csv"), index=False)
        log(f"[{name}] WARNING: {len(missing)} scenarios could not be read")
    if not parts:
        raise RuntimeError(f"{name}: no scenario could be read, e.g. {missing[0]['error']}")
    data = pd.concat(parts, ignore_index=True)
    data["YEAR"] = data["YEAR"].astype(int)
    data = data.merge(inputs, left_on="run_id", right_index=True, how="left", validate="many_to_one")
    data["active"] = (data["VALUE"] > ACTIVE_THRESHOLD).astype(np.uint8)
    data = data.sort_values(["run_id", "GROUP", "YEAR"]).reset_index(drop=True)
    data.to_pickle(out_file)
    per_run = data.groupby(["GROUP", "run_id"])["active"].max()
    pd.DataFrame({"share_zero_rows": 1 - data.groupby("GROUP")["active"].mean(),
                  "share_runs_nonzero": per_run.groupby("GROUP").mean()}).reset_index() \
        .to_csv(Path(out_file).with_name(f"{name}_zero_share.csv"), index=False)
    log(f"[{name}] {data['run_id'].nunique()} scenarios, {len(data)} rows, {(time.time() - start) / 60:.1f} min")


def build_datasets(P, which=("zambia", "transfer")):
    """Data tables of the Zambia surrogate (data/zambia/<CATEGORY>_long.pkl) and of the transfer
    (data/transfer/<Country>_<CATEGORY>.pkl)."""
    if "zambia" in which:
        folder = Path(P["data"]) / "zambia"
        folder.mkdir(parents=True, exist_ok=True)
        for category, spec in ZAMBIA_OUTPUTS.items():
            build_table(P, "Zambia", spec, spec["filter"], spec["groups"], folder / f"{category}_long.pkl")
    if "transfer" in which:
        folder = Path(P["data"]) / "transfer"
        folder.mkdir(parents=True, exist_ok=True)
        for country in COUNTRIES:
            for category, spec in TRANSFER_OUTPUTS.items():
                build_table(P, country, spec, spec["filter"][country], spec["groups"][country],
                            folder / f"{country}_{category}.pkl")


def make_splits(P, seed=2026):
    """Check the fixed splits of the study (splits/zambia, splits/ghana); draw new ones only if missing
    (Zambia: 1,000 test, 500 calibration, 6,500 train pool; Ghana: 400 test, 300 calibration, the rest pool)."""
    z, g = Path(P["splits"]) / "zambia", Path(P["splits"]) / "ghana"
    z_names = ["pool_ids.csv", "test_ids.csv", "calibration_ids.csv", "train_pool_ids.csv"]
    if not all((z / n).exists() for n in z_names):
        log("Zambia split files missing: drawing a NEW split (it differs from the study)")
        z.mkdir(parents=True, exist_ok=True)
        order = np.array(sorted(load_scenarios(P, "Zambia").index))
        np.random.default_rng(seed).shuffle(order)
        test, pool = sorted(order[:1000]), sorted(order[1000:])
        cal = np.array(pool)
        np.random.default_rng(seed).shuffle(cal)
        write_ids(z / "pool_ids.csv", pool)
        write_ids(z / "test_ids.csv", test)
        write_ids(z / "calibration_ids.csv", cal[:500], ordered=True)
        write_ids(z / "train_pool_ids.csv", sorted(set(pool) - set(cal[:500])))
    if not all((g / f"{n}.csv").exists() for n in ["test", "calibration", "pool"]):
        log("Ghana split files missing: drawing a NEW split (it differs from the study)")
        g.mkdir(parents=True, exist_ok=True)
        order = np.array(sorted(load_scenarios(P, "Ghana").index))
        np.random.default_rng(seed).shuffle(order)
        write_ids(g / "test.csv", sorted(order[:400]))
        write_ids(g / "calibration.csv", order[400:700])
        write_ids(g / "pool.csv", sorted(order[700:]))
    train, cal, test = (set(x) for x in zambia_split(P))
    assert not (train & cal or train & test or cal & test), "Zambia split is inconsistent"
    s = {k: set(v) for k, v in ghana_split(P).items()}
    assert not (s["test"] & s["calibration"] or s["test"] & s["pool"] or s["calibration"] & s["pool"]), \
        "Ghana split is inconsistent"
    log(f"splits: Zambia {len(train)} train pool, {len(cal)} calibration, {len(test)} test; "
        f"Ghana {len(s['pool'])} pool, {len(s['calibration'])} calibration, {len(s['test'])} test")


def compare_tables(P, old_zambia, old_transfer, tolerance=1e-9):
    """Compare the data tables in P['data'] with those used in the study (folders with <CATEGORY>_long.pkl
    and <Country>_<CATEGORY>.pkl): same rows and values up to a relative difference of `tolerance`."""
    keys = ["run_id", "GROUP", "YEAR"]
    pairs = [(Path(P["data"]) / "zambia" / f"{c}_long.pkl", Path(old_zambia) / f"{c}_long.pkl") for c in ZAMBIA_OUTPUTS]
    pairs += [(Path(P["data"]) / "transfer" / f"{k}_{c}.pkl", Path(old_transfer) / f"{k}_{c}.pkl")
              for k in COUNTRIES for c in TRANSFER_OUTPUTS]
    for new_file, old_file in pairs:
        if not (new_file.exists() and old_file.exists()):
            print(f"  {new_file.name:24s} not compared (file missing)")
            continue
        new, old = pd.read_pickle(new_file), pd.read_pickle(old_file)
        m = new.merge(old.assign(run_id=old["run_id"].astype(str))[keys + ["VALUE"]], on=keys, how="outer",
                      suffixes=("", "_old"), indicator=True)
        only_new, only_old = int((m["_merge"] == "left_only").sum()), int((m["_merge"] == "right_only").sum())
        both = m[m["_merge"] == "both"]
        gap = float((both["VALUE"] - both["VALUE_old"]).abs().max() / max(both["VALUE_old"].abs().max(), EPS))
        same = only_new == only_old == 0 and gap <= tolerance
        print(f"  {new_file.name:24s} {'IDENTICAL' if same else 'DIFFERENT'}  rows {len(both)}, only new {only_new}, "
              f"only old {only_old}, largest relative difference {gap:.1e}")
