"""
CLEWs scenarios: solver files, modifying and solving the scenarios (step 1).
"""

import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .definitions import (CAPACITY_FACTOR_PREFIXES, COUNTRIES, INPUT_COLUMNS, INPUT_RANGES, RESULT_FILES,
    SOLVER_SCRIPT_FILES, SOLVER_SCRIPT_URL)
from .utils import log, normalize_run_id


def get_solver_files(P):
    """Download the model formulation, the otoole configuration and the preprocessing script of the
    OSeMOSYS-Solver-script (S. Kumar, MIT licence) into P['clews'], unless they are already there."""
    folder = Path(P["clews"])
    folder.mkdir(parents=True, exist_ok=True)
    for name in SOLVER_SCRIPT_FILES:
        if not (folder / name).exists():
            urllib.request.urlretrieve(SOLVER_SCRIPT_URL + name, folder / name)
            log(f"downloaded {name} from the OSeMOSYS-Solver-script")


def load_scenarios(P, country):
    """The 9 input values of every scenario of a country, indexed by run_id."""
    df = pd.read_csv(P["scenarios"][country], encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    df["run_id"] = df["run_id"].map(normalize_run_id)
    if df["run_id"].duplicated().any():
        raise ValueError(f"{country}: duplicate run_id values in the scenario table")
    return df.set_index("run_id")[INPUT_COLUMNS].astype(float)


def sample_scenarios(n, seed, first_id):
    """Latin hypercube sample of the 9 inputs over INPUT_RANGES (to explore new scenarios)."""
    from scipy.stats import qmc
    u = qmc.LatinHypercube(d=len(INPUT_COLUMNS), seed=seed).random(n=n)
    rows = {"run_id": [f"run_{first_id + i:03d}" for i in range(n)]}
    for j, name in enumerate(INPUT_COLUMNS):
        lo, hi = INPUT_RANGES[name]
        rows[name] = lo + (hi - lo) * u[:, j]
    return pd.DataFrame(rows).set_index("run_id")


def fix_data_file(data_file, out_file):
    """Rewrite a few empty parameter blocks of a CLEWs data file into a form otoole can read
    (as fix_txt_file in Run_CLEWs_5.2.py of the OSeMOSYS-Solver-script)."""
    single_line = ["UDCTag", "OperationalLifeStorage", "StorageLevelStart"]
    lines = Path(data_file).read_text(encoding="utf-8").splitlines(keepends=True)
    fixed, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        step = 0
        if any(line.startswith(f"param {p} default") for p in single_line):
            if i + 3 < len(lines) and lines[i + 1].strip() == ":=" and lines[i + 3].strip() == ";":
                step = 4
        elif line.startswith("param UDCConstant default"):
            if i + 5 < len(lines) and lines[i + 5].strip() == ";":
                step = 6
        if step:
            head = re.sub(r"\s*:=?\s*$", "", line)
            fixed.append(f"{head} :=\n;\n")
            i += step
        else:
            fixed.append(lines[i])
            i += 1
    Path(out_file).write_text("".join(fixed), encoding="utf-8")


def baseline_csv(P, country):
    """The unmodified model data of a country as CSV files (made once with otoole)."""
    folder = Path(P["temp"]) / f"{country}_baseline_csv"
    if not folder.exists():
        fixed = Path(P["temp"]) / f"{country}_data_fixed.txt"
        fix_data_file(P["data_files"][country], fixed)
        subprocess.run(f'otoole convert datafile csv "{fixed}" "{folder}" "{Path(P["clews"]) / "config_com.yaml"}"',
                       shell=True, check=True, cwd=str(P["clews"]))
    return folder


def apply_inputs(csv_dir, country, x):
    """Apply the 9 input values of one scenario to the CSV model data (see INPUT_RANGES)."""
    spec, years = COUNTRIES[country], COUNTRIES[country]["years"]

    def demand(df):
        df["VALUE"] *= x["demand_multiplier"]

    def discount(df):
        df["VALUE"] = x["discount_rate"]

    def fuel(df):
        tech = df["TECHNOLOGY"].astype(str)
        df.loc[tech.isin(spec["fuel_techs"]) | tech.str.startswith("LND"), "VALUE"] *= x["fuel_cost_multiplier"]

    def capital(df):
        df["VALUE"] *= x["capital_cost_multiplier"]

    def capacity_factor(df):
        df.loc[df["TECHNOLOGY"].astype(str).str.startswith(CAPACITY_FACTOR_PREFIXES), "VALUE"] *= \
            x["capacity_factor_multiplier"]

    def iar(df):
        df.loc[df["COMMODITY"].astype(str).str.startswith("AGRWAT"), "VALUE"] *= x["iar_multiplier"]

    def oar(df):
        crops = df["COMMODITY"].astype(str).str.contains("CRP", na=False)
        df.loc[crops & df["YEAR"].astype(int).between(years[0], years[-1]), "VALUE"] *= x["oar_multiplier"]

    def accumulated(df):
        comm = df["COMMODITY"].astype(str)
        df.loc[comm.isin(spec["water_demand"]), "VALUE"] *= x["water_accumulated_multiplier"]
        df.loc[comm.isin(spec["crop_demand"]), "VALUE"] *= x["crop_accumulated_multiplier"]

    for name, change in [("SpecifiedAnnualDemand.csv", demand), ("DiscountRate.csv", discount),
                         ("VariableCost.csv", fuel), ("CapitalCost.csv", capital),
                         ("CapacityFactor.csv", capacity_factor), ("InputActivityRatio.csv", iar),
                         ("OutputActivityRatio.csv", oar), ("AccumulatedAnnualDemand.csv", accumulated)]:
        df = pd.read_csv(csv_dir / name)
        change(df)
        df.to_csv(csv_dir / name, index=False)


def ensure_annual_results(results, csv_dir):
    """Write ProductionByTechnologyAnnual / UseByTechnologyAnnual if the solver did not: from the time-slice
    files, or from RateOfActivity x activity ratio x YearSplit."""
    keys = ["REGION", "TECHNOLOGY", "COMMODITY", "YEAR"]
    for annual, timeslice, ratio_file in [("ProductionByTechnologyAnnual.csv", "ProductionByTechnology.csv",
                                           "OutputActivityRatio.csv"),
                                          ("UseByTechnologyAnnual.csv", "UseByTechnology.csv", "InputActivityRatio.csv")]:
        if (results / annual).exists():
            continue
        if (results / timeslice).exists():
            pd.read_csv(results / timeslice).groupby(keys, as_index=False)["VALUE"].sum().to_csv(results / annual, index=False)
            continue
        rate = pd.read_csv(results / "RateOfActivity.csv").rename(columns={"VALUE": "RATE"})
        ratio = pd.read_csv(csv_dir / ratio_file).rename(columns={"VALUE": "RATIO"})
        split = pd.read_csv(csv_dir / "YearSplit.csv").rename(columns={"VALUE": "SPLIT"})
        for df in (rate, ratio):
            df["MODE_OF_OPERATION"] = df["MODE_OF_OPERATION"].astype(str)
            df["YEAR"] = df["YEAR"].astype(int)
        split["YEAR"] = split["YEAR"].astype(int)
        m = rate.merge(ratio, on=["REGION", "TECHNOLOGY", "MODE_OF_OPERATION", "YEAR"], how="inner")
        m = m.merge(split[["TIMESLICE", "YEAR", "SPLIT"]], on=["TIMESLICE", "YEAR"], how="left")
        m["VALUE"] = m["RATE"].astype(float) * m["RATIO"].astype(float) * m["SPLIT"].fillna(0.0).astype(float)
        m.groupby(keys, as_index=False)["VALUE"].sum().to_csv(results / annual, index=False)


def scenario_solved(P, country, run_id):
    results = Path(P["runs"][country][0]) / run_id / "results"
    return all((results / name).exists() for name in RESULT_FILES)


def solve_scenario(P, country, run_id, x, base_csv, time_limit=1800):
    """Modify the model data for one scenario, solve it and keep only RESULT_FILES. The solution steps are
    those of Run_CLEWs_5.2.py of the OSeMOSYS-Solver-script: otoole (CSV -> data file), preprocess_data.py,
    GLPK to write the problem, Gurobi to solve it, otoole (solution -> CSV results). Temporary files are
    always removed."""
    clews = Path(P["clews"])
    config_file = clews / "config_com.yaml"
    work = Path(P["temp"]) / country / run_id
    out = Path(P["runs"][country][0]) / run_id
    csv_dir, all_results = work / "csv", work / "results"
    data_txt, data_pre, lp_file, sol_file = (work / n for n in ["data.txt", "data_pre.txt", "model.lp", "result.sol"])
    start, status, error = time.perf_counter(), "success", ""

    def call(command, timeout=time_limit):
        subprocess.run(command, shell=True, check=True, timeout=timeout, cwd=str(clews))

    try:
        for leftover in (work, out):
            shutil.rmtree(leftover, ignore_errors=True)
        work.mkdir(parents=True)
        shutil.copytree(base_csv, csv_dir)
        apply_inputs(csv_dir, country, x)
        call(f'otoole convert csv datafile "{csv_dir}" "{data_txt}" "{config_file}"')
        call(f'"{sys.executable}" "{clews / "preprocess_data.py"}" otoole "{data_txt}" "{data_pre}"')
        call(f'glpsol -d "{data_pre}" -m "{clews / "model.v.5.2.txt"}" --wlp "{lp_file}" --check')
        call(f'gurobi_cl TimeLimit={time_limit} Threads=1 ResultFile="{sol_file}" "{lp_file}"', timeout=time_limit + 120)
        all_results.mkdir()
        call(f'otoole results gurobi csv "{sol_file}" "{all_results}" csv "{csv_dir}" "{config_file}"')
        ensure_annual_results(all_results, csv_dir)
        missing = [n for n in RESULT_FILES if not (all_results / n).exists()]
        if missing:
            raise FileNotFoundError(f"missing result files: {missing}")
        (out / "results").mkdir(parents=True)
        for name in RESULT_FILES:
            shutil.copy2(all_results / name, out / "results" / name)
    except subprocess.TimeoutExpired as exc:
        status, error = "timeout", str(exc)
    except Exception as exc:
        status, error = "failed", str(exc)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if status != "success":
        out.mkdir(parents=True, exist_ok=True)
        (out / "generation_error.json").write_text(json.dumps({"status": status, "error": error}, indent=2))
    return {"run_id": run_id, **x, "status": status, "running_time_min": (time.perf_counter() - start) / 60,
            "error_message": error[:500]}


def solve_scenarios(P, country, scenarios, workers=16, time_limit=1800):
    """Solve every scenario of the table that is not solved yet (in parallel threads). Can be stopped
    and started again; failed scenarios are tried again."""
    get_solver_files(P)
    Path(P["runs"][country][0]).mkdir(parents=True, exist_ok=True)
    Path(P["temp"]).mkdir(parents=True, exist_ok=True)
    todo = [r for r in scenarios.index if not scenario_solved(P, country, r)]
    log(f"{country}: {len(scenarios)} scenarios, {len(scenarios) - len(todo)} solved, {len(todo)} to solve")
    if not todo:
        return
    base = baseline_csv(P, country)
    summary_file = Path(P["runs"][country][0]) / "solve_summary.csv"
    rows, start = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(solve_scenario, P, country, r, scenarios.loc[r].to_dict(), base, time_limit) for r in todo]
        for i, f in enumerate(as_completed(futures), start=1):
            row = f.result()
            rows.append(row)
            log(f"{i}/{len(todo)} {row['run_id']} -> {row['status']} ({row['running_time_min']:.2f} min) | "
                f"about {(time.time() - start) / i * (len(todo) - i) / 60:.0f} min left")
            if i % 25 == 0 or i == len(todo):
                old = pd.read_csv(summary_file) if summary_file.exists() else pd.DataFrame()
                pd.concat([old, pd.DataFrame(rows)]).drop_duplicates("run_id", keep="last").to_csv(summary_file, index=False)
                rows = []
