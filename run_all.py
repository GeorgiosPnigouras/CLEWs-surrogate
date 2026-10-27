"""
Runs the whole study of the paper
  "Surrogate Modeling for Fast Scenario Exploration in Climate-Land-Water-Energy Systems:
   From Zambia to Ghana via Transfer Learning"
from start to end. All functions are in the folder surrogate/ (one module per step); this file
holds only the settings and the order of the steps.

  1. solve           solve the 8,000 Zambia and 2,000 Ghana scenarios with CLEWs
  2. data            fixed splits and the data tables of all output categories
  3. tune            hyperparameter search of the Zambia surrogate (validation data only)
  4. zambia          Zambia surrogates for all sample sizes and repetitions (models are saved)
  5. transfer        Ghana surrogates with and without transfer from Zambia
  6. evaluate        pair accuracy and the number of CLEWs runs needed for each level of use
  7. demonstration   scenario exploration with the 1,000-run Zambia surrogate
  8. figures         figures and tables of the paper

Run it from a terminal, as it starts parallel processes (on Windows, Spyder's console cannot do this):
    python run_all.py
Finished work is saved and skipped when the script is started again, so it can be stopped at any time.
Set a step to False to skip it.
"""

import time
from pathlib import Path

from surrogate import data, demonstration, evaluation, figures, scenarios, transfer, tuning, zambia
from surrogate.definitions import COUNTRIES
from surrogate.utils import log

# ================================================================================ SETTINGS
HERE = Path(__file__).resolve().parent
WORK = HERE / "work"                    # everything the study writes

P = {
    "clews": WORK / "clews",            # solver files of the OSeMOSYS-Solver-script (downloaded)
    "data_files": {                     # CLEWs data files of the two models (Zenodo, see README)
        "Zambia": HERE / "clews_models" / "Zambia.txt",
        "Ghana": HERE / "clews_models" / "Ghana.txt",
    },
    "scenarios": {                      # the 9 input values of every scenario of the study
        "Zambia": HERE / "scenarios" / "zambia_scenarios.csv",
        "Ghana": HERE / "scenarios" / "ghana_scenarios.csv",
    },
    "runs": {                           # solved scenarios; more folders can be listed (searched in order)
        "Zambia": [WORK / "runs" / "Zambia"],
        "Ghana": [WORK / "runs" / "Ghana"],
    },
    "splits": HERE / "splits",          # fixed training, calibration and test scenarios of the study
    "temp": WORK / "temp",              # temporary solver files (large; a fast local disk is best)
    "data": WORK / "data",
    "tuning": WORK / "tuning",
    "results": WORK / "results",
    "figures": WORK / "figures",
}

STEPS = {
    "solve": True,
    "data": True,
    "verify": False,                    # compare the data tables with those of the study (folders below)
    "tune": True,
    "zambia": True,
    "transfer": True,
    "evaluate": True,
    "demonstration": True,
    "figures": True,
}

QUICK = False                           # True: short test of steps 3-8 (few sizes, seeds and epochs),
                                        # written to work/quicktest; the numbers mean nothing
SOLVE_WORKERS = 16                      # scenarios solved at the same time (one core each)
SOLVE_TIME_LIMIT = 1800                 # seconds per solver step (longest Zambia scenario of the study: 10.7 min)
WORKERS, THREADS = 12, 2                # tuning and transfer: jobs at the same time x cores per job
ZAMBIA_WORKERS, ZAMBIA_THREADS = 24, 1  # Zambia surrogates
OLD_ZAMBIA_DATA = None                  # for "verify": folder with <CATEGORY>_long.pkl of the study
OLD_TRANSFER_DATA = None                # for "verify": folder with <Country>_<CATEGORY>.pkl of the study
# ================================================================================


def main():
    start = time.time()
    if QUICK:
        for key in ["tuning", "results", "figures"]:
            P[key] = WORK / "quicktest" / key
    demo_size = 250 if QUICK else demonstration.DEMO_SIZE

    if STEPS["solve"]:
        log("1. solving the CLEWs scenarios")
        for country in COUNTRIES:
            table = scenarios.load_scenarios(P, country)
            scenarios.solve_scenarios(P, country, table, SOLVE_WORKERS, SOLVE_TIME_LIMIT)
    if STEPS["data"]:
        log("2. splits and data tables")
        data.make_splits(P)
        data.build_datasets(P)
    if STEPS["verify"]:
        log("   comparison with the data tables of the study")
        data.compare_tables(P, OLD_ZAMBIA_DATA, OLD_TRANSFER_DATA)
    if STEPS["tune"]:
        log("3. tuning of the Zambia surrogate")
        tuning.tune(P, WORKERS, THREADS, QUICK)
    if STEPS["zambia"]:
        log("4. Zambia surrogates")
        zambia.train_zambia(P, None, ZAMBIA_WORKERS, ZAMBIA_THREADS, QUICK)
    if STEPS["transfer"]:
        log("5. transfer Zambia -> Ghana")
        transfer.transfer(P, None, WORKERS, THREADS, QUICK)
    if STEPS["evaluate"]:
        log("6. evaluation")
        evaluation.evaluate(P)
    if STEPS["demonstration"]:
        log("7. demonstration")
        demonstration.demonstration(P, size=demo_size)
    if STEPS["figures"]:
        log("8. figures and tables")
        figures.figures_and_tables(P, heatmap_size=demo_size)
    log(f"done in {(time.time() - start) / 3600:.1f} h; results in {P['results']}, figures in {P['figures']}")


if __name__ == "__main__":
    main()
