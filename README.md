# Surrogate modeling for fast scenario exploration in CLEWs: from Zambia to Ghana via transfer learning

Code of the paper *Surrogate Modeling for Fast Scenario Exploration in Climate-Land-Water-Energy Systems: From Zambia to Ghana via Transfer Learning* by Georgios Pnigouras, Francesco Gardumi and Shravan Kumar (submitted).

The code trains machine-learning surrogates of CLEWs (Climate, Land, Energy and Water) models built in
OSeMOSYS, evaluates them for two levels of use (screening and detailed use), transfers the surrogate
from the Zambia model to the Ghana model, and reproduces the demonstration, figures and tables of the
paper. The whole study runs from one script, from solving the CLEWs scenarios to the figures.

## Files

| File or folder | Content |
|---|---|
| `run_all.py` | Settings (folders, steps to run, parallel workers) and the order of the steps. This is the only file to edit and run. |
| `surrogate/` | All functions of the study, each defined once, in one module per part of the work (listed below). |
| `scenarios/` | The 9 input values of the 8,000 Zambia and 2,000 Ghana scenarios of the study. |
| `splits/` | The fixed training, calibration and test scenarios of the study. |
| `clews_models/` | Place for the CLEWs data files of the two models (not included, see Data). |

## Steps of `run_all.py`

| Step | Module in `surrogate/` | What it does | Output (in `work/`) |
|---|---|---|---|
| 1. solve | `scenarios.py` | Solves the CLEWs scenarios of both countries | `runs/<country>/<run_id>/results/` |
| 2. data | `data.py` | Checks the fixed splits and builds the data tables of all output categories, including the total discounted cost | `data/` |
| 3. tune | `tuning.py` | Hyperparameter search of the Zambia surrogate (validation data only) | `tuning/chosen_config.json` |
| 4. zambia | `zambia.py` | Zambia surrogates for all sample sizes and repetitions: measures, prediction intervals, saved models | `results/zambia/` |
| 5. transfer | `transfer.py` | Ghana surrogates with and without transfer from Zambia | `results/ghana/` |
| 6. evaluate | `evaluation.py` | Pair accuracy and the number of CLEWs runs needed for each level of use | `results/*/ranking.csv`, `minimum_runs.csv`, `figures/table_minimum_runs_*.tex` |
| 7. demonstration | `demonstration.py` | Scenario exploration with the 1,000-run Zambia surrogate | `results/demonstration/` |
| 8. figures | `figures.py` | Figures and supplementary tables of the paper | `figures/` |

The other modules hold what the steps share: `definitions.py` (inputs and their ranges, the two case
studies, the output categories and the example criteria of the levels of use), `utils.py` (parallel jobs,
splits and samples), `measures.py` (WMAPE, R², prediction intervals, pair accuracy) and `networks.py`
(the training loop used by all neural networks).

### Where the results of the paper come from

| Paper | Function (module) |
|---|---|
| Zambia learning curves (`fig:learning-curves`) | `fig_learning_curves` (`figures.py`) |
| Error per technology group (`fig:heatmap`) | `fig_heatmap` (`figures.py`) |
| Prediction intervals (`fig:uncertainty`) | `fig_uncertainty` (`figures.py`) |
| Levels of use, Zambia and Ghana (`tab:minimum-runs`, `tab:minimum-runs-ghana`) | `minimum_runs` (`evaluation.py`) |
| Ghana WMAPE, pair accuracy and R² (`fig:ghana-wmape`, `fig:ghana-pairs`, `fig:ghana-r2`) | `fig_ghana` (`figures.py`) |
| Demonstration (`fig:demonstration`) | `demonstration` (`demonstration.py`), `fig_demonstration` (`figures.py`) |
| Supplementary tables (zero detection, intervals and pair accuracy per technology group) | `supplementary_tables` (`figures.py`) |

## Data and code used from other sources

- **CLEWs models.** The data files of the two models are published on Zenodo and are not included here.
  Download them and set their paths in `run_all.py` (`P["data_files"]`):
  - C. Lo Giudice and M. Mulunda, *CLEWs Zambia dataset*, Zenodo, 2026. doi:[10.5281/zenodo.19206729](https://doi.org/10.5281/zenodo.19206729)
  - S. G. Savanur, *Ghana CLEWs model and dataset*, Zenodo, 2026. doi:[10.5281/zenodo.20267395](https://doi.org/10.5281/zenodo.20267395)
- **Solving the models.** The conversion and solution of the models follow the
  [OSeMOSYS-Solver-script](https://github.com/ShravanKumar23/OSeMOSYS-Solver-script) by S. Kumar (MIT licence).
  Its `model.v.5.2.txt` (OSeMOSYS model), `config_com.yaml` (otoole configuration) and
  `preprocess_data.py` are downloaded from that repository when step 1 starts (`get_solver_files`), and
  the steps of its `Run_CLEWs_5.2.py` are followed for every scenario: the data file is fixed for otoole
  and converted to CSV files, the 9 inputs of the scenario are applied to the CSV files, and the files are
  converted back to a data file, preprocessed, written as an LP problem with GLPK, solved with Gurobi and
  converted to CSV results with otoole (`fix_data_file`, `apply_inputs`, `solve_scenario`).
- **Scenarios.** `scenarios/` holds the input values of every scenario of the study, drawn by Latin
  hypercube sampling over the ranges in `INPUT_RANGES` (`surrogate/definitions.py`); step 1 solves exactly these.
  `sample_scenarios` draws new scenarios in the same way.

## Installation

Python 3.9 or 3.10 and the packages in `requirements.txt`, which lists the versions used in the study. A
separate environment keeps them apart from other projects, for example with conda:

```
conda create -n clews-surrogate python=3.10
conda activate clews-surrogate
pip install -r requirements.txt
```

Step 1 also needs [otoole](https://github.com/OSeMOSYS/otoole), GLPK (`glpsol`) and Gurobi (`gurobi_cl`,
with a licence) on the PATH. Steps 2-8 do not need them.

## Running

1. Download the CLEWs data files (see above) and set their paths in `run_all.py`.
2. Check the other settings at the top of `run_all.py`: folders, steps (`STEPS`), parallel workers.
3. Run from a terminal:

```
python run_all.py
```

The settings run several jobs at the same time and suit a computer with 24 cores: 16 scenarios are
solved at once (`SOLVE_WORKERS = 16`) and up to 24 networks are trained at once (`WORKERS`,
`ZAMBIA_WORKERS`). On a computer with fewer cores, lower these numbers; set them all to 1 to run one
job at a time. In the study, solving took on average 0.69 minutes per Zambia scenario and 1.29 minutes
per Ghana scenario on one core.

## Reproducibility

- Steps 1, 2 and 6-8 are deterministic. With `STEPS["verify"] = True`, the data tables built in step 2
  are compared with those of the study (`compare_tables`).
- Steps 3-5 train neural networks. With the same seeds, samples and settings, training is not exactly
  repeatable across computers and package versions, so a new run gives slightly different numbers from
  those in the paper. The settings and package versions of each run are recorded in `run_info.json` of
  each results folder.

## Citation

The full reference will be added after publication.

## Licence

MIT licence, see `LICENSE`.
