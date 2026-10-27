"""
Functions of the study
  "Surrogate Modeling for Fast Scenario Exploration in Climate-Land-Water-Energy Systems:
   From Zambia to Ghana via Transfer Learning":
machine-learning surrogates of CLEWs models for scenario exploration, with transfer learning from Zambia
to Ghana. Every function is defined once, in one of the modules below;
run_all.py holds the settings (folders, steps to run, parallel workers) and runs the whole study from
start to end.

Modules
  definitions     inputs, case studies, output categories, evaluation criteria
  utils           run ids, parallel jobs, splits and samples
  scenarios       solver files, modifying and solving the CLEWs scenarios      (step 1)
  data            reading the results, total discounted cost, splits           (step 2)
  measures        WMAPE, R2, prediction intervals, pair accuracy
  networks        the training loop shared by all neural networks
  tuning          hyperparameter search of the Zambia surrogate                 (step 3)
  zambia          Zambia surrogates, saved models                               (step 4)
  transfer        Zambia -> Ghana, with and without transfer                    (step 5)
  evaluation      pair accuracy, levels of use                                  (step 6)
  demonstration   scenario exploration with the 1,000-run surrogate             (step 7)
  figures         figures and tables of the paper                               (step 8)

Sources
  CLEWs models   C. Lo Giudice and M. Mulunda, CLEWs Zambia dataset, Zenodo, 2026, doi:10.5281/zenodo.19206729
                 S. G. Savanur, Ghana CLEWs model and dataset, Zenodo, 2026, doi:10.5281/zenodo.20267395
  Solving        OSeMOSYS-Solver-script, S. Kumar, MIT licence,
                 https://github.com/ShravanKumar23/OSeMOSYS-Solver-script: model.v.5.2.txt, config_com.yaml
                 and preprocess_data.py are downloaded from it (get_solver_files), and the solution steps of
                 its Run_CLEWs_5.2.py are followed (fix_data_file, solve_scenario).

Paths are passed as a dict P (built in run_all.py) with the keys
  clews       folder for model.v.5.2.txt, config_com.yaml, preprocess_data.py
  data_files  {country: CLEWs data file (.txt)}
  scenarios   {country: scenario table (.csv)}
  runs        {country: [folders with the solved scenarios, <folder>/<run_id>/results/]}
  splits, temp, data, tuning, results, figures   folders read or written by the steps
"""
