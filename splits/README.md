Fixed splits of the study (read by run_all.py; new splits are drawn only if these files are missing,
and the results then differ from the paper).

zambia/
  pool_ids.csv          the 7,000 development scenarios (tuning uses only these)
  test_ids.csv          the 1,000 test scenarios
  calibration_ids.csv   500 of the development scenarios, in a fixed order ("order"); the first
                        250 are used for the prediction intervals
  train_pool_ids.csv    the other 6,500 development scenarios; the training samples are drawn from these
ghana/
  test.csv              the 400 test scenarios
  calibration.csv       300 calibration scenarios in a fixed order; the first 250 are used
  pool.csv              the remaining 1,300 scenarios; the training samples are drawn from these
