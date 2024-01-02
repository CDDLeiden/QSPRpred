# Change Log

From v2.1.0 to v3.0.0

## Fixes

- Fixed random seeds to give reproducible results. Each dataset is initialized with a
  single random state (either from the constructor or a random number generator) which
  is used in all subsequent random operations. Each model is initialized with a single
  random state as well: it uses the random state from the dataset, unless it's overriden
  in the constructor. When a dataset is saved to a file so is its random state, which is
  used by the dataset when the dataset is reloaded.
- fixed error with serialization of the `DNNModel.params` attribute, when no parameters
  are set.
- Fix bug with saving predictions from classification model
  when `ModelAssessor.useProba` set to `False`.
- Add missing implementation of `QSPRDataset.removeProperty`
- Improved behavior of the Papyrus data source (does not attempt to connect to the
  internet if the data set already exists).

## Changes

- The whole package was refactored to simplify certain commonly used imports. The
  tutorial code was adjusted to reflect that.
- The jupyter notebooks in the tutorial now pass a random state to ensure consistent
  results.
- The default parameter values for `STFullyConnected` have changed from `n_epochs` =
  1000 to `n_epochs` = 100, from `neurons_h1` = 4000 to `neurons_h1` = 256
  and `neurons_hx` = 1000 to `neurons_hx` = 128.
- Rename `HyperParameterOptimization` to `HyperparameterOptimization`.
- `TargetProperty.fromList` and `TargetProperty.fromDict` now accept a both a string and
  a `TargetTask` as the `task` argument,
  without having to set the `task_from_str` argument, which is now deprecated.
- Make `EarlyStopping.mode` flexible for `QSPRModel.fitAttached`.
- `save_params` argument added to `OptunaOptimization` to save the best hyperparameters
  to the model (default: `True`).
- We now use `jsonpickle` for object serialization, which is more flexible than the
  non-standard approach before, but it also means previous models will not be compatible
  with this version.
- `SklearnMetric` was renamed to `SklearnMetrics`, it now also accepts an scikit-learn
  scorer name as input.
- `QSPRModel.fitAttached` now accepts a `save_model` (default: `True`)
  and `save_dataset` (default: `False`) argument to save the model and dataset to a file
  after fitting.
- Tutorials were completely rewritten and expanded. They can now be found in
  the `tutorials` folder instead of the `tutorial` folder.
- `MetricsPlot` now supports multi-class and multi-task classification models.
- `CorrelationPlot` now supports multi-task regression models.
<<<<<<< CHANGELOG.md
- `RepeatsFilter` argument `year_name` renamed to `time_col` and
  arugment `additional_cols` added.
- The behaviour of `QSPRDataset` was changed with regards to target properties. It now
  remembers the original state of any target property and all changes are performed in
  place on the original property column (i.e. conversion to multi-class classification).
  This is to always maintain the same property name and always have the option to reset
  it to the raw original state (i.e. if we switch to regression or want to repeat a
  transformation).
- The default log level for the package was changed from `INFO` to `WARNING`. A new
  tutorial
  was added to explain how to change the log level.
- `RepeatsFilter` argument `year_name` renamed to `time_col` and arugment `additional_cols` added.
- The behaviour of `QSPRDataset` was changed with regards to target properties. It now remembers the original state of any target property and all changes are performed in place on the original property column (i.e. conversion to multi-class classification). This is to always maintain the same property name and always have the option to reset it to the raw original state (i.e. if we switch to regression or want to repeat a transformation).
- The `perc` argument of `BorutaPy` can now be set from the CLI.

## New Features

- The `qsprpred.benchmarks` module was added, which contains functions to easily
  benchmark
  models on datasets.
- Most unit tests now have a variant that checks whether using a fixed random seed gives
  reproducible results.
- The build pipeline now contains a check that the jupyter notebooks give the same
  results as ones that were observed before.
- Added `FitMonitor`, `AssessorMonitor`, and `HyperparameterOptimizationMonitor` base
  classes to monitor the progress of fitting, assessing, and hyperparameter
  optimization, respectively.
- Added `BaseMonitor` class to internally keep track of the progress of a fitting,
  assessing, or hyperparameter optimization process.
- Added `FileMonitor` class to save the progress of a fitting, assessing, or
  hyperparameter optimization process to files.
- Added `WandBMonitor` class to save the progress of a fitting, assessing, or
  hyperparameter optimization process to [Weights & Biases](https://wandb.ai/).
- Added `NullMonitor` class to ignore the progress of a fitting, assessing, or
  hyperparameter optimization process.
- Added `ListMonitor` class to combine multiple monitors.
- Cross-validation, testing, hyperparameter optimization and early-stopping were made
  more flexible by allowing custom splitting and fold generation strategies. A tutorial
  showcasing these features was created.
- Added a `reset` method to `QSPRDataset`, which resets splits and loads all descriptors
  into the training set matrix again.
- Added `ConfusionMatrixPlot` to plot confusion matrices.
- Added the `searchWithIndex`, `searchOnProperty`, `searchWithSMARTS` and `sample`
  to `MoleculeTable` to facilitate more advanced sampling from data.
- Assessors now have the `split_multitask_scores` flag that can be used to evaluate each
  task seperately with single-task metrics.
- `MoleculeDataSet`s now has the `smiles` property to easily get smiles.
- A Docker-based runner in `testing/runner` can now be used to test GPU-enabled features
  and run the full CI pipeline.
- It is now possible to save `PandasDataTable`s to a CSV file instead of the default
  pickle format (slower, but more human-readable).

## Removed Features

- The `Metric` interface has been simplified in order to make it easier to implement
  custom metrics. The `Metric` interface now only requires the implementation of
  the `__call__` method, which takes predictions and returns a `float`. The `Metric`
  interface no longer requires the implementation
  of `needsDiscreteToScore`, `needsProbaToScore` and `supportsTask`. However, this means
  the base functionality of `checkMetricCompatibility`, `isClassificationMetric`
  and `isRegressionMetric` are no longer available.
- Default hyperparameter search space file, no longer available.
