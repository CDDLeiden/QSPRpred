# Change Log

From v2.1.0.a2 to v2.2.0

## Fixes
- Fixed random seeds to give reproducible results. Each dataset is initialized with a single random state (either from the constructor or a random number generator) which is used in all subsequent random operations. Each model is initialized with a single random state as well: it uses the random state from the dataset, unless it's overriden in the constructor. When a dataset is saved to a file so is its random state, which is used by the dataset when the dataset is reloaded.
- fixed error with serialization of the `DNNModel.params` attribute, when no parameters are set.
- Fix bug with saving predictions from classification model when `ModelAssessor.useProba` set to `False`.
- Add missing implementation of `QSPRDataset.removeProperty`

## Changes
- The jupyter notebooks now pass a random state to ensure consistent results.
- The default parameter values for `STFullyConnected` have changed from `n_epochs` = 1000 to `n_epochs` = 100, from `neurons_h1` = 4000 to `neurons_h1` = 256 and `neurons_hx` = 1000 to `neurons_hx` = 128.
- Rename `HyperParameterOptimization` to `HyperparameterOptimization`.
- `TargetProperty.fromList` and `TargetProperty.fromDict` now accept a both a string and a `TargetTask` as the `task` argument,
without having to set the `task_from_str` argument, which is now deprecated.
- `save_params` argument added to `OptunaOptimization` to save the best hyperparameters to the model (default: `True`).

## New Features
- Most unit tests now have a variant that checks whether using a fixed random seed gives reproducible results.
- The build pipeline now contains a check that the jupyter notebooks give the same results as ones that were observed before.
- Added `FitMonitor`, `AssessorMonitor`, and `HyperparameterOptimizationMonitor` base classes to monitor the progress of fitting, assessing, and  hyperparameter optimization, respectively.
- Added `BaseMonitor` class to internally keep track of the progress of a fitting, assessing, or hyperparameter optimization process.
- Added `FileMonitor` class to save the progress of a fitting, assessing, or hyperparameter optimization process to files.
- Added `WandBMonitor` class to save the progress of a fitting, assessing, or hyperparameter optimization process to [Weights & Biases](https://wandb.ai/).
- Added `NullMonitor` class to ignore the progress of a fitting, assessing, or hyperparameter optimization process.
- Added `ListMonitor` class to combine multiple monitors.
- Cross-validation, testing, hyperparameter optimization and early-stopping were made more flexible by allowing custom splitting and fold generation strategies. A tutorial showcasing these features was created. 

## Removed Features
