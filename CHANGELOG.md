# Change Log

From v2.0.0.dev1 to v2.0.0.dev2

## Fixes


## Changes

- Hyperparameter optimization moved to a separate class from `QSPRModel.bayesOptimization` and `QSPRModel.gridSearch` to `OptunaOptimization` and `GridSearchOptimization` in the new module `qsprpred.models.param_optimzation` with a base clase `HyperParameterOptimization` in `qsprpred.models.interfaces`.
- `QSPRModel` attribute `model` now called `estimator`, which is always an instance of `alg`, while `alg` may no longer be an instance but only a Type.
- Converting input data for `qsprpred.models.neural_network.Base` to dataloaders now executed in the `fit` and `predict` functions instead of in the `qspred.deep.models.QSPRDNN` class.
- 'LowVarianceFilter` now includes boundary in the filtered features, e.g. if threshold is 0.1, also features that
  have a variance of 0.1 will be removed.

## New Features

