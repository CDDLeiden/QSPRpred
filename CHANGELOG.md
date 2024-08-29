# Change Log

From v3.1.1 to v3.2.0

## Fixes

- Fixed a bug in `ChempropModel` that caused it not to work with missing values in the
  target column.

## Changes

- `calibration_score` is now implemented under the `Metric` class as `CalibrationScore`.

## New Features

- Added a range of new
  metrics: `BEDROC`, `EnrichmentFactor`, `RobustInitialEnhancement`,
  `Prevalence`, `Sensitivity`, `Specificity`, `PositivePredictivity`, `NegativePredictivity`,
  `CohenKappa`, `BalancedPositivePredictivity`, `BalancedNegativePredictivity`,
  `BalancedMatthewsCorrcoeff`, `BalancedCohenKappa`, `KSlope`, `R20`, `KPrimeSlope`,
  `RPrime20`, `Pearson`, `Spearman`, `Kendall`, `AverageFoldError`,
  `AbsoluteAverageFoldError`, `PercentageWithinFoldError`
- Added `MaskedMetric` which can be wrapped around any metric to mask datapoints
  when a target value is missing.
- Added a tutorial on model and data serialization.
- `ApplicabilityDomain` now has a `transform` method that can be used to transform
  a dataset to a continuous applicability domain score, such as the distance to the
  nearest neighbor in the training set (an example was added to the
  [tutorials](./tutorials/basics/data/applicability_domain.ipynb)).

## Removed Features

None.

