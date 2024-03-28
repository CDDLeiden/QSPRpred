# Change Log

From v3.0.1 to v3.0.2

## Fixes

- Fixed a bug where an attached standardizer would be refit when calling
  `QSPRModel.predictMols` with `use_applicability_domain=True`.
- Fixed a bug with `use_applicability_domain=True` in `QSPRModel.predictMols`
  where an error would be raised if there were invalid molecules in the input.
- Fixed a bug where dataset type was not properly set to numeric
  in `MlChemADWrapper.contains`
- Fixed a bug in `QSPRDataset` where property transformations were not applied.
- Fixed a bug where an attached standardizer would be refit when calling
  `QSPRModel.predictMols` with `use_applicability_domain=True`.
- Fixed random seed not set in `FoldsFromDataSplit.iterFolds` for `ClusterSplit`.
- Fixed a bug where class ratios were shuffled in the `RatioDistributionAlgorithm`.

## Changes

- The module containing the sole model base class (`QSPRModel`) was renamed
  from `models` to `model`.
- Restrictions on `numpy` versions were removed to allow for more flexibility in
  package installations. However, the `BorutaFilter` feature selection method does not
  function with `numpy` versions 1.24.0 and above. Therefore, this functionality now
  requires a downgrade to `numpy` version 1.23.0 or lower. This was reflected in the
  documentation and `numpy` itself outputs a reasonable error message if the version is
  incompatible.
- Data type in `MlChemADWrapper` is now set to `float64` by default, instead
  of `float32`.
- Saving of models after hyperparameter optimization was improved to ensure parameters
  are always propagated to the underlying estimator as well.

## New Features

- The `DataFrameDescriptorSet` class was extended to allow more flexibility when joining
  custom descriptor sets.
- Added the `prepMols` method to `DescriptorSet` to allow separated customization of
  molecule preparation before descriptor calculation.
- The package can now be installed from the PyPI repository 🐍📦.
- New argument (`refit_optimal`) was added to `HyperparameterOptimization.optimize()`
  method to make refitting of the model with optimal parameters easier.

## Removed Features

None.
