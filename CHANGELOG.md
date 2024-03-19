# Change Log

From v3.0.2 to v3.0.3

## Fixes

- Fixed a bug where an attached standardizer would be refit when calling 
`QSPRModel.predictMols` with `use_applicability_domain=True`.
- Fixed a bug with `use_applicability_domain=True` in `QSPRModel.predictMols` 
  where an error would be raised if there were invalid molecules in the input.
- Fixed a bug where dataset type was not properly set to numeric in `MlChemADWrapper.contains`
- Fixed a bug where an attached standardizer would be refit when calling
  `QSPRModel.predictMols` with `use_applicability_domain=True`.
- Fixed random seed not set in `FoldsFromDataSplit.iterFolds` for `ClusterSplit`.

## Changes

- Data type in `MlChemADWrapper` is now set to `float64` by default, instead of `float32`.

## New Features

- Added the `prepMols` method to `DescriptorSet` to allow separated customization of
  molecule preparation before descriptor calculation.

## Removed Features

None.
