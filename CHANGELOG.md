# Change Log

From v3.0.1 to v3.0.2

## Fixes

- Fixed a bug in `QSPRDataset` where property transformations were not applied.
- Fixed a bug where an attached standardizer would be refit when calling
  `QSPRModel.predictMols` with `use_applicability_domain=True`.
- Fixed random seed not set in `FoldsFromDataSplit.iterFolds` for `ClusterSplit`.

## Changes

- Restrictions on `numpy` versions were removed to allow for more flexibility in
  package installations. However, the `BorutaFilter` feature selection method does not
  function with `numpy` versions 1.24.0 and above. Therefore, this functionality now
  requires a downgrade to `numpy` version 1.23.0 or lower. This was reflected in the
  documentation and `numpy` itself outputs a reasonable error message if the version is
  incompatible.

## New Features

- The `DataFrameDescriptorSet` class was extended to allow more flexibility when joining
  custom descriptor sets.
- Added the `prepMols` method to `DescriptorSet` to allow separated customization of
  molecule preparation before descriptor calculation.
- The package can now be installed from the PyPI repository 🐍📦.

## Removed Features

None.
