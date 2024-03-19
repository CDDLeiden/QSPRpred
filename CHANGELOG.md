# Change Log

From v3.0.1 to v3.0.2

## Fixes

- Fixed a bug in `QSPRDataset` where property transformations were not applied.
- Fixed a bug where an attached standardizer would be refit when calling
  `QSPRModel.predictMols` with `use_applicability_domain=True`.
- Fixed random seed not set in `FoldsFromDataSplit.iterFolds` for `ClusterSplit`.

## Changes

None.

## New Features

- The `DataFrameDescriptorSet` class was extended to allow more flexibility when joining
  custom descriptor sets.
- Added the `prepMols` method to `DescriptorSet` to allow separated customization of
  molecule preparation before descriptor calculation.

## Removed Features

None.
