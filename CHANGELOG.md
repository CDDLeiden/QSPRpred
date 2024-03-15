# Change Log

From v3.0.2 to v3.0.3

## Fixes

- Fixed a bug where an attached standardizer would be refit when calling
  `QSPRModel.predictMols` with `use_applicability_domain=True`.
- Fixed random seed not set in `FoldsFromDataSplit.iterFolds` for `ClusterSplit`.

## Changes

- The module containing the sole model base class (`QSPRModel`) was renamed
  from `models` to `model`.

## New Features

None.

## Removed Features

None.
