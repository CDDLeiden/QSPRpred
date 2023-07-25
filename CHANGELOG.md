# Change Log

From v2.0.0 to v2.0.1

## Fixes

- Requirement python version in pyproject.toml updated to 3.10, as older version of python don't support the type hinting used in the code.
- Corrected type hinting for `QSPRModel.handleInvalidsInPredictions`, which resulted in an error when importing the package in google colab.
- The `predictMols` method returned random predictions in v2.0.0 due to unpatched shuffling code. This has now been fixed.
- fixed error with serialization of the `DataFrameDescriptorSet` (#63)
- Papyrus descriptors are not fetched by default anymore from the `Papyrus`  adapter, which caused fetching of unnecessary data.

## Changes
- `PCMSplit` replaces `StratifiedPerTarget` and is compatible with `RandomSplit`, `ScaffoldSplit` and `ClusterSplit`.
- In the case single-task dataset, the `RandomSplit` now uses `StratifiedShuffleSplit` in case of classification.

## New Features
- `ClusterSplit` - splits data based clustering of molecular fingerprints.
- raise error if search space for optuna optimization is missing search space type annotation or if type not in list

## Removed Features
- `StratifiedPerTarget` is replaced by `PCMSplit`.
