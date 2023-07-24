# Change Log

From v2.0.0 to v2.0.1

## Fixes

- Requirement python version in pyproject.toml updated to 3.10, as older version of python don't support the type hinting used in the code.
- Corrected type hinting for `QSPRModel.handleInvalidsInPredictions`, which resulted in an error when importing the package in google colab.
- The `predictMols` method returned random predictions in v2.0.0 due to unpatched shuffling code. This has now been fixed.
- fixed error with serialization of the `DataFrameDescriptorSet` (#63)
- Papyrus descriptors are not fetched by default anymore from the `Papyrus`  adapter, which caused fetching of unnecessary data.

## Changes

## New Features
- raise error if search space for optuna optimization is missing search space type annotation or if type not in list
