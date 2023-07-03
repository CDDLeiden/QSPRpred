# Change Log

From v2.0.0 to v2.0.1

## Fixes

- Requirement python version in pyproject.toml updated to 3.10, as older version of python don't support the type hinting used in the code.
- Corrected type hinting for `QSPRModel.handleInvalidsInPredictions`, which resulted in an error when importing the package in google colab.

## Changes

## New Features
- raise error if search space for optuna optimization is missing search space type annotation or if type not in list
