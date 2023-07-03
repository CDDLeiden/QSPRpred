# Change Log

From v2.0.0 to v2.0.1

## Fixes
- Corrected type hinting for `QSPRModel.handleInvalidsInPredictions`, which resulted in an error when importing the package in google colab.

## Changes

## New Features
- raise error if search space for optuna optimization is missing search space type annotation or if type not in list
