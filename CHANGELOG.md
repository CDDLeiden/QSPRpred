# Change Log
From v1.0.0 to v1.1.0

## Fixes

- Fix issue with Mordred descriptor


## Changes

- The FeatureStandardizer was added for scaler fitting, applying, saving and loading
- The Predictor was added to get model predictions for a set of molecules, is compatible with DrugEx scorers
- The standardization of features is now possible with the `standardize_features` argument of `QSPRDataset` by supplying a list of standardizers rather than modifying the matrix directly
    - standardization is now also done separately for training and test sets in cross-validation as well

## New Features

- Tutorials for training and using the QSPR models
- Depiction of results for classification models (see `qsprpred.plotting.classification`)
- The `precomputed` flag was added to `QSPRDataset`
