# Change Log
From v1.0.0 to v1.1.0

## Fixes

- Fix issue with Mordred descriptor


## Changes

- Some features not specific to machine learning were extracted from `QSPRDataset` to a new class called `MoleculeTable`
  - `MoleculeTable` is mainly to hold data about molecules, including their  descriptors, scaffolds, bioactivities and other data
    - this class will be used as the base class for other data set classes that need molecule data, but have to perform their own transformations to do their job
  - `QSPRDataset` derives from `MoleculeTable` an object describing the training and test set for modelling and also handles data preparation 
- `QSPRDataset` now handles saving of its metadata and other related files (i.e. standardizers and other data transformers) with one method (`save`) -> names of the files start with a chosen prefix, which is a name given to the data set 
- The FeatureStandardizer was added for scaler fitting, applying, saving and loading
- The Predictor was added to get model predictions for a set of molecules, is compatible with DrugEx scorers
- The standardization of features is now possible with the `standardize_features` argument of `QSPRDataset` by supplying a list of standardizers rather than modifying the matrix directly
    - standardization is now also done separately for training and test sets in cross-validation as well

## New Features

- Tutorials for training and using the QSPR models
- Depiction of results for classification models (see `qsprpred.plotting.classification`)
- The `precomputed` flag was added to `QSPRDataset`
