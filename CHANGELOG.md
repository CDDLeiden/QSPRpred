# Change Log
From v1.0.0 to v1.1.0

## Fixes

- Fix issue with Mordred descriptor
- Descriptor sets now process a list of molecules instead of just one at a time (prevents performance issues if multiple sets are calculated in parallel)


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
- The `DescriptorSet` interface was updated and all built-in descriptors were adapted to reflect this change. 
  - The presence of `descriptors` property getter and setter is now enforced.
  - When called the `DescriptorSet` implementations now strictly return lists.
  - Conversion to descriptor data frame is now handled exclusively in `DescriptorsCalculator`
- Default `chunk_size` for `MoleculeTable.addDescriptors` was set to 50 so that smaller data sets can take advantage of more CPUs as well.
- The `datasplit` interface was changed to mimic the `sklearn.model_selection.BaseCrossValidator` interface so all `sklearn` cross-validation methods can be used with QSPRPred out of the box to either generate train/test split or cross-validation splits (see the new features below)
- Default `chunk_size` for `MoleculeTable` was set to 50 so that smaller data sets can take advantage of more CPUs as well.
- The number of CPUs to use for parallel operations  by `MoleculeTable` is now set in the `__init__` of the class
- `DescriptorSets` are now initialized with the specific arguments instead of args and kwargs.
- `MorganFP` was replaced by a more general class `FingerprintSet` which uses an object from the `Fingerprint` class as its fingerprint type
- Passing invalid SMILES to the `getScores` function of  `Predictor` will now return a score of None. 

## New Features

- Tutorials for training and using the QSPR models
- Depiction of results for classification models (see `qsprpred.plotting.classification`)
- The `precomputed` flag was added to `QSPRDataset`
- Added an option to directly fetch `QSPRDataset` from Papyrus with accession IDs (see `qsprpred.data.sources.papyrus`)
- The `datasplit` interface is now used to both generate train/test split and also the cross-validation splits
- Train/test split of the data set is now saved in the matrix itself and is reloaded upon deserialization
- `MoleculeTable` was updated with new features to generate scaffolds of molecules
- `TanimotoDistances` was added as descriptortype.
- Balanced class weighing was added as an option to the CLI
- `PredictorDesc` was added as a new `DescriptorSet` type. It uses a QSPRpred model as descriptor.
