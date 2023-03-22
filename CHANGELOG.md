# Change Log

From v1.3.1 to v1.4.0

## Fixes

- more robust error handling of invalid molecules in `MoleculeTable`
- Not all scorers in `supported_scoring` were actually working in the multi-class case, the scorer support is now
  divided by single and multiclass support (moved to `metrics.py`, see also New Features).
- Instead of all smiles, only invalid smiles are now printed to the log when they are removed.
- problems with PaDEL descriptors and fingerprints on Linux were fixed

## Changes

- `MoleculeTable` now uses a custom index. When a `MoleculeTable` is created a new column (`QSPRID`) is added (overwritten if already present), which is then used as the index of the underlying data frame.
  - It is possible to override this with a custom index by passing `index_cols` to the `MoleculeTable` constructor. These columns will be then used as index or a multi-index if more than one column is passed.
  - Due to this change, `scaffoldsplit` now uses these IDs instead of unreliable SMILES strings (see documentation for the new API). 
- If there are invalid molecules in `MoleculeTable`, `addDescriptors` now fails by default. You can disable this by passing `fail_on_invalid=False` to the method.
- To support multitask modelling, the representation of the target in the `QSPRdataset` has changed to a list of 
  `TargetProperty`s (see New Features). These can be automatically initizalid from dictionaries in the `QSPRdataset`
  init.

## New Features

- The index of the `MoleculeTable` can now be used to relate cross-validation and test outputs to the original molecules. Therefore, the index is now also saved in the model training outputs.
- Added the `checkMols` method to `MoleculeTable` to use for indication of invalid molecules in the data.
- Support for Sklearn Multitask modelling
- New class abstract class `Metric`, which is an abstract base class that allows for creating custom scorers.
- Subclass `SklearnMetric` of the `Metric` class, this class wraps the sklearn metrics, to allow for checking 
  the compatibility of each Sklearn scoring function with the `QSPRSklearn` model type.
- New class `TargetProperty`, to allow for multitask modelling, a `QSPRdataset` has to have the option of multiple
  targetproperties. To support this a targer property is now defined seperatly from the dataset as a `TargetProperty`
  instance, which holds the information on name,  `TargetTask` (see also Changes) and threshold of the property.
