# Change Log

From v1.3.1 to v2.0.0

## Fixes

- more robust error handling of invalid molecules in `MoleculeTable`
- Not all scorers in `supported_scoring` were actually working in the multi-class case, the scorer support is now
  divided by single and multiclass support (moved to `metrics.py`, see also New Features).
- Instead of all smiles, only invalid smiles are now printed to the log when they are removed.
- problems with PaDEL descriptors and fingerprints on Linux were fixed
- fixed a serialization issue with `DataFrameDescriptorSet` 

## Changes

- Hyperparameter optimization moved to a separate class from `QSPRModel.bayesOptimization` and `QSPRModel.gridSearch` to `OptunaOptimization` and `GridSearchOptimization` in the new module `qsprpred.models.param_optimzation` with a base clase `HyperParameterOptimization` in `qsprpred.models.interfaces`.
- ⚠️ Important! ⚠️ `QSPRModel` attribute `model` now called `estimator`, which is always an instance of `alg`, while `alg` may no longer be an instance but only a Type.
- Converting input data for `qsprpred.models.neural_network.Base` to dataloaders now executed in the `fit` and `predict` functions instead of in the `qspred.deep.models.QSPRDNN` class.
- `MoleculeTable` now uses a custom index. When a `MoleculeTable` is created a new column (`QSPRID`) is added (overwritten if already present), which is then used as the index of the underlying data frame.
  - It is possible to override this with a custom index by passing `index_cols` to the `MoleculeTable` constructor. These columns will be then used as index or a multi-index if more than one column is passed.
  - Due to this change, `scaffoldsplit` now uses these IDs instead of unreliable SMILES strings (see documentation for the new API). 
- If there are invalid molecules in `MoleculeTable`, `addDescriptors` now fails by default. You can disable this by passing `fail_on_invalid=False` to the method.
- To support multitask modelling, the representation of the target in the `QSPRdataset` has changed to a list of 
  `TargetProperty`s (see New Features). These can be automatically initizalid from dictionaries in the `QSPRdataset`
  init.
- A `fill_value` argument was also added to the `predict_CLI` script to allow for filling missing values in the
  prediction data set as well.
- ⚠️ Important! ⚠️ `setup.py` and `setup.cfg` were substituted with `pyproject.toml` and `MANIFEST.in`. A lighter version of the package is now the default installation option!!!
  - Installation options for the optional dependencies are described in README.md
  - CI scripts were modified to test the package on the full version. See changes in `.gitlab-ci.yml`.
  - Features using the extra dependencies were moved to `qsprpred.extra` and `qsprpred.deep` subpackages. The structure of the subpackages is the same as of the main package, so you just need to remember to use `qsprpred.extra` or `qsprpred.deep` instead of just `qsprpred` in your imports if you were using these features from the main package before. 
- The way descriptors are stored in `MoleculeTable` was changed. They now reside in their own `DescriptorTable` instances that are linked to the orginal `MoleculeTable`
  - This change was made to allow several types of descriptors to be calculated and used efficiently (facilitated by a the `DescriptorsCalculators` interface)
  - Unfortunately, this change is not backwards compatible, so previously pickled `MoleculeTable` instances will not work with this version. There were also changes to how models handle multiple descriptor types, which also makes them incompatible with previous versions. However, this can be fixed by modifying the old JSON files as illustrated in commits 7d3f8633 and 6564f024.
- 'LowVarianceFilter` now includes boundary in the filtered features, e.g. if threshold is 0.1, also features that
  have a variance of 0.1 will be removed.

## New Features
- New feature split `ManualSplit` for splitting data by a user-defined column
- The index of the `MoleculeTable` can now be used to relate cross-validation and test outputs to the original molecules. Therefore, the index is now also saved in the model training outputs.
- the `Papyrus.getData()` method now accepts `activity_types` parameter to select a list of activity types to get.
- Added the `checkMols` method to `MoleculeTable` to use for indication of invalid molecules in the data.
- Support for Sklearn Multitask modelling
- New class abstract class `Metric`, which is an abstract base class that allows for creating custom scorers.
- Subclass `SklearnMetric` of the `Metric` class, this class wraps the sklearn metrics, to allow for checking 
  the compatibility of each Sklearn scoring function with the `QSPRSklearn` model type.
- New class `TargetProperty`, to allow for multitask modelling, a `QSPRdataset` has to have the option of multiple
  targetproperties. To support this a targer property is now defined seperatly from the dataset as a `TargetProperty`
  instance, which holds the information on name,  `TargetTask` (see also Changes) and threshold of the property.
- Support for protein descriptors and PCM modeling was added.
  - The `PCMDataset` class was introduced that extends `QSPRDataset` and adds the `addProteinDescriptors` method, which can be used to calculate protein descriptors by linking information from the table with sequencing data.
- Support for precalculated descriptors was added with `addCustomDescriptors` method of `MoleculeTable`.
  - It allows for adding precalculated descriptors to the `MoleculeTable` by linking the information from the table with external precalculated descriptors.
- The [tutorial](tutorial) was improved with more detailed sections on data preparation and PCM modelling added.
