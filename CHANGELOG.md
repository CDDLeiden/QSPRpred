# Change Log

From v3.2.1 to v4.0.0

## Fixes

- The random state was not set correctly in the `DNNModel` class. `torch.manual_seed`
  was only called when initializing the model, but not when calling `loadEstimator`.
  This could lead to different results when running the same assessment multiple times 
  in the same session. Thus, results generated with older versions were reproducible 
  across sessions, but not within the same session.
- Fix conversion of continous target properties to classification targets in the `QSPRTable`
  class when missing values are present. The conversion was not done correctly, where
  NaN values were converted to a class label, which is not desired. Now, NaN values are
  ignored during conversion.
- When initializing a `QSPRModel` without setting `random_state` a message would be
  displayed that the random state was set to a random integer. However, the random
  state would not actually be set. This message is now removed.

## Changes

- Renamed `QSPRDataset` to `QSPRTable` to better reflect its purpose.
- New API definition (`ChemStore`) was added. It describes a chemical storage system for
  easier interoperability between QSPRpred and other packages.
  The `PandasChemStore` implementation was added to provide storage
  for `MoleculeTable` and `QSPRTable`, which still function as before, but take a
  storage object for initialization. As a result the `fromDF` method
  of `MoleculeTable`/`QSPRTable` now serves as a factory method for creating the
  objects from a `pandas.DataFrame`. This is now covered more in-depth in
  the [data representation tutorial](./tutorials/basics/data/data_representation.ipynb).
- A new argument `drop_empty_target_props` was added to the `QSPRTable.fromDF` method to
  pass through to the init of `QSPRTable`.
- Renamed `TargetProperty` to `TargetSpec` to better reflect its purpose and avoid
  confusion with properties as referred to in the context of `MoleculeTable` and 
  `QSPRTable`. The `QSPRTable.getTargetProperties` function was renamed to 
  `QSPRTable.getTargetSpecs` and a function `QSPRTable.getTargetSpec` was added to 
  retrieve the `TargetSpec` of a single specified target property.
- `QSPRTable.getTargetPropertyNames` was renamed to `QSPRTable.getTargetPropertiesNames`
- `SKlearnStandardizer` was replaced by `SklearnStep` which makes use of the new
  `Step` API and accepts any type of scikit-learn transformer that implements the
  `fit` and `transform` methods.
- `QSPRTable.dropEmptyProperties` was removed and this functionality is now covered by
  the new `MoleculeTable.dropEmptyEntries` method.
- `MLChemADWrapper` was renamed to `MLChemAD`
- `TemporalPerTarget` split arguments `year_col`, `split_years`, 
  `firts_year_per_compound`, and `dataset` were renamed to `time_prop`, `split_time`,
  `first_time_per_compound`, and `data_set`, respectively.
- `CrossValAssessor` and `TestSetAssessor` are replaced by one class `Assessor`. which 
  provides a unified interface for assessing models on different data splits. The 
  `Assessor` class can be used with any `DataSetPipeline` and `DataSplit`. Also 
  training set predictions are now saved in addition to the test set predictions.
- The `QSPRTable.split` method no longer saves the train-test split, but returns the
  an generator that can iterate over the train-test indices for multiple folds.
  To save splits to the `QSPRTable` a function `addSplit` was added. Now multiple
  different splits can be added to a `QSPRTable` instance, which may be retrieved using
  the new `iterSplit` method by name. For more information, see the 
  [data splitting tutorial](./tutorials/basics/data/data_splitting.ipynb).
- Option was added to `ManualSplit` that makes it possible to return multiple splits. 
  Is used when `splitprop` is list.


## New Features

- Thanks to the new storage API, standardization of molecules is now more flexible and
  the `PapyrusStandardizer` class was added that provides standardization of molecules
  as done in the Papyrus database.
- A new class `Pipeline` was added. It provides a way to chain together multiple
  data processing steps operating on `Pandas.DataFrame` objects into a single workflow.
  In extension a class `DataSetPipeline` was added which operates on `QSPRTable` objects.
  See the [data preparation tutorial](./tutorials/basics/data/data_preparation.ipynb) for
  examples. These replace the `prepareDataset` method of `QSPRTable`.
- A new API definition (`Step`) was added. It describes a single processing step in a
  data pipeline and ensures all preprocessing steps can easily be applied in a
  customizable but consistent way.
- `DummyStep` (does nothing) and `Shuffle` (shuffles the entries) were added as basic 
  `Step` data processing steps.
- All implementations of `DataFilter` (`CategoryFilter` and `RepeatsFilter`)
  now conform to the `Step` API. Also `NaNFilter` (removes rows with NaN values) and
  `OutlierFilter` (removes rows with outlier feature values) were added.
- All implementations of `FeatureFilter` (`LowVarianceFilter`, `HighCorrelationFilter`,
  `BorutaFilter`) now conform to the `Step` API.
- A new API definition `Imputer` (implementation of `Step` API) was added to handle 
  missing values in the dataset. `TargetImputer` and `FeatureImputer` classes were 
  introduced to specifically address missing values in target and feature columns.
- A new API definition `TargetTransformer` (implementation of `Step` API) was added to
  transform target variables in the dataset. `SimpleTargetTransformer` was added which
  provides a way to apply a number of basic transformations to the target variables (
  e.g. log transformation).
- The `QSPRTable.getTargets` and `QSPRTable.getTarget` functions were added, which
  return the values of the specified target properties/property.
- A new descriptor set `RandomDescs` was added. It returns random numbers as
  descriptors for testing purposes.

## Removed Features
- The `prepareDataset` method of `QSPRTable` is removed in favour of the new
  `DatasetPipeline` class, which provides a more flexible way to preprocess data for
  QSPR modeling. See the [New Features](#new-features) section for more information.
- The `QSPRTable.hasFeatures`, `QSPRTable.getFeatureNames`, `QSPRTable.getFeatures`,
  `QSPRTable.shuffle`, `QSPRTable.fillMissing`, `QSPRTable.filterFeatures`,
  `QSPRTable.transformProperties`, `QSPRTable.imputeProperties`\
  `MoleculeTable.imputeProperties`, `QSPRTable.getApplicability`, 
  `QSPRTable.dropOutliers`, and `QSPRTable.setApplicabilityDomain` methods as well as 
  the `QSPRTable.X`, `QSPRTable.y`, `QSPRTable.X_ind`, and `QSPRTable.y_ind` attributes 
  were removed due to the change to the new `DataSetPipeline` class mentioned above. 
  Alternatives to these functions are available as `Step` objects in the new pipeline 
  framework.


  