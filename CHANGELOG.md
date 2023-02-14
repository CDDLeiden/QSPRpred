# Change Log
From v1.1.0 to v1.2.0

## Fixes

- Fix issue with Mordred descriptor
- Descriptor sets now process a list of molecules instead of just one at a time (prevents performance issues if multiple sets are calculated in parallel)
- Empty values of descriptors are now not imputed with 0 automatically, but are left as `NaN` or `None` instead


## Changes

- Some features not specific to machine learning were extracted from `QSPRDataset` to a new class called `MoleculeTable`
  - `MoleculeTable` is mainly to hold data about molecules, including their  descriptors, scaffolds, bioactivities and other data
    - this class also now manages settings for parallelization and chunking in the constructor rather than on per method basis
    - this class will be used as the base class for other data set classes that need molecule data, but have to perform their own transformations to do their job
  - `QSPRDataset` derives from `MoleculeTable` an object describing the training and test set for modelling and also handles data preparation 
- `QSPRDataset` now handles saving of its metadata and other related files (i.e. standardizers and other data transformers) with one method (`save`) -> names of the files start with a chosen prefix, which is a name given to the data set 
- The `SKLearnStandardizer` was added for scaler fitting, applying, saving and loading
  - The standardization of features is now possible with the `feature_standardizer` argument of `QSPRDataset.prepareDataset` by supplying an instance of `SKLearnStandardizer` or directly a `StandardScaler` or any other standardizer from `sklearn.preprocessing` with `BaseEstimator` interface
      - standardization is now also done separately for training and test sets in cross-validation as well
- The `DescriptorSet` interface was updated and all built-in descriptors were adapted to reflect this change. 
  - The presence of `descriptors` property getter and setter is now enforced.
  - When called the `DescriptorSet` implementations now strictly return lists.
  - Conversion to descriptor data frame is now handled exclusively in `DescriptorsCalculator`
- The `datasplit` interface was changed to mimic the `sklearn.model_selection.BaseCrossValidator` interface so all `sklearn` cross-validation methods can be used with QSPRPred out of the box to either generate train/test split or cross-validation splits (see the new features below)
- Default `chunk_size` for `MoleculeTable` was set to 50 so that smaller data sets can take advantage of more CPUs as well.
- The number of CPUs to use for parallel operations  by `MoleculeTable` is now set in the `__init__` of the class and is 1 by default so that the default behaviour is to not use parallelism.
- `DescriptorSets` are now initialized with the specific arguments instead of args and kwargs.
- `MorganFP` was replaced by a more general class `FingerprintSet` which uses an object from the `Fingerprint` class as its fingerprint type
- The `Predictor` class was replaced, its features are now accessible with the models directly:
  - ```python
    from qsprpred import QSPRsklearn # QSPRDNN can be used the same way
    from qsprpred import QSPRDataset
    
    # creation and loading 
    model = QSPRsklearn( # or QSPRDNN
        name="any_name",
        base_dir="/some/path"
    )
  
    # loading directly from meta file also possible 
    model = QSPRsklearn.fromFile("/path/to/any_name_meta.json")
  
    # predictions can be done directly on a list of SMILES
    model.predictMols([
        'CC(=C)C1CC2=C(O1)C=CC3=C2OC4COC5=CC(=C(C=C5C4C3=O)OC)OC',
        'CCOC(=O)C1=C2CN(C(=O)C3=C(N2C=N1)C=CC(=C3)F)C'
    ])
  
    # classifiers can also use predict_probas=True to get probablities
    model.predictMols([
        'CC(=C)C1CC2=C(O1)C=CC3=C2OC4COC5=CC(=C(C=C5C4C3=O)OC)OC',
        'CCOC(=O)C1=C2CN(C(=O)C3=C(N2C=N1)C=CC(=C3)F)C'
    ], use_probas=True)
  
    # it is also possible to give a QSPRDataset directly:
    dataset = QSPRDataset(name="data")
    model.predict(dataset)

    ```
  - Calls to `predict`, `predictProba` or `predictMols` with `use_probas=True` will now return a score of `None` for invalid molecules.

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
- New submodule for evaluation metric custom (`qsprpred.metrics`) with `calibration_error` function to estimate the calibration of a classifier
- Added the [Mold2](https://pubs.acs.org/doi/10.1021/ci800038f) and [PaDEL](https://onlinelibrary.wiley.com/doi/10.1002/jcc.21707) molecular descriptors
