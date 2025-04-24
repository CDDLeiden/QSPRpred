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

## New Features

- Thanks to the new storage API, standardization of molecules is now more flexible and
  the `PapyrusStandardizer` class was added that provides standardization of molecules
  as done in the Papyrus database.

## Removed Features

None.

