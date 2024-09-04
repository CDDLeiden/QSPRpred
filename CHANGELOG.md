# Change Log

From v3.2.0 to v4.0.0

## Fixes

None.

## Changes

- Renamed `QSPRDataset` to `QSPRTable` to better reflect its purpose.
- New API definition (`ChemStore`) was added. It describes a chemical storage system for
  easier interoperability between QSPRpred and other packages.
  The `TabularStorageBasic` implementation was added to provide storage
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

