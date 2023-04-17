# Change Log
From v1.3.1 to v1.4.0

## Fixes

None.

## Changes

- `MoleculeTable` now uses a custom index. When a `MoleculeTable` is created a new column (`QSPRID`) is added (overwritten if already present), which is then used as the index of the underlying data frame.
  - It is possible to override this with a custom index by passing `index_cols` to the `MoleculeTable` constructor. These columns will be then used as index or a multi-index if more than one column is passed.
  - Due to this change, `scaffoldsplit` now uses these IDs instead of unreliable SMILES strings (see documentation for the new API). 

## New Features

- The index of the `MoleculeTable` can now be used to relate cross-validation and test outputs to the original molecules. Therefore, the index is now also saved in the model training outputs.
- the `Papyrus.getData()` method now accepts `activity_types` parameter to select a list of activity types to get.
