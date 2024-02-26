# Change Log

From v3.0.0 to v3.0.1

## Fixes
- Fixed a bug in `QSPRDataset` where property transformations were not applied.

## Changes

- renamed `PandasDataTable.transform` to `PandasDataTable.transformProperties`
- moved `imputeProperties`, `dropEmptyProperties` and `hasProperty` from `MoleculeTable`
  to `PandasDataTable`.
- removed `getProperties`, `addProperty`, `removeProperty`, now use `PandasDataTable`
  methods directly.

## New Features

## Removed Features
