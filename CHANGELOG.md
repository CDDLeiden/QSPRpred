# Change Log

From v3.0.2 to v3.1.0

## Fixes

- Fixed a bug in `QSPRDataset` where property transformations were not applied.
- Fixed a bug where an attached standardizer would be refit when calling
  `QSPRModel.predictMols` with `use_applicability_domain=True`.
- Fixed random seed not set in `FoldsFromDataSplit.iterFolds` for `ClusterSplit`.

## Changes

- renamed `PandasDataTable.transform` to `PandasDataTable.transformProperties`
- moved `imputeProperties`, `dropEmptyProperties` and `hasProperty` from `MoleculeTable`
  to `PandasDataTable`.
- removed `getProperties`, `addProperty`, `removeProperty`, now use `PandasDataTable`
  methods directly.
- Since the way descriptors are saved has changed, this release is incompatible with
  previous data sets and models. However, these can be easily converted to the new
  format by adding
  a prefix with descriptor set name to the old descriptor tables. Feel free to contact
  us if you require assistance with this.
- Due to some changes in `rdkit-2023.9.6`, the `add_rdkit`
  option for molecule tables temporarily might not work.
  This also affects the current ChemProp integration, which was not adapted to 2.0.0 yet.  
  In order to prevent these issues, QSPRpred now forces rdkit version `rdkit-2023.9.5`, 
  but we will be working on resolving these.

## New Features

- Descriptors are now saved with prefixes to indicate the descriptor sets. This reduces
  the chance of name collisions when using multiple descriptor sets.
- Added new methods to `MoleculeTable` and `QSARDataset` for more fine-grained control
  of clearing, dropping and restoring of descriptor sets calculated for the dataset.
    - `dropDescriptorSets` will drop descriptors associated with the given descriptor
      sets.
    - `dropDescriptors` will drop individual descriptors associated with the given
      descriptor sets and properties.
    - All drop actions are restorable with `restoreDescriptorSets` unless explicitly
      cleared from the data set with the `clear` parameter of `dropDescriptorSets`.
- Added a proper API for parallelization backend selection and configuration (see
  documentation of `ParallelGenerator` and `JITParallelGenerator` for more information).
- Clusters can now be added to a `MoleculeTable` with `addClusters` and retrieved with
  `getClusters`, similar to scaffolds.

## Removed Features

- removed support for PyBoost since the project was abandoned by the original developers and is [no longer maintained](https://github.com/sb-ai-lab/Py-Boost/graphs/contributors)
