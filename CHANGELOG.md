# Change Log
From v1.3.0 to v1.3.1

## Fixes
- Fix not re-initiating model weights during DNN training
- Feature values converted to `np.float32` and then np.inf are converted to `nan` on `DescriptorsCalculator.__call__`.

## Changes

- `QSPRDataset.prepareDataset` changed attributes from `standardize` and `sanitize` to only `standardizer`.
  - Accepted parameters are either `chembl`, `old`, or a function that reads and standardizes smiles.
  - None is now also supported to allow skipping smiles standardization.
  - SMILES standardization now runs in parallel, but if the input function is not pickable, will just run on a single core.
- `QSPRModel.predictMols` now accepts parameters `smiles_standardizer`, `n_jobs` and `fill_value`.

## New Features