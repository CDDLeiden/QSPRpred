# Change Log

From v2.1.0 to v2.1.1

## Fixes

- ⚠️ Important! ⚠️ Fixed bug in `predictMols` where the `feature_standardizer` was 
  not being applied to the calculated features. This bug was introduced in v2.1.0.
  Models trained with v2.1.0 are compatible with v2.1.1, make sure to update 
  QSPRpred to v2.1.1 to ensure that the `feature_standardizer` is applied when
  predicting on new molecules.

## Changes

## New Features

## Removed Features
