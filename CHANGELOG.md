# Change Log

From v2.0.0.dev1 to v2.0.0.dev2

## Fixes


## Changes
- 'LowVarianceFilter` now includes boundary in the filtered features, e.g. if threshold is 0.1, also features that
  have a variance of 0.1 will be removed.

## New Features
- New feature split `ManualSplit` for splitting data by a user-defined column
