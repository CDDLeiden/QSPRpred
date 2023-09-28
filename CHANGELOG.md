# Change Log

From v2.1.0.a2 to v2.2.0

## Fixes
- Fixed random seeds to give reproducible results. Each dataset is initialized with a single random state (either from the constructor or a random number generator) which is used in all subsequent random operations. Each model is initialized with a single random state as well: it uses the random state from the dataset, unless it's overriden in the constructor. When a dataset is saved to a file so is its random state, which is used by the dataset when the dataset is reloaded.

## Changes
- The jupyter notebooks now pass a random state to ensure consistent results.

## New Features
- Most unit tests now have a variant that checks whether using a fixed random seed gives reproducible results.
- The build pipeline now contains a check that the jupyter notebooks give the same results as ones that were observed before.

## Removed Features
