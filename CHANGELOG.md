# Change Log
From v1.3.0 to v1.4.0.dev0

## Fixes

- Not all scorers in `supported_scoring` were actually working in the multi-class case, the scorer support is now
  divided by single and multiclass support (moved to `metrics.py`, see also New Features).
- problems with PaDEL descriptors and fingerprints on Linux were fixed

## Changes
- To support multitask modelling, the representation of the target in the `QSPRdataset` has changed to a list of 
  `TargetProperty`s (see New Features). These can be automatically initizalid from dictionaries in the `QSPRdataset`
  init.

## New Features
.
- Support for Sklearn Multitask modelling
- New class abstract class `Metric`, which is an abstract base class that allows for creating custom scorers.
- Subclass `SklearnMetric` of the `Metric` class, this class wraps the sklearn metrics, to allow for checking 
  the compatibility of each Sklearn scoring function with the `QSPRSklearn` model type.
- New class `TargetProperty`, to allow for multitask modelling, a `QSPRdataset` has to have the option of multiple
  targetproperties. To support this a targer property is now defined seperatly from the dataset as a `TargetProperty`
  instance, which holds the information on name,  `TargetTask` (see also Changes) and threshold of the property. 
