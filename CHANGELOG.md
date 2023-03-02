# Change Log
From v1.2.0 to v1.3.0

## Fixes

- Not all scorers in `supported_scoring` were actually working in the multi-class case, the scorer support is now
  divided by single and multiclass support (moved to `metrics.py`, see also New Features).
- problems with PaDEL descriptors and fingerprints on Linux were fixed

## Changes
- To support multitask modelling, the representation of the target in the `QSPRdataset` has changed to a list of 
  `TargetProperty`s (see New Features). These can be automatically initizalid from dictionaries in the `QSPRdataset`
  init.


- `QSPRModel` metadata now contains two extra entries:
  1. `model_class` - the fully qualified class name of the model
  2. `version` - the version of QSPRPred used to save the model
  - this change is not compatible with older files, but you can manually add these two entries and it should work fine in the newer version

## New Features

- The `QSPRModel.fromFile()` method can now instantiate a model from a file directly without knowing the underlying model type. It simply uses the class path stored in the model metadata file now.
- New class abstract class `Metric`, which is an abstract base class that allows for creating custom scorers.
- Subclass `SklearnMetric` of the `Metric` class, this class wraps the sklearn metrics, to allow for checking 
  the compatibility of each Sklearn scoring function with the `QSPRSklearn` model type.
- New class `TargetProperty`, to allow for multitask modelling, a `QSPRdataset` has to have the option of multiple
  targetproperties. To support this a targer property is now defined seperatly from the dataset as a `TargetProperty`
  instance, which holds the information on name,  `TargetTask` (see also Changes) and threshold of the property. 
- Support for Sklearn Multitask modelling
