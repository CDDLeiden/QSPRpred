"""This module is used for standardizing feature sets."""
from abc import ABC, abstractmethod

import numpy as np
import sklearn_json as skljson
from qsprpred.logs import logger
from sklearn.preprocessing import StandardScaler as Scaler


class FeatureStandardizer(ABC):
    """Standardizer for molecular features."""

    def __init__(self, scaler):
        self.scaler=scaler

    def __call__(self, features: np.array) -> np.array:
        """Standardize features.

        Args:
            features: array of features to be standardized

        Returns:
            features: array of standardized features
        """
        features = self.scaler.transform(features)
        logger.debug("Data standardized")
        return features

    @classmethod
    def fromFile(cls, fname: str):
        """Construct standardizer from json file.

        Args:
            fname: file name to load standardizer from
        """
        return StandardStandardizer(skljson.from_json(fname))

    @classmethod
    def fromFit(cls, features: np.array):
        """Construct standardizer by fitting on feature set.

        Args:
            features: array of features to fit standardizer on
        """
        pass

    def toFile(self, fname) -> None:
        """Save standardizer to json file.

        Args:
            fname: file name to save standardizer to
        """
        skljson.to_json(self.scaler, fname)


class StandardStandardizer(FeatureStandardizer):
    """Standard scaler standardizer for molecular features."""
    
    @classmethod
    def fromFit(cls, features: np.array):
        """Construct standardizer by fitting on feature set.

        Args:
            features: array of features to fit standardizer on
        """
        scaler = Scaler()
        scaler.fit(features)
        return StandardStandardizer(scaler)
