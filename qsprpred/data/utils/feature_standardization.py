"""This module is used for standardizing feature sets."""
import numpy as np
import sklearn_json as skljson
from qsprpred.logs import logger


class SKLearnStandardizer:
    """Standardizer for molecular features."""

    def __init__(self, scaler):
        """
        Initialize the standardizer.

        Args:
            scaler: sklearn object
        """

        self.scaler = scaler

    def getInstance(self):
        """Get scaler object."""
        return self.scaler

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

    def toFile(self, fname) -> None:
        """Save standardizer to json file.

        Args:
            fname: file name to save standardizer to
        """
        skljson.to_json(self.scaler, fname)

    @staticmethod
    def fromFile(fname: str):
        """Construct standardizer from json file.

        Args:
            fname: file name to load standardizer from
        """
        return SKLearnStandardizer(skljson.from_json(fname))


    @staticmethod
    def fromFit(features: np.array, scaler):
        """Construct standardizer by fitting on feature set.

        Args:
            features: array of features to fit standardizer on
            scaler: sklearn object to fit
        """
        scaler.fit(features)
        return SKLearnStandardizer(scaler)
