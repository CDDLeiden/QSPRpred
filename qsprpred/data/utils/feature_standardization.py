"""This module is used for standardizing feature sets."""
import numpy as np
import sklearn_json as skljson

from ...logs import logger


class SKLearnStandardizer:
    """Standardizer for molecular features."""
    def __init__(self, scaler):
        """
        Initialize the standardizer.

        Args:
            scaler: sklearn object
        """

        self.scaler = scaler

    def __str__(self):
        """Return string representation."""
        return f"SKLearnStandardizer_{self.scaler.__class__.__name__}"

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


def apply_feature_standardizer(feature_standardizer, X, fit=True):
    """
    Apply and/or fit feature standardizers.

    Arguments:
        feature_standardizer (SKLearnStandardizer): standardizes and/or scales features
            (i.e `StandardScaler` from scikit-learn or `SKLearnStandardizer`)
        X (pd.DataFrame): feature matrix to standardize
        fit (bool): fit the standardizer on the data instead of just applying it

    Returns:
        pd.DataFrame: standardized feature matrix of the same dimensions as X
        SKLearnStandardizer: (fitted) feature standardizer
    """
    if X.shape[1] == 0:
        raise ValueError("No features to standardize.")

    standardizer = feature_standardizer
    if isinstance(standardizer, SKLearnStandardizer):
        standardizer = standardizer.getInstance()

    if fit:
        standardizer = SKLearnStandardizer.fromFit(X, standardizer)
    else:
        standardizer = SKLearnStandardizer(standardizer)

    X = standardizer(X)

    return X, standardizer
