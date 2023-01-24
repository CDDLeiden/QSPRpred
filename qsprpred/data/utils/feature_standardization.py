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

def apply_feature_standardizers(feature_standardizers, X, fit=True):
    """
    Apply and/or fit feature standardizers.

    Arguments:
        feature_standardizers (list of feature standardizer objs): standardizes and/or scales features (i.e `StandardScaler` from scikit-learn or `SKLearnStandardizer`)
        X (pd.DataFrame): feature matrix to standardize
        fit (bool): fit the standardizer on the data instead of just applying it

    Returns:
        pd.DataFrame: standardized feature matrix of the same dimensions as X
        list: list of (fitted) feature standardizers
    """
    fitted_standardizers = []
    for idx, standardizer in enumerate(feature_standardizers):
        if isinstance(standardizer, SKLearnStandardizer):
            standardizer = standardizer.getInstance()

        if fit:
            standardizer = SKLearnStandardizer.fromFit(X, standardizer)
        else:
            standardizer = SKLearnStandardizer(standardizer)

        X = standardizer(X)
        fitted_standardizers.append(standardizer)

    return X, fitted_standardizers
