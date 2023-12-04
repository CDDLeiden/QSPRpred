"""This module is used for standardizing feature sets."""
import numpy as np
import pandas as pd
import ml2json

from ...logs import logger
from ...utils.serialization import JSONSerializable


class SKLearnStandardizer(JSONSerializable):
    """Standardizer for molecular features."""
    def __init__(self, scaler):
        """
        Initialize the standardizer.

        Args:
            scaler: sklearn object
        """

        self.scaler = scaler

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["scaler"] = ml2json.to_dict(self.scaler)
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.scaler = ml2json.from_dict(state["scaler"])

    def __str__(self):
        """Return string representation."""
        return f"SKLearnStandardizer_{self.scaler.__class__.__name__}"

    def __call__(self, features: np.array) -> np.array:
        """Standardize features.

        Args:
            features: array of features to be standardized

        Returns:
            features: array of standardized features
        """
        # if isinstance(features, np.ndarray):
        #     features = pd.DataFrame(features)
        # print(features)
        features = self.scaler.transform(features)
        logger.debug("Data standardized")
        return features

    def getInstance(self):
        """Get scaler object."""
        return self.scaler

    @classmethod
    def fromFit(cls, features: np.array, scaler):
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

    X_std = standardizer(X)
    X = pd.DataFrame(X_std, index=X.index, columns=X.columns)

    return X, standardizer
