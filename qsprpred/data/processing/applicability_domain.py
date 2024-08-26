from abc import ABC, abstractmethod
from math import floor

import ml2json
import numpy as np
import pandas as pd
from mlchemad.base import ApplicabilityDomain as MLChemADApplicabilityDomain
from scipy.spatial.distance import _METRICS as dist_fns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)

from ...logs import logger
from ...utils.serialization import JSONSerializable


class ApplicabilityDomain(JSONSerializable, ABC):
    """Define the applicability domain for a dataset.

    A class to define the applicability domain for a dataset.
    A fitted applicability domain can be used to filter out molecules that are not in
    in the applicability domain or just to check if a molecule is in the applicability
    domain.
    """
    def __init__(
        self, threshold: float | None = None, direction: str | None = None
    ) -> None:
        """Initialize the applicability domain with a threshold.

        Args:
            threshold (float | None): threshold value
            direction (str | None): direction of the threshold, should be set if
                threshold is set
        """
        self.threshold = threshold
        self._direction = direction

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> None:
        """Fit the applicability domain model.

        Args:
            X (pd.DataFrame): array of features to fit model on
        """

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the features to a score for the applicability domain.

        The result could be a boolean array indicating if the features are in the
        applicability domain or a score indicating how much the features are in the
        applicability domain
        (e.g., a probability or a distance).

        Args:
            X (pd.DataFrame): array of features

        Returns:
            pd.Series: scores for the applicability domain
        """

    @property
    @abstractmethod
    def fitted(self) -> bool:
        """Return whether the applicability domain is fitted or not."""

    def contains(self, X: pd.DataFrame) -> pd.DataFrame:
        """Check if the applicability domain contains the features.

        Args:
            X (pd.DataFrame): array of features to check

        Returns:
            pd.Series: pd.Series of booleans indicating if the features are in the
                applicability domain
        """
        if self.threshold is not None:
            return self._apply_threshold(self.transform(X))
        else:
            X_transformed = self.transform(X)
            # Check if the transformed features are boolean or could be converted to
            # boolean
            try:
                return X_transformed.astype(bool)
            except ValueError:
                raise ValueError(
                    "Features cannot be converted to boolean,"
                    "set threshold and direction to apply threshold"
                )

    @property
    def direction(self) -> str:
        """Return the direction of the threshold.

        The direction should be '>', '<', '>=', '<='
        """
        return self._direction

    @direction.setter
    def direction(self, direction: str) -> None:
        """Set the direction of the threshold.

        Args:
            direction (str): direction of the threshold
        """
        if direction not in [">", "<", ">=", "<="]:
            raise ValueError("Direction must be one of '>', '<', '>=', '<='")
        self._direction = direction

    def _apply_threshold(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply a threshold to the applicability domain.

        Args:
            X (pd.Series): array of transformed features
        """
        if self.direction == ">":
            return X > self.threshold
        elif self.direction == "<":
            return X < self.threshold
        elif self.direction == ">=":
            return X >= self.threshold
        elif self.direction == "<=":
            return X <= self.threshold
        else:
            raise ValueError("Direction must be set to apply threshold")


class MLChemADWrapper(ApplicabilityDomain):
    """Define the applicability domain for a dataset using the MLChemAD package.

    This class uses the MLChemAD package to filter out molecules that are not in the
    applicability domain. The MLChemAD package is available at
    https://github.com/OlivierBeq/MLChemAD

    Attributes:
        applicabilityDomain (MLChemApplicabilityDomain): applicability domain object
        fitted (bool): whether the applicability domain is fitted or not
    """
    def __init__(
        self,
        applicability_domain: MLChemADApplicabilityDomain,
        astype: str | None = "float64",
    ) -> None:
        """Initialize the MLChemADFilter with the domain_type attribute.

        Args:
            applicability_domain (MLChemAD): applicability domain object
            astype (str | None): type to cast the features to before fitting or
                checking the applicability domain
        """
        self.applicabilityDomain = applicability_domain
        self.astype = astype
        self.threshold = None
        self._direction = None

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["applicabilityDomain"] = ml2json.to_dict(self.applicabilityDomain)
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.applicabilityDomain = ml2json.from_dict(state["applicabilityDomain"])

    def fit(self, X: pd.DataFrame) -> None:
        """Fit the applicability domain model.

        Args:
            X (pd.DataFrame): array of features to fit model on
        """
        if self.astype is not None:
            try:
                X = X.astype(self.astype)
            except ValueError:
                logger.warning(
                    f"Cannot convert X to {self.astype}, fitting with raw data"
                )
        self.applicabilityDomain.fit(X)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Check if the applicability domain contains the features.

        Args:
            X (pd.DataFrame): array of features to check

        Returns:
            pd.Series: pd.Series of booleans indicating if the features are in the
                applicability domain
        """
        if not self.fitted:
            raise RuntimeError("Applicability domain not fitted, call fit first")
        if self.astype is not None:
            try:
                X = X.astype(self.astype)
            except ValueError:
                logger.warning(
                    f"Cannot convert X to {self.astype}, checking with raw data."
                    "Note. if the data type is different from the one used for fitting,"
                    "the result may be incorrect"
                )
        return pd.Series(
            self.applicabilityDomain.contains(X.copy()),
            index=X.index,
        )

    @property
    def fitted(self) -> bool:
        """Return whether the applicability domain is fitted or not."""
        return self.applicabilityDomain.fitted_


class KNNApplicabilityDomain(ApplicabilityDomain):
    """Applicability domain defined using K-nearest neighbours.

    This class is adapted from the `KNNApplicabilityDomain` class in the
    `mlchemad` package.
    """
    def __init__(
        self,
        k: int = 5,
        alpha: float = None,
        hard_threshold: float = None,
        scaling: str | None = "robust",
        dist: str = "euclidean",
        scaler_kwargs=None,
        njobs: int = 1,
        astype: str | None = "float64",
    ):
        f"""Create the k-Nearest Neighbor applicability domain.

        :param k: number of nearest neighbors
        :param alpha: ratio of inlier samples calculated from the training set;
            ignored if hard_threshold is set
        :param hard_threshold: samples with a distance greater or equal to this
            threshold will be considered outliers
        :param scaling: scaling method; must be one of 'robust', 'minmax', 'maxabs',
            'standard' or None (default: 'robust')
        :param dist: kNN distance to be calculated (default: euclidean); one of
            {list(dist_fns.keys())}; jaccard is recommended for binary fingerprints.
        :param scaler_kwargs: additional parameters to supply to the scaler
        :param njobs: number of parallel processes used to fit the kNN model
        """
        super().__init__()
        if scaler_kwargs is None:
            scaler_kwargs = {}
        if alpha is not None and (alpha > 1 or alpha < 0):
            raise ValueError("alpha must lie between 0 and 1")
        scaling_methods = ("robust", "minmax", "maxabs", "standard", None)
        if scaling not in scaling_methods:
            raise ValueError(f"scaling method must be one of {scaling_methods}")
        # Scaler for input data
        if scaling == "robust":
            self.scaler = RobustScaler(**scaler_kwargs)
        elif scaling == "minmax":
            self.scaler = MinMaxScaler(**scaler_kwargs)
        elif scaling == "maxabs":
            self.scaler = MaxAbsScaler(**scaler_kwargs)
        elif scaling == "standard":
            self.scaler = StandardScaler(**scaler_kwargs)
        elif scaling is None:
            self.scaler = None
        else:
            raise NotImplementedError("scaling method not implemented")
        if dist not in dist_fns.keys():
            raise NotImplementedError("distance type is not available")
        else:
            self.dist = dist
        self.k = k
        self.alpha = alpha
        self.hard_threshold = hard_threshold
        self.nn = NearestNeighbors(n_neighbors=k, metric=dist, n_jobs=njobs)
        self._fitted = False
        self.astype = astype

    def fit(self, X):
        """Fit the applicability domain to the given feature matrix

        :param X: feature matrix
        """
        # Normalize the data
        self.X_norm = self.scaler.fit_transform(X) if self.scaler is not None else X
        # Fit the NN
        self.nn.fit(self.X_norm)
        # Find the distance to the kNN neighbors
        # (ignoring the first neighbor, which is the sample itself)
        self.kNN_dist = self.nn.kneighbors(
            self.X_norm, return_distance=True, n_neighbors=self.k + 1
        )[0][:, 1:].mean(axis=1)
        kNN_train_distance_sorted_ = np.trim_zeros(np.sort(self.kNN_dist))
        # Find the confidence threshold
        if self.hard_threshold:
            self.threshold_ = self.hard_threshold
            self.direction = "<"
        elif self.alpha:
            self.threshold = kNN_train_distance_sorted_[
                floor(kNN_train_distance_sorted_.shape[0] * self.alpha) - 1]
            self.direction = "<="

        self._fitted = True
        return self

    def transform(self, X):
        """Get the distance to the kNN neighbors for the given feature matrix

        :param X: feature matrix

        :return: array of distances to the kNN neighbors
        """
        try:
            X = X.astype(self.astype)
        except ValueError:
            logger.warning(
                f"Cannot convert X to {self.astype}, fitting with raw data"
            )
        
        # Scale input features

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X.copy())
        else:
            X_scaled = X.copy()

        X_transformed = self.nn.kneighbors(X_scaled, return_distance=True)[0].mean(axis=1)
        return pd.Series(X_transformed, index=X.index)

    @property
    def fitted(self) -> bool:
        """Return whether the applicability domain is fitted or not."""
        return self._fitted
