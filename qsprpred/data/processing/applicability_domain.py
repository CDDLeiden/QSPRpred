from abc import ABC, abstractmethod
import numpy as np
from ...utils.serialization import JSONSerializable
from mlchemad.base import ApplicabilityDomain as MLChemApplicabilityDomain

class ApplicabilityDomain(JSONSerializable, ABC):
    """Define the applicability domain for a dataset.

    A class to define the applicability domain for a dataset.
    A fitted applicability domain can be used to filter out molecules that are not in
    in the applicability domain or just to check if a molecule is in the applicability
    domain.
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Fit the applicability domain model.

        Args:
            X (np.ndarray): array of features to fit model on
        """

    @abstractmethod
    def contains(self, X: np.ndarray) -> np.ndarray:
        """Check if the applicability domain contains the features.

        Args:
            X (np.ndarray): array of features to check

        Returns:
            np.ndarray: array of booleans indicating if the features are in the
                applicability domain
        """

    def filter(self, X: np.ndarray) -> np.ndarray:
        """Filter out some rows from a dataframe.

        Args:
            X (np.ndarray): array of features to filter

        Returns:
            The filtered np.ndarray
        """
        return X[self.contains(X)]

class MLChemAD(ApplicabilityDomain):
    """To filter out molecules that are not in the applicability domain.

    This class uses the MLChemAD package to filter out molecules that are not in the
    applicability domain. The MLChemAD package is available at
    https://github.com/OlivierBeq/MLChemAD
    """
    def __init__(self, applicability_domain: MLChemApplicabilityDomain) -> None:
        """Initialize the MLChemADFilter with the domain_type attribute.

        Args:
            applicability_domain (MLChemAD): applicability domain object
        """
        self.ApplicabilityDomain = applicability_domain
        self.fitted = False

    def fit(self, X: np.ndarray) -> None:
        """Fit the applicability domain model.

        Args:
            X (np.ndarray): array of features to fit model on
        """
        self.ApplicabilityDomain.fit(X)
        self.fitted = True

    def contains(self, X: np.ndarray) -> np.ndarray:
        """Check if the applicability domain contains the features.

        Args:
            X (np.ndarray): array of features to check

        Returns:
            np.ndarray: array of booleans indicating if the features are in the
                applicability domain
        """
        if not self.fitted:
            raise RuntimeError("Applicability domain not fitted, call fit first")
        return self.ApplicabilityDomain.contains(X)
