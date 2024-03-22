from abc import ABC, abstractmethod

import ml2json
import pandas as pd
from mlchemad.base import ApplicabilityDomain as MLChemADApplicabilityDomain

from ...logs import logger
from ...utils.serialization import JSONSerializable


class ApplicabilityDomain(JSONSerializable, ABC):
    """Define the applicability domain for a dataset.

    A class to define the applicability domain for a dataset.
    A fitted applicability domain can be used to filter out molecules that are not in
    in the applicability domain or just to check if a molecule is in the applicability
    domain.
    """
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> None:
        """Fit the applicability domain model.

        Args:
            X (pd.DataFrame): array of features to fit model on
        """

    @abstractmethod
    def contains(self, X: pd.DataFrame) -> pd.DataFrame:
        """Check if the applicability domain contains the features.

        Args:
            X (pd.DataFrame): array of features to check

        Returns:
            pd.DataFrame: array of booleans indicating if the features are in the
                applicability domain
        """

    def filter(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filter out some rows from a dataframe.

        Args:
            X (pd.DataFrame): array of features to filter

        Returns:
            The filtered pd.DataFrame
        """
        if X.shape[0] == 0:
            logger.warning("Empty dataframe, nothing to filter")
            return X
        return X[self.contains(X)]

    @property
    @abstractmethod
    def fitted(self) -> bool:
        """Return whether the applicability domain is fitted or not."""


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

    def contains(self, X: pd.DataFrame) -> pd.DataFrame:
        """Check if the applicability domain contains the features.

        Args:
            X (pd.DataFrame): array of features to check

        Returns:
            pd.DataFrame: array of booleans indicating if the features are in the
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
        return pd.DataFrame(
            self.applicabilityDomain.contains(X), index=X.index, columns=["in_domain"]
        )

    @property
    def fitted(self) -> bool:
        """Return whether the applicability domain is fitted or not."""
        return self.applicabilityDomain.fitted_
