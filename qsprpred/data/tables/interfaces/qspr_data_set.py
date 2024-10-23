from abc import ABC, abstractmethod
from typing import Callable, Generator, Optional

import numpy as np
import pandas as pd
from mlchemad.applicability_domains import (
    ApplicabilityDomain as MLChemADApplicabilityDomain,
)

from qsprpred import TargetProperty
from qsprpred.data.processing.applicability_domain import ApplicabilityDomain
from qsprpred.data.processing.feature_standardizers import SKLearnStandardizer
from qsprpred.data.tables.interfaces.molecule_data_set import MoleculeDataSet


class QSPRDataSet(MoleculeDataSet, ABC):
    """Interface for storing and managing QSPR-specific data sets."""
    @abstractmethod
    def setTargetProperties(
        self,
        target_props: list[TargetProperty | dict],
        drop_empty: bool = True,
    ):
        """Set the target properties for the dataset.

        Args:
            target_props (list[TargetProperty | dict]): The target properties to add.
            drop_empty (bool): If True, drop rows with missing target properties.
        """

    @property
    @abstractmethod
    def hasFeatures(self) -> bool:
        """Indicates if the dataset has features."""

    @abstractmethod
    def getFeatureNames(self) -> list[str]:
        """Get the names of the features that are currently in the dataset.

        Returns:
            (list): list of feature names
        """

    @abstractmethod
    def makeRegression(self, target_property: str):
        """Make this a regression dataset for the given target property.

        Args:
            target_property (str): The name of the target property.
        """

    @abstractmethod
    def makeClassification(self, target_property: str, threshold: float):
        """Make this a classification dataset for the given target property.

        Args:
            target_property (str): The name of the target property.
            threshold (float): The threshold for the classification.
        """

    @abstractmethod
    def restoreTargetProperty(self, prop: TargetProperty | str):
        """Restore a target property to the original state.

        Args:
            prop (TargetProperty | str): The target property to restore.
        """

    @abstractmethod
    def transformProperties(self, targets: list[str], transformer: Callable):
        """Transform the target properties using the given transformer.

        Args:
            targets (list[str]): list of target properties names to transform
            transformer (Callable): transformer function
        """

    @abstractmethod
    def addTargetProperty(self, prop: TargetProperty | dict, drop_empty: bool = True):
        """Add a target property to the dataset.

        Args:
            prop (TargetProperty):
                name of the target property to add
            drop_empty (bool):
                whether to drop rows with empty target property values. Defaults to
                `True`.
        """

    @abstractmethod
    def iterFolds(
        self,
        split: "DataSplit",  # noqa: F821
        concat: bool = False,
    ) -> Generator[
        tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame | pd.Series,
            pd.DataFrame | pd.Series,
            list[int],
            list[int],
        ],
        None,
        None,
    ]:
        """Iterate over the folds of the dataset.

        Args:
            split (DataSplit):
                split instance orchestrating the split
            concat (bool):
                whether to concatenate the training and test feature matrices

        Yields:
            (tuple):
                training and test feature matrices and target vectors
                for each fold
        """

    @property
    @abstractmethod
    def X(self) -> pd.DataFrame:
        """Get the training feature matrix."""

    @property
    @abstractmethod
    def y(self) -> pd.Series | pd.DataFrame:
        """Get the training target vector/matrix."""

    @property
    @abstractmethod
    def X_ind(self) -> pd.DataFrame:
        """Get the test feature matrix."""

    @property
    @abstractmethod
    def y_ind(self) -> pd.Series | pd.DataFrame:
        """Get the test target vector/matrix."""

    @abstractmethod
    def prepareDataset(
        self,
        split: type["DataSplit"] = None,
        feature_calculators: list["DescriptorSet"] | None = None,
        feature_fill_value: float = np.nan,
        pipeline: "Pipeline" = None,
        applicability_domain: (
            ApplicabilityDomain | MLChemADApplicabilityDomain | None
        ) = None,
        recalculate_features: bool = False,
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        """Prepare the dataset for use in QSPR model.

        Arguments:
            split (datasplitter obj): splits the dataset into train and test set
            feature_calculators (list[DescriptorSet]): descriptor sets to add to the data set
            feature_fill_value (float): value to fill missing values with.
                Defaults to `numpy.nan`
            pipeline (Pipeline): pipeline to apply to the calculated features
            applicability_domain (applicabilityDomain obj): attaches an
                applicability domain calculator to the dataset and fits it on
                the training set
            recalculate_features (bool): recalculate features even if they are already
                present in the file
            shuffle (bool): whether to shuffle the created training and test sets
            random_state (int): random state for shuffling
        """

    @abstractmethod
    def getFeatures(
        self,
        inplace: bool = False,
        concat: bool = False,
        raw: bool = False,
        ordered: bool = False,
        refit_pipeline: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get the current feature and response values (training and test) from the 
        dataset.

        This method also applies any pre-processing pipeline that have been set on the
        dataset during preparation. Outliers are dropped from the test set if they are
        present, unless `concat` is `True`.

        Args:
            inplace (bool): If `True`, the created feature matrices will be saved to the
                dataset object itself as 'X' and 'X_ind' attributes. Note that this will
                overwrite any existing feature matrices and if the data preparation
                workflow changes, these are not kept up to date. Therefore, it is
                recommended to generate new feature sets after any data set changes.
            concat (bool): If `True`, the training and test feature matrices will be
                concatenated into a single matrix. This is useful for training models
                that do not require separate training and test sets (i.e. the final
                optimized models).
            raw (bool): If `True`, the raw feature matrices will be returned without
                any standardization applied.
            ordered (bool):
                If `True`, the returned feature matrices will be ordered
                according to the original order of the data set. This is only relevant
                if `concat` is `True`.
            refit_pipeline (bool): If `True`, the pipeline will be
                refit on the training set upon this call. If `False`, the previously
                fitted pipeline will be used. Defaults to `True`. Use `False` if
                this dataset is used for prediction only and the pipeline has
                been initialized already.

        Returns:
            (tuple):
                - (pd.DataFrame): training feature matrix
                - (pd.DataFrame): test feature matrix
                - (pd.DataFrame): training target vector/matrix
                - (pd.DataFrame): test target vector/matrix
            or if `concat` is `True`:
                - (pd.DataFrame): feature matrix
                - (pd.DataFrame): target vector/matrix
        """

    @abstractmethod
    def getTargetProperties(self, names: list) -> list[TargetProperty]:
        """Get the target properties with the given names.

        Args:
            names (list): list of target property names

        Returns:
            (list): list of target properties
        """

    @property
    @abstractmethod
    def isMultiTask(self) -> bool:
        """Indicates if the dataset is a multi-task dataset."""

    @abstractmethod
    def unsetTargetProperty(self, name: str | TargetProperty):
        """Unset the target property with the given name.

        Args:
            (str | TargetProperty): name of the target property to unset
        """
