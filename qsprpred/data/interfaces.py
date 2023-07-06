"""Abstract base classes for data preparation classes."""
from abc import ABC, abstractmethod
from typing import Callable, Iterable

import numpy as np
import pandas as pd


class StoredTable(ABC):
    """Abstract base class for tables that are stored in a file."""
    @abstractmethod
    def save(self):
        """Save the table to a file."""

    @abstractmethod
    def reload(self):
        """Reload the table from a file."""

    @abstractmethod
    def clearFiles(self):
        """Delete the files associated with the table."""

    @staticmethod
    @abstractmethod
    def fromFile(filename: str) -> "StoredTable":
        """Load a `StoredTable` object from a file.

        Args:
            filename (str): The name of the file to load the object from.

        Returns:
            The `StoredTable` object itself.
        """


class DataSet(StoredTable):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def getProperties(self):
        """Get the properties of the dataset."""

    @abstractmethod
    def addProperty(self, name: str, data: list):
        """Add a property to the dataset.

        Args:
            name (str): The name of the property.
            data (list): The data of the property.
        """

    @abstractmethod
    def removeProperty(self, name: str):
        """Remove a property from the dataset.

        Args:
            name (str): The name of the property.
        """

    @abstractmethod
    def getSubset(self, prefix: str):
        """Get a subset of the dataset.

        Args:
            prefix (str): The prefix of the subset.
        """

    @abstractmethod
    def apply(
        self,
        func: callable,
        func_args: list = None,
        func_kwargs: dict = None,
        *args,
        **kwargs
    ):
        """Apply a function to the dataset.

        Args:
            func (callable): The function to apply.
            func_args (list, optional): The positional arguments of the function.
            func_kwargs (dict, optional): The keyword arguments of the function.
        """

    @abstractmethod
    def transform(self, targets, transformers):
        pass

    @abstractmethod
    def filter(self, table_filters: list[Callable]):
        """Filter the dataset.

        Args:
            table_filters (List[Callable]): The filters to apply.
        """


class MoleculeDataSet(DataSet):
    @abstractmethod
    def addDescriptors(self, calculator: "DescriptorsCalculator"):  # noqa: F821
        """
        Add descriptors to the dataset.

        Args:
            calculator (DescriptorsCalculator): An instance of the
                `DescriptorsCalculator` class that wraps the descriptors to be
                calculated.
        """

    @abstractmethod
    def getDescriptors(self) -> pd.DataFrame:
        """
        Get the table of descriptors that are currently in the dataset.

        Returns:
            a pd.DataFrame with the descriptors
        """

    @abstractmethod
    def getDescriptorNames(self) -> list[str]:
        """
        Get the names of the descriptors that are currently in the dataset.

        Returns:
            a `list` of descriptor names
        """

    @property
    @abstractmethod
    def hasDescriptors(self):
        """Indicates if the dataset has descriptors."""


class DataSetDependant:  # Note: this shouldn't be ABC; no abstract methods defined
    """Classes that need a data set to operate have to implement this."""
    def __init__(self, dataset) -> None:
        self.dataSet = dataset

    def setDataSet(self, dataset: MoleculeDataSet):
        """
        Set the data sets.
        """
        self.dataSet = dataset

    @property
    def hasDataSet(self):
        return self.dataSet is not None

    def getDataSet(self):
        if self.hasDataSet:
            return self.dataSet
        else:
            raise ValueError("Data set not set.")


class DataSplit(ABC):
    """Defines a function split a dataframe into train and test set."""
    @abstractmethod
    def split(self, X : np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame | pd.Series) -> Iterable[tuple[list[int], list[int]]]:
        """Split the given data into one or multiple train/test subsets.

        These classes handle partitioning of a feature matrix
        by returning an iterator of train
        and test indices. It is compatible with the approach taken
        in the `sklearn` package (see `sklearn.model_selection._BaseKFold`).
        This can be used for both cross-validation or a one time train/test split.

        Args:
            X (np.ndarray | pd.DataFrame): the input data matrix
            y (np.ndarray | pd.DataFrame | pd.Series): the target variable(s)

        Returns:
            an iterator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix X (note that these are integer indices, rather than a
            pandas index!)
        """


class DataFilter(ABC):
    """Filter out some rows from a dataframe."""
    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out some rows from a dataframe.

        Args:
            df (pd.DataFrame): dataframe to be filtered

        Returns:
            The filtered pd.DataFrame
        """


class FeatureFilter(ABC):
    """Filter out uninformative featureNames from a dataframe."""
    @abstractmethod
    def __call__(self, df: pd.DataFrame, y_col: pd.DataFrame = None):
        """Filter out uninformative features from a dataframe.

        Args:
            df (pd.DataFrame): dataframe to be filtered
            y_col (pd.DataFrame, optional): output dataframe if the filtering method
                requires it

        Returns:
            The filtered pd.DataFrame
        """
