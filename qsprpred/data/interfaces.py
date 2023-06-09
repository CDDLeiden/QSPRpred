"""Abstract base classes for data preparation classes."""
from abc import ABC, abstractmethod
from typing import Callable, List

import pandas as pd


class StoredTable(ABC):
    """Abstract base class for tables that are stored in a file."""
    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def reload(self):
        pass

    @abstractmethod
    def clearFiles(self):
        pass

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
        pass

    @abstractmethod
    def addProperty(self, name, data):
        pass

    @abstractmethod
    def removeProperty(self, name):
        pass

    @abstractmethod
    def getSubset(self, prefix: str):
        pass

    @abstractmethod
    def apply(self, func, func_args=None, func_kwargs=None, *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, targets, transformers):
        pass

    @abstractmethod
    def filter(self, table_filters: List[Callable]):
        pass


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
        pass


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
    def split(self, X, y):
        """
        Split the given data into multiple subsets.

        Args:
            X (DataFrame): the input data matrix
            y (Series): the target variable

        Returns:
            an iterator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix X
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
