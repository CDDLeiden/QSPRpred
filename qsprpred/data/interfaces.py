"""Abstract base classes for data preparation classes."""
from abc import ABC, abstractmethod

import pandas as pd

from qsprpred.data.utils.descriptorcalculator import Calculator

class DataSet(ABC):
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
    def getSubset(self, prefix : str):
        pass

    @abstractmethod
    def apply(self, func, func_args=None, func_kwargs=None, *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, targets, transformers):
        pass

class MoleculeDataSet(DataSet):

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def addDescriptors(self, calculator : Calculator):
        """
        Add descriptors to the dataset.

        Args:
            calculator: The descriptor calculator class wrapping the descriptors to calculate.
        Returns:
            `None`
        """
        pass

    @abstractmethod
    def getDescriptors(self):
        """
        Get the table of descriptors that are currently in the dataset.

        Returns:
            a `DataFrame` with the descriptors
        """

        pass

    @abstractmethod
    def getDescriptorNames(self):
        """
        Get the names of the descriptors that are currently in the dataset.

        Returns:
            a `list` of descriptor names
        """
        pass

    @property
    @abstractmethod
    def hasDescriptors(self):
        pass

    @staticmethod
    @abstractmethod
    def fromFile(filename) -> 'MoleculeDataSet':
        pass

class datasplit(ABC):
    """Defines a function split a dataframe into train and test set."""

    @abstractmethod
    def __call__(self, df, Xcol, ycol):
        """Split dataframe df into train and test set.

        Args:
            df: pandas dataframe to split
            Xcol: input column name, e.g. "SMILES"
            ycol: output column name, e.g. "Cl"

        Returns:
            X: training set input
            X_ind: test set input
            y: training set output
            y_ind: test set output
        """
        pass

class datafilter(ABC):
    """Filter out some rows from a dataframe."""

    @abstractmethod
    def __call__(self, df):
        """Filter out some rows from a dataframe.

        Args:
            df: pandas dataframe to filter

        Returns:
            df: filtered pandas dataframe
        """
        pass

class featurefilter(ABC):
    """Filter out uninformative features from a dataframe."""

    @abstractmethod
    def __call__(self, df, y_col : pd.DataFrame = None):
        """Filter out uninformative features from a dataframe.
        
        Args:
            df: pandas dataframe to filter
            y_col: output variable column name if the method requires it
        Returns:
            df: filtered pandas dataframe
        """
        pass

