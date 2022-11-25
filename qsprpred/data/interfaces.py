"""Abstract base classes for data preparation classes."""
from abc import ABC, abstractmethod


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
    def __call__(self, df):
        """Filter out uninformative features from a dataframe.
        
        Args:
            df: pandas dataframe to filter
        Returns:
            df: filtered pandas dataframe
        """
        pass

