from abc import ABC, abstractmethod
from typing import Callable, Generator

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
        func_args: list | None = None,
        func_kwargs: dict | None = None,
        *args,
        **kwargs,
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

    @property
    @abstractmethod
    def smiles(self) -> Generator[str, None, None]:
        """Get the SMILES strings of the molecules in the dataset.

        Returns:
            list[str]: The SMILES strings of the molecules in the dataset.
        """



class DataSetDependant:
    """Classes that need a data set to operate have to implement this."""
    def __init__(self, dataset: MoleculeDataSet | None = None) -> None:
        self.dataSet = dataset

    def setDataSet(self, dataset: MoleculeDataSet):
        self.dataSet = dataset

    @property
    def hasDataSet(self) -> bool:
        """Indicates if this object has a data set attached to it."""
        return self.dataSet is not None

    def getDataSet(self):
        """Get the data set attached to this object.

        Raises:
            ValueError: If no data set is attached to this object.
        """
        if self.hasDataSet:
            return self.dataSet
        else:
            raise ValueError("Data set not set.")
