from abc import ABC, abstractmethod
from typing import Callable, Generator

import pandas as pd

from qsprpred.data.descriptors.sets import DescriptorSet


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


class DataTable(StoredTable):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def getProperty(self, name: str):
        """Get values of a given property."""

    @abstractmethod
    def getProperties(self):
        """Get the property names contained in the dataset."""

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
    def transformProperties(self, names, transformers):
        """Transform property values using a transformer function.

        Args:
            targets (list[str]): list of column names to transform.
            transformer (Callable): Function that transforms the data in target columns
                to a new representation.
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
        on_props: list[str] | None = None,
        func_args: list | None = None,
        func_kwargs: dict | None = None,
    ):
        """Apply a function on all or selected properties. The properties are supplied
        as the first positional argument to the function.

        Args:
            func (callable): The function to apply.
            on_props (list, optional): The properties to include.
            func_args (list, optional): The positional arguments of the function.
            func_kwargs (dict, optional): The keyword arguments of the function.
        """

    @abstractmethod
    def filter(self, table_filters: list[Callable]):
        """Filter the dataset.

        Args:
            table_filters (List[Callable]): The filters to apply.
        """


class MoleculeDataTable(DataTable):
    @abstractmethod
    def addDescriptors(self, descriptors: DescriptorSet, *args, **kwargs):
        """
        Add descriptors to the dataset.

        Args:
            descriptors (list[DescriptorSet]): The descriptors to add.
            args: Additional positional arguments to be passed to each descriptor set.
            kwargs: Additional keyword arguments to be passed to each descriptor set.
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

    def __init__(self, dataset: MoleculeDataTable | None = None) -> None:
        self.dataSet = dataset

    def setDataSet(self, dataset: MoleculeDataTable):
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
