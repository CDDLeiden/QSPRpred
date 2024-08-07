from abc import ABC, abstractmethod
from typing import Iterable, Any, Sized, Generator

import pandas as pd

from qsprpred.data.storage.interfaces.chunk_iterable import ChunkIterable
from qsprpred.data.storage.interfaces.data_store import DataStorage


class PropertyStorage(DataStorage, ChunkIterable, ABC):
    """A simple `DataStorage` that maps property names to arbitrary data."""

    @property
    @abstractmethod
    def idProp(self) -> str:
        """Get the name of the property that contains the molecule IDs."""

    @abstractmethod
    def getProperty(self, name: str, ids: tuple[str] | None = None) -> Iterable[Any]:
        """Get values of a given property.

        Args:
            name (str): The name of the property.
            ids (list, optional): The IDs of the entries to get the property for.
        """

    @abstractmethod
    def getProperties(self) -> list[str]:
        """Get the property names contained in the storage."""

    @abstractmethod
    def addProperty(self, name: str, data: Sized, ids: list[str] | None = None):
        """Add a property to the dataset. The supplied data should be a sized list
        of values of the same length as the number of entries in the storage.

        Args:
            name (str): The name of the property.
            data (list): The data of the property.
            ids (list, optional): The IDs of the entries to add the property for.
        """

    def hasProperty(self, name: str) -> bool:
        """Check whether a property is present in the data frame.

        Args:
            name (str): Name of the property.

        Returns:
            bool: Whether the property is present.
        """

    @abstractmethod
    def removeProperty(self, name: str):
        """Remove a property from the dataset.

        Args:
            name (str): The name of the property.
        """

    @abstractmethod
    def getSubset(self, subset: list[str],
                  ids: list[str] | None = None) -> "PropertyStorage":
        """Get a subset of the storage for the given properties.

        Args:
            subset (list): The list of property names to include in the subset.
            ids (list, optional): The IDs of the entries to include in the subset.
        """

    # @abstractmethod
    # def transformProperties(
    #         self,
    #         names: list[str],
    #         transformer: Callable[[Iterable[Any]], Iterable[Any]]
    # ):
    #     """Transform property values using a transformer function. The transformer
    #     function is applied to the values of the properties in the storage in order of
    #     appearance in the `names` list. The transformer function should take a single
    #     iterable argument and return a new iterable of the same length with the
    #     transformed values. The transformed values will replace the original values
    #     in the storage. If the original values should be preserved, is up to the
    #     downstream implementation.
    #
    #     Args:
    #         names (list[str]): list of column names to transform.
    #         transformer (Callable): Function that transforms the data in target columns
    #             to a new representation.
    #     """

    @abstractmethod
    def getDF(self) -> pd.DataFrame:
        """Get the stored properties as a pandas DataFrame.

        Returns:
            pd.DataFrame: The data as a pandas DataFrame.
        """

    @classmethod
    @abstractmethod
    def fromDF(cls, df: pd.DataFrame, *args, **kwargs) -> "PropertyStorage":
        """Create a new `PropertyStorage` object from a pandas DataFrame.

        Args:
            df (pd.DataFrame): The data to create the storage from.
            args: Additional positional arguments to be passed to the constructor.
            kwargs: Additional keyword arguments to be passed to the constructor.

        Returns:
            PropertyStorage: The new storage object.
        """

    @abstractmethod
    def apply(
            self,
            func: callable,
            func_args: list | None = None,
            func_kwargs: dict | None = None,
            on_props: tuple[str, ...] | None = None,
            as_df: bool = False,
    ) -> Generator[Iterable[Any], None, None]:
        """Apply a function on all or selected properties of the chunks of data.
        The properties are supplied as the first positional argument to the function.
        The format of the properties is up to the downstream implementation, but it
        should always be a single object supplied as the first parameter.

        Args:
            func (callable): The function to apply.
            func_args (list, optional): The positional arguments of the function.
            func_kwargs (dict, optional): The keyword arguments of the function.
            on_props (list, optional): The properties to apply the function on.
            as_df (bool, optional): Provide properties as a DataFrame to the function.
            *args: Additional positional arguments to the function.
            **kwargs: Additional keyword arguments to the function.
        """

    @abstractmethod
    def dropEntries(self, ids: tuple[str, ...]):
        """Drop entries from the storage.

        Args:
            ids (list): The IDs of the entries to drop.
        """

    @abstractmethod
    def addEntries(self, ids: list[str], props: dict[str, list],
                   raise_on_existing: bool = True):
        """Add entries to the storage.

        Args:
            ids (list):
                The IDs of the entries to add.
            props (dict):
                The properties to add.
            raise_on_existing (bool):
                Overwrite existing entries. If `True`,
                an exception is raised if an entry already exists.

        Raises:
            ValueError: If an entry already exists and `overwrite` is `False`.
        """

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __contains__(self, item):
        pass

    def __iter__(self):
        return self.iterChunks(1)

    @abstractmethod
    def iterChunks(
            self,
            size: int | None = None,
            on_props: list | None = None
    ) -> Generator[list[Any], None, None]:
        """
        Iterate over chunks of molecules across the store.

        :return: an iterable of lists of stored molecules
        """
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    def __delitem__(self, key):
        return self.dropEntries((key,))

    def __setitem__(self, key, value):
        return self.addEntries((key,), value)

    def __str__(self):
        return f"{self.__class__.__name__} ({len(self)})"

    def __repr__(self):
        return str(self)

    def __bool__(self):
        return len(self) > 0
