from abc import ABC, abstractmethod

import pandas as pd

from qsprpred.data.descriptors.sets import DescriptorSet


class DescriptorProvider(ABC):

    @property
    @abstractmethod
    def descriptorSets(self) -> list[DescriptorSet]:
        """Get the descriptor sets that are currently in the storage.
        
        Returns:
            a `list` of descriptor sets
        """
        pass

    @abstractmethod
    def dropDescriptorSets(
            self,
            descriptors: list[DescriptorSet | str],
    ):
        """Drop descriptor sets from the storage.
        
        Args:
            descriptors:
                The descriptor sets to drop.
        """
        pass

    @abstractmethod
    def addDescriptors(self, descriptors: DescriptorSet, *args, **kwargs):
        """Add descriptors to the dataset.

        Args:
            descriptors (list[DescriptorSet]): The descriptors to add.
            args: Additional positional arguments to be passed to each descriptor set.
            kwargs: Additional keyword arguments to be passed to each descriptor set.
        """

    @abstractmethod
    def getDescriptors(self) -> pd.DataFrame:
        """Get the table of descriptors that are currently in the storage.

        Returns:
            a pd.DataFrame with the descriptors
        """

    @abstractmethod
    def getDescriptorNames(self) -> list[str]:
        """Get the names of the descriptors that are currently in the storage.

        Returns:
            a `list` of descriptor names
        """

    @abstractmethod
    def hasDescriptors(self):
        """Indicates if the storage has descriptors."""
