from abc import ABC, abstractmethod
from typing import Callable, Iterable, Any, Generator

from qsprpred.data.chem.identifiers import Identifiable
from qsprpred.data.chem.standardizers.base import Standardizable
from qsprpred.data.storage.interfaces.chunk_iterable import ChunkIterable
from qsprpred.data.storage.interfaces.descriptor_provider import DescriptorProvider
from qsprpred.data.storage.interfaces.mol_processable import MolProcessable
from qsprpred.data.storage.interfaces.property_storage import PropertyStorage
from qsprpred.data.storage.interfaces.searchable import SMARTSSearchable
from qsprpred.utils.interfaces.randomized import Randomized
from qsprpred.utils.interfaces.summarizable import Summarizable


class MoleculeDataSet(
    PropertyStorage,
    DescriptorProvider,
    MolProcessable,
    SMARTSSearchable,
    Summarizable,
    ChunkIterable,
    Randomized,
    Identifiable,
    Standardizable,
    ABC
):

    @property
    @abstractmethod
    def smilesProp(self) -> str:
        """Get the name of the property that contains the SMILES strings."""

    @property
    @abstractmethod
    def smiles(self) -> Generator[str, None, None]:
        """Get the SMILES strings of the molecules in the data frame.

        Returns:
            Generator[str, None, None]: Generator of SMILES strings.
        """

    @abstractmethod
    def imputeProperties(self, names: list[str], imputer: Callable):
        """Impute missing values in the target properties using the given imputer.

        Args:
            names (list[str]): list of target properties names to impute
            imputer (Callable): imputer function
        """

    @abstractmethod
    def transformProperties(self, names: list[str],
                            transformer: Callable[[Iterable[Any]], Iterable[Any]]):
        pass
