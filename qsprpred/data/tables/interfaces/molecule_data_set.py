from abc import ABC, abstractmethod
from typing import Any, Callable, Generator, Iterable

from qsprpred.data.chem.identifiers import Identifiable
from qsprpred.data.chem.standardizers.base import Standardizable
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
    Randomized,
    Identifiable,
    Standardizable,
    ABC,
):
    """Interface for storing and managing chemical data sets for machine learning."""
    @property
    @abstractmethod
    def smilesProp(self) -> str:
        """Get the name of the property that contains the SMILES strings."""

    @property
    @abstractmethod
    def smiles(self) -> Generator[str, None, None]:
        """Get the SMILES strings of the molecules in this instance.

        Returns:
            Generator[str, None, None]: Generator of SMILES strings.
        """
