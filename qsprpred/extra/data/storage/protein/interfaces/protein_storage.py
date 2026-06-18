from abc import ABC, abstractmethod
from typing import Iterable

from qsprpred.data.storage.interfaces.property_storage import PropertyStorage
from qsprpred.extra.data.storage.protein.interfaces.storedprotein import StoredProtein


class ProteinStorage(PropertyStorage, ABC):
    """Storage for proteins.

    Attributes:
        sequenceProp (str): name of the property that contains all protein sequences
        proteins (Iterable[StoredProtein]): all proteins in the store
    """

    @property
    @abstractmethod
    def sequenceProp(self) -> str:
        """Get the name of the property that contains all protein sequences."""

    @abstractmethod
    def add_protein(
            self, protein: StoredProtein, raise_on_existing=True
    ) -> StoredProtein:
        """Add a protein to the store.

        Args:
            protein (StoredProtein): protein sequence
            raise_on_existing (bool):
                raise an exception if the protein already exists in the store

        Returns:
            StoredProtein: instance of the added protein
        """

    @property
    @abstractmethod
    def proteins(self) -> Iterable[StoredProtein]:
        """Get all proteins in the store.

        Returns:
            Iterable[StoredProtein]: iterable of `Protein` instances
        """

    @abstractmethod
    def getProtein(self, protein_id: str) -> StoredProtein:
        """Get a protein from the store using its name.

        Args:
            protein_id (str): name of the protein to search

        Returns:
            StoredProtein: instance of `Protein`
        """

    @abstractmethod
    def getPCMInfo(self) -> tuple[dict[str, str], dict]:
        """Return a dictionary mapping of protein ids to their sequences and a
        dictionary with metadata for each. This is mainly for compatibility with
        QSPRpred's PCM modelling API.

        Returns:
            sequences (dict): Dictionary of protein sequences.
            metadata (dict): Dictionary of metadata for each protein.
        """
