from abc import ABC, abstractmethod
from typing import Any, Iterable

from rdkit import Chem

from qsprpred.data.storage.interfaces.chem_store import ChemStore
from qsprpred.data.storage.interfaces.property_storage import PropertyStorage
from qsprpred.data.storage.interfaces.stored_mol import StoredMol
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


class DockableStore(ChemStore, ABC):
    """Storage for dockable molecules.

    Attributes:
        proteins (Iterable[StoredProtein]): all proteins in the store
    """
    @abstractmethod
    def add_target(
        self, target: StoredProtein, raise_on_existing=True
    ) -> StoredProtein:
        """Add a target to the store.

        Args:
            target (StoredProtein): target protein
            raise_on_existing (bool):
                raise an exception if the target already exists in the store

        Returns:
            (StoredProtein): instance of the added target
        """

    @abstractmethod
    def add_poses(
        self,
        mol_id: str,
        poses: Chem.Mol,
        target: StoredProtein,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[StoredMol]:
        """Add poses to the store.

        Args:
            mol_id (str): identifier of the molecule to add poses for
            poses (Chem.Mol): dictionary of target identifiers and poses
            target (StoredProtein): target protein
            metadata (list[dict[str, Any]]): additional metadata to store with the poses

        Returns:
            list[StoredMol]: Added poses represented as `StoredMol`
        """

    @abstractmethod
    def get_poses(self, mol_id: str, target_id: str) -> list[StoredMol]:
        """Get poses from the store.

        Args:
            mol_id (str): identifier of the molecule to get poses for
            target_id (str): identifier of the target to get poses for

        Returns:
            list[StoredMol]: poses for the molecule and target
        """

    @abstractmethod
    def get_complex_for_pose(self, mol_id: str, target_id: str) -> Chem.Mol:
        """Get the complex for a pose.

        Args:
            mol_id (str): identifier of the molecule to get the complex for
            target_id (str): identifier of the target to get the complex for

        Returns:
            tuple: complex and the target
        """

    @abstractmethod
    def get_target(self, target_id: str) -> StoredProtein:
        """Get a target from the store using its ID.

        Args:
            target_id (str): identifier of the target to search

        Returns:
            StoredProtein: instance of `Protein`
        """
