from abc import ABC, abstractmethod
from typing import Iterable, Any

from rdkit import Chem

from qsprpred.data.storage.interfaces.chem_store import ChemStore
from qsprpred.data.storage.interfaces.property_storage import PropertyStorage
from qsprpred.data.storage.interfaces.stored_mol import StoredMol
from qsprpred.extra.data.storage.protein.interfaces.storedprotein import StoredProtein


class ProteinStorage(PropertyStorage, ABC):

    @property
    @abstractmethod
    def sequenceProp(self) -> str:
        """Get the name of the property that contains all protein sequences."""

    @abstractmethod
    def add_protein(self, protein: StoredProtein, raise_on_existing=True):
        """
        Add a protein to the store.

        :param protein: protein sequence
        :param raise_on_existing: raise an exception if the protein already exists in the store
        :return: `StoredProtein` instance of the added protein
        """

    @property
    @abstractmethod
    def proteins(self) -> Iterable[StoredProtein]:
        """
        Get all proteins in the store.

        :return: iterable of `Protein` instances
        """

    @abstractmethod
    def getProtein(self, protein_id: str) -> StoredProtein:
        """
        Get a protein from the store using its name.

        :param protein_id: name of the protein to search
        :return: instance of `Protein`
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

    @abstractmethod
    def add_target(self, target: StoredProtein,
                   raise_on_existing=True) -> StoredProtein:
        """
        Add a target to the store.

        :param target: target protein
        :param raise_on_existing: raise an exception if the target already exists in the store
        :return: `Protein` instance of the added target
        """

    @abstractmethod
    def add_poses(self, mol_id: str, poses: Chem.Mol, target: StoredProtein,
                  metadata: list[dict[str, Any]] | None = None) -> list[StoredMol]:
        """
        Add poses to the store.

        :param mol_id: identifier of the molecule to add poses for
        :param poses: dictionary of target identifiers and poses
        :param target: target protein
        :param metadata: additional metadata to store with the poses
        :return: Added poses represented as `StoredMol`
        """

    @abstractmethod
    def get_poses(self, mol_id: str, target_id: str) -> list[StoredMol]:
        """
        Get poses from the store.

        :param mol_id: identifier of the molecule to get poses for
        :param target_id: identifier of the target to get poses for
        :return:
        """

    @abstractmethod
    def get_complex_for_pose(self, mol_id: str, target_id: str) -> Chem.Mol:
        """
        Get the complex for a pose.

        :param mol_id: identifier of the molecule to get the complex for
        :param target_id: identifier of the target to get the complex for
        :return: tuple of the complex and the target
        """

    @abstractmethod
    def get_target(self, target_id: str) -> StoredProtein:
        """
        Get a target from the store using its ID.

        :param target_id: identifier of the target to search
        :return: instance of `Protein`
        """
