from abc import ABC, abstractmethod
from typing import Optional, Any

from rdkit import Chem


class StoredMol(ABC):
    """
    A simple interface for a molecule that can be stored in a chemstore.
    """

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.id}, {self.smiles})"

    @property
    @abstractmethod
    def parent(self) -> Optional["StoredMol"]:
        """
        Get the parent molecule of this representation.

        Returns:
            The parent molecule of this representation as a `StoredMol` instance.
        """

    @property
    @abstractmethod
    def smiles(self) -> str:
        """
        Get the SMILES of the molecule.

        :return: SMILES of the molecule
        """

    @property
    @abstractmethod
    def id(self) -> str:
        """
        Get the identifier of the molecule.

        :return: identifier of the molecule
        """

    @property
    @abstractmethod
    def props(self) -> dict[str, Any] | None:
        """
        Get the metadata of the molecule.

        :return: metadata of the molecule
        """

    @property
    @abstractmethod
    def representations(self) -> list["StoredMol"] | None:
        pass

    def as_rd_mol(self) -> Chem.Mol:
        """
        Get the RDKit molecule object of the standardized representation of this instance.

        :return: `rdkit.Chem.Mol` instance
        """

        return Chem.MolFromSmiles(self.smiles)
