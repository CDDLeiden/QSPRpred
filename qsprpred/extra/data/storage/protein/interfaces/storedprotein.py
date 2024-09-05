from abc import ABC, abstractmethod
from typing import Any, Iterable

from rdkit import Chem


class StoredProtein(ABC):
    """A protein object.

    Attributes:
        id (str): id of the protein
        sequence (str): sequence of the protein
        props (dict[str, Any]): properties of the protein
        representations (Iterable[StoredProtein]): representations of the protein
    """
    @property
    @abstractmethod
    def id(self) -> str:
        """Get the id of the protein."""

    @property
    @abstractmethod
    def sequence(self) -> str | None:
        """Get the sequence of the protein."""

    @property
    @abstractmethod
    def props(self) -> dict[str, Any] | None:
        """Get the properties of the protein."""

    @abstractmethod
    def as_pdb(self) -> str | None:
        """Return the protein as a PDB file."""

    @abstractmethod
    def as_fasta(self) -> str | None:
        """Return the protein as a FASTA file."""

    def as_rd_mol(self) -> Chem.Mol | None:
        """Return the protein as an RDKit molecule."""
        pdb = self.as_pdb()
        if pdb is not None:
            return Chem.MolFromPDBBlock(self.as_pdb())

    @property
    @abstractmethod
    def representations(self) -> Iterable["StoredProtein"]:
        """Get all representations of the protein."""
