from abc import ABC, abstractmethod
from typing import Any, Iterable

from rdkit import Chem


class StoredProtein(ABC):
    """
    A protein object
    """

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @property
    @abstractmethod
    def sequence(self) -> str | None:
        pass

    @property
    @abstractmethod
    def props(self) -> dict[str, Any] | None:
        pass

    @abstractmethod
    def as_pdb(self) -> str | None:
        pass

    @abstractmethod
    def as_fasta(self) -> str | None:
        pass

    def as_rd_mol(self) -> Chem.Mol | None:
        pdb = self.as_pdb()
        if pdb is not None:
            return Chem.MolFromPDBBlock(self.as_pdb())

    @property
    @abstractmethod
    def representations(self) -> Iterable["StoredProtein"]:
        pass
