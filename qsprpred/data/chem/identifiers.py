from abc import ABC, abstractmethod

from rdkit import Chem


class ChemIdentifier(ABC):

    @abstractmethod
    def __call__(self, smiles: str) -> str:
        """
        Get the identifier of the molecule represented by the given SMILES.

        :param smiles: input SMILES
        :return: calculated identifier
        """


class Identifiable(ABC):

    @property
    @abstractmethod
    def identifier(self) -> ChemIdentifier:
        """
        Get the identifier used by the store.

        :return: `ChemIdentifier` instance
        """

    @abstractmethod
    def applyIdentifier(self, identifier: ChemIdentifier):
        pass


class InchiIdentifier(ChemIdentifier):

    def __call__(self, smiles: str) -> str:
        """
        Get the InChIKey of the molecule represented by the given SMILES.

        :param smiles: input SMILES
        :return: calculated InChIKey
        """
        return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


class IndexIdentifier(ChemIdentifier):
    """
    Implementation of a `ChemIdentifier` that returns an index as the identifier.
    """

    def __init__(self, zfill: int = 5):
        self.index = 0
        self.zfill = zfill

    def __call__(self, smiles: str) -> str:
        """
        Get the InChIKey of the molecule represented by the given SMILES.

        :param smiles: input SMILES
        :return: calculated InChIKey
        """
        self.index += 1
        return str(self.index).zfill(self.zfill)
