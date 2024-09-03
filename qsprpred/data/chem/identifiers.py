from abc import ABC, abstractmethod

from rdkit import Chem


class ChemIdentifier(ABC):
    """Interface for identifiers of molecules."""

    @abstractmethod
    def __call__(self, smiles: str) -> str:
        """ Get the identifier of the molecule represented by the given SMILES.

        Args:
            smiles (str): input SMILES
            
        Returns:
            str: calculated identifier
        """


class Identifiable(ABC):
    """Interface for objects which have molecule identifiers.
    
    Attributes:
        identifier (ChemIdentifier): The identifier used by the store.
    """

    @property
    @abstractmethod
    def identifier(self) -> ChemIdentifier:
        """Get the identifier used by the store.

        Returns:
            ChemIdentifier: The identifier used by the store.
        """

    @abstractmethod
    def applyIdentifier(self, identifier: ChemIdentifier):
        """Apply an identifier to the SMILES in the store.
        
        Args:
            identifier (ChemIdentifier): The identifier to apply.
        """
        pass


class InchiIdentifier(ChemIdentifier):
    """Class for InChI identifiers of molecules."""

    def __call__(self, smiles: str) -> str:
        """Get the InChIKey of the molecule represented by the given SMILES.

        Args:
            smiles (str): input SMILES
            
        Returns:
            str: calculated InChIKey
        """
        return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


class IndexIdentifier(ChemIdentifier):
    """Implementation of a `ChemIdentifier` that returns an index as the identifier.
    
    Attributes:
        index (int): The current index.
        zfill (int): The number of digits to zero-fill the index
    """

    def __init__(self, zfill: int = 5):
        """Initialize the index identifier.
        
        Args:
            zfill (int): The number of digits to zero-fill the index
        """
        self.index = 0
        self.zfill = zfill

    def __call__(self, smiles: str) -> str:
        """Get the index as the molecule identifier.

        Args:
            smiles (str): input SMILES
            
        Returns:
            str: calculated identifier
        """
        self.index += 1
        return str(self.index).zfill(self.zfill)
