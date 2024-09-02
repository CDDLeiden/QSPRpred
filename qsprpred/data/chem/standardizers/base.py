from abc import ABC, abstractmethod


class ChemStandardizer(ABC):
    """Standardizer to convert SMILES to a standardized form."""

    def __call__(self, smiles: str) -> tuple[str | None, str]:
        """Convert the SMILES to a standardized form.
        
        Args:
            smiles (str): SMILES to be converted
        
        Returns:
            (tuple[str | None, str]): 
                a tuple where the first element is the 
                standardized SMILES and the second element is the original SMILES
        """
        return self.convert_smiles(smiles)

    @abstractmethod
    def convert_smiles(self, smiles: str) -> tuple[str | None, str]:
        """Convert the SMILES to a standardized form.

        Args:
            smiles (str): SMILES to be converted
            
        Returns:
            (tuple[str | None, str]): 
                a tuple where the first element is the 
                standardized SMILES and the second element is the original SMILES
        """
        pass

    @property
    @abstractmethod
    def settings(self):
        """Settings of the standardizer."""
        pass

    @abstractmethod
    def get_id(self) -> str:
        """Return the unique identifier of the standardizer."""
        pass

    @classmethod
    @abstractmethod
    def from_settings(cls, settings: dict) -> "ChemStandardizer":
        """Create a standardizer from settings."""
        pass

    @classmethod
    def from_settings_file(cls, path: str) -> "ChemStandardizer":
        """Load the standardizer from a settings file in JSON format.

        Args:
            path (str): Path to the settings file.
            
        Returns:
            ChemStandardizer: The standardizer loaded from the settings file.
        """
        import json

        with open(path, "r") as f:
            settings = json.load(f)
        return cls.from_settings(settings)

    def get_hash_id(self) -> str:
        """Get the hash ID of the standardizer.
        
        Returns:
            str: The hash ID of the standardizer
        """
        import hashlib

        return hashlib.md5(self.get_id()).hexdigest()


class Standardizable(ABC):
    """Interface for objects that can be standardized."""

    @property
    @abstractmethod
    def standardizer(self) -> ChemStandardizer:
        """Get the standardizer used by the store.

        Returns:
            ChemStandardizer: The standardizer used by the store.
        """

    @abstractmethod
    def applyStandardizer(self, standardizer: ChemStandardizer):
        """Apply a standardizer to the SMILES in the store.
        
        Args:
            standardizer (ChemStandardizer): The standardizer to apply
        """
        pass


class ChemStandardizationException(Exception):
    """Exception raised when standardization fails."""
    pass
