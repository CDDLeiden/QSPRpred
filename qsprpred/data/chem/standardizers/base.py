from abc import ABC, abstractmethod


class ChemStandardizationException(Exception):
    """Exception raised when standardization fails."""


class ChemStandardizer(ABC):
    """Standardizer to convert SMILES to a standardized form.

    This class defines an interface of a uniquely identifiable standardizer.
    The `getID` method should return a unique identifier for the standardizer
    based on its settings. Standardizes that have the same ID should produce
    the same standardized form for a given SMILES.

    The main method of the class is `convertSMILES`, which should convert
    a given SMILES to a standardized form based on the settings of the standardizer.
    """
    def __call__(self, smiles: str) -> str | None:
        """Convert the SMILES to a standardized form. Simply calls `convertSMILES`.

        Args:
            smiles (str): SMILES to be converted

        Returns:
            str | None:
                The standardized SMILES string or `None` if standardization fails or
                the molecule is deemed invalid.

        Raises:
            ChemStandardizationException:
                if standardization fails, but the upstream code should be notified
                and handle the exception.
        """
        return self.convertSMILES(smiles)

    @abstractmethod
    def convertSMILES(self, smiles: str) -> str | None:
        """Convert the SMILES to a standardized form.

        Args:
            smiles (str): SMILES to be converted

        Returns:
            str | None:
                The standardized SMILES string or `None` if standardization fails or
                the molecule is deemed invalid.

        Raises:
            ChemStandardizationException:
                if standardization fails, but the upstream code should be notified
                and handle the exception.
        """

    @property
    @abstractmethod
    def settings(self) -> dict:
        """Settings of the standardizer. It should contain complete
        information needed to initialize another equivalent standardizer.
        """

    @abstractmethod
    def getID(self) -> str:
        """Return the unique identifier of the standardizer. This method should
        return a unique identifier based on the settings of the standardizer.

        Two standardizers with the same settings should have the same ID and
        produce the same standardized form for a given SMILES.

        Returns:
            str: The unique identifier of the standardizer.
        """

    @classmethod
    @abstractmethod
    def fromSettings(cls, settings: dict) -> "ChemStandardizer":
        """Create a new standardizer from a settings dictionary."""

    @classmethod
    def fromSettingsFile(cls, path: str) -> "ChemStandardizer":
        """Load the standardizer from a settings file in JSON format.

        Args:
            path (str): Path to the settings file.

        Returns:
            ChemStandardizer: The standardizer loaded from the settings file.
        """
        import json

        with open(path, "r") as f:
            settings = json.load(f)
        return cls.fromSettings(settings)

    def getHashID(self) -> str:
        """Get the hash ID of the standardizer. This is simply the MD5 hash of the
        unique identifier of the standardizer.

        Returns:
            str: The hash ID of the standardizer
        """
        import hashlib

        return hashlib.md5(self.getID().encode("utf-8")).hexdigest()


class Standardizable(ABC):
    """Interface for objects that use chemical standardization with `
    `ChemStandardizer` objects.
    """
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
