"""Fingerprint classes."""

from abc import ABC, abstractmethod

from rdkit.Chem import AllChem


class fingerprint(ABC):
    """Base class for fingerprints."""

    def __call__(self, mol):
        """Actual call method.

        Args:
            mol: molecule to fingerprint

        Returns:
            fingerprint (list): `list` of fingerprint for "mol"
        """
        return self.getFingerprint(mol)

    @abstractmethod
    def settings(self):
        """Return settings of fingerprint."""
        pass

    @abstractmethod
    def __len__(self):
        """Return length of fingerprint."""
        pass

    @abstractmethod
    def getKey(self):
        """Return identifier of fingerprint."""
        pass


class MorganFP(fingerprint):
    """Morgan fingerprint."""

    def __init__(self, radius=2, nBits=2048):
        self.radius = radius
        self.nBits = nBits

    def getFingerprint(self, mol):
        """Return Morgan fingerprint for the input molecule.

        Args:
            mol: molecule to fingerprint

        Returns:
            fingerprint (list): `list` of fingerprint for "mol"
        """
        return AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nBits)

    @property
    def settings(self):
        return {"radius": self.radius, "nBits": self.nBits}

    def __len__(self):
        return self.nBits

    def getKey(self):
        return "MorganFP"


class _FingerprintRetriever:
    """Based on recipe 8.21 of the book "Python Cookbook".

    To support a new type of fingerprint, just add a function "get_fingerprintname(self, *args, **kwargs)".
    """

    def get_fingerprint(self, fp_type, *args, **kwargs):
        method_name = "get_" + fp_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f"{fp_type} is not a supported descriptor set type.")
        return method(*args, **kwargs)

    def get_MorganFP(self, *args, **kwargs):
        return MorganFP(*args, **kwargs)


def get_fingerprint(fp_type: str, *args, **kwargs):
    return _FingerprintRetriever().get_fingerprint(fp_type, *args, **kwargs)
