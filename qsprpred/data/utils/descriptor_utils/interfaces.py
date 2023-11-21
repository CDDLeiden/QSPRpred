"""Definitions of interfaces pertaining to descriptor and fingerprint sets."""

from abc import ABC, abstractmethod


class Fingerprint(ABC):
    """Base class for fingerprints."""
    def __call__(self, mols):
        """Actual call method.

        Args:
            mols: molecules to obtain the fingerprints of

        Returns:
            fingerprint (list): `list` of fingerprints for "mols"
        """
        return self.getFingerprints(mols)

    @abstractmethod
    def settings(self):
        """Return settings of fingerprint."""

    @abstractmethod
    def __len__(self):
        """Return length of fingerprint."""

    @abstractmethod
    def getKey(self):
        """Return identifier of fingerprint."""
