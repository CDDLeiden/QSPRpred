"""Fingerprint classes."""

from abc import ABC, abstractmethod

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem


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


class MorganFP(Fingerprint):
    """Morgan fingerprint."""
    def __init__(self, radius=2, nBits=2048):
        self.radius = radius
        self.nBits = nBits

    def getFingerprints(self, mols):
        """Return the Morgan fingerprints for the input molecules.

        Args:
            mols: molecules to obtain the fingerprint of

        Returns:
            fingerprint (list): `list` of fingerprints for "mols"
        """
        convertFP = DataStructs.ConvertToNumpyArray

        ret = np.zeros((len(mols), len(self)))
        for idx, mol in enumerate(mols):
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.nBits
            )
            np_fp = np.zeros(len(fp))
            convertFP(fp, np_fp)
            ret[idx] = np_fp

        return ret

    @property
    def settings(self):
        return {"radius": self.radius, "nBits": self.nBits}

    def __len__(self):
        return self.nBits

    def getKey(self):
        return "MorganFP"


class _FingerprintRetriever:
    """Based on recipe 8.21 of the book "Python Cookbook".

    To support a new type of fingerprint, just add a function
    `getFingerprintName(self, *args, **kwargs)`.
    """
    def getFingerprint(self, fp_type, *args, **kwargs):
        if fp_type.lower() == "fingerprint":
            raise Exception("Please specify the type of fingerprint you want to use.")
        method_name = "get" + fp_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f"{fp_type} is not a supported descriptor set type.")
        return method(*args, **kwargs)

    def getMorganFP(self, *args, **kwargs):
        return MorganFP(*args, **kwargs)

    def getCDKFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import CDKFP

        return CDKFP(*args, **kwargs)

    def getCDKExtendedFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import (
            CDKExtendedFP,
        )

        return CDKExtendedFP(*args, **kwargs)

    def getCDKEStateFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import CDKEStateFP

        return CDKEStateFP()

    def getCDKGraphOnlyFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import (
            CDKGraphOnlyFP,
        )

        return CDKGraphOnlyFP(*args, **kwargs)

    def getCDKMACCSFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import CDKMACCSFP

        return CDKMACCSFP()

    def getCDKPubchemFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import CDKPubchemFP

        return CDKPubchemFP()

    def getCDKSubstructureFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import (
            CDKSubstructureFP,
        )

        return CDKSubstructureFP(*args, **kwargs)

    def getCDKKlekotaRothFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import (
            CDKKlekotaRothFP,
        )

        return CDKKlekotaRothFP(*args, **kwargs)

    def getCDKAtomPairs2DFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import (
            CDKAtomPairs2DFP,
        )

        return CDKAtomPairs2DFP(*args, **kwargs)


AVAIL_FPS = [
    m.lstrip("get")
    for m in dir(_FingerprintRetriever) if m.startswith("get") and m != "getFingerprint"
]


def getFingerprint(fp_type: str, *args, **kwargs):
    return _FingerprintRetriever().getFingerprint(fp_type, *args, **kwargs)
