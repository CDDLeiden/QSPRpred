"""Fingerprint classes."""

from abc import ABC, abstractmethod

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys


class fingerprint(ABC):
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
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nBits)
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

class RDkitMACCSFP(fingerprint):
    """RDkits implementation of MACCS keys fingerprint."""

    def getFingerprints(self, mols):
        """Return the MACCS fingerprints for the input molecules.

        Args:
            mols: molecules to obtain the fingerprint of

        Returns:
            fingerprint (list): `list` of fingerprints for "mols"
        """
        convertFP = DataStructs.ConvertToNumpyArray

        ret = np.zeros((len(mols), len(self)))
        for idx, mol in enumerate(mols):
            fp = MACCSkeys.GenMACCSKeys(mol)
            np_fp = np.zeros(len(fp))
            convertFP(fp, np_fp)
            ret[idx] = np_fp

        return ret

    @property
    def settings(self):
        return {}

    def __len__(self):
        return 167

    def getKey(self):
        return "RDKitMACCSFP"

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
    
    def get_RDKitMACCSFP(self, *args, **kwargs):
        return RDkitMACCSFP(*args, **kwargs)

    def get_CDKFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import CDKFP
        return CDKFP(*args, **kwargs)

    def get_CDKExtendedFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import CDKExtendedFP
        return CDKExtendedFP(*args, **kwargs)

    def get_CDKEStateFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import CDKEStateFP
        return CDKEStateFP()

    def get_CDKGraphOnlyFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import CDKGraphOnlyFP
        return CDKGraphOnlyFP(*args, **kwargs)

    def get_CDKMACCSFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import CDKMACCSFP
        return CDKMACCSFP()

    def get_CDKPubchemFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import CDKPubchemFP
        return CDKPubchemFP()

    def get_CDKSubstructureFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import CDKSubstructureFP
        return CDKSubstructureFP(*args, **kwargs)

    def get_CDKKlekotaRothFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import CDKKlekotaRothFP
        return CDKKlekotaRothFP(*args, **kwargs)

    def get_CDKAtomPairs2DFP(self, *args, **kwargs):
        from qsprpred.extra.data.utils.descriptor_utils.fingerprints import CDKAtomPairs2DFP
        return CDKAtomPairs2DFP(*args, **kwargs)


def get_fingerprint(fp_type: str, *args, **kwargs):
    return _FingerprintRetriever().get_fingerprint(fp_type, *args, **kwargs)
