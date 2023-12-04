"""Fingerprint classes."""
from abc import ABC, abstractmethod

import numpy as np
from rdkit import DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, rdmolops


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
    def __init__(self, radius=2, nBits=2048, **kwargs):
        self.radius = radius
        self.nBits = nBits
        self.kwargs = kwargs

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
                mol, self.radius, nBits=self.nBits, **self.kwargs
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


class RDKitMACCSFP(Fingerprint):
    """RDKits implementation of MACCS keys fingerprint."""
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


class MaccsFP(Fingerprint):
    def __init__(self, nBits=167, **kwargs):
        self.nBits = nBits
        self.kwargs = kwargs

    def getFingerprints(self, mols):
        convertFP = DataStructs.ConvertToNumpyArray
        ret = np.zeros((len(mols), len(self)))
        for idx, mol in enumerate(mols):
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol, **self.kwargs)
            np_fp = np.zeros(len(fp))
            convertFP(fp, np_fp)
            ret[idx] = np_fp
        return ret

    @property
    def settings(self):
        return {"nBits": self.nBits}

    def __len__(self):
        return self.nBits

    def getKey(self):
        return "MACCSFP"


class AvalonFP(Fingerprint):
    def __init__(self, nBits=1024, **kwargs):
        self.nBits = nBits
        self.kwargs = kwargs

    def getFingerprints(self, mols):
        convertFP = DataStructs.ConvertToNumpyArray
        ret = np.zeros((len(mols), len(self)))
        for idx, mol in enumerate(mols):
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=self.nBits, **self.kwargs)
            np_fp = np.zeros(len(fp))
            convertFP(fp, np_fp)
            ret[idx] = np_fp
        return ret

    @property
    def settings(self):
        return {"nBits": self.nBits}

    def __len__(self):
        return self.nBits

    def getKey(self):
        return "AvalonFP"


class TopologicalFP(Fingerprint):
    def __init__(self, nBits=2048, **kwargs):
        self.nBits = nBits
        self.kwargs = kwargs

    def getFingerprints(self, mols):
        convertFP = DataStructs.ConvertToNumpyArray
        ret = np.zeros((len(mols), len(self)))
        for idx, mol in enumerate(mols):
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                mol, nBits=self.nBits, **self.kwargs
            )
            np_fp = np.zeros(len(fp))
            convertFP(fp, np_fp)
            ret[idx] = np_fp
        return ret

    @property
    def settings(self):
        return {"nBits": self.nBits}

    def __len__(self):
        return self.nBits

    def getKey(self):
        return "TopologicalFP"


class AtomPairFP(Fingerprint):
    def __init__(self, nBits=2048, **kwargs):
        self.nBits = nBits
        self.kwargs = kwargs

    def getFingerprints(self, mols):
        convertFP = DataStructs.ConvertToNumpyArray
        ret = np.zeros((len(mols), len(self)))
        for idx, mol in enumerate(mols):
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol, nBits=self.nBits, **self.kwargs
            )
            np_fp = np.zeros(len(fp))
            convertFP(fp, np_fp)
            ret[idx] = np_fp
        return ret

    @property
    def settings(self):
        return {"nBits": self.nBits}

    def __len__(self):
        return self.nBits

    def getKey(self):
        return "AtomPairFP"


class RDKitFP(Fingerprint):
    def __init__(self, minPath=1, maxPath=7, nBits=2048, **kwargs):
        self.minPath = minPath
        self.maxPath = maxPath
        self.nBits = nBits
        self.kwargs = kwargs

    def getFingerprints(self, mols):
        convertFP = DataStructs.ConvertToNumpyArray
        ret = np.zeros((len(mols), len(self)))
        for idx, mol in enumerate(mols):
            fp = rdmolops.RDKFingerprint(
                mol,
                minPath=self.minPath,
                maxPath=self.maxPath,
                fpSize=self.nBits,
                **self.kwargs,
            )
            np_fp = np.zeros(len(fp))
            convertFP(fp, np_fp)
            ret[idx] = np_fp
        return ret

    @property
    def settings(self):
        return {"minPath": self.minPath, "maxPath": self.maxPath, "nBits": self.nBits}

    def __len__(self):
        return self.nBits

    def getKey(self):
        return "RDKitFP"


class PatternFP(Fingerprint):
    def __init__(self, nBits=2048, **kwargs):
        self.nBits = nBits
        self.kwargs = kwargs

    def getFingerprints(self, mols):
        convertFP = DataStructs.ConvertToNumpyArray

        ret = np.zeros((len(mols), len(self)))
        for idx, mol in enumerate(mols):
            fp = rdmolops.PatternFingerprint(mol, fpSize=self.nBits, **self.kwargs)
            np_fp = np.zeros(len(fp))
            convertFP(fp, np_fp)
            ret[idx] = np_fp

        return ret

    @property
    def settings(self):
        return {"nBits": self.nBits}

    def __len__(self):
        return self.nBits

    def getKey(self):
        return "PatternFP"


class LayeredFP(Fingerprint):
    def __init__(self, minPath=1, maxPath=7, nBits=2048, **kwargs):
        self.minPath = minPath
        self.maxPath = maxPath
        self.nBits = nBits
        self.kwargs = kwargs

    def getFingerprints(self, mols):
        convertFP = DataStructs.ConvertToNumpyArray
        ret = np.zeros((len(mols), len(self)))
        for idx, mol in enumerate(mols):
            fp = rdmolops.LayeredFingerprint(
                mol,
                minPath=self.minPath,
                maxPath=self.maxPath,
                fpSize=self.nBits,
                **self.kwargs,
            )
            np_fp = np.zeros(len(fp))
            convertFP(fp, np_fp)
            ret[idx] = np_fp
        return ret

    @property
    def settings(self):
        return {"minPath": self.minPath, "maxPath": self.maxPath, "nBits": self.nBits}

    def __len__(self):
        return self.nBits

    def getKey(self):
        return "LayeredFP"


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

    def getMaccsFP(self, *args, **kwargs):
        return MaccsFP(*args, **kwargs)

    def getAvalonFP(self, *args, **kwargs):
        return AvalonFP(*args, **kwargs)

    def getTopologicalFP(self, *args, **kwargs):
        return TopologicalFP(*args, **kwargs)

    def getAtomPairFP(self, *args, **kwargs):
        return AtomPairFP(*args, **kwargs)

    def getRDKitFP(self, *args, **kwargs):
        return RDKitFP(*args, **kwargs)

    def getPatternFP(self, *args, **kwargs):
        return PatternFP(*args, **kwargs)

    def getLayeredFP(self, *args, **kwargs):
        return LayeredFP(*args, **kwargs)

    def getRDKitMACCSFP(self, *args, **kwargs):
        return RDKitMACCSFP(*args, **kwargs)

    def getCDKFP(self, *args, **kwargs):
        from qsprpred.extra.data.descriptors.fingerprints import CDKFP
        return CDKFP(*args, **kwargs)

    def getCDKExtendedFP(self, *args, **kwargs):
        from qsprpred.extra.data.descriptors.fingerprints import CDKExtendedFP
        return CDKExtendedFP(*args, **kwargs)

    def getCDKEStateFP(self, *args, **kwargs):
        from qsprpred.extra.data.descriptors.fingerprints import CDKEStateFP
        return CDKEStateFP()

    def getCDKGraphOnlyFP(self, *args, **kwargs):
        from qsprpred.extra.data.descriptors.fingerprints import CDKGraphOnlyFP
        return CDKGraphOnlyFP(*args, **kwargs)

    def getCDKMACCSFP(self, *args, **kwargs):
        from qsprpred.extra.data.descriptors.fingerprints import CDKMACCSFP
        return CDKMACCSFP()

    def getCDKPubchemFP(self, *args, **kwargs):
        from qsprpred.extra.data.descriptors.fingerprints import CDKPubchemFP
        return CDKPubchemFP()

    def getCDKSubstructureFP(self, *args, **kwargs):
        from qsprpred.extra.data.descriptors.fingerprints import \
            CDKSubstructureFP
        return CDKSubstructureFP(*args, **kwargs)

    def getCDKKlekotaRothFP(self, *args, **kwargs):
        from qsprpred.extra.data.descriptors.fingerprints import \
            CDKKlekotaRothFP
        return CDKKlekotaRothFP(*args, **kwargs)

    def getCDKAtomPairs2DFP(self, *args, **kwargs):
        from qsprpred.extra.data.descriptors.fingerprints import CDKAtomPairs2DFP
        return CDKAtomPairs2DFP(*args, **kwargs)


AVAIL_FPS = [
    m.lstrip("get")
    for m in dir(_FingerprintRetriever) if m.startswith("get") and m != "getFingerprint"
]


def get_fingerprint(fp_type: str, *args, **kwargs):
    return _FingerprintRetriever().getFingerprint(fp_type, *args, **kwargs)
