"""Fingerprint classes."""

from abc import ABC, abstractmethod

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem
from PaDEL_pywrapper import PaDEL as PaDEL_calculator
import PaDEL_pywrapper.descriptor as cdk_fps


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


class CDKFP(fingerprint):
    """CDK fingerprint."""

    def __init__(self, size=1024, searchDepth=7):
        self.size = size
        self.searchDepth = searchDepth
        fp = cdk_fps.FP(size=size, searchDepth=searchDepth)
        self._padel = PaDEL_calculator([fp])

    def getFingerprints(self, mols):
        """Return the CDK fingerprint for the input molecules.

        Args:
            mols: molecules to obtain the fingerprint of

        Returns:
            fingerprint (list): `list` of fingerprints for "mols"
        """
        return self._padel.calculate(mols, show_banner=False).values

    @property
    def settings(self):
        return {"size": self.size, "searchDepth": self.searchDepth}

    def __len__(self):
        return self.size

    def getKey(self):
        return "CDK-FP"


class CDKExtendedFP(fingerprint):
    """CDK extended fingerprint with 25 additional ring features and isotopic masses."""

    def __init__(self):
        fp = cdk_fps.ExtendedFP
        self._padel = PaDEL_calculator([fp])

    def getFingerprints(self, mols):
        """Return the CDK extended fingerprint for the input molecules.

        Args:
            mols: molecules to obtain the fingerprint of

        Returns:
            fingerprint (list): `list` of fingerprints for "mols"
        """
        return self._padel.calculate(mols, show_banner=False).values

    @property
    def settings(self):
        return {}

    def __len__(self):
        return 1024

    def getKey(self):
        return "CDK-ExtendedFP"


class CDKEStatedFP(fingerprint):
    """CDK EState fingerprint."""

    def __init__(self):
        fp = cdk_fps.EStateFP
        self._padel = PaDEL_calculator([fp])

    def getFingerprints(self, mols):
        """Return the CDK extended fingerprint for the input molecules.

        Args:
            mols: molecules to obtain the fingerprint of

        Returns:
            fingerprint (list): `list` of fingerprints for "mols"
        """
        return self._padel.calculate(mols, show_banner=False).values

    @property
    def settings(self):
        return {}

    def __len__(self):
        return 79

    def getKey(self):
        return "CDK-EStateFP"


class CDKGraphOnlyFP(fingerprint):
    """CDK fingerprint ignoring bond orders."""

    def __init__(self, searchDepth=7, size=1024):
        self.size = size
        self.searchDepth = searchDepth
        fp = cdk_fps.GraphOnlyFP(size=size, searchDepth=searchDepth)
        self._padel = PaDEL_calculator([fp])

    def getFingerprints(self, mols):
        """Return the CDK graph only fingerprint for the input molecules.

        Args:
            mols: molecules to obtain the fingerprint of

        Returns:
            fingerprint (list): `list` of fingerprints for "mols"
        """
        return self._padel.calculate(mols, show_banner=False).values

    @property
    def settings(self):
        return {"size": self.size, "searchDepth": self.searchDepth}

    def __len__(self):
        return self.size

    def getKey(self):
        return "CDK-GraphOnlyFP"


class CDKMACCSFP(fingerprint):
    """CDK MACCS fingerprint."""

    def __init__(self):
        fp = cdk_fps.MACCSFP
        self._padel = PaDEL_calculator([fp])

    def getFingerprints(self, mols):
        """Return the CDK MACCS fingerprint for the input molecules.

        Args:
            mols: molecules to obtain the fingerprint of

        Returns:
            fingerprint (list): `list` of fingerprints for "mols"
        """
        return self._padel.calculate(mols, show_banner=False).values

    @property
    def settings(self):
        return {}

    def __len__(self):
        return 166

    def getKey(self):
        return "CDK-MACCSFP"


class CDKPubchemFP(fingerprint):
    """CDK PubChem fingerprint."""

    def __init__(self):
        fp = cdk_fps.PubchemFP
        self._padel = PaDEL_calculator([fp])

    def getFingerprints(self, mols):
        """Return the CDK PubChem fingerprint for the input molecules.

        Args:
            mols: molecules to obtain the fingerprint of

        Returns:
            fingerprint (list): `list` of fingerprints for "mols"
        """
        return self._padel.calculate(mols, show_banner=False).values

    @property
    def settings(self):
        return {}

    def __len__(self):
        return 881

    def getKey(self):
        return "CDK-PubchemFP"


class CDKSubstructureFP(fingerprint):
    """CDK Substructure fingerprint.

    Based on SMARTS patterns for functional group classification by Christian Laggner.
    """

    def __init__(self, useCounts: bool = False):
        self.useCounts = useCounts
        if useCounts:
            fp = cdk_fps.SubstructureFPCount
        else:
            fp = cdk_fps.SubstructureFP
        self._padel = PaDEL_calculator([fp])

    def getFingerprints(self, mols):
        """Return the CDK Substructure fingerprint for the input molecules.

        Args:
            mols: molecules to obtain the fingerprint of

        Returns:
            fingerprint (list): `list` of fingerprints for "mols"
        """
        return self._padel.calculate(mols, show_banner=False).values

    @property
    def settings(self):
        return {'useCounts': self.useCounts}

    def __len__(self):
        return 307

    def getKey(self):
        return "CDK-SubstructureFP"


class CDKKlekotaRothFP(fingerprint):
    """CDK Klekota & Roth fingerprint."""

    def __init__(self, useCounts: bool = False):
        self.useCounts = useCounts
        if useCounts:
            fp = cdk_fps.KlekotaRothFPCount
        else:
            fp = cdk_fps.KlekotaRothFP
        self._padel = PaDEL_calculator([fp])

    def getFingerprints(self, mols):
        """Return the CDK Klekota & Roth fingerprint for the input molecules.

        Args:
            mols: molecules to obtain the fingerprint of

        Returns:
            fingerprint (list): `list` of fingerprints for "mols"
        """
        return self._padel.calculate(mols, show_banner=False).values

    @property
    def settings(self):
        return {'useCounts': self.useCounts}

    def __len__(self):
        return 4860

    def getKey(self):
        return "CDK-KlekotaRothFP"


class CDKAtomPairs2DFP(fingerprint):
    """CDK atom pairs and topological fingerprint."""

    def __init__(self, useCounts: bool = False):
        self.useCounts = useCounts
        if useCounts:
            fp = cdk_fps.AtomPairs2DFPCount
        else:
            fp = cdk_fps.AtomPairs2DFP
        self._padel = PaDEL_calculator([fp])

    def getFingerprints(self, mols):
        """Return the CDK atom pairs and topological fingerprint for the input molecules.

        Args:
            mols: molecules to obtain the fingerprint of

        Returns:
            fingerprint (list): `list` of fingerprints for "mols"
        """
        return self._padel.calculate(mols, show_banner=False).values

    @property
    def settings(self):
        return {'useCounts': self.useCounts}

    def __len__(self):
        return 780

    def getKey(self):
        return "CDK-AtomPairs2DFP"


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

    def get_CDKFP(self, *args, **kwargs):
        return CDKFP(*args, **kwargs)

    def get_CDKExtendedFP(self, *args, **kwargs):
        return CDKExtendedFP(*args, **kwargs)

    def get_CDKEStatedFP(self, *args, **kwargs):
        return CDKEStatedFP()

    def get_CDKGraphOnlyFP(self, *args, **kwargs):
        return CDKGraphOnlyFP(*args, **kwargs)

    def get_CDKMACCSFP(self, *args, **kwargs):
        return CDKMACCSFP()

    def get_CDKPubchemFP(self, *args, **kwargs):
        return CDKPubchemFP()

    def get_CDKSubstructureFP(self, *args, **kwargs):
        return CDKSubstructureFP(*args, **kwargs)

    def get_CDKKlekotaRothFP(self, *args, **kwargs):
        return CDKKlekotaRothFP(*args, **kwargs)

    def get_CDKAtomPairs2DFP(self, *args, **kwargs):
        return CDKAtomPairs2DFP(*args, **kwargs)


def get_fingerprint(fp_type: str, *args, **kwargs):
    return _FingerprintRetriever().get_fingerprint(fp_type, *args, **kwargs)
