"""
fingerprints

Created by: Martin Sicho
On: 12.05.23, 16:25
"""
from PaDEL_pywrapper import PaDEL as PaDEL_calculator
from PaDEL_pywrapper import descriptor as cdk_fps

from qsprpred.data.utils.descriptor_utils.interfaces import Fingerprint


class CDKFP(Fingerprint):
    """Class for calculating CDK fingerprint.

    Attributes:
        size (int): Number of bits in the CDK fingerprints (ignored for others)
        searchDepth (int): Search depth for the CDK fingerprints (ignored for others)
    """
    def __init__(self, size: int = 1024, searchDepth: int = 7):
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
        return "CDKFP"


class CDKExtendedFP(Fingerprint):
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
        return "CDKExtendedFP"


class CDKEStateFP(Fingerprint):
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
        return "CDKEStateFP"


class CDKGraphOnlyFP(Fingerprint):
    """CDK fingerprint ignoring bond orders.

    Attributes:
        size (int): Number of bits in the CDK fingerprints (ignored for others)
        searchDepth (int): Search depth for the CDK fingerprints (ignored for others)
    """
    def __init__(self, size: int = 1024, searchDepth: int = 7):
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
        return "CDKGraphOnlyFP"


class CDKMACCSFP(Fingerprint):
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
        return "CDKMACCSFP"


class CDKPubchemFP(Fingerprint):
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
        return "CDKPubchemFP"


class CDKSubstructureFP(Fingerprint):
    """CDK Substructure fingerprint.

    Based on SMARTS patterns for functional group classification by Christian Laggner.

    Attributes:
        useCounts (bool): If True, use the count version of the fingerprint.
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
        return {"useCounts": self.useCounts}

    def __len__(self):
        return 307

    def getKey(self):
        if self.useCounts:
            return "CDKSubstructureFPCount"
        return "CDKSubstructureFP"


class CDKKlekotaRothFP(Fingerprint):
    """CDK Klekota & Roth fingerprint.

    Attributes:
        useCounts (bool): If True, use the count version of the fingerprint.
    """
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
        return {"useCounts": self.useCounts}

    def __len__(self):
        return 4860

    def getKey(self):
        if self.useCounts:
            return "CDKKlekotaRothFPCount"
        return "CDKKlekotaRothFP"


class CDKAtomPairs2DFP(Fingerprint):
    """CDK atom pairs and topological fingerprint."""
    def __init__(self, useCounts: bool = False):
        self.useCounts = useCounts
        if useCounts:
            fp = cdk_fps.AtomPairs2DFPCount
        else:
            fp = cdk_fps.AtomPairs2DFP
        self._padel = PaDEL_calculator([fp])

    def getFingerprints(self, mols):
        """Return the CDK atom pairs and topological fingerprint for the input
        molecules.

        Args:
            mols: molecules to obtain the fingerprint of

        Returns:
            fingerprint (list): `list` of fingerprints for "mols"
        """
        return self._padel.calculate(mols, show_banner=False).values

    @property
    def settings(self):
        return {"useCounts": self.useCounts}

    def __len__(self):
        return 780

    def getKey(self):
        if self.useCounts:
            return "CDKAtomPairs2DFPCount"
        return "CDKAtomPairs2DFP"
