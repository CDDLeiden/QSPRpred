"""Extra fingerprints from various packages:

- `CDKFP`: CDK fingerprint
- `CDKExtendedFP`: CDK extended fingerprint
- `CDKEStateFP`: CDK EState fingerprint
- `CDKGraphOnlyFP`: CDK fingerprint ignoring bond orders
- `CDKMACCSFP`: CDK MACCS fingerprint
- `CDKPubchemFP`: CDK PubChem fingerprint
- `CDKSubstructureFP`: CDK Substructure fingerprint
- `CDKAtomPairs2DFP`: CDK hashed atom pair fingerprint
- `CDKKlekotaRothFP`: CDK hashed Klekota-Roth fingerprint

"""
from typing import Any

import numpy as np
from PaDEL_pywrapper import PaDEL as PaDEL_calculator
from PaDEL_pywrapper import descriptor as cdk_fps
from rdkit import Chem
from rdkit.Chem import Mol

from qsprpred.data.descriptors.fingerprints import Fingerprint


class CDKFP(Fingerprint):
    """The CDK fingerprint.

    Attributes:
        size (int): size of the fingerprint
        searchDepth (int): search depth of the fingerprint
    """

    def __init__(self, size: int = 1024, search_depth: int = 7):
        """Initialize the CDK fingerprint.

        Args:
            size (int): size of the fingerprint
            search_depth (int): search depth of the fingerprint
        """
        self.size = size
        super().__init__()
        self.searchDepth = search_depth
        fp = cdk_fps.FP(size=size, searchDepth=search_depth)
        self._padel = PaDEL_calculator([fp])

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Return the CDK fingerprint for the input molecules.

        Args:
            mols (list[Chem.Mol]):
                molecules to obtain the fingerprint of

        Returns:
            np.ndarray: `np.ndarray` of fingerprints for `mols`

        """
        return self._padel.calculate(mols, show_banner=False).values

    @property
    def settings(self):
        return {"size": self.size, "search_depth": self.searchDepth}

    def __len__(self):
        return self.size

    def __str__(self):
        return "CDKFP"


class CDKExtendedFP(Fingerprint):
    """CDK extended fingerprint with 25 additional ring features and isotopic masses."""

    def __init__(self):
        super().__init__()
        fp = cdk_fps.ExtendedFP
        self._padel = PaDEL_calculator([fp])

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Return the CDK extended fingerprint for the input molecules.

        Args:
            mols (list[Chem.Mol]):
                molecules to obtain the fingerprint of

        Returns:
            np.ndarray: `np.ndarray` of fingerprints for `mols`
        """
        return self._padel.calculate(mols, show_banner=False).values

    def __len__(self):
        return 1024

    def __str__(self):
        return "CDKExtendedFP"


class CDKEStateFP(Fingerprint):
    """CDK EState fingerprint."""

    def __init__(self):
        super().__init__()
        fp = cdk_fps.EStateFP
        self._padel = PaDEL_calculator([fp])

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Return the CDK estate fingerprint for the input molecules.

        Args:
            mols (list[Chem.Mol]):
                molecules to obtain the fingerprint of

        Returns:
            np.ndarray: `np.ndarray` of fingerprints for `mols`
        """
        return self._padel.calculate(mols, show_banner=False).values

    def __len__(self):
        return 79

    def __str__(self):
        return "CDKEStateFP"


class CDKGraphOnlyFP(Fingerprint):
    """CDK fingerprint ignoring bond orders.

    Attributes:
        size (int): Number of bits in the CDK fingerprints (ignored for others)
        searchDepth (int): Search depth for the CDK fingerprints (ignored for others)
    """

    def __init__(self, size: int = 1024, search_depth: int = 7):
        self.size = size
        super().__init__()
        self.searchDepth = search_depth
        fp = cdk_fps.GraphOnlyFP(size=size, searchDepth=self.searchDepth)
        self._padel = PaDEL_calculator([fp])

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Return the CDK graph only fingerprint for the input molecules.

        Args:
            mols (list[Chem.Mol]):
                molecules to obtain the fingerprint of

        Returns:
            np.ndarray: `np.ndarray` of fingerprints for `mols`
        """
        return self._padel.calculate(mols, show_banner=False).values

    def __len__(self):
        return self.size

    def __str__(self):
        return "CDKGraphOnlyFP"


class CDKMACCSFP(Fingerprint):
    """CDK MACCS fingerprint."""

    def __init__(self):
        super().__init__()
        fp = cdk_fps.MACCSFP
        self._padel = PaDEL_calculator([fp])

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Return the CDK MACCS fingerprint for the input molecules.

        Args:
            mols (list[Chem.Mol]):
                molecules to obtain the fingerprint of

        Returns:
            np.ndarray: `np.ndarray` of fingerprints for `mols`
        """
        return self._padel.calculate(mols, show_banner=False).values

    def __len__(self):
        return 166

    def __str__(self):
        return "CDKMACCSFP"


class CDKPubchemFP(Fingerprint):
    """CDK PubChem fingerprint."""

    def __init__(self):
        super().__init__()
        fp = cdk_fps.PubchemFP
        self._padel = PaDEL_calculator([fp])

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Return the CDK PubChem fingerprint for the input molecules.

        Args:
            mols: molecules to obtain the fingerprint of

        Returns:
            fingerprint (list): `list` of fingerprints for "mols"
        """
        return self._padel.calculate(mols, show_banner=False).values

    def __len__(self):
        return 881

    def __str__(self):
        return "CDKPubchemFP"


class CDKSubstructureFP(Fingerprint):
    """CDK Substructure fingerprint.

    Based on SMARTS patterns for functional group classification by Christian Laggner.

    Attributes:
        useCounts (bool):
            whether to use counts instead of presence/absence
    """

    def __init__(self, use_counts: bool = False):
        super().__init__()
        self.useCounts = use_counts
        if use_counts:
            fp = cdk_fps.SubstructureFPCount
        else:
            fp = cdk_fps.SubstructureFP
        self._padel = PaDEL_calculator([fp])

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Return the CDK Substructure fingerprint for the input molecules.

        Args:
            mols (list[Chem.Mol]):
                molecules to obtain the fingerprint of

        Returns:
            np.ndarray: `np.ndarray` of fingerprints for `mols`
        """
        return self._padel.calculate(mols, show_banner=False).values

    def __len__(self):
        return 307

    def __str__(self):
        if self.useCounts:
            return "CDKSubstructureFPCount"
        return "CDKSubstructureFP"


class CDKKlekotaRothFP(Fingerprint):
    """CDK Klekota & Roth fingerprint."""

    def __init__(self, use_counts: bool = False):
        """Initialise the fingerprint.

        Args:
            use_counts (bool):
                whether to use counts instead of presence/absence
        """
        super().__init__()
        self.useCounts = use_counts
        if use_counts:
            fp = cdk_fps.KlekotaRothFPCount
        else:
            fp = cdk_fps.KlekotaRothFP
        self._padel = PaDEL_calculator([fp])

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Return the CDK Klekota & Roth fingerprint for the input molecules.

        Args:
            mols (list[Chem.Mol]):
                molecules to obtain the fingerprint of

        Returns:
            np.ndarray: `np.ndarray` of fingerprints for `mols`
        """
        return self._padel.calculate(mols, show_banner=False).values

    def __len__(self):
        return 4860

    def __str__(self):
        if self.useCounts:
            return "CDKKlekotaRothFPCount"
        return "CDKKlekotaRothFP"


class CDKAtomPairs2DFP(Fingerprint):
    """CDK atom pairs and topological fingerprint.

    Attributes:
        useCounts (bool):
            whether to use counts instead of presence/absence
    """

    def __init__(self, use_counts: bool = False):
        """
        Initialise the fingerprint.

        Args:
            use_counts:
                whether to use counts instead of presence/absence
        """
        super().__init__()
        self.useCounts = use_counts
        if use_counts:
            fp = cdk_fps.AtomPairs2DFPCount
        else:
            fp = cdk_fps.AtomPairs2DFP
        self._padel = PaDEL_calculator([fp])

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """
        Return the CDK atom pairs and topological fingerprint for the input molecules.

        Args:
            mols (list[Chem.Mol]):
                molecules to obtain the fingerprint of

        Returns:
            np.ndarray: `np.ndarray` of fingerprints for `mols`
        """
        return self._padel.calculate(mols, show_banner=False).values

    def __len__(self):
        return 780

    def __str__(self):
        if self.useCounts:
            return "CDKAtomPairs2DFPCount"
        return "CDKAtomPairs2DFP"
