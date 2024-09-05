"""Fingerprint classes."""

from abc import ABC
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, MACCSkeys, Mol, rdMolDescriptors, rdmolops

from qsprpred.data.descriptors.sets import DescriptorSet
from qsprpred.data.storage.interfaces.stored_mol import StoredMol


class Fingerprint(DescriptorSet, ABC):
    """Base class for calculation of binary fingerprints.

    Attributes:
        usedBits (list): list of bits of the fingerprint currently being used
        descriptors (list): list of descriptors
        isFP (bool): Whether the descriptor is a fingerprint
        dtype (type): Data type of the descriptor
    """
    def __init__(self, used_bits: list[int] | None = None):
        """Initialize the fingerprint.

        Args:
            used_bits (list): list of bits of the fingerprint currently being used
        """
        super().__init__()
        self.usedBits = used_bits or list(range(len(self)))

    @property
    def usedBits(self) -> list[int] | None:
        """List of bits of the fingerprint currently being used."""
        return self._usedBits

    @usedBits.setter
    def usedBits(self, value: list[int]):
        """Set the list of bits of the fingerprint currently being used.

        Args:
            value (list): list of bits of the fingerprint currently being used
        """
        self._usedBits = sorted(value)

    @property
    def descriptors(self) -> list[str]:
        """list of descriptors."""
        return [f"{self}_{i}" for i in self.usedBits]

    @descriptors.setter
    def descriptors(self, value: list[str]):
        """Set the list of descriptors

        Args:
            value (list[str]): list of descriptors
        """
        self.usedBits = [int(x.split("_")[-1]) for x in sorted(value)]

    @property
    def isFP(self):
        """Whether the descriptor is a fingerprint."""
        return True

    @property
    def dtype(self):
        """Data type of the descriptor."""
        return bool

    def prepMols(self, mols: list[str | Mol]) -> list[Mol]:
        """Prepare the molecules by adding hydrogens.

        Args:
            mols (list[str | Mol]): list of SMILES or RDKit molecules

        Returns:
            (list[Mol]): list of RDKit molecules
        """
        return [Chem.AddHs(mol) for mol in self.iterMols(mols)]

    def __call__(
        self,
        mols: list[str | Mol | StoredMol],
        props: dict[str, list[Any]] | None = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """Calculate binary fingerprints for the input molecules. Only the bits
        specified by `usedBits` will be returned if more bits are calculated.

        Before calculating the fingerprints, the molecules are
        prepared by adding hydrogens (see `Fingerprint.prepMols`).
        If this is undesirable, the user can prepare the molecules
        themselves and call `Fingerprint.getDescriptors` directly.

        Args:
            mols(list): list of SMILES or RDKit molecules
            props(dict): dictionary of properties
            *args: positional arguments to pass to `Fingerprint.getDescriptors`
            **kwargs: keyword arguments to pass to `Fingerprint.getDescriptors`

        Returns:
            (pd.DataFrame) descriptor values of shape (n_mols, n_descriptors)
        """
        mols, props = self.parsePropsAndMols(mols, props)
        values = self.getDescriptors(self.prepMols(mols), props, *args, **kwargs)
        values = values[:, self.usedBits]
        values = values.astype(self.dtype)
        df = pd.DataFrame(
            values,
            index=pd.Index(props[self.idProp], name=self.idProp),
            columns=self.transformToFeatureNames(),
        )
        return df


class MorganFP(Fingerprint):
    """Morgan fingerprint.

    Attributes:
        radius (int): radius of the fingerprint
        nBits (int): number of bits in the fingerprint
        kwargs (dict):
            additional keyword arguments to pass to RDKit's
            `GetMorganFingerprintAsBitVect` function
    """
    def __init__(self, radius=2, nBits=2048, **kwargs):
        """Initialize the Morgan fingerprint.

        Args:
            radius (int): radius of the fingerprint
            nBits (int): number of bits in the fingerprint
            kwargs (dict): additional keyword arguments
        """
        super().__init__(used_bits=list(range(nBits)))
        self.radius = radius
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Calculate the Morgan fingerprints for the input molecules.

        Args:
            mols (list): list of RDKit molecules
            props (dict): dictionary of properties
            *args: positional arguments (not used)
            **kwargs: keyword arguments (not used, set in the constructor)

        Returns:
            (np.ndarray): descriptor values of shape (n_mols, n_descriptors)
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

    def __len__(self):
        return self.nBits

    def __str__(self):
        return "MorganFP"


class RDKitMACCSFP(Fingerprint):
    """RDKits implementation of MACCS keys fingerprint."""
    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Calculate the MACCS keys fingerprints for the input molecules.

        Args:
            mols (list): list of RDKit molecules
            props (dict): dictionary of properties (not used)
            *args: positional arguments (not used)
            **kwargs: keyword arguments (not used)

        Returns:
            np.ndarray of shape (n_mols, n_descriptors)
        """
        convertFP = DataStructs.ConvertToNumpyArray

        ret = np.zeros((len(mols), len(self)))
        for idx, mol in enumerate(mols):
            fp = MACCSkeys.GenMACCSKeys(mol)
            np_fp = np.zeros(len(fp))
            convertFP(fp, np_fp)
            ret[idx] = np_fp

        return ret

    def __len__(self):
        return 167

    def __str__(self):
        return "RDKitMACCSFP"


class MaccsFP(Fingerprint):
    """MACCS keys fingerprint.

    Attributes:
        nBits (int): number of bits in the fingerprint
        kwargs (dict):
            additional keyword arguments to pass to RDKit's `GetMACCSKeysFingerprint`
            function
    """
    def __init__(self, nBits=167, **kwargs):
        """Initialize the MACCS keys fingerprint.

        Args:
            nBits (int): number of bits in the fingerprint
            kwargs (dict):
                additional keyword arguments to pass to RDKit's
                `GetMACCSKeysFingerprint` function
        """
        super().__init__(used_bits=list(range(nBits)))
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Calculate the MACCS keys fingerprints for the input molecules.

        Args:
            mols (list): list of RDKit molecules
            props (dict): dictionary of properties (not used)
            *args: positional arguments (not used)
            **kwargs: keyword arguments (not used, set in the constructor)

        Returns:
            (np.ndarray): descriptor values of shape (n_mols, n_descriptors)
        """
        convertFP = DataStructs.ConvertToNumpyArray
        ret = np.zeros((len(mols), len(self)))
        for idx, mol in enumerate(mols):
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol, **self.kwargs)
            np_fp = np.zeros(len(fp))
            convertFP(fp, np_fp)
            ret[idx] = np_fp
        return ret

    def __len__(self):
        return self.nBits

    def __str__(self):
        return "MACCSFP"


class AvalonFP(Fingerprint):
    """Avalon fingerprint.

    Attributes:
        nBits (int): number of bits in the fingerprint
        kwargs (dict): additional keyword arguments to pass to Avalon's `GetAvalonFP`
    """
    def __init__(self, nBits=1024, **kwargs):
        """Initialize the Avalon fingerprint.

        Args:
            nBits (int): number of bits in the fingerprint
            kwargs (dict):
                additional keyword arguments to pass to Avalon's `GetAvalonFP`
        """
        super().__init__(used_bits=list(range(nBits)))
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Calculate the Avalon fingerprints for the input molecules.

        Args:
            mols (list): list of RDKit molecules
            props (dict): dictionary of properties (not used)
            *args: positional arguments (not used)
            **kwargs: keyword arguments (not used, set in the constructor)

        Returns:
            (np.ndarray): descriptor values of shape (n_mols, n_descriptors)
        """
        convertFP = DataStructs.ConvertToNumpyArray
        ret = np.zeros((len(mols), len(self)))
        for idx, mol in enumerate(mols):
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=self.nBits, **self.kwargs)
            np_fp = np.zeros(len(fp))
            convertFP(fp, np_fp)
            ret[idx] = np_fp
        return ret

    def __len__(self):
        return self.nBits

    def __str__(self):
        return "AvalonFP"


class TopologicalFP(Fingerprint):
    """Topological torsion fingerprint.

    Attributes:
        nBits (int): number of bits in the fingerprint
        kwargs (dict):
            additional keyword arguments to pass to RDKit's
            `GetHashedTopologicalTorsionFingerprintAsBitVect` function
    """
    def __init__(self, nBits=2048, **kwargs):
        """Initialize the topological torsion fingerprint.

        Args:
            nBits (int): number of bits in the fingerprint
            kwargs (dict): additional keyword arguments
        """
        super().__init__(used_bits=list(range(nBits)))
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Calculate the topological torsion fingerprints for the input molecules.

        Args:
            mols (list): list of RDKit molecules
            props (dict): dictionary of properties (not used)
            *args: positional arguments (not used)
            **kwargs: keyword arguments (not used)

        Returns:
            (np.ndarray): descriptor values of shape (n_mols, n_descriptors)
        """
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

    def __len__(self):
        return self.nBits

    def __str__(self):
        return "TopologicalFP"


class AtomPairFP(Fingerprint):
    """Atom pair fingerprint.

    Attributes:
        nBits (int): number of bits in the fingerprint
        kwargs (dict):
            additional keyword arguments to pass to RDKit's
            `GetHashedAtomPairFingerprintAsBitVect` function
    """
    def __init__(self, nBits=2048, **kwargs):
        """Initialize the atom pair fingerprint.

        Args:
            nBits (int): number of bits in the fingerprint
            kwargs (dict):
                additional keyword arguments to pass to RDKit's
                `GetHashedAtomPairFingerprintAsBitVect`
        """
        super().__init__(used_bits=list(range(nBits)))
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Calculate the atom pair fingerprints for the input molecules.

        Args:
            mols (list): list of RDKit molecules
            props (dict): dictionary of properties (not used)
            *args: positional arguments (not used)
            **kwargs: keyword arguments (not used, set in the constructor)

        Returns:
            (np.ndarray): descriptor values of shape (n_mols, n_descriptors)
        """
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

    def __len__(self):
        return self.nBits

    def __str__(self):
        return "AtomPairFP"


class RDKitFP(Fingerprint):
    """RDKit fingerprint.

    This is a wrapper around RDKit's RDKFingerprint function.

    Attributes:
        minPath (int): minimum path length
        maxPath (int): maximum path length
        nBits (int): number of bits in the fingerprint
        kwargs (dict): additional keyword arguments to pass to RDKit's `RDKFingerprint`
    """
    def __init__(self, minPath=1, maxPath=7, nBits=2048, **kwargs):
        """Initialize the RDKit fingerprint.

        Args:
            minPath (int): minimum path length
            maxPath (int): maximum path length
            nBits (int): number of bits in the fingerprint
            kwargs (dict):
                additional keyword arguments to pass to RDKit's `RDKFingerprint`
        """
        super().__init__(used_bits=list(range(nBits)))
        self.minPath = minPath
        self.maxPath = maxPath
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Calculate the RDKit fingerprints for the input molecules.

        Args:
            mols (list): list of RDKit molecules
            props (dict): dictionary of properties (not used)
            *args: positional arguments (not used)
            **kwargs: keyword arguments (not used, set in the constructor)

        Returns:
            (np.ndarray): descriptor values of shape (n_mols, n_descriptors)
        """
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

    def __len__(self):
        return self.nBits

    def __str__(self):
        return "RDKitFP"


class PatternFP(Fingerprint):
    """Pattern fingerprint.

    Attributes:
        nBits (int): number of bits in the fingerprint
        kwargs (dict):
            additional keyword arguments to pass to RDKit's `PatternFingerprint`
    """
    def __init__(self, nBits=2048, **kwargs):
        """Initialize the pattern fingerprint.

        Args:
            nBits (int): number of bits in the fingerprint
            kwargs (dict):
                additional keyword arguments to pass to RDKit's `PatternFingerprint`
        """
        super().__init__(used_bits=list(range(nBits)))
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Calculate the pattern fingerprints for the input molecules.

        Args:
            mols (list): list of RDKit molecules
            props (dict): dictionary of properties (not used)
            *args: positional arguments (not used)
            **kwargs: keyword arguments (not used, set in the constructor)

        Returns:
            (np.ndarray): descriptor values of shape (n_mols, n_descriptors)
        """
        convertFP = DataStructs.ConvertToNumpyArray

        ret = np.zeros((len(mols), len(self)))
        for idx, mol in enumerate(mols):
            fp = rdmolops.PatternFingerprint(mol, fpSize=self.nBits, **self.kwargs)
            np_fp = np.zeros(len(fp))
            convertFP(fp, np_fp)
            ret[idx] = np_fp

        return ret

    def __len__(self):
        return self.nBits

    def __str__(self):
        return "PatternFP"


class LayeredFP(Fingerprint):
    """Layered fingerprint.

    Attributes:
        minPath (int): minimum path length
        maxPath (int): maximum path length
        nBits (int): number of bits in the fingerprint
        kwargs (dict):
            additional keyword arguments to pass to RDKit's `LayeredFingerprint`
    """
    def __init__(self, minPath=1, maxPath=7, nBits=2048, **kwargs):
        """Initialize the layered fingerprint.

        Args:
            minPath (int): minimum path length
            maxPath (int): maximum path length
            nBits (int): number of bits in the fingerprint
            kwargs (dict):
                additional keyword arguments to pass to RDKit's `LayeredFingerprint`
        """
        super().__init__(used_bits=list(range(nBits)))
        self.minPath = minPath
        self.maxPath = maxPath
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Calculate the layered fingerprints for the input molecules.

        Args:
            mols (list): list of RDKit molecules
            props (dict): dictionary of properties (not used)
            *args: positional arguments (not used)
            **kwargs: keyword arguments (not used, set in the constructor)
        """
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

    def __len__(self):
        return self.nBits

    def __str__(self):
        return "LayeredFP"
