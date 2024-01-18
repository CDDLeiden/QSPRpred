"""Fingerprint classes."""
from abc import ABC
from typing import Any

import numpy as np
import pandas as pd
from rdkit import DataStructs, Chem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors, rdmolops, Mol

from qsprpred.data.descriptors.sets import DescriptorSet


class Fingerprint(DescriptorSet, ABC):
    """Base class for fingerprints."""

    def __init__(self, used_bits: list[int] | None = None):
        super().__init__()
        self.usedBits = used_bits or list(range(len(self)))

    @property
    def usedBits(self) -> list[int] | None:
        return self._usedBits

    @usedBits.setter
    def usedBits(self, value: list[int]):
        self._usedBits = sorted(value)

    @property
    def descriptors(self) -> list[str]:
        return [f"{self}_{i}" for i in self.usedBits]

    @descriptors.setter
    def descriptors(self, value: list[str]):
        self.usedBits = [int(x.split("_")[-1]) for x in sorted(value)]

    @property
    def isFP(self):
        return True

    @property
    def dtype(self):
        return bool

    def __call__(
        self, mols: list[str | Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> pd.DataFrame:
        """Calculate the descriptors for a list of molecules.

        Args:
            mols(list): list of SMILES or RDKit molecules
            props(dict): dictionary of properties
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:
            data frame of descriptor values of shape (n_mols, n_descriptors)
        """
        mols = list(self.iterMols(mols, to_list=True))
        mols = [Chem.AddHs(mol) for mol in self.iterMols(mols)]
        values = self.getDescriptors(mols, props, *args, **kwargs)
        values = values[:, self.usedBits]
        values = values.astype(self.dtype)
        df = pd.DataFrame(values, index=props[self.idProp])
        df.columns = self.descriptors
        return df


class MorganFP(Fingerprint):
    """Morgan fingerprint."""

    def __init__(self, radius=2, nBits=2048, **kwargs):
        super().__init__(used_bits=list(range(nBits)))
        self.radius = radius
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        """Return the Morgan fingerprints for the input molecules.

        Args:
            mols: molecules to obtain the fingerprint of
            props: dictionary of properties

        Returns:
            array: `np.ndarray` of fingerprints for "mols", shape (n_mols, n_bits)
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

    def __len__(self):
        return 167

    def __str__(self):
        return "RDKitMACCSFP"


class MaccsFP(Fingerprint):
    def __init__(self, nBits=167, **kwargs):
        super().__init__()
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
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
    def __init__(self, nBits=1024, **kwargs):
        super().__init__()
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
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
    def __init__(self, nBits=2048, **kwargs):
        super().__init__()
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
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
    def __init__(self, nBits=2048, **kwargs):
        super().__init__()
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
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
    def __init__(self, minPath=1, maxPath=7, nBits=2048, **kwargs):
        super().__init__()
        self.minPath = minPath
        self.maxPath = maxPath
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
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
    def __init__(self, nBits=2048, **kwargs):
        super().__init__()
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
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
    def __init__(self, minPath=1, maxPath=7, nBits=2048, **kwargs):
        super().__init__()
        self.minPath = minPath
        self.maxPath = maxPath
        self.nBits = nBits
        self.kwargs = kwargs

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
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
