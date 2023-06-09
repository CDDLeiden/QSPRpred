"""Descriptorset: a collection of descriptors that can be calculated for a molecule.

To add a new descriptor or fingerprint calculator:
* Add a descriptor subclass for your descriptor calculator
* Add a function to retrieve your descriptor by name to the descriptor retriever class
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Mol

from ...models.interfaces import QSPRModel
from .descriptor_utils import fingerprints
from .descriptor_utils.drugexproperties import Property
from .descriptor_utils.rdkitdescriptors import RdkitDescriptors


class DescriptorSet(ABC):
    __len__ = lambda self: self.getLen()

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Calculate the descriptors for a given input.

        Args:
            *args: arguments to be passed to perform the calculation
            **kwargs: keyword arguments to be passed to perform the calculation

        Returns:
            DataFrame or np.array of descriptors with shape [n_inputs, n_descriptors]
        """

    @property
    @abstractmethod
    def descriptors(self):
        """Return a list of descriptor names."""

    @descriptors.setter
    @abstractmethod
    def descriptors(self, value):
        """Set the descriptor names."""

    def getLen(self):
        """Return the number of descriptors."""
        return len(self.descriptors)

    @property
    @abstractmethod
    def isFP(self):
        """Return True if descriptorset is fingerprint."""

    @property
    @abstractmethod
    def settings(self):
        """Return dictionary with arguments used to initialize the descriptorset."""

    @abstractmethod
    def __str__(self):
        """Return string representation of the descriptorset."""


class MoleculeDescriptorSet(DescriptorSet):
    """Abstract base class for descriptorsets.

    Descriptorset: a collection of descriptors that can be calculated for a molecule.
    """
    @abstractmethod
    def __call__(self, mols: list[str | Mol]):
        """
        Calculate the descriptor for a molecule.

        Args:
            mols: list of molecules (SMILES `str` or RDKit Mol)

        Returns:
            an array or data frame of descriptor values of shape (n_mols, n_descriptors)
        """

    @staticmethod
    def iterMols(mols: list[str | Mol], to_list=False):
        """
        Create a molecule iterator or list from RDKit molecules or SMILES.

        Args:
            mols: list of molecules (SMILES `str` or RDKit Mol)
            to_list: if True, return a list instead of an iterator

        Returns:
            an array or data frame of descriptor values of shape (n_mols, n_descriptors)
        """
        ret = (Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol for mol in mols)
        if to_list:
            ret = list(ret)
        return ret


class DataFrameDescriptorSet(DescriptorSet):
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._descriptors = df.columns.tolist()

    def getDF(self):
        return self._df

    def getIndex(self):
        return self._df.index

    def __call__(self, index, *args, **kwargs):
        ret = pd.DataFrame(index=index)
        ret = ret.merge(self._df, how="left", left_index=True, right_index=True)
        return ret[self.descriptors]

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, value):
        self._descriptors = value

    @property
    def isFP(self):
        return False

    @property
    def settings(self):
        return {}

    def __str__(self):
        return "DataFrame"


class FingerprintSet(MoleculeDescriptorSet):
    """Generic fingerprint descriptorset can be used to calculate any fingerprint type
    defined in descriptorutils.fingerprints.
    """
    def __init__(self, fingerprint_type, *args, **kwargs):
        """
        Initialize the descriptor with the same arguments as you would pass to your
        fingerprint type of choice.

        Args:
            fingerprint_type: fingerprint type
            *args: fingerprint specific arguments
            **kwargs: fingerprint specific arguments keyword arguments
        """
        self._isFP = True
        self.fingerprintType = fingerprint_type
        self.getFingerprint = fingerprints.get_fingerprint(
            self.fingerprintType, *args, **kwargs
        )

        self._keepindices = None

    def __call__(self, mols):
        """Calculate the fingerprint for a list of molecules."""
        mols = [Chem.AddHs(mol) for mol in self.iterMols(mols)]
        ret = self.getFingerprint(mols)

        if self.keepindices:
            ret = ret[:, self.keepindices]

        return ret

    @property
    def keepindices(self):
        """Return the indices of the fingerprint to keep."""
        return self._keepindices

    @keepindices.setter
    def keepindices(self, val):
        """Set the indices of the fingerprint to keep."""
        self._keepindices = [int(x) for x in val] if val else None

    @property
    def isFP(self):
        """Return True if descriptorset is fingerprint."""
        return self._isFP

    @property
    def settings(self):
        """Return dictionary with arguments used to initialize the descriptorset."""
        return {
            "fingerprint_type": self.fingerprintType,
            **self.getFingerprint.settings,
        }

    def getLen(self):
        """Return the length of the fingerprint."""
        return len(self.getFingerprint)

    def __str__(self):
        return f"FingerprintSet_{self.fingerprintType}"

    @property
    def descriptors(self):
        """Return the indices of the fingerprint that are kept."""
        indices = self.keepindices if self.keepindices else range(self.getLen())
        return [f"{idx}" for idx in indices]

    @descriptors.setter
    def descriptors(self, value):
        """Set the indices of the fingerprint to keep."""
        self.keepindices(value)


class DrugExPhyschem(MoleculeDescriptorSet):
    """
    Physciochemical properties originally used in DrugEx for QSAR modelling.

    Args:
        props: list of properties to calculate
    """
    def __init__(self, physchem_props=None):
        """Initialize the descriptorset with Property arguments (a list of properties to
        calculate) to select a subset.

        Args:
            physchem_props: list of properties to calculate
        """
        self._isFP = False
        self.props = list(Property(physchem_props).props)

    def __call__(self, mols):
        """Calculate the DrugEx properties for a molecule."""
        calculator = Property(self.props)
        return calculator.getScores(self.iterMols(mols, to_list=True))

    @property
    def isFP(self):
        return self._isFP

    @property
    def settings(self):
        return {"physchem_props": self.props}

    @property
    def descriptors(self):
        return self.props

    @descriptors.setter
    def descriptors(self, props):
        """Set new props as a list of names."""
        self.props = list(Property(props).props)

    def __str__(self):
        return "DrugExPhyschem"


class RDKitDescs(MoleculeDescriptorSet):
    """
    Calculate RDkit descriptors.

    Args:
        rdkit_descriptors: list of descriptors to calculate, if none, all 2D rdkit
            descriptors will be calculated
        compute_3Drdkit: if True, 3D descriptors will be calculated
    """
    def __init__(self, rdkit_descriptors=None, compute_3Drdkit=False):
        self._isFP = False
        # TODO: RdkitDescriptors probably needs refactoring; see definition
        self._calculator = RdkitDescriptors(rdkit_descriptors, compute_3Drdkit)
        self._descriptors = self._calculator.descriptors
        self.compute3Drdkit = compute_3Drdkit

    def __call__(self, mols):
        return self._calculator.getScores(self.iterMols(mols, to_list=True))

    @property
    def isFP(self):
        return self._isFP

    @property
    def settings(self):
        return {
            "rdkit_descriptors": self.descriptors,
            "compute_3Drdkit": self.compute3Drdkit,
        }

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, descriptors):
        self._calculator.descriptors = descriptors
        self._descriptors = descriptors

    def __str__(self):
        return "RDkit"


class TanimotoDistances(MoleculeDescriptorSet):
    """
    Calculate Tanimoto distances to a list of SMILES sequences.

    Args:
        list_of_smiles (list of strings): list of SMILES to calculate the distances.
        fingerprint_type (str): fingerprint type to use.
        *args: `fingerprint` arguments
        **kwargs: `fingerprint` keyword arguments, should contain fingerprint_type
    """
    def __init__(self, list_of_smiles, fingerprint_type, *args, **kwargs):
        """Initialize the descriptorset with a list of SMILES sequences and a
        fingerprint type.

        Args:
            list_of_smiles (list of strings): list of SMILES sequences to calculate
                distance to
            fingerprint_type (str): fingerprint type to use
        """
        self._descriptors = list_of_smiles
        self.fingerprintType = fingerprint_type
        self._args = args
        self._kwargs = kwargs
        self._isFP = False

        # intialize fingerprint calculator
        self.getFingerprint = fingerprints.get_fingerprint(
            self.fingerprintType, *self._args, **self._kwargs
        )
        self.calculate_fingerprints(list_of_smiles)

    def __call__(self, mols):
        """Calculate the Tanimoto distances to the list of SMILES sequences.

        Args:
            mols (List[str] or List[rdkit.Chem.rdchem.Mol]): SMILES sequences or RDKit
                molecules to calculate the distances.
        """
        mols = [
            Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol for mol in mols
        ]
        # Convert np.arrays to BitVects
        fps = [
            DataStructs.CreateFromBitString("".join(map(str, x)))
            for x in self.getFingerprint(mols)
        ]
        return [
            list(1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, self.fps)))
            for fp in fps
        ]

    def calculate_fingerprints(self, list_of_smiles):
        """Calculate the fingerprints for the list of SMILES sequences."""
        # Convert np.arrays to BitVects
        self.fps = [
            DataStructs.CreateFromBitString("".join(map(str, x))) for x in self.
            getFingerprint([Chem.MolFromSmiles(smiles) for smiles in list_of_smiles])
        ]

    @property
    def isFP(self):
        return self._isFP

    @property
    def settings(self):
        return {
            "fingerprint_type": self.fingerprintType,
            "list_of_smiles": self._descriptors,
            "args": self._args,
            "kwargs": self._kwargs,
        }

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, list_of_smiles):
        """Set new list of SMILES sequences to calculate distance to."""
        self._descriptors = list_of_smiles
        self.list_of_smiles = list_of_smiles
        self.fps = self.calculate_fingerprints(self.list_of_smiles)

    def __str__(self):
        return "TanimotoDistances"


class PredictorDesc(MoleculeDescriptorSet):
    """MoleculeDescriptorSet that uses a Predictor object to calculate descriptors from
    a molecule."""
    def __init__(self, model: list[QSPRModel | str]):
        """
        Initialize the descriptorset with a `QSPRModel` object.

        Args:
            model: a fitted model instance or a path to the model's meta file
        """

        if isinstance(model, str):
            from qsprpred.models.interfaces import QSPRModel

            self.model = QSPRModel.fromFile(model)
        else:
            self.model = model

        self._descriptors = [self.model.name]

    def __call__(self, mols):
        """
        Calculate the descriptor for a list of molecules.

        Args:
            mols (list): list of smiles or rdkit molecules

        Returns:
            an array of descriptor values
        """
        mols = list(mols)
        if type(mols[0]) != str:
            mols = [Chem.MolToSmiles(mol) for mol in mols]
        return self.model.predictMols(mols, use_probas=False)

    @property
    def isFP(self):
        return False

    @property
    def settings(self):
        """Return args and kwargs used to initialize the descriptorset."""
        return {
            "model":
                self.model.
                metaFile  # FIXME: we save absolute path to meta file so this descriptor
            # set is not really portable
        }

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, descriptors):
        self._descriptors = descriptors

    def getLen(self):
        return 1

    def __str__(self):
        return "PredictorDesc"


class _DescriptorSetRetriever:
    """Based on recipe 8.21 of the book "Python Cookbook".

    To support a new type of descriptor, just add a function "get_descname(self, *args,
    **kwargs)".
    """
    def getDescriptor(self, desc_type, *args, **kwargs):
        if desc_type.lower() == "descriptor":
            raise Exception("Please specify the type of fingerprint you want to use.")
        method_name = "get" + desc_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f"{desc_type} is not a supported descriptor set type.")
        return method(*args, **kwargs)

    def getFingerprintSet(self, *args, **kwargs):
        """Wrapper to get a fingerprint set."""
        return FingerprintSet(*args, **kwargs)

    def getDrugExPhyschem(self, *args, **kwargs):
        """Wrapper to get DrugEx physichochemical properties."""
        return DrugExPhyschem(*args, **kwargs)

    def getMordred(self, *args, **kwargs):
        """Wrapper to get Mordred descriptors - depends on optional dependency."""
        from qsprpred.extra.data.utils.descriptorsets import Mordred

        return Mordred(*args, **kwargs)

    def getMold2(self, *args, **kwargs):
        """Wrapper to get Mold2 descriptors - depends on optional dependency."""
        from qsprpred.extra.data.utils.descriptorsets import Mold2

        return Mold2(*args, **kwargs)

    def getPaDEL(self, *args, **kwargs):
        """Wrapper to get PaDEL descriptors - depends on optional dependency."""
        from qsprpred.extra.data.utils.descriptorsets import PaDEL

        return PaDEL(*args, **kwargs)

    def getRDKit(self, *args, **kwargs):
        """Wrapper to get rdkit descriptors."""
        return RDKitDescs(*args, **kwargs)

    def getPredictorDesc(self, *args, **kwargs):
        """Wrapper to get predictors as descriptors."""
        return PredictorDesc(*args, **kwargs)

    def getProDec(self, *args, **kwargs):
        """Wrapper to get protein descriptorsfrom prodec - depends on optional
        dependency."""
        from qsprpred.extra.data.utils.descriptorsets import ProDecDescriptorSet

        return ProDecDescriptorSet(*args, **kwargs)

    def getTanimotoDistances(self, *args, **kwargs):
        """Wrapper to get bulk tanimoto distances."""
        return TanimotoDistances(*args, **kwargs)


def get_descriptor(desc_type: str, *args, **kwargs):
    """Wrapper to get a descriptorfunction from _DescriptorSetRetriever."""
    return _DescriptorSetRetriever().getDescriptor(desc_type, *args, **kwargs)
