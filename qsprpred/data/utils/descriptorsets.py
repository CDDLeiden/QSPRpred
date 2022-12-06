"""Descriptorssets.

To add a new descriptor or fingerprint calculator:
* Add a descriptor subclass for your descriptor calculator
* Add a function to retrieve your descriptor by name to the descriptor retriever class
"""

from abc import ABC, abstractmethod

import mordred
import numpy as np
import pandas as pd
from mordred import descriptors as mordreddescriptors
from qsprpred.data.utils.descriptor_utils.drugexproperties import Property
from qsprpred.data.utils.descriptor_utils.rdkitdescriptors import RDKit_desc
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


class DescriptorSet(ABC):
    """Abstract base class for descriptorssets."""

    @abstractmethod
    def __call__(self, mol):
        """
        Calculate the descriptor for a molecule.

        Args:
            mol: smiles or rdkit molecule
            *args: optional arguments
            **kwargs: optional keyword arguments

        Returns:
            a `list` of descriptor values
        """
        pass

    @abstractmethod
    def is_fp(self):
        """Return True if descriptorset is fingerprint."""
        pass

    @abstractmethod
    def settings(self):
        """Return args and kwargs used to initialize the descriptorset."""
        pass

    @abstractmethod
    def __str__(self):
        pass


class MorganFP(DescriptorSet):
    def __init__(self, *args, **kwargs):
        """
        Initialize the descriptor with the same arguments as you would pass to `GetMorganFingerprintAsBitVect` function of RDKit.

        Args:
            *args: `GetMorganFingerprintAsBitVect` arguments
            **kwargs: `GetMorganFingerprintAsBitVect` keyword arguments
        """
        self._args = args
        self._kwargs = kwargs
        self._is_fp = True

        self.keepindices = None

    def __call__(self, mol):
        convertMol = Chem.MolFromSmiles
        convertFP = DataStructs.ConvertToNumpyArray
        morgan = AllChem.GetMorganFingerprintAsBitVect

        mol = convertMol(mol) if isinstance(mol, str) else mol
        fp = morgan(mol, *self._args, **self._kwargs)
        self.ln = len(fp)
        ret = np.zeros(self.ln)
        convertFP(fp, ret)
        ret = pd.DataFrame(
            ret.reshape(1, -1), columns=[f"{idx}" for idx in range(ret.shape[0])]
        )
        if self.keepindices:
            ret = ret[self.keepindices]
        return ret

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        return self._args, self._kwargs

    def get_len(self):
        return((self.__call__("C")).shape[1])

    def __str__(self):
        return "MorganFP"


class Mordred(DescriptorSet):
    """Descriptors from molecular descriptor calculation software Mordred.

    From https://github.com/mordred-descriptor/mordred.
    Initialize the descriptor with the same arguments as you would pass to `Calculator` function of Mordred.
    If no mordred descriptor object passed, the all descriptors will be calculated.

    Args:
        *args: `Calculator` arguments
        **kwargs: `Calculator` keyword arguments
    """
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._process_args(*args, **kwargs)

        self._is_fp = False

        self._mordred = None
        self.descriptors = self._args[0]

    def __call__(self, mol):
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return self._mordred.pandas([mol], quiet=True)

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        return self._args, self._kwargs

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, names):
        calc = mordred.Calculator(mordreddescriptors)
        self._mordred = mordred.Calculator(
            [d for d in calc.descriptors if str(d) in names], **self._kwargs
        )
        self._descriptors = names

    def _process_args(self, descs=None, version=None, ignore_3D=False, config=None):
        if descs:
            if not isinstance(descs, list):
                descs = (mordred.Calculator(descs).descriptors)
        else:
            descs = mordred.Calculator(mordreddescriptors).descriptors
        self._args = [[str(d) for d in descs]]
        self._kwargs = {"version": version, "ignore_3D": ignore_3D, "config": config}

    def get_len(self):
        return(len(self.descriptors))

    def __str__(self):
        return "Mordred"


class DrugExPhyschem(DescriptorSet):
    """
    Pysciochemical properties originally used in DrugEx for QSAR modelling
    Initialize the descriptor with Property arguments (a list of properties to calculate) to select a subset.

    Args:
        *args: `Property` arguments
        **kwargs: `Property` keyword arguments
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._is_fp = False
        self._calculator = Property(*args, **kwargs)
        self._descriptors = self._calculator.props

    def __call__(self, mol):
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        physchem = pd.DataFrame(
            self._calculator.getScores([mol]), columns=self._descriptors
        )
        return physchem

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        return self._args, self._kwargs

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, props):
        self._calculator.props = props
        self._descriptors = props

    def get_len(self):
        return(len(self.descriptors))

    def __str__(self):
        return "DrugExPhyschem"


class rdkit_descs(DescriptorSet):
    """
    RDkit descriptors
    Initialize the descriptor names (a list of properties to calculate) to select a subset of the rdkit descriptors.
    Add compute_3Drdkit argument to indicate if 3D descriptors should also be calculated

    Args:
        *args: `Property` arguments
        **kwargs: `Property` keyword arguments
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._is_fp = False
        self._calculator = RDKit_desc(*args, **kwargs)
        self._descriptors = self._calculator.descriptors

    def __call__(self, mol):
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        physchem = pd.DataFrame(
            self._calculator.getScores([mol]), columns=self._descriptors
        )
        return physchem

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        return self._args, self._kwargs

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, descriptors):
        self._calculator.descriptors = descriptors
        self._descriptors = descriptors

    def get_len(self):
        return(len(self.descriptors))

    def __str__(self):
        return "RDkit"


class _DescriptorSetRetriever:
    """
    Based on recipe 8.21 of the book "Python Cookbook".
    To support a new type of descriptor, just add a function "get_descname(self, *args, **kwargs)".
    """

    def get_descriptor(self, desc_type, *args, **kwargs):
        method_name = "get_" + desc_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f"{desc_type} is not a supported descriptor set type.")
        return method(*args, **kwargs)

    def get_MorganFP(self, *args, **kwargs):
        return MorganFP(*args, **kwargs)

    def get_DrugExPhyschem(self, *args, **kwargs):
        return DrugExPhyschem(*args, **kwargs)

    def get_Mordred(self, *args, **kwargs):
        return Mordred(*args, **kwargs)


def get_descriptor(desc_type: str, *args, **kwargs):
    return _DescriptorSetRetriever().get_descriptor(desc_type, *args, **kwargs)
