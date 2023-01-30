"""Descriptorssets.

To add a new descriptor or fingerprint calculator:
* Add a descriptor subclass for your descriptor calculator
* Add a function to retrieve your descriptor by name to the descriptor retriever class
"""

from abc import ABC, abstractmethod

import mordred
import numpy as np
from mordred import descriptors as mordreddescriptors
from qsprpred.data.utils.descriptor_utils import fingerprints
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

    @property
    @abstractmethod
    def descriptors(self):
        """Return a list of descriptor names."""
        pass

    @descriptors.setter
    @abstractmethod
    def descriptors(self, value):
        """Set the descriptor names."""
        pass

    def get_len(self):
        return len(self.descriptors)

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


class FingerprintSet(DescriptorSet):

    def __init__(self, *args, **kwargs):
        """
        Initialize the descriptor with the same arguments as you would pass to your fingerprint type of choice.

        Args:
            *args: fingerprint type arguments
            **kwargs: fingerprint type keyword arguments, make sure to include also "fingerprint_type" as a keyword argument.
        """
        self._args = args
        self._kwargs = kwargs
        self._is_fp = True
        self.fingerprint_type = kwargs["fingerprint_type"]
        self._kwargs.pop("fingerprint_type")
        self.get_fingerprint = fingerprints.get_fingerprint(self.fingerprint_type, *args, **kwargs)

        self._keepindices = None

    def __call__(self, mol):
        convertMol = Chem.MolFromSmiles
        convertFP = DataStructs.ConvertToNumpyArray

        mol = convertMol(mol) if isinstance(mol, str) else mol
        fp = self.get_fingerprint(mol)
        ret = np.zeros(len(fp))
        convertFP(fp, ret)
        if self.keepindices:
            ret = ret[self.keepindices]
        return list(ret)

    @property
    def keepindices(self):
        return self._keepindices

    @keepindices.setter
    def keepindices(self, val):
        self._keepindices = [int(x) for x in val] if val else None

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        # add fingerprint type to kwargs
        self._kwargs["fingerprint_type"] = self.fingerprint_type
        return self._args, self._kwargs

    def get_len(self):
        return len(self.__call__("C"))

    def __str__(self):
        return f"FingerprintSet"

    @property
    def descriptors(self):
        return [f"{idx}" for idx in range(self.get_len())]

    @descriptors.setter
    def descriptors(self, value):
        """
        Sets the number of bits to a given value.
        """
        self._kwargs["nBits"] = value


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
        return self._mordred.pandas([mol], quiet=True, nproc=1).iloc[0].to_list()

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
        self.props = [x for x in Property(*args, **kwargs).props]

    def __call__(self, mol):
        calculator = Property(self.props)
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return list(calculator.getScores([mol])[0])

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        return self._args, self._kwargs

    @property
    def descriptors(self):
        return self.props

    @descriptors.setter
    def descriptors(self, props):
        """Set new props as a list of names."""
        self.props = [x for x in Property(props).props]

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
        return list(self._calculator.getScores([mol])[0])

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

    def __str__(self):
        return "RDkit"


class TanimotoDistances(DescriptorSet):
    """
    RDkit descriptors
    Initialize the descriptor names (a list of properties to calculate) to select a subset of the rdkit descriptors.
    Add compute_3Drdkit argument to indicate if 3D descriptors should also be calculated

    Args:
        list_of_smiles (list of strings): list of SMILES sequences to calculate distance to
        *args: `fingerprint` arguments
        **kwargs: `fingerprint` keyword arguments, should contain fingerprint_type
    """

    def __init__(self, list_of_smiles, *args, **kwargs):
        self._args = args
        self.list_of_smiles = list_of_smiles
        self._kwargs = kwargs
        self._is_fp = False
        self._descriptors = list_of_smiles
        self.fingerprint_type = kwargs["fingerprint_type"]
        self._kwargs.pop("fingerprint_type")
        self.get_fingerprint = fingerprints.get_fingerprint(self.fingerprint_type, *args, **kwargs)
        mols = [Chem.MolFromSmiles(smiles) for smiles in list_of_smiles]
        self.fps = [self.get_fingerprint(mol) for mol in mols]

    def __call__(self, mol):
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return list(1 - np.array(DataStructs.BulkTanimotoSimilarity(self.get_fingerprint(mol), self.fps)))

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        self._kwargs.update({"fingerprint_type": self.fingerprint_type, "list_of_smiles": self.list_of_smiles})
        return self._args, self._kwargs

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, descriptors):
        self._calculator.descriptors = descriptors
        self._descriptors = descriptors

    def __str__(self):
        return "TanimotoDistances"


class _DescriptorSetRetriever:
    """Based on recipe 8.21 of the book "Python Cookbook".

    To support a new type of descriptor, just add a function "get_descname(self, *args, **kwargs)".
    """

    def get_descriptor(self, desc_type, *args, **kwargs):
        method_name = "get_" + desc_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f"{desc_type} is not a supported descriptor set type.")
        return method(*args, **kwargs)

    def get_FingerprintSet(self, *args, **kwargs):
        return FingerprintSet(*args, **kwargs)

    def get_DrugExPhyschem(self, *args, **kwargs):
        return DrugExPhyschem(*args, **kwargs)

    def get_Mordred(self, *args, **kwargs):
        return Mordred(*args, **kwargs)

    def get_RDkit(self, *args, **kwargs):
        return rdkit_descs(*args, **kwargs)

    def get_TanimotoDistances(self, *args, **kwargs):
        return TanimotoDistances(*args, **kwargs)


def get_descriptor(desc_type: str, *args, **kwargs):
    return _DescriptorSetRetriever().get_descriptor(desc_type, *args, **kwargs)
