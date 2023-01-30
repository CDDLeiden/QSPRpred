"""Descriptorssets. A descriptorset is a collection of descriptors that can be calculated for a molecule.

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


class DescriptorSet(ABC):
    """Abstract base class for descriptorsets.

    A descriptorset is a collection of descriptors that can be calculated for a molecule.
    """

    @abstractmethod
    def __call__(self, mol):
        """
        Calculate the descriptor for a molecule.

        Args:
            mol: smiles or rdkit molecule

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
        """Return the number of descriptors."""
        return len(self.descriptors)

    @abstractmethod
    def is_fp(self):
        """Return True if descriptorset is fingerprint."""
        pass

    @abstractmethod
    def settings(self):
        """Return dictionary with arguments used to initialize the descriptorset."""
        pass

    @abstractmethod
    def __str__(self):
        """Return string representation of the descriptorset."""
        pass


class FingerprintSet(DescriptorSet):
    """Generic fingerprint descriptorset can be used to calculate any fingerprint type defined in descriptorutils.fingerprints."""

    def __init__(self, fingerprint_type, *args, **kwargs):
        """
        Initialize the descriptor with the same arguments as you would pass to your fingerprint type of choice.

        Args:
            fingerprint_type: fingerprint type
            *args: fingerprint specific arguments
            **kwargs: fingerprint specific arguments keyword arguments
        """
        self._args = args
        self._kwargs = kwargs
        self._is_fp = True
        self.fingerprint_type = fingerprint_type
        self.get_fingerprint = fingerprints.get_fingerprint(self.fingerprint_type, *args, **kwargs)

        self._keepindices = None

    def __call__(self, mol):
        """Calculate the fingerprint for a molecule."""
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
        """Return the indices of the fingerprint to keep."""
        return self._keepindices

    @keepindices.setter
    def keepindices(self, val):
        """Set the indices of the fingerprint to keep."""
        self._keepindices = [int(x) for x in val] if val else None

    @property
    def is_fp(self):
        """Return True if descriptorset is fingerprint."""
        return self._is_fp

    @property
    def settings(self):
        """Return dictionary with arguments used to initialize the descriptorset."""
        return {"fingerprint_type": self.fingerprint_type, "args": self._args, "kwargs": self._kwargs}

    def get_len(self):
        """Return the length of the fingerprint."""
        return len(self.__call__("C"))

    def __str__(self):
        return f"FingerprintSet"

    @property
    def descriptors(self):
        """Return the indices of the fingerprint that are kept."""
        return [f"{idx}" for idx in range(self.get_len())]

    @descriptors.setter
    def descriptors(self, value):
        """Set the indices of the fingerprint to keep."""
        self.keepindices(value)


class Mordred(DescriptorSet):
    """Descriptors from molecular descriptor calculation software Mordred.

    From https://github.com/mordred-descriptor/mordred.

    Args:
        descs (list): list of mordred descriptor names
        version (str): version of mordred
        ignore_3D (bool): ignore 3D information
        config (str): path to config file
    """

    def __init__(self, descs=None, version=None, ignore_3D=False, config=None):
        """
        Initialize the descriptor with the same arguments as you would pass to `Calculator` function of Mordred.

        With the exception of the `descs` argument which can also be a list of mordred descriptor names instead
        of a mordred descriptor module.

        Args:
            descs: List of Mordred descriptor names, a Mordred descriptor module or None for all mordred descriptors
            version (str): version of mordred
            ignore_3D (bool): ignore 3D information
            config (str): path to config file?
        """
        if descs:
            # if mordred descriptor module is passed, convert to list of descriptor instances
            if not isinstance(descs, list):
                descs = (mordred.Calculator(descs).descriptors)
        else:
            # use all mordred descriptors if no descriptors are specified
            descs = mordred.Calculator(mordreddescriptors).descriptors

        self.version = version
        self.ignore_3D = ignore_3D
        self.config = config

        self._is_fp = False

        self._mordred = None

        # convert to list of descriptor names if descriptor instances are passed and initiate mordred calulator
        self.descriptors = [[str(d) for d in descs]]

    def __call__(self, mol):
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return self._mordred.pandas([mol], quiet=True, nproc=1).iloc[0].to_list()

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        return {"descs": self.descriptors, "version": self.version, "ignore_3D": self.ignore_3D, "config": self.config}

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, names):
        """Set the descriptors to calculate.

        Converts a list of Mordred descriptor names to Mordred descriptor instances which is used to initialize the
        a Mordred calculator with the specified descriptors.

        Args:
            names: List of Mordred descriptor names.
        """
        calc = mordred.Calculator(mordreddescriptors)
        self._mordred = mordred.Calculator(
            [d for d in calc.descriptors if str(d) in names],
            version=self.version, ignore_3D=self.ignore_3D, config=self.config)
        self._descriptors = names

    def __str__(self):
        return "Mordred"


class DrugExPhyschem(DescriptorSet):
    """
    Pysciochemical properties originally used in DrugEx for QSAR modelling.

    Args:
        props: list of properties to calculate
    """

    def __init__(self, physchem_props=None):
        """Initialize the descriptorset with Property arguments (a list of properties to calculate) to select a subset.

        Args:
            physchem_props: list of properties to calculate
        """
        self._is_fp = False
        self.props = [x for x in Property(physchem_props).props]

    def __call__(self, mol):
        """Calculate the DrugEx properties for a molecule."""
        calculator = Property(self.props)
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return list(calculator.getScores([mol])[0])

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        return {"physchem_props": self.props}

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
    Calculate RDkit descriptors.

    Args:
        rdkit_descriptors: list of descriptors to calculate, if none, all 2D rdkit descriptors will be calculated
        compute_3Drdkit: if True, 3D descriptors will be calculated
    """

    def __init__(self, rdkit_descriptors=None, compute_3Drdkit=False):
        self._is_fp = False
        self._calculator = RDKit_desc(rdkit_descriptors, compute_3Drdkit)
        self._descriptors = self._calculator.descriptors

    def __call__(self, mol):
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return list(self._calculator.getScores([mol])[0])

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        return {"rdkit_descriptors": self._descriptors, "compute_3Drdkit": self._calculator.compute_3Drdkit}

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
    Calculate Tanimoto distances to a list of SMILES sequences.

    Args:
        list_of_smiles (list of strings): list of SMILES sequences to calculate distance to
        fingerprint_type (str): fingerprint type to use
        *args: `fingerprint` arguments
        **kwargs: `fingerprint` keyword arguments, should contain fingerprint_type
    """

    def __init__(self, list_of_smiles, fingerprint_type, *args, **kwargs):
        """Initialize the descriptorset with a list of SMILES sequences and a fingerprint type.

        Args:
            list_of_smiles (list of strings): list of SMILES sequences to calculate distance to
            fingerprint_type (str): fingerprint type to use
        """
        self._descriptors = list_of_smiles
        self.fingerprint_type = fingerprint_type
        self._args = args
        self._kwargs = kwargs
        self._is_fp = False

        # intialize fingerprint calculator
        self.get_fingerprint = fingerprints.get_fingerprint(self.fingerprint_type, *self._args, **self._kwargs)
        self.fps = self.calculate_fingerprints(self.list_of_smiles)

    def __call__(self, mol):
        """Calculate the Tanimoto distances to the list of SMILES sequences.

        Args:
            mol (str or rdkit.Chem.rdchem.Mol): SMILES sequence or RDKit molecule to calculate distances to
        """
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return list(1 - np.array(DataStructs.BulkTanimotoSimilarity(self.get_fingerprint(mol), self.fps)))

    def calculate_fingerprints(self, list_of_smiles):
        """Calculate the fingerprints for the list of SMILES sequences."""
        self.fps = [self.get_fingerprint(Chem.MolFromSmiles(smiles)) for smiles in list_of_smiles]

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        return {"fingerprint_type": self.fingerprint_type,
                "list_of_smiles": self.list_of_smiles, "args": self._args, "kwargs": self._kwargs}

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
