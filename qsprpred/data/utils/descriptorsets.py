"""Descriptorssets. A descriptorset is a collection of descriptors that can be calculated for a molecule.

To add a new descriptor or fingerprint calculator:
* Add a descriptor subclass for your descriptor calculator
* Add a function to retrieve your descriptor by name to the descriptor retriever class
"""

from abc import ABC, abstractmethod
from typing import List, Union

import mordred
import numpy as np
from mordred import descriptors as mordreddescriptors
from qsprpred.data.utils.descriptor_utils import fingerprints
from qsprpred.data.utils.descriptor_utils.drugexproperties import Property
from qsprpred.data.utils.descriptor_utils.rdkitdescriptors import RDKit_desc
from rdkit import Chem, DataStructs
from rdkit.Chem import Mol


class DescriptorSet(ABC):
    """Abstract base class for descriptorsets.

    A descriptorset is a collection of descriptors that can be calculated for a molecule.
    """

    @abstractmethod
    def __call__(self, mols: List[Union[str, Mol]]):
        """
        Calculate the descriptor for a molecule.

        Args:
            mols: list of molecules (SMILES `str` or RDKit Mol)

        Returns:
            an array or data frame of descriptor values of shape (n_mols, n_descriptors)
        """
        pass

    def iterMols(self, mols: List[Union[str, Mol]], to_list=False):
        """
        Calculate the descriptor for a molecule.

        Args:
            mols: list of molecules (SMILES `str` or RDKit Mol)

        Returns:
            an array or data frame of descriptor values of shape (n_mols, n_descriptors)
        """
        ret = (Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol for mol in mols)
        if to_list:
            ret = list(ret)
        return ret

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
        self._is_fp = True
        self.fingerprint_type = fingerprint_type
        self.get_fingerprint = fingerprints.get_fingerprint(self.fingerprint_type, *args, **kwargs)

        self._keepindices = None

    def __call__(self, mols):
        """Calculate the fingerprint for a list of molecules."""
        convertFP = DataStructs.ConvertToNumpyArray

        ret = np.zeros((len(mols), self.get_len() if not self.keepindices else len(self.keepindices)))
        for idx, mol in enumerate(self.iterMols(mols)):
            fp = self.get_fingerprint(mol)
            np_fp = np.zeros(len(fp))
            convertFP(fp, np_fp)
            if self.keepindices:
                np_fp = np_fp[self.keepindices]
            ret[idx] = np_fp

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
    def is_fp(self):
        """Return True if descriptorset is fingerprint."""
        return self._is_fp

    @property
    def settings(self):
        """Return dictionary with arguments used to initialize the descriptorset."""
        return {"fingerprint_type": self.fingerprint_type, **self.get_fingerprint.settings}

    def get_len(self):
        """Return the length of the fingerprint."""
        return len(self.get_fingerprint(Chem.MolFromSmiles("C")))

    def __str__(self):
        return f"FingerprintSet"

    @property
    def descriptors(self):
        """Return the indices of the fingerprint that are kept."""
        indices = self.keepindices if self.keepindices else range(self.get_len())
        return [f"{idx}" for idx in indices]

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
        self.descriptors = [str(d) for d in descs]

    def __call__(self, mols):
        return self._mordred.pandas(self.iterMols(mols), quiet=True, nproc=1).values

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

    def __call__(self, mols):
        """Calculate the DrugEx properties for a molecule."""
        calculator = Property(self.props)
        return calculator.getScores(self.iterMols(mols, to_list=True))

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
        self.compute_3Drdkit = compute_3Drdkit

    def __call__(self, mols):
        return self._calculator.getScores(self.iterMols(mols, to_list=True))

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        return {"rdkit_descriptors": self.descriptors, "compute_3Drdkit": self.compute_3Drdkit}

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
        self.fps = self.calculate_fingerprints(list_of_smiles)

    def __call__(self, mols):
        """Calculate the Tanimoto distances to the list of SMILES sequences.

        Args:
            mols (list of rdkit.Chem.rdchem.Mol): SMILES sequence or RDKit molecule to calculate distances to
        """
        ret = np.zeros((len(mols), self.get_len()))
        for idx, mol in enumerate(self.iterMols(mols, to_list=True)):
            ret[idx] = 1 - np.array(DataStructs.BulkTanimotoSimilarity(self.get_fingerprint(mol), self.fps))

        return ret

    def calculate_fingerprints(self, list_of_smiles):
        """Calculate the fingerprints for the list of SMILES sequences."""
        return [self.get_fingerprint(Chem.MolFromSmiles(smiles)) for smiles in list_of_smiles]

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        return {"fingerprint_type": self.fingerprint_type,
                "list_of_smiles": self.descriptors, "args": self._args, "kwargs": self._kwargs}

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, list_of_smiles):
        """Set new list of SMILES sequences to calculate distance to."""
        self._descriptors = list_of_smiles
        self.fps = self.calculate_fingerprints(list_of_smiles)

    def get_len(self):
        return len(self.descriptors)

    def __str__(self):
        return "TanimotoDistances"


class PredictorDesc(DescriptorSet):
    """DescriptorSet that uses a Predictor object to calculate the descriptors for a molecule."""

    def __init__(self, model_path: str, metadata_path: str):
        """
        Initialize the descriptorset with a Predictor object.

        Args:
            model_path: path to the model file
            metadata_path: path to the metadata file
        """
        self._is_fp = False
        self.model_path = model_path
        self.metadata_path = metadata_path
        from qsprpred.scorers.predictor import Predictor
        self._predictor = Predictor.fromFile(model_path, metadata_path)
        self._descriptors = [self._predictor.getKey()]

    def __call__(self, mols):
        """
        Calculate the descriptor for a molecule.

        Args:
            mol: smiles or rdkit molecule

        Returns:
            a `list` of descriptor values
        """
        return list(self._predictor.getScores(self.iterMols(mols, to_list=True)))

    @ property
    def is_fp(self):
        return self._is_fp

    @ property
    def settings(self):
        """Return args and kwargs used to initialize the descriptorset."""
        return {"model_path": self.model_path, "metadata_path": self.metadata_path}

    @ property
    def descriptors(self):
        return self._descriptors

    @ descriptors.setter
    def descriptors(self, descriptors):
        self._descriptors = descriptors

    def get_len(self):
        return 1

    def __str__(self):
        return "PredictorDesc"


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

    def get_PredictorDesc(self, *args, **kwargs):
        return PredictorDesc(*args, **kwargs)

    def get_TanimotoDistances(self, *args, **kwargs):
        return TanimotoDistances(*args, **kwargs)


def get_descriptor(desc_type: str, *args, **kwargs):
    return _DescriptorSetRetriever().get_descriptor(desc_type, *args, **kwargs)
