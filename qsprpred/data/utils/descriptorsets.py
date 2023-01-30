"""Descriptorssets.

To add a new descriptor or fingerprint calculator:
* Add a descriptor subclass for your descriptor calculator
* Add a function to retrieve your descriptor by name to the descriptor retriever class
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List

import mordred
import numpy as np
from mordred import descriptors as mordreddescriptors
from Mold2_pywrapper import Mold2 as Mold2_calculator
from PaDEL_pywrapper import PaDEL as PaDEL_calculator
from PaDEL_pywrapper.descriptors import descriptors as PaDEL_descriptors
from qsprpred.data.utils.descriptor_utils.drugexproperties import Property
from qsprpred.data.utils.descriptor_utils.rdkitdescriptors import RDKit_desc
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Mol


class DescriptorSet(ABC):
    """Abstract base class for descriptorssets."""

    @abstractmethod
    def __call__(self, mols : List[Union[str, Mol]]):
        """
        Calculate the descriptor for a molecule.

        Args:
            mols: list of molecules (SMILES `str` or RDKit Mol)

        Returns:
            an array or data frame of descriptor values of shape (n_mols, n_descriptors)
        """
        pass

    def iterMols(self, mols : List[Union[str, Mol]], to_list=False):
        """
        Create a molecule iterator or list from RDKit molecules or SMILES.

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

        self._keepindices = None

    def __call__(self, mols):
        convertFP = DataStructs.ConvertToNumpyArray
        morgan = AllChem.GetMorganFingerprintAsBitVect

        ret = np.zeros((len(mols), self.get_len()))
        for idx,mol in enumerate(self.iterMols(mols)):
            fp = morgan(mol, *self._args, **self._kwargs)
            np_fp = np.zeros(len(fp))
            convertFP(fp, np_fp)
            if self.keepindices:
                np_fp = np_fp[self.keepindices]
            ret[idx] = np_fp

        return ret

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
        return self._args, self._kwargs

    def get_len(self):
        return len(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles("C"), *self._args, **self._kwargs))

    def __str__(self):
        return "MorganFP"

    @property
    def descriptors(self):
        indices = self.keepindices if self.keepindices else range(self.get_len())
        return [f"{idx}" for idx in indices]

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
    If no mordred descriptor object passed, then all descriptors are calculated.

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

    def __call__(self, mols):
        return self._mordred.pandas(self.iterMols(mols), quiet=True, nproc=1).values

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
    Physicochemical properties originally used in DrugEx for QSAR modelling
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

    def __call__(self, mols):
        calculator = Property(self.props)
        return calculator.getScores(self.iterMols(mols, to_list=True))

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
        *args: `RDKit_desc` arguments
        **kwargs: `RDKit_desc` keyword arguments
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._is_fp = False
        self._calculator = RDKit_desc(*args, **kwargs)
        self._descriptors = self._calculator.descriptors

    def __call__(self, mols):
        return self._calculator.getScores(self.iterMols(mols, to_list=True))

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


class Mold2(DescriptorSet):
    """Descriptors from molecular descriptor calculation software Mold2.

    From https://github.com/OlivierBeq/Mold2_pywrapper.
    Initialize the descriptor with no arguments.
    All descriptors are always calculated.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a PaDEL calculator

        Args:
            *args: ignored
            **kwargs: ignored
        """
        # args and kwargs are ignored
        self._args = None
        self._kwargs = None

        self._is_fp = False

        self._mold2 = Mold2_calculator()
        self._descriptors = self._mold2.calculate([Chem.MolFromSmiles("C")], show_banner=False).columns.tolist()
        self._keepindices = list(range(len(self._descriptors)))

    def __call__(self, mols):
        values = self._mold2.calculate(self.iterMols(mols), show_banner=False, njobs=1).values
        # Drop columns
        return values[:, self._keepindices]

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
        # Remove names not listed
        for i, index, desc_name in zip(range(-1, -len(self._keepindices)-1, -1),
                                       self._keepindices[::-1],
                                       self.descriptors[::-1]):
            if desc_name not in names:
                del self._keepindices[i], self._descriptors[i]

    def __str__(self):
        return "Mold2"


class PaDEL(DescriptorSet):
    """Descriptors from molecular descriptor calculation software PaDEL.

    From https://github.com/OlivierBeq/PaDEL_pywrapper.
    """

    def __init__(self, descs: Optional[List[str]] = None, **kwargs):
        """Initialize a PaDEL calculator

        Args:
            descs: list of PaDEL descriptor short names
            **kwargs: keyword arguments of the `PaDEL_pywrapper.PaDEL` constructor
        """
        # args and kwargs are ignored
        self._args = descs
        self._process_kwargs(**kwargs)

        self._is_fp = False

        # Create calculator and obtain default descriptor names
        dummy_mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
        AllChem.EmbedMolecule(dummy_mol)  # Required for line below, since 3D descs would raise without 3D coords
        self._name_mapping = {name: descriptor
                              for descriptor in PaDEL_descriptors
                              for name in descriptor().calculate(dummy_mol)}

        # Initialize descriptor names
        if descs is None:
            self._descriptors = PaDEL_descriptors
        else:
            self.descriptors = descs
        # Instantiate calculator
        self._padel = PaDEL_calculator(self._descriptors, ignore_3D=self._ignore_3D)
        # Obtain filtered descriptors from calculator
        self._descriptors = self._padel.descriptors

    def __call__(self, mols):
        values = self._padel.calculate(list(self.iterMols(mols)), show_banner=False, njobs=1).values
        return values

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        return self._args, self._kwargs

    @property
    def descriptors(self):
        names = [name for name, desc in self._name_mapping.items() if desc in self._descriptors]
        return names

    @descriptors.setter
    def descriptors(self, names):
        self._descriptors = [self._name_mapping[name] for name in names]
        self._padel = PaDEL_calculator(self._descriptors, ignore_3D=self._ignore_3D)

    def get_names(self):
        return self._padel.calculate([Chem.MolFromSmiles("C")], show_banner=False).columns.tolist()

    def _process_kwargs(self, **kwargs):
        self._ignore_3D = kwargs.get("ignore_3D", True)
        self._kwargs = kwargs

    def __str__(self):
        return "PaDEL"


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

    def get_MorganFP(self, *args, **kwargs):
        return MorganFP(*args, **kwargs)

    def get_DrugExPhyschem(self, *args, **kwargs):
        return DrugExPhyschem(*args, **kwargs)

    def get_Mordred(self, *args, **kwargs):
        return Mordred(*args, **kwargs)

    def get_RDkit(self, *args, **kwargs):
        return rdkit_descs(*args, **kwargs)


def get_descriptor(desc_type: str, *args, **kwargs):
    return _DescriptorSetRetriever().get_descriptor(desc_type, *args, **kwargs)
