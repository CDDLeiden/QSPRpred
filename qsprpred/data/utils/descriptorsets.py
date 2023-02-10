"""Descriptorssets. A descriptorset is a collection of descriptors that can be calculated for a molecule.

To add a new descriptor or fingerprint calculator:
* Add a descriptor subclass for your descriptor calculator
* Add a function to retrieve your descriptor by name to the descriptor retriever class
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List

import mordred
import numpy as np
from mordred import descriptors as mordreddescriptors
from qsprpred.data.utils.descriptor_utils import fingerprints
from Mold2_pywrapper import Mold2 as Mold2_calculator
from PaDEL_pywrapper import PaDEL as PaDEL_calculator
from PaDEL_pywrapper.descriptors import _descs_2D as PaDEL_2D_descriptors, descriptors as PaDEL_All_descriptors
from qsprpred.data.utils.descriptor_utils.drugexproperties import Property
from qsprpred.data.utils.descriptor_utils.rdkitdescriptors import RDKit_desc
from rdkit import Chem, DataStructs
from rdkit.Chem import Mol, AllChem


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
        ret = self.get_fingerprint(self.iterMols(mols, to_list=True))

        if self.keepindices:
            ret = ret[:,self.keepindices]

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
        return len(self.get_fingerprint)

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
    Physciochemical properties originally used in DrugEx for QSAR modelling.

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
        self.calculate_fingerprints(list_of_smiles)

    def __call__(self, mols):
        """Calculate the Tanimoto distances to the list of SMILES sequences.

        Args:
            mols (List[str] or List[rdkit.Chem.rdchem.Mol]): SMILES sequences or RDKit molecules to calculate distances to
        """
        mols = [Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol for mol in mols]
        # Convert np.arrays to BitVects
        fps = list(map(lambda x: DataStructs.CreateFromBitString(''.join(map(str, x))),
                   self.get_fingerprint(mols)))
        return [list(1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, self.fps)))
                for fp in fps]

    def calculate_fingerprints(self, list_of_smiles):
        """Calculate the fingerprints for the list of SMILES sequences."""
        # Convert np.arrays to BitVects
        self.fps = list(map(lambda x: DataStructs.CreateFromBitString(''.join(map(str, x))),
                            self.get_fingerprint([Chem.MolFromSmiles(smiles) for smiles in list_of_smiles])
                            ))

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        return {"fingerprint_type": self.fingerprint_type,
                "list_of_smiles": self._descriptors, "args": self._args, "kwargs": self._kwargs}

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


class Mold2(DescriptorSet):
    """Descriptors from molecular descriptor calculation software Mold2.

    From https://github.com/OlivierBeq/Mold2_pywrapper.
    Initialize the descriptor with no arguments.
    All descriptors are always calculated.
    """

    def __init__(self, descs: Optional[List[str]] = None):
        """Initialize a PaDEL calculator.

        Args:
            descs: names of Mold2 descriptors to be calculated (e.g. D001)
        """
        self._is_fp = False
        self._descs = descs
        self._mold2 = Mold2_calculator()
        self._default_descs = self._mold2.calculate([Chem.MolFromSmiles("C")], show_banner=False).columns.tolist()
        self._descriptors = self._default_descs[:]
        self._keepindices = list(range(len(self._descriptors)))

    def __call__(self, mols):
        values = self._mold2.calculate(self.iterMols(mols), show_banner=False)
        # Drop columns
        values = values[self._descriptors].values
        return values

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        return {'descs': self._descs}

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, names: Optional[List[str]] = None):
        if names is None:
            self._descriptors = self._default_descs[:]
            self._keepindices = list(range(len(self._descriptors)))
        # Find descriptors not part of Mold2
        remainder = set(names).difference(set(self._default_descs))
        if len(remainder) > 0:
            raise ValueError(f'names are not valid Mold2 descriptor names: {", ".join(remainder)}')
        else:
            new_indices = []
            new_descs = []
            for i, desc_name in enumerate(self._default_descs):
                if desc_name in names:
                    new_indices.append(i)
                    new_descs.append(self._default_descs[i])

    def __str__(self):
        return "Mold2"


class PaDEL(DescriptorSet):
    """Descriptors from molecular descriptor calculation software PaDEL.

    From https://github.com/OlivierBeq/PaDEL_pywrapper.
    """

    def __init__(self, descs: Optional[List[str]] = None, ignore_3D: bool = True):
        """Initialize a PaDEL calculator

        Args:
            descs: list of PaDEL descriptor short names
            ignore_3D (bool): skip 3D descriptor calculation
        """
        self._descs = descs
        self._ignore_3D = ignore_3D

        self._is_fp = False

        # Create calculator and obtain default descriptor names
        dummy_mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
        AllChem.EmbedMolecule(dummy_mol)  # Required for line below, since 3D descs would raise without 3D coords
        self._name_mapping = {name: descriptor
                              for descriptor in (PaDEL_2D_descriptors if ignore_3D else PaDEL_All_descriptors)
                              for name in descriptor().calculate(dummy_mol)}

        # Initialize descriptors and calculator
        if descs is None:
            self.descriptors = None
        else:
            self.descriptors = descs

    def __call__(self, mols):
        values = self._padel.calculate(self.iterMols(mols, to_list=True), show_banner=False, njobs=1)
        intersection = list(set(self._keep).intersection(values.columns))
        values = values[intersection]
        return values

    @property
    def is_fp(self):
        return self._is_fp

    @property
    def settings(self):
        return {'descs': self._descs, 'ignore_3D': self._ignore_3D}

    @property
    def descriptors(self):
        return self._keep

    @descriptors.setter
    def descriptors(self, names: Optional[List[str]] = None):
        # From name to PaDEL descriptor sub-classes
        if names is None:
            self._descriptors = list(set(self._name_mapping.values()))
        else:
            remainder = set(names).difference(set(self._name_mapping.keys()))
            if len(remainder) > 0:
                raise ValueError(f'names are not valid PaDEL descriptor names: {", ".join(remainder)}')
            self._descriptors = list(set(self._name_mapping[name] for name in names))
        # Instantiate calculator
        self._padel = PaDEL_calculator(self._descriptors, ignore_3D=self._ignore_3D)
        # Set names to keep when calculating
        if names is None:
            self._keep = [name for name, desc in self._name_mapping.items() if desc in self._descriptors]
        else:
            self._keep = names

    def get_names(self):
        return self._padel.calculate([Chem.MolFromSmiles("C")], show_banner=False, njobs=1).columns.tolist()

    def __str__(self):
        return "PaDEL"


class PredictorDesc(DescriptorSet):
    """DescriptorSet that uses a Predictor object to calculate the descriptors for a molecule."""

    def __init__(self, model : "QSPRModel"):
        """
        Initialize the descriptorset with a `QSPRModel` object.

        Args:
            model: a fitted model instance
        """
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
    def is_fp(self):
        return False

    @property
    def settings(self):
        """Return args and kwargs used to initialize the descriptorset."""
        return {'model': self.model}

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
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

    def get_Mold2(self, *args, **kwargs):
        return Mold2(*args, **kwargs)

    def get_PaDEL(self, *args, **kwargs):
        return PaDEL(*args, **kwargs)

    def get_RDkit(self, *args, **kwargs):
        return rdkit_descs(*args, **kwargs)

    def get_PredictorDesc(self, *args, **kwargs):
        return PredictorDesc(*args, **kwargs)

    def get_TanimotoDistances(self, *args, **kwargs):
        return TanimotoDistances(*args, **kwargs)


def get_descriptor(desc_type: str, *args, **kwargs):
    return _DescriptorSetRetriever().get_descriptor(desc_type, *args, **kwargs)
