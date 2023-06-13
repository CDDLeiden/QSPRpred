"""
descriptorsets

Created by: Martin Sicho
On: 12.05.23, 16:25
"""
from abc import ABC, abstractmethod

import mordred
import pandas as pd
import prodec
from Mold2_pywrapper import Mold2 as Mold2_calculator
from mordred import descriptors as mordreddescriptors
from PaDEL_pywrapper import PaDEL as PaDEL_calculator
from PaDEL_pywrapper import descriptors as PaDEL_descriptors
from rdkit import Chem

from ....data.utils.descriptorsets import DescriptorSet, MoleculeDescriptorSet


class Mordred(MoleculeDescriptorSet):
    """Descriptors from molecular descriptor calculation software Mordred.

    From https://github.com/mordred-descriptor/mordred.

    Arguments:
        descs (list): list of mordred descriptor names
        version (str): version of mordred
        ignore_3D (bool): ignore 3D information
        config (str): path to config file
    """
    def __init__(self, descs=None, version=None, ignore_3D=False, config=None):
        """
        Initialize the descriptor with the same arguments as you would pass to
        `DescriptorsCalculator` function of Mordred.

        With the exception of the `descs` argument which can also be a list of mordred
        descriptor names instead of a mordred descriptor module.

        Args:
            descs: list of Mordred descriptor names, a Mordred descriptor module or None
                for all mordred descriptors.
            version (str): version of mordred.
            ignore_3D (bool): ignore 3D information.
            config (str): path to config file?
        """
        if descs:
            # if mordred descriptor module is passed, convert to list of descriptor instances # noqa: E501
            if not isinstance(descs, list):
                descs = mordred.Calculator(descs).descriptors
        else:
            # use all mordred descriptors if no descriptors are specified
            descs = mordred.Calculator(mordreddescriptors).descriptors

        self.version = version
        self.ignore3D = ignore_3D
        self.config = config

        self._isFP = False

        self._mordred = None

        # convert to list(descriptor names) if descriptor instances are passed
        self.descriptors = [str(d) for d in descs]

    def __call__(self, mols):
        return self._mordred.pandas(self.iterMols(mols), quiet=True, nproc=1).values

    @property
    def isFP(self):
        return self._isFP

    @property
    def settings(self):
        return {
            "descs": self.descriptors,
            "version": self.version,
            "ignore_3D": self.ignore3D,
            "config": self.config,
        }

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, names):
        """Set the descriptors to calculate.

        Converts a list of Mordred descriptor names to Mordred descriptor instances
        which is used to initialize the a Mordred calculator with the specified
        descriptors.

        Args:
            names: list of Mordred descriptor names.
        """
        calc = mordred.Calculator(mordreddescriptors)
        self._mordred = mordred.Calculator(
            [d for d in calc.descriptors if str(d) in names],
            version=self.version,
            ignore_3D=self.ignore3D,
            config=self.config,
        )
        self._descriptors = names

    def __str__(self):
        return "Mordred"


class Mold2(MoleculeDescriptorSet):
    """Descriptors from molecular descriptor calculation software Mold2.

    From https://github.com/OlivierBeq/Mold2_pywrapper.
    Initialize the descriptor with no arguments.
    All descriptors are always calculated.

    Arguments:
        descs: names of Mold2 descriptors to be calculated (e.g. D001)
    """
    def __init__(self, descs: list[str] | None = None):
        """Initialize a PaDEL calculator.

        Args:
            descs: names of Mold2 descriptors to be calculated (e.g. D001)
        """
        self._isFP = False
        self._descs = descs
        self._mold2 = Mold2_calculator()
        self._default_descs = self._mold2.calculate(
            [Chem.MolFromSmiles("C")], show_banner=False
        ).columns.tolist()
        self._descriptors = self._default_descs[:]
        self._keepindices = list(range(len(self._descriptors)))

    def __call__(self, mols):
        values = self._mold2.calculate(self.iterMols(mols), show_banner=False)
        # Drop columns
        values = values[self._descriptors].values
        return values

    @property
    def isFP(self):
        return self._isFP

    @property
    def settings(self):
        return {"descs": self._descs}

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, names: list[str] | None = None):
        if names is None:
            self._descriptors = self._default_descs[:]
            self._keepindices = list(range(len(self._descriptors)))
            return
        # Find descriptors not part of Mold2
        remainder = set(names).difference(set(self._default_descs))
        if len(remainder) > 0:
            raise ValueError(
                f'names are not valid Mold2 descriptor names: {", ".join(remainder)}'
            )
        else:
            new_indices = []
            new_descs = []
            for i, desc_name in enumerate(self._default_descs):
                if desc_name in names:
                    new_indices.append(i)
                    new_descs.append(self._default_descs[i])
            self._descriptors = new_descs
            self._keepindices = new_indices

    def __str__(self):
        return "Mold2"


class PaDEL(MoleculeDescriptorSet):
    """Descriptors from molecular descriptor calculation software PaDEL.

    From https://github.com/OlivierBeq/PaDEL_pywrapper.

    Arguments:
        descs: list of PaDEL descriptor short names
        ignore_3D (bool): skip 3D descriptor calculation
    """
    def __init__(self, descs: list[str] | None = None, ignore_3D: bool = True):
        """Initialize a PaDEL calculator

        Args:
            descs: list of PaDEL descriptor short names
            ignore_3D (bool): skip 3D descriptor calculation
        """
        self._descs = descs
        self._ignore3D = ignore_3D

        self._isFP = False

        # Obtain default descriptor names
        self._name_mapping = {}
        for descriptor in PaDEL_descriptors:
            # Skipt if desc is 3D and set to be ignored
            if ignore_3D and descriptor.is_3D:
                continue
            for name in descriptor.description.name:
                self._name_mapping[name] = descriptor

        # Initialize descriptors and calculator
        if descs is None:
            self.descriptors = None
        else:
            self.descriptors = descs

    def __call__(self, mols):
        mols = [Chem.AddHs(mol) for mol in self.iterMols(mols)]
        values = self._padel.calculate(mols, show_banner=False, njobs=1)
        intersection = list(set(self._keep).intersection(values.columns))
        values = values[intersection]
        return values

    @property
    def isFP(self):
        return self._isFP

    @property
    def settings(self):
        return {"descs": self._descs, "ignore_3D": self._ignore3D}

    @property
    def descriptors(self):
        return self._keep

    @descriptors.setter
    def descriptors(self, names: list[str] | None = None):
        # From name to PaDEL descriptor sub-classes
        if names is None:
            self._descriptors = list(set(self._name_mapping.values()))
        else:
            remainder = set(names).difference(set(self._name_mapping.keys()))
            if len(remainder) > 0:
                raise ValueError(
                    "names are not valid PaDEL descriptor names: "
                    f"{', '.join(remainder)}"
                )
            self._descriptors = list({self._name_mapping[name] for name in names})
        # Instantiate calculator
        self._padel = PaDEL_calculator(self._descriptors, ignore_3D=self._ignore3D)
        # Set names to keep when calculating
        if names is None:
            self._keep = [
                name for name, desc in self._name_mapping.items()
                if desc in self._descriptors
            ]
        else:
            self._keep = names

    def __str__(self):
        return "PaDEL"


class ProteinDescriptorSet(DescriptorSet):
    """
    Abstract base class for protein descriptor sets.

    Arguments:
        acc_keys: target accession keys, the resulting data frame will be indexed by
        these keys
        sequences: optional list of protein sequences matched to the accession keys
        **kwargs: additional data mapped to the accession keys, each parameter
            should follow the same format as the sequences (dict(str : str))
    """
    @abstractmethod
    def __call__(self, acc_keys, sequences: dict[str:str] = None, **kwargs):
        """
        Calculate the protein descriptors for a given target.

        Args:
            acc_keys: target accession keys, the resulting data frame will be indexed by
            these keys
            sequences: optional list of protein sequences matched to the accession keys
            **kwargs: additional data mapped to the accession keys, each parameter
                should follow the same format as the sequences (dict(str : str))

        Returns:
            a data frame of descriptor values of shape (acc_keys, n_descriptors),
                indexed by acc_keys
        """


class NeedsMSAMixIn(ABC):  # FIXME - Inherit ABC but no abstract method # noqa: B024
    """Abstract base class for descriptorsets that need a multiple sequence alignment.

    Arguments
        msa (dict): mapping of accession keys to sequences from the multiple
            sequence alignment."""
    def __init__(self):
        self.msa = None

    def setMSA(self, msa: dict[str:str]):
        """
        Set the multiple sequence alignment for the protein descriptor set.

        Args:
            msa (dict): mapping of accession keys to sequences from the multiple
                sequence alignment
        """
        self.msa = msa


class ProDecDescriptorSet(ProteinDescriptorSet, NeedsMSAMixIn):
    def __init__(self, sets: list[str] | None = None):
        super().__init__()
        self._settings = {"sets": sets}
        self.factory = prodec.ProteinDescriptors()
        self.sets = self.factory.available_descriptors if sets is None else sets
        self._descriptors = None

    @staticmethod
    def calculate_descriptor(factory, msa, descriptor):
        """
        Calculate a protein descriptor for given targets using given multiple sequence
        alignment.

        Args:
            factory (ProteinDescriptors): factory to create the descriptor
            msa (dict): mapping of accession keys to sequences from the multiple
                sequence alignment
            descriptor (str): name of the descriptor to calculate

        Returns:
            a data frame of descriptor values of shape (acc_keys, n_descriptors),
                indexed by acc_keys
        """

        # Get protein descriptor from ProDEC
        prodec_descriptor = factory.get_descriptor(descriptor)

        # Calculate descriptor features for aligned sequences of interest
        protein_features = prodec_descriptor.pandas_get(msa.values(), ids=msa.keys())
        protein_features.set_index("ID", inplace=True)

        return protein_features

    def __call__(self, acc_keys, sequences: dict[str:str] = None, **kwargs):
        df = pd.DataFrame(index=pd.Index(acc_keys, name="ID"))
        for descriptor in self.sets:
            df = df.merge(
                self.calculate_descriptor(self.factory, self.msa, descriptor),
                left_index=True,
                right_index=True,
            )

        if not self._descriptors:
            self._descriptors = df.columns.tolist()
        else:
            df.drop(
                columns=[col for col in df.columns if col not in self._descriptors],
                inplace=True,
            )

        return df

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
        return self._settings

    def __str__(self):
        return "ProDec"
