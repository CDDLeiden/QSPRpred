"""Module with definitions of various extra descriptor sets:

- `Mordred`: Descriptors from molecular descriptor calculation software Mordred.
- `Mold2`: Descriptors from molecular descriptor calculation software Mold2.
- `PaDEL`: Descriptors from molecular descriptor calculation software PaDEL.
- `ProDec`: Protein descriptors from the ProDec package.

"""
import os
from abc import abstractmethod
from typing import Optional, Any

import mordred
import numpy as np
import pandas as pd
import prodec
from Mold2_pywrapper import Mold2 as Mold2_calculator
from PaDEL_pywrapper import PaDEL as PaDELCalculator
from PaDEL_pywrapper import descriptors as PaDEL_descriptors
from Signature_pywrapper import Signature as Signature_calculator
from mordred import descriptors as Mordred_descriptors
from rdkit import Chem
from rdkit.Chem import Mol

from qsprpred.data.descriptors.sets import DescriptorSet
from qsprpred.extra.data.utils.msa_calculator import MSAProvider, ClustalMSA


class Mordred(DescriptorSet):
    """Descriptors from molecular descriptor calculation software Mordred.

    From https://github.com/mordred-descriptor/mordred.


    Attributes:
        descs (list[str]): List of Mordred descriptor names.
        version (str): version of mordred
        ignore_3D (bool): ignore 3D information
        config (str): path to config file if available

    """

    def __init__(
        self,
        descs: list[str] | None = None,
        version: str | None = None,
        ignore_3D: bool = False,
        config: str | None = None,
    ):
        """
        Initialize the descriptor with the same arguments as you would pass
        to `DescriptorsCalculator` function of Mordred, except the
        `descs` argument, which can also be a `list` of mordred descriptor names instead
        of a mordred descriptor module.

        Args:
            descs (list[str]): List of Mordred descriptor names, a Mordred descriptor
                module or `None` for all mordred descriptors
            version (str): version of mordred
            ignore_3D (bool): ignore 3D information
            config (str): path to config file?
        """
        super().__init__()
        if descs:
            # if mordred descriptor module is passed,
            # convert to list of descriptor instances
            if not isinstance(descs, list):
                descs = mordred.Calculator(descs).descriptors
        else:
            # use all mordred descriptors if no descriptors are specified
            descs = mordred.Calculator(Mordred_descriptors).descriptors
        # init member variables
        self.version = version
        self.ignore3D = ignore_3D
        self.config = config
        self._mordred = None
        # convert to list of descriptor names if descriptor instances are passed
        self.descriptors = [str(d) for d in descs]

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        df = self._mordred.pandas(self.iterMols(mols), quiet=True, nproc=1)
        df = df.apply(pd.to_numeric, errors="coerce")  # replace errors by nan values
        return df.values

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, names: list[str]):
        """Set the descriptors to calculate.

        Converts a list of Mordred descriptor names to Mordred descriptor instances,
        which is used to initialize a Mordred calculator with the specified descriptors.

        Args:
            names (list[str]): List of Mordred descriptor names.
        """
        calc = mordred.Calculator(Mordred_descriptors)
        self._mordred = mordred.Calculator(
            [d for d in calc.descriptors if str(d) in names],
            version=self.version,
            ignore_3D=self.ignore3D,
            config=self.config,
        )
        self._descriptors = names

    def __str__(self):
        return "Mordred"


class Mold2(DescriptorSet):
    """Descriptors from molecular descriptor calculation software Mold2.

    From https://github.com/OlivierBeq/Mold2_pywrapper.
    Initialize the descriptor with no arguments.
    All descriptors are always calculated.

    Arguments:
        descs: names of Mold2 descriptors to be calculated (e.g. D001)
    """

    def __init__(self, descs: list[str] | None = None):
        """Initialize a Mold2 descriptor calculator.

        Args:
            descs (list[str] | None):
                names of Mold2 descriptors to be calculated (e.g. D001)
        """
        super().__init__()
        self._descs = descs
        self._mold2 = Mold2_calculator()
        self._defaultDescs = self._mold2.calculate(
            [Chem.MolFromSmiles("C")], show_banner=False
        ).columns.tolist()
        self._descriptors = self._defaultDescs[:]
        self._keepindices = list(range(len(self._descriptors)))

    @property
    def supportsParallel(self) -> bool:
        return False

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        values = self._mold2.calculate(self.iterMols(mols), show_banner=False)
        # Drop columns
        values = values[self._descriptors].values
        return values

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, names: list[str] | None = None):
        """Set the descriptors to calculate.

        Args:
            names (list[str] | None): list of Mold2 descriptor names
        """
        if names is None:
            self._descriptors = self._defaultDescs[:]
            self._keepindices = list(range(len(self._descriptors)))
            return
        # Find descriptors not part of Mold2
        remainder = set(names).difference(set(self._defaultDescs))
        if len(remainder) > 0:
            raise ValueError(
                f'names are not valid Mold2 descriptor names: {", ".join(remainder)}'
            )
        else:
            new_indices = []
            new_descs = []
            for i, desc_name in enumerate(self._defaultDescs):
                if desc_name in names:
                    new_indices.append(i)
                    new_descs.append(self._defaultDescs[i])
            self._descriptors = new_descs
            self._keepindices = new_indices

    def __str__(self):
        return "Mold2"


class PaDEL(DescriptorSet):
    """Descriptors from molecular descriptor calculation software PaDEL.

    From https://github.com/OlivierBeq/PaDEL_pywrapper.

    Attributes:
        descriptors (list[str]): list of PaDEL descriptor names
    """

    _notJSON = ["_nameMapping", "_padel", "_descriptors", *DescriptorSet._notJSON]

    def __init__(
        self,
        descs: list[str] | None = None,
        ignore_3d: bool = True,
        n_jobs: int | None = None,
    ):
        """Initialize a PaDEL calculator

        Args:
            descs: list of PaDEL descriptor short names
            ignore_3d (bool): skip 3D descriptor calculation
        """
        super().__init__()
        self.nJobs = n_jobs or os.cpu_count()
        self._descs = descs
        self._ignore3D = ignore_3d
        # Initialize name mapping
        self._initMapping()
        # Initialize descriptors and calculator
        if descs is None:
            self.descriptors = None
        else:
            self.descriptors = descs

    @property
    def supportsParallel(self) -> bool:
        return False

    def _initMapping(self):
        # Obtain default descriptor names
        self._nameMapping = {}
        for descriptor in PaDEL_descriptors:
            # Skip if desc is 3D and set to be ignored
            if self._ignore3D and descriptor.is_3D:
                continue
            for name in descriptor.description.name:
                self._nameMapping[name] = descriptor

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._initMapping()
        self.descriptors = state["_keep"]

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        mols = [Chem.AddHs(mol) for mol in self.iterMols(mols)]
        df = self._padel.calculate(mols, show_banner=False, njobs=self.nJobs)
        intersection = list(set(self._keep).intersection(df.columns))
        df = df[intersection]
        return df.values

    @property
    def descriptors(self):
        return self._keep

    @descriptors.setter
    def descriptors(self, names: list[str] | None = None):
        """Set the descriptors to calculate.

        Args:
            names (list[str] | None): list of PaDEL descriptor names
        """
        # convert from name to PaDEL descriptor sub-classes
        if names is None:
            self._descriptors = list(set(self._nameMapping.values()))
        else:
            remainder = set(names).difference(set(self._nameMapping.keys()))
            if len(remainder) > 0:
                raise ValueError(
                    "names are not valid PaDEL descriptor names: "
                    f"{', '.join(remainder)}"
                )
            self._descriptors = list({self._nameMapping[name] for name in names})
        # Instantiate calculator
        self._padel = PaDELCalculator(self._descriptors, ignore_3D=self._ignore3D)
        # Set names to keep when calculating
        if names is None:
            self._keep = [
                name
                for name, desc in self._nameMapping.items()
                if desc in self._descriptors
            ]
        else:
            self._keep = names

    def __str__(self):
        return "PaDEL"


class ExtendedValenceSignature(DescriptorSet):
    """SMILES signature based on extended valence sequence from

    The Signature Molecular Descriptor.

    1. Using Extended Valence Sequences in QSAR and QSPR StudiesJean-Loup
    Faulon, Donald P. Visco, and Ramdas S. Pophale
    Journal of Chemical Information and Computer Sciences 2003 43 (3), 707-720
    DOI: 10.1021/ci020345w
    """

    def __init__(self, depth: int | list[int]):
        """Initialize a ExtendedValenceSignature calculator

        Args:
            depth: depth of the signature
        """
        super().__init__()
        self._depth = depth
        self._signature = Signature_calculator()
        self._descriptors = []
        # Flag initialization of descriptors after first calculation
        self._descriptors_init = False

    @property
    def supportsParallel(self) -> bool:
        return False

    def getDescriptors(
        self, mols: list[Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> np.ndarray:
        mols = [Chem.AddHs(mol) for mol in self.iterMols(mols)]
        df = self._signature.calculate(
            mols, depth=self._depth, show_banner=False, njobs=1
        ).fillna(0)
        if not self._descriptors_init:
            self.descriptors = df.columns.tolist()
            self._descriptors_init = True
        else:
            intersection = list(set(self.descriptors).intersection(df.columns))
            df = df[intersection]
        return df.values

    @property
    def descriptors(self):
        return self._descriptors

    @descriptors.setter
    def descriptors(self, names: list[str] | None = None):
        if names is None:
            self._descriptors = []
        else:
            self._descriptors = names
            self._descriptors_init = True

    def __str__(self):
        return "ExtendedValenceSignature"


class ProteinDescriptorSet(DescriptorSet):
    """Abstract base class for protein descriptor sets."""

    @abstractmethod
    def getProteinDescriptors(
        self, acc_keys: list[str], sequences: Optional[dict[str, str]] = None, **kwargs
    ) -> pd.DataFrame:
        """
        Calculate the protein descriptors for a given target.

        Args:
            acc_keys (list[str]):
                target accession keys, the resulting data frame
                will be indexed by these keys
            sequences (dict[str, str]):
                optional list of protein sequences matched to the accession keys
            **kwargs:
                additional data passed from `ProteinDescriptorCalculator`

        Returns:
            pd.DataFrame:
                a data frame of descriptor values of shape (acc_keys, n_descriptors),
                indexed by `acc_keys`
        """

    def getDescriptors(
        self,
        mols: list[Mol],
        props: dict[str, list[Any] | dict[str, str]],
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Get array of calculated protein descriptors for given targets.

        Args:
            mols (list[Mol]): list of molecules, not used
            props (dict[str, list[Any]  |  dict[str, str]]): dictionary of properties
                for the molecules, including the accession keys
            *args: additional arguments, not used
            **kwargs: additional keyword arguments, passed to `getProteinDescriptors`

        Returns:
            np.ndarray: array of calculated protein descriptors
        """
        # Get array of calculated protein descriptors
        acc_keys = sorted(set(props["acc_keys"]))
        values = self.getProteinDescriptors(acc_keys, **kwargs).reset_index()
        values.rename(columns={"ID": "acc_keys"}, inplace=True)
        # create a data frame with the same order of acc_keys as in props
        df = pd.DataFrame({"acc_keys": props["acc_keys"]})
        # merge the calculated values with the data frame to attach them to the rows
        df = df.merge(values, left_on="acc_keys", right_on="acc_keys",
                      how="left").set_index("acc_keys")
        return df.values

    @property
    def requiredProps(self) -> list[str]:
        existing = super().requiredProps
        return ["acc_keys", *existing]

    def supportsParallel(self) -> bool:
        return False


class ProDec(ProteinDescriptorSet):
    """Protein descriptors from the ProDec package.

    See https://github.com/OlivierBeq/ProDEC.

    Attributes:
        sets (list[str]):
            list of ProDec descriptor names  (see https://github.com/OlivierBeq/ProDEC)
        factory (prodec.ProteinDescriptors):
            factory to calculate descriptors
    """

    def __init__(
        self, sets: list[str] | None = None, msa_provider: MSAProvider = ClustalMSA()
    ):
        """Initialize a ProDec calculator.

        Args:
            sets:
                list of ProDec descriptor names, if `None`, all available are used
                (see https://github.com/OlivierBeq/ProDEC)
        """
        super().__init__()
        self.factory = prodec.ProteinDescriptors()
        self.sets = self.factory.available_descriptors if sets is None else sets
        self._descriptors = []
        self.msaProvider = msa_provider
        self.msa = None

    def __getstate__(self):
        o_dict = super().__getstate__()
        # Remove factory from state
        del o_dict["factory"]
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        # Add factory to state
        self.factory = prodec.ProteinDescriptors()

    @staticmethod
    def calculateDescriptor(
        factory: prodec.ProteinDescriptors, msa: dict[str, str], descriptor: str
    ):
        """
        Calculate a protein descriptor for given targets
        using a given multiple sequence alignment.

        Args:
            factory (ProteinDescriptors):
                factory to create the descriptor
            msa (dict):
                mapping of accession keys to sequences from the multiple
                sequence alignment
            descriptor (str):
                name of the descriptor to calculate (see
                https://github.com/OlivierBeq/ProDEC)

        Returns:
            a data frame of descriptor values of shape (acc_keys, n_descriptors),
            indexed by acc_keys
        """
        # Get protein descriptor from ProDEC
        prodec_descriptor = factory.get_descriptor(descriptor)
        # Calculate descriptor features for aligned sequences of interest
        protein_features = prodec_descriptor.pandas_get(msa.values(), ids=msa.keys())
        return protein_features

    def getProteinDescriptors(
        self, acc_keys: list[str], sequences: Optional[dict[str, str]] = None, **kwargs
    ) -> pd.DataFrame:
        """
        Calculate the protein descriptors for a given target.

        Args:
            acc_keys:
                target accession keys, defines the resulting index of
                the returned `pd.DataFrame`
            sequences:
                optional list of protein sequences matched to the accession keys
            **kwargs:
                any additional data passed from `ProteinDescriptorCalculator`

        Returns:
            a data frame of descriptor values of shape (acc_keys, n_descriptors),
        """
        # calculate MSA
        if not self.msa:
            self.msa = self.msaProvider(sequences, **kwargs)
        # calculate descriptors
        dfs = []
        for descriptor in self.sets:
            dfs.append(self.calculateDescriptor(self.factory, self.msa, descriptor))
        df = pd.concat(dfs, axis=1)
        df.set_index("ID", inplace=True, drop=True)
        # Keep only descriptors that were requested to keep
        if not self._descriptors:
            self._descriptors = sorted(df.columns.tolist())
        else:
            df.drop(
                columns=[col for col in df.columns if col not in self._descriptors],
                inplace=True,
            )
        # reorder columns to reflect the order of descriptors
        df = df[self.descriptors]
        return df

    @property
    def descriptors(self):
        return sorted(self._descriptors)

    @descriptors.setter
    def descriptors(self, value):
        self._descriptors = value

    def __str__(self):
        return "ProDec_" + "_".join(self.sets)
