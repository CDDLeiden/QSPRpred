"""This module is used for calculating molecular descriptors using descriptorsets."""
import json
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd

from qsprpred.data.utils.descriptor_utils.msa_calculator import ClustalMSA
from qsprpred.data.utils.descriptorsets import get_descriptor, DescriptorSet, \
    ProteinDescriptorSet
from qsprpred.logs import logger
from rdkit.Chem.rdchem import Mol

class DescriptorsCalculator(ABC):
    """Calculator for various descriptors of molecules."""

    __in__ = __contains__ = lambda self, x: x in self.descsets

    def __str__(self):
        return f"{self.__class__.__name__}_{'_'.join([str(x) for x in self.descsets])}"

    def __init__(self, descsets: List[DescriptorSet]) -> None:
        """Set the descriptorsets to be calculated with this calculator."""
        self.descsets = list(descsets)

    @classmethod
    def loadDescriptorSets(cls, fname: str) -> List[DescriptorSet]:
        with open(fname, "r") as infile:
            descset_dict = json.load(infile)

        descsets = []
        for key, value in descset_dict.items():
            if key.startswith("FingerprintSet_"):
                key = "FingerprintSet"
            descset = get_descriptor(key, **value["settings"])
            if descset.is_fp:
                descset.keepindices = value["keepindices"]
            else:
                descset.descriptors = value["descriptors"]
            descsets.append(descset)

        return descsets

    @classmethod
    def fromFile(cls, fname: str):
        """Initialize descriptorset from a json file.

        Args:
            fname: file name of json file with descriptor names and settings
        """

        descsets = cls.loadDescriptorSets(fname)
        return cls(descsets)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        """
        Calculate features as a pandas dataframe.

        Args:
            *args: arguments to be passed to the calculator to perform the calculation
            **kwargs: keyword arguments to be passed to the calculator to perform the calculation

        Returns:
            matrix (pd.DataFrame): a data frame of shape (n_inputs, n_descriptors), if one of the inputs is expected to be an index, the output data frame should use the same index to map outputs to inputs.
        """

    def toFile(self, fname: str) -> None:
        """Save descriptorset to json file.

        Args:
            fname: file name of json file with descriptor names and settings
        """
        descset_dict = {}
        for descset in self.descsets:
            if descset.is_fp:
                descset_dict[descset.__str__()] = {
                    "settings": descset.settings,
                    "keepindices": descset.keepindices,
                }
            else:
                descset_dict[descset.__str__()] = {
                    "settings": descset.settings,
                    "descriptors": descset.descriptors,
                }
        with open('%s' % fname, "w") as outfile:
            json.dump(descset_dict, outfile)

    def keepDescriptors(self, descriptors: List[str]) -> None:
        """Drop all descriptors/descriptorsets not in descriptor list.

        Args:
            descriptors: list of descriptornames with descriptorset prefix to keep
        """
        to_remove = []
        for idx, descriptorset in enumerate(self.descsets):
            # Find all descriptors in current descriptorset
            descs_from_curr_set = [
                f.replace(f"Descriptor_{descriptorset}_", "")
                for f in descriptors
                if f.startswith(f"Descriptor_{descriptorset}_")
            ]
            # if there are none to keep from current descriptors set, skip the whole set
            if not descs_from_curr_set:
                to_remove.append(idx)
                continue

            if descriptorset.is_fp:
                # if the set is a fingerprint, set indices to keep
                self.descsets[idx].keepindices = [
                    f for f in descs_from_curr_set
                ]
            else:
                # if the set is not a fingerprint, set descriptors to keep
                self.descsets[idx].descriptors = descs_from_curr_set

        # remove all descriptorsets that are not in the list of descriptors to keep
        self.descsets = [x for i, x in enumerate(
            self.descsets) if i not in to_remove]

    def get_len(self):
        """Return number of descriptors calculated by all descriptorsets."""
        length = 0
        for descset in self.descsets:
            length += descset.get_len()
        return length

    @staticmethod
    def treatInfs(df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values by NaNs.

        Args:
            df: dataframe to treat

        Returns:
            dataframe with infinite values replaced by NaNs
        """
        if np.isinf(df).any().any():
            col_names = df.columns
            x_loc, y_loc = np.where(np.isinf(df.values))
            inf_cols = np.take(col_names, np.unique(y_loc))
            logger.debug("Infinite values in dataframe at columns:"
                         f"\n{inf_cols}"
                         "And rows:"
                         f"\n{np.unique(x_loc)}"
                         )
            # Convert absurdly high values to NaNs
            df = df.replace([np.inf, -np.inf], np.NAN)
        return df

class MoleculeDescriptorsCalculator(DescriptorsCalculator):
    """Calculator for molecule properties."""

    def __call__(self, mols: List[Mol], dtype=np.float32) -> pd.DataFrame:
        """Calculate descriptors for list of mols.

        Args:
            mols: list of rdkit mols
        """
        df = pd.DataFrame()
        for descset in self.descsets:
            values = descset(mols)
            values = pd.DataFrame(values, columns=descset.descriptors)
            if descset.is_fp:
                values.add_prefix(f"{descset.fingerprint_type}_")
            values = values.astype(dtype)
            values = self.treatInfs(values)
            df = pd.concat([df, values.add_prefix(
                f"Descriptor_{descset}_")], axis=1)

        # replace errors by nan values
        df = df.apply(pd.to_numeric, errors='coerce')

        return df

class ProteinDescriptorCalculator(DescriptorsCalculator):

    def __init__(self, descsets: List[ProteinDescriptorSet], msa_provider = ClustalMSA("qsprpred@example.com")) -> None:
        super().__init__(descsets)
        self.msaProvider = msa_provider

    def __call__(self, acc_keys, sequences: dict[str: str] = None, dtype=np.float32, **kwargs) -> pd.DataFrame:
        df = pd.DataFrame(index=acc_keys)
        msa = None
        for descset in self.descsets:
            if hasattr(descset, "setMSA"):
                if msa is None:
                    msa = self.msaProvider(sequences, **kwargs)
                descset.setMSA(msa)
            values = descset(acc_keys, sequences, **kwargs)

            if descset.is_fp:
                values.add_prefix(f"{descset.fingerprint_type}_")
            values = values.astype(dtype)
            values = self.treatInfs(values)
            values = values.add_prefix(f"Descriptor_PCM_{descset}_")
            df = df.merge(values, left_index=True, right_index=True)

        return df