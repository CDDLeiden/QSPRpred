"""This module is used for calculating molecular descriptors using descriptorsets."""
import json
from abc import ABC, abstractmethod
from typing import List, Union

import pandas as pd
from qsprpred.data.utils.descriptorsets import DescriptorSet, get_descriptor
from rdkit.Chem.rdchem import Mol


class Calculator(ABC):
    """Calculator for molecule properties."""

    @abstractmethod
    def __call__(self, mols: List[Union[Mol, str]]) -> pd.DataFrame:
        """Calculate all specified descriptors for a list of rdkit mols.

        Args:
            mols: list of rdkit mols or smiles strings

        Returns:
            a numpy array with the calculated descriptors of shape (n_mols, n_descriptors)
        """
        pass

    @classmethod
    @abstractmethod
    def fromFile(cls, fname: str):
        """Construct calculator from json file.

        Args:
            fname: file name to save featurecalculator to
        """
        pass

    @abstractmethod
    def toFile(self, fname: str) -> None:
        """Save calculator with settings to json file.

        Args:
            fname: file name to save featurecalculator to
        """
        pass

    @abstractmethod
    def keepDescriptors(self, descriptors: List[str]) -> None:
        """Drop all descriptors/descriptorsets not in descriptor list.

        Args:
            descriptors: list of descriptornames to keep
        """
        pass


class DescriptorsCalculator(Calculator):
    """Calculator for molecule properties."""

    def __init__(self, descsets: List[DescriptorSet]) -> None:
        """Set the descriptorsets to be calculated with this calculator."""
        self.descsets = list(descsets)

    __in__ = __contains__ = lambda self, x: x in self.descsets

    @classmethod
    def fromFile(cls, fname: str):
        """Initialize descriptorset from a json file.

        Args:
            fname: file name of json file with descriptor names and settings
        """
        with open(fname, "r") as infile:
            descset_dict = json.load(infile)

        descsets = []
        for key, value in descset_dict.items():
            descset = get_descriptor(key, **value["settings"])
            if descset.is_fp:
                descset.keepindices = value["keepindices"]
            else:
                descset.descriptors = value["descriptors"]
            descsets.append(descset)
        return DescriptorsCalculator(descsets)

    def __call__(self, mols: List[Mol]) -> pd.DataFrame:
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
            df = pd.concat([df, values.add_prefix(f"Descriptor_{descset}_")], axis=1)

        # replace errors by nan values
        df = df.apply(pd.to_numeric, errors='coerce')

        return df

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
        for idx, descriptorset in enumerate(self.descsets):
            # Find all descriptors in current descriptorset
            descs_from_curr_set = [
                f.replace(f"Descriptor_{descriptorset}_", "")
                for f in descriptors
                if f.startswith(f"Descriptor_{descriptorset}_")
            ]
            # if there are none to keep from current descriptors set, drop the whole set
            if not descs_from_curr_set:
                self.descsets.remove(descriptorset)
            # if the set is a fingerprint, set indices to keep
            elif descriptorset.is_fp:
                self.descsets[idx].keepindices = [
                    f for f in descs_from_curr_set
                ]
            # if the set is not a fingerprint, set descriptors to keep
            else:
                self.descsets[idx].descriptors = descs_from_curr_set

    def get_len(self):
        """Return number of descriptors calculated by all descriptorsets."""
        length = 0
        for descset in self.descsets:
            length += descset.get_len()
        return length
