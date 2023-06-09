"""This module is used for calculating molecular descriptors using descriptorsets."""
import json
from abc import ABC, abstractmethod
from types import TypeVar

import numpy as np
import pandas as pd
from rdkit.Chem.rdchem import Mol

from ...logs import logger
from ...utils.inspect import import_class
from .descriptorsets import get_descriptor

DataFrameDescriptorSet = TypeVar("DataFrameDescriptorSet")
DescriptorSet = TypeVar("DescriptorSet")


class DescriptorsCalculator(ABC):
    """Calculator for various descriptors of molecules."""

    __in__ = __contains__ = lambda self, x: x in self.descSets

    def __str__(self):
        return self.getPrefix()

    @abstractmethod
    def getPrefix(self) -> str:
        """Return prefix for descriptor names of the calculator."""

    def __init__(self, descsets: list[DescriptorSet]) -> None:
        """Set the descriptorsets to be calculated with this calculator."""
        self.descSets = list(descsets)

    @classmethod
    def loadDescriptorSets(cls, fname: str) -> list[DescriptorSet]:
        """Loads the descriptor sets from a json file.

        Args:
            fname (str): name of the json file with descriptor names and settings

        Returns:
            list[DescriptorSet]: list of descriptor sets
        """
        with open(fname, "r") as infile:
            descset_dict = json.load(infile)

        descsets = []
        for key, value in descset_dict.items():
            if key == "calculator":
                continue
            name = value["name"]
            if name.startswith("FingerprintSet_"):
                name = "FingerprintSet"
            descset = get_descriptor(name, **value["settings"])
            if descset.is_fp:
                descset.keepindices = value["keepindices"]
            else:
                descset.descriptors = value["descriptors"]
            descsets.append(descset)

        return descsets

    @classmethod
    def classFromFile(cls, fname: str):
        """Initialize descriptorset from a json file.

        Args:
            fname: file name of json file with descriptor names and settings
        """

        with open(fname, "r") as infile:
            descset_dict = json.load(infile)
            # Assume calculator is a molecule descriptor calculator if not specified
            # (for backwards compatibility before introduction of protein descriptor calculator) # noqa: 501
            if "calculator" not in descset_dict:
                descset_dict["calculator"] = (
                    "qsprpred.data.utils.descriptorcalculator."
                    "MoleculeDescriptorsCalculator"
                )

            return import_class(descset_dict["calculator"])

    @classmethod
    def fromFile(cls, fname: str):
        """Initialize descriptorset from a json file.

        Args:
            fname: file name of json file with descriptor names and settings
        """

        cl = cls.classFromFile(fname)
        descsets = cl.loadDescriptorSets(fname)
        return cl(descsets)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> pd.DataFrame:
        """
        Calculate features as a pandas dataframe.

        Args:
            *args: arguments to be passed to the calculator to perform the calculation
            **kwargs: keyword arguments to be passed to the calculator to perform the
                calculation

        Returns:
            matrix (pd.DataFrame): a data frame of shape (n_inputs, n_descriptors), if
                one of the inputs is expected to be an index, the output data frame
                should use the same index to map outputs to inputs.
        """

    def toFile(self, fname: str) -> None:
        """Save descriptorset to json file.

        Args:
            fname: name of the json file with descriptor names and settings
        """
        descset_dict = {}
        for idx, descset in enumerate(self.descSets):
            idx = str(idx)
            if descset.is_fp:
                descset_dict[idx] = {
                    "name": str(descset),
                    "settings": descset.settings,
                    "keepindices": descset.keepindices,
                }
            else:
                descset_dict[idx] = {
                    "name": str(descset),
                    "settings": descset.settings,
                    "descriptors": descset.descriptors,
                }
            descset_dict[idx]["class"] = descset.__class__.__name__
        # save fully qualified class name of calculator
        descset_dict["calculator"] = (
            self.__class__.__module__ + "." + self.__class__.__name__
        )
        with open("%s" % fname, "w") as outfile:
            json.dump(descset_dict, outfile)

    def keepDescriptors(self, descriptors: list[str]) -> None:
        """Drop all descriptors/descriptorsets not in descriptor list.

        Args:
            descriptors: list of descriptornames with descriptorset prefix to keep
        """
        to_remove = []
        for idx, descriptorset in enumerate(self.descSets):
            # Find all descriptors in current descriptorset
            descs_from_curr_set = [
                f.replace(f"{self.getPrefix()}_{descriptorset}_", "")
                for f in descriptors
                if f.startswith(f"{self.getPrefix()}_{descriptorset}_") and (
                    f.replace(f"{self.getPrefix()}_{descriptorset}_", "") in
                    descriptorset.descriptors
                )
            ]
            # if there are none to keep from current descriptors set, skip the whole set
            if not descs_from_curr_set:
                to_remove.append(idx)
                continue

            if descriptorset.is_fp:
                # if the set is a fingerprint, set indices to keep
                self.descSets[idx].keepindices = list(descs_from_curr_set)
            else:
                # if the set is not a fingerprint, set descriptors to keep
                self.descSets[idx].descriptors = descs_from_curr_set

        # remove all descriptorsets that are not in the list of descriptors to keep
        self.descSets = [x for i, x in enumerate(self.descSets) if i not in to_remove]

    def get_len(self):
        """Return number of descriptors calculated by all descriptorsets."""
        length = 0
        for descset in self.descSets:
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
            logger.debug(
                "Infinite values in dataframe at columns:"
                f"\n{inf_cols}"
                "And rows:"
                f"\n{np.unique(x_loc)}"
            )
            # Convert absurdly high values to NaNs
            df = df.replace([np.inf, -np.inf], np.NAN)
        return df


class MoleculeDescriptorsCalculator(DescriptorsCalculator):
    """Calculator for molecule properties."""
    def __call__(self, mols: list[Mol], dtype=np.float32) -> pd.DataFrame:
        """Calculate descriptors for list of mols.

        Args:
            mols: list of rdkit mols
        """
        df = pd.DataFrame()
        for descset in self.descSets:
            values = descset(mols)
            values = pd.DataFrame(values, columns=descset.descriptors)
            if descset.is_fp:
                values.add_prefix(f"{descset.fingerprint_type}_")
            values = values.astype(dtype)
            values = self.treatInfs(values)
            df = pd.concat(
                [df, values.add_prefix(f"{self.getPrefix()}_{descset}_")], axis=1
            )
        # replace errors by nan values
        df = df.apply(pd.to_numeric, errors="coerce")

        return df

    def getPrefix(self) -> str:
        return "Descriptor"


class CustomDescriptorsCalculator(DescriptorsCalculator):
    """Calculator for custom descriptors."""
    def __init__(self, descsets: list[DataFrameDescriptorSet]) -> None:
        """Initialize calculator.

        Args:
            descsets (list[DataFrameDescriptorSet]): list of descriptorsets
        """
        super().__init__(descsets)

    def getPrefix(self) -> str:
        return "Descriptor_Custom"

    def __call__(self, index, dtype=np.float32) -> pd.DataFrame:
        df = pd.DataFrame(index=index)
        for descset in self.descSets:
            values = descset(index)
            if descset.is_fp:
                values.add_prefix(f"{descset.fingerprint_type}_")
            values = values.astype(dtype)
            values = self.treatInfs(values)
            values = values.add_prefix(f"{self.getPrefix()}_{descset}_")
            df = df.merge(values, left_index=True, right_index=True)

        return df
