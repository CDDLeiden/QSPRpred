"""This module is used for calculating molecular descriptors using descriptorsets."""
import json
from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from qsprpred.data.utils.descriptorsets import DescriptorSet, get_descriptor
from rdkit.Chem.rdchem import Mol


class Calculator(ABC):
    """Calculator for molecule properties."""

    @abstractmethod
    def __call__(self, mols: List[Mol]) -> pd.DataFrame:
        """Calculate all specified descriptors for a list of rdkit mols.

        Args:
            df: dataframe containing a column with SMILES


        Returns:
            df: original dataframe with added descriptor columns
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


class descriptorsCalculator(Calculator):
    """Calculator for molecule properties."""

    def __init__(self, descsets: List[DescriptorSet]) -> None:
        """Set the descriptorsets to be calculated with this calculator."""
        self.descsets = descsets

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
            descset = get_descriptor(key, *value["settings"][0], **value["settings"][1])
            if descset.is_fp:
                descset.keepindices = value["keepindices"]
            else:
                descset.descriptors = value["descriptors"]
            descsets.append(descset)
        return descriptorsCalculator(descsets)

    def __call__(self, mols: List[Mol]) -> pd.DataFrame:
        """Calculate descriptors for list of mols.
        
        Args:
            mols: list of rdkit mols
        """
        df = pd.DataFrame()
        for descset in self.descsets:
            descs = []
            for mol in mols:
                try:
                    descs.append(descset(mol))
                except: pass
            values = pd.concat(descs)
            df = pd.concat([df, values.add_prefix(f"{descset}_")], axis=1)
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
        with open(fname, "w") as outfile:
            json.dump(descset_dict, outfile)
