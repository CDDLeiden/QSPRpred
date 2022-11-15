"""This module is used for calculating molecular descriptors using descriptorsets."""
import json
from abc import ABC, abstractmethod
from typing import List

import numpy as np
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
        valid_mols = [mol for mol in mols if not mol is None]
        df_valid = pd.DataFrame()
        for descset in self.descsets:
            values = pd.concat([descset(mol) for mol in valid_mols], ignore_index=True)
            df_valid = pd.concat([df_valid, values.add_prefix(f"{descset}_")], axis=1)

        # Add invalid mols back as rows of zero
        df = pd.DataFrame(np.zeros((len(mols), df_valid.shape[1])), columns=df_valid.columns)
        df.iloc[pd.notnull(mols),:] = df_valid

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
        with open(fname, "w") as outfile:
            json.dump(descset_dict, outfile)
    
    def get_len(self):
        """Return number of descriptors calculated by all descriptorsets."""
        length = 0
        for descset in self.descsets:
            length += descset.get_len()
        return length
