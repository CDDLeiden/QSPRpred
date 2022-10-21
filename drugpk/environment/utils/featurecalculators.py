"""
descriptors

Created by: Martin Sicho
On: 05.10.22, 10:13
"""
from abc import abstractmethod, ABC

from typing import List
import pandas as pd
import json
from drugpk.environment.utils import descriptors
from rdkit.Chem.rdchem import Mol

class FeaturesCalculator(ABC):

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all specified descriptors for a pandas series of SMILES

        Args:
            df: dataframe containing a column with SMILES


        Returns:
            df: original dataframe with added descriptor columns
        """
        pass

    @classmethod
    @abstractmethod
    def fromFile(cls, fname: str):
        """
        Construct featurescalculator from json file

        Args:
            fname: file name to save featurecalculator to
        """
        pass

    @abstractmethod
    def toFile(self, fname: str) -> None:
        """
        Save featurescalculator to json file

        Args:
            fname: file name to save featurecalculator to
        """
        pass


class FeaturesCalculator():
    def __init__(self, descriptors: List[descriptors.Descriptor]) -> None:
        self.descriptors = descriptors

    @classmethod
    def fromFile(cls, fname: str):
        with open(fname, 'r') as infile:
            descriptor_dict = json.load(infile)

        descriptors = []
        for name, settings in descriptor_dict.items():
            descriptor = descriptors.get_descriptor(name, *settings['_args'], **settings['_kwargs'])
            if 'keepindices' in settings.keys():
                descriptor.keepindices = settings['keepindices']
            descriptors.append(descriptor)
        return FeaturesCalculator(descriptors)
    
    def __call__(self, mols: List[Mol]) -> pd.DataFrame:
        for descriptor in self.descriptors:
            desc_calc = [descriptor(mol) for mol in mols] 
        return desc_calc
    
    def toFile(self, fname):
        descriptor_dict = {}
        for descriptor in self.descriptors:
            descriptor_dict[descriptor.__str__()] = {key:descriptor.__dict__[key] for key in ['_args', '_kwargs', 'keepindices']}
        with open(fname, 'w') as outfile:
            json.dump(descriptor_dict, outfile)

