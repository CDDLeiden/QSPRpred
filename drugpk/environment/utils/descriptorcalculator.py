from abc import abstractmethod, ABC

from typing import List
import pandas as pd
import json
from drugpk.environment.utils.descriptors import DescriptorSet, get_descriptor
from rdkit.Chem.rdchem import Mol

class Calculator(ABC):

    @abstractmethod
    def __call__(self, mols: List[Mol]) -> pd.DataFrame:
        """
        Calculate all specified descriptors for a list of rdkit mols

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
        Construct calculator from json file

        Args:
            fname: file name to save featurecalculator to
        """
        pass

    @abstractmethod
    def toFile(self, fname: str) -> None:
        """
        Save calculator with settings to json file

        Args:
            fname: file name to save featurecalculator to
        """
        pass


class descriptorsCalculator(Calculator):
    def __init__(self, descriptors: List[DescriptorSet]) -> None:
        self.descriptors = descriptors

    @classmethod
    def fromFile(cls, fname: str):
        with open(fname, 'r') as infile:
            descriptor_dict = json.load(infile)

        descriptors = []
        for name, settings in descriptor_dict.items():
            descriptor = get_descriptor(name, *settings['_args'], **settings['_kwargs'])
            if 'keepindices' in settings.keys():
                descriptor.keepindices = settings['keepindices']
            descriptors.append(descriptor)
        return descriptorsCalculator(descriptors)
    
    def __call__(self, mols: List[Mol]) -> pd.DataFrame:
        df = pd.DataFrame()
        for descriptor in self.descriptors:
            values = pd.concat([descriptor(mol) for mol in mols])
            df = pd.concat([df, values.add_prefix(f"{descriptor}_")], axis=1)
        return df

    def toFile(self, fname: str) -> None:
        descriptor_dict = {}
        for descriptor in self.descriptors:
            save_keys = [key for key in ['_args', '_kwargs', 'keepindices', '_descriptors'] if key in descriptor.__dict__.keys()]
            descriptor_dict[descriptor.__str__()] = {key:descriptor.__dict__[key] for key in save_keys}
        with open(fname, 'w') as outfile:
            json.dump(descriptor_dict, outfile)

