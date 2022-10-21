"""
descriptors.py

To add a new descriptor or fingerprint calculator:
* Add a descriptor subclass for your descriptor calculator
* Add a function to retrieve your descriptor by name to the descriptor retriever class

"""

from abc import abstractmethod, ABC
import numpy as np

class Descriptor(ABC):
    """
    Abstract base class for descriptors.

    """

    @abstractmethod
    def __call__(self, mol):
        """
        Calculate the descriptor for a molecule.

        Args:
            mol: smiles or rdkit molecule
            *args: optional arguments
            **kwargs: optional keyword arguments

        Returns:
            a `list` of descriptor values
        """
        pass

    @abstractmethod
    def __str__(self):
        pass

class MorganFP(Descriptor):

    def __init__(self, *args, **kwargs):
        """
        Initialize the descriptor with the same arguments as you would pass to `GetMorganFingerprintAsBitVect` function of RDKit.

        Args:
            *args: `GetMorganFingerprintAsBitVect` arguments
            **kwargs: `GetMorganFingerprintAsBitVect` keyword arguments
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        self._convertMol = Chem.MolFromSmiles
        self._convertFP = DataStructs.ConvertToNumpyArray
        self._morgan = AllChem.GetMorganFingerprintAsBitVect
        self._args = args
        self._kwargs = kwargs
        self.keepindices = None

    def __call__(self, mol):
        mol = self._convertMol(mol) if isinstance(mol, str) else mol
        fp = self._morgan(mol, *self._args, **self._kwargs)
        self.ln = len(fp)
        ret = np.zeros(self.ln)
        self._convertFP(fp, ret)
        if self.keepindices:
            ret = ret[self.keepindices]
        return ret

    def __str__(self):
        return "MorganFP"

class _DescriptorRetriever:
    """
    Based on recipe 8.21 of the book "Python Cookbook".
    To support a new type of descriptor, just add a function "get_descname(self, *args, **kwargs)".
    """

    def get_descriptor(self, desc_type, *args, **kwargs):
        method_name = 'get_' + desc_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f'{desc_type} is not a supported fingerprint type.')
        return method(*args, **kwargs)

    def get_MorganFP(self, *args, **kwargs):
        return MorganFP(*args, **kwargs)

def get_descriptor(desc_type: str, *args, **kwargs):
    return _DescriptorRetriever().get_descriptor(desc_type, *args, **kwargs)