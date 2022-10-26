"""
descriptors.py

To add a new descriptor or fingerprint calculator:
* Add a descriptor subclass for your descriptor calculator
* Add a function to retrieve your descriptor by name to the descriptor retriever class

"""

from abc import abstractmethod, ABC
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from drugpk.environment.utils.properties import Property
import mordred

class DescriptorSet(ABC):
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

class MorganFP(DescriptorSet):

    def __init__(self, *args, **kwargs):
        """
        Initialize the descriptor with the same arguments as you would pass to `GetMorganFingerprintAsBitVect` function of RDKit.

        Args:
            *args: `GetMorganFingerprintAsBitVect` arguments
            **kwargs: `GetMorganFingerprintAsBitVect` keyword arguments
        """
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
        ret = pd.DataFrame(ret.reshape(1,-1), columns=[f"MorganFP_{idx}" for idx in range(ret.shape[0])])
        if self.keepindices:
            ret = ret[self.keepindices]
        return ret

    def __str__(self):
        return "MorganFP"

class Mordred(DescriptorSet):

    def __init__(self, *args, **kwargs):
        """
        Initialize the descriptor with the same arguments as you would pass to `Calculator` function of Mordred.
        If no mordred descriptor object passed, the all descriptors will be calculated

        Args:
            *args: `Calculator` arguments
            **kwargs: `Calculator` keyword arguments
        """

        self._args = args
        self._kwargs = kwargs
        self._process_args()

        self._is_fp = False
        
        self._mordred = None
        self._descriptors = self._args

    def __call__(self, mol):
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        return self._mordred.pandas([mol])

    @property
    def is_fp(self):
        return self._is_fp()
    
    @property
    def settings(self):
        return self._args, self._kwargs

    @property
    def descriptors(self):
        return self.descriptors

    @descriptors.setter
    def descriptors(self, names):
            calc = mordred.Calculator(mordred.descriptors)
            self._mordred = mordred.Calculator([d for d in calc.descriptors if str(d) in names], **self._kwargs)
            self.descriptors = names

    def _process_args(self, descs=None, version=None, ignore_3D=False, config=None):
        self._args = mordred.Calculator(descs).descriptors if descs else mordred.Calculator(mordred.descriptors).descriptors
        self._kwargs = {"version": version, "ignore_3D":ignore_3D, "config":config}

    def __str__(self):
        return "Mordred"


class physchem(Descriptor):
    """
        Initialize the descriptor with prop_list 
        If no mordred descriptor object passed, the all descriptors will be calculated

        Args:
            *args: `Calculator` arguments
            **kwargs: `Calculator` keyword arguments
    """
    def __init__(self, *args, **kwargs):
        self._convertMol = Chem.MolFromSmiles
        self._args = args
        self._kwargs = kwargs

    def __call__(self, mol):
        mol = self._convertMol(mol) if isinstance(mol, str) else mol
        physchem = self.calc_physchem([mol], *self._args, **self._kwargs)
        return physchem
    
    def __str__(self):
        return "physchem"

    @staticmethod
    def calc_physchem(mols, prop_list):
        '''
            Calculated physicochemical properties for a list of rdkit molecules using the Property class
            arguments:
                mols: list of rdkit molecules
                prop_list: list of properties to calculate
        '''
        fps = np.zeros((len(mols), len(prop_list)))
        props = Property()
        for i, prop in enumerate(prop_list):
            props.prop = prop
            fps[:, i] = props(mols)
        return pd.DataFrame(data=fps, columns=prop_list)


class DrugExDescriptors(Descriptor):
    """
        Pysciochemical properties originally used in DrugEx for QSAR modelling
    """
    def __init__(self, *args, **kwargs):
        self._convertMol = Chem.MolFromSmiles
        self.prop_list = ['MW', 'logP', 'HBA', 'HBD', 'Rotable', 'Amide', 'Bridge', 'Hetero', 'Heavy', 'Spiro',
                          'FCSP3', 'Ring', 'Aliphatic', 'Aromatic', 'Saturated', 'HeteroR', 'TPSA', 'Valence', 'MR']
        self._args = args
        self._kwargs = kwargs

    def __call__(self, mol):
        mol = self._convertMol(mol) if isinstance(mol, str) else mol
        physchem_props = physchem.calc_physchem([mol], prop_list=self.prop_list)
        return physchem_props
    
    def __str__(self):
        return "DrugExPhyschem"


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

    def get_physchem(self, *args, **kwargs):
        return physchem(*args, **kwargs)
    
    def get_DrugExPhyschem(self, *args, **kwargs):
        return DrugExDescriptors(*args, **kwargs)
    
    def get_Mordred(self, *args, **kwargs):
        return Mordred(*args, **kwargs)

def get_descriptor(desc_type: str, *args, **kwargs):
    return _DescriptorRetriever().get_descriptor(desc_type, *args, **kwargs)


