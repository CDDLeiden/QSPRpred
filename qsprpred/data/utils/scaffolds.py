"""
scaffold

Created by: Martin Sicho
On: 05.10.22, 10:07
"""
from abc import ABC, abstractmethod

class Scaffold(ABC):
    """
    Abstract base class for calculating molecular scaffolds of different kinds.
    """

    @abstractmethod
    def __call__(self, mol):
        """
        Calculate the scaffold for a molecule.

        Args:
            mol: smiles or rdkit molecule

        Returns:
            smiles of the scaffold
        """
        pass

    @abstractmethod
    def __str__(self):
        pass

class Murcko(Scaffold):

    def __call__(self, mol):
        """
        Calculate the Murcko scaffold for a molecule.

        Args:
            mol: SMILES as `str` or an instance of `Mol`

        Returns:
            SMILES of the Murcko scaffold as `str`
        """

        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        scaff = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaff)

    def __str__(self):
        return "Murcko"