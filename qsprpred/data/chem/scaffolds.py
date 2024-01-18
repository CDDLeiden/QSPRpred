from abc import ABC, abstractmethod

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold

from qsprpred.data.processing.mol_processor import MolProcessorWithID


class Scaffold(MolProcessorWithID, ABC):
    """
    Abstract base class for calculating molecular scaffolds of different kinds.
    """

    @abstractmethod
    def __call__(self, mols: list[str | Mol], props, *args, **kwargs):
        """
        Calculate the scaffold for a molecule.

        Args:
            mol (str | Mol): SMILES or RDKit molecule to calculate the scaffold for.

        Returns:
            smiles of the scaffold
        """

    @abstractmethod
    def __str__(self):
        pass

    def supportsParallel(self) -> bool:
        return True


class Murcko(Scaffold):
    """Class for calculating Murcko scaffolds of a given molecule."""

    def __call__(self, mols, props, *args, **kwargs):
        """
        Calculate the Murcko scaffold for a molecule.

        Args:
            mol: SMILES as `str` or an instance of `Mol`

        Returns:
            SMILES of the Murcko scaffold as `str`
        """
        res = []
        for mol in mols:
            mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
            scaff = MurckoScaffold.GetScaffoldForMol(mol)
            res.append(Chem.MolToSmiles(scaff))
        return pd.Series(res, index=props[self.idProp])

    def __str__(self):
        return "Murcko"


class BemisMurcko(Scaffold):
    """
    Reimplementation of Bemis-Murcko scaffolds based on a function described in the
    discussion here: https://sourceforge.net/p/rdkit/mailman/message/37269507/

    This implementation allows more flexibility in terms of what substructures should be
    kept, converted, and how.

    Credit: Francois Berenger
    Ref.: Bemis, G. W., & Murcko, M. A. (1996). "The properties of known drugs. 1.
    Molecular frameworks." Journal of medicinal chemistry, 39(15), 2887-2893.

    """

    def __init__(
        self,
        convert_hetero=True,
        force_single_bonds=True,
        remove_terminal_atoms=True,
        id_prop=None,
    ):
        """
        Initialize the scaffold generator.

        Args:
            convert_hetero (bool):
                Convert hetero atoms to carbons.
            force_single_bonds (bool):
                Convert all scaffold's bonds to single ones.
            remove_terminal_atoms (bool):
                Remove all terminal atoms, keep only
                ring linkers.
            id_prop (str):
                Name of the property that contains the molecule's unique identifier.
        """
        super().__init__(id_prop=id_prop)
        self.convertHetero = convert_hetero
        self.forceSingleBonds = force_single_bonds
        self.removeTerminalAtoms = remove_terminal_atoms

    @staticmethod
    def findTerminalAtoms(mol):
        res = []

        for a in mol.GetAtoms():
            if len(a.GetBonds()) == 1:
                res.append(a)
        return res

    def __call__(self, mols, props, *args, **kwargs):
        """
        Calculate the Bemis-Murcko scaffold for a molecule.

        Args:
            mol: SMILES as `str` or an instance of `Mol`

        Returns:
            SMILES of the Bemis-Murcko scaffold as `str`
        """
        res = []
        for mol in mols:
            mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
            only_HA = Chem.rdmolops.RemoveHs(mol)
            rw_mol = Chem.RWMol(only_HA)

            # switch all HA to Carbon
            if self.convertHetero:
                for i in range(rw_mol.GetNumAtoms()):
                    rw_mol.ReplaceAtom(i, Chem.Atom(6))

            # switch all non single bonds to single
            if self.forceSingleBonds:
                non_single_bonds = []
                for b in rw_mol.GetBonds():
                    if b.GetBondType() != Chem.BondType.SINGLE:
                        non_single_bonds.append(b)
                for b in non_single_bonds:
                    j = b.GetBeginAtomIdx()
                    k = b.GetEndAtomIdx()
                    rw_mol.RemoveBond(j, k)
                    rw_mol.AddBond(j, k, Chem.BondType.SINGLE)

            # as long as there are terminal atoms, remove them
            if self.removeTerminalAtoms:
                terminal_atoms = self.findTerminalAtoms(rw_mol)
                while terminal_atoms:
                    for a in terminal_atoms:
                        for b in a.GetBonds():
                            rw_mol.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                        rw_mol.RemoveAtom(a.GetIdx())
                    terminal_atoms = self.findTerminalAtoms(rw_mol)
                res.append(Chem.MolToSmiles(rw_mol.GetMol()))
        return pd.Series(res, index=props[self.idProp])

    def __str__(self):
        return "Bemis-Murcko"
