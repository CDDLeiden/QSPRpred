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

    @abstractmethod
    def __str__(self):
        pass


class Murcko(Scaffold):
    """Class for calculating Murcko scaffolds of a given molecule."""
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
        self, convert_hetero=True, force_single_bonds=True, remove_terminal_atoms=True
    ):
        """
        Initialize the scaffold generator.

        Args:
            convert_hetero (bool): Convert hetero atoms to carbons.
            force_single_bonds (bool): Convert all scaffold's bonds to single ones.
            remove_terminal_atoms (bool): Remove all terminal atoms, keep only
                ring linkers.
        """
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

    def __call__(self, mol):
        """
        Calculate the Bemis-Murcko scaffold for a molecule.

        Args:
            mol: SMILES as `str` or an instance of `Mol`

        Returns:
            SMILES of the Bemis-Murcko scaffold as `str`
        """

        from rdkit import Chem

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
            return Chem.MolToSmiles(rw_mol.GetMol())

    def __str__(self):
        return "Bemis-Murcko"
