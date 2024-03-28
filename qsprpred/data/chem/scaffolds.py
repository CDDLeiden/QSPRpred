from abc import ABC, abstractmethod

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol, ReplaceSubstructs
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


class BemisMurckoRDKit(Scaffold):
    """Class for calculating Murcko scaffolds of a given molecule
    using the default implementation in RDKit. If you want, an implementation
    closer to the original paper, see the `BemisMurcko` class.

    """

    def __call__(self, mols, props, *args, **kwargs):
        res = []
        for mol in mols:
            mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
            scaff = MurckoScaffold.GetScaffoldForMol(mol)
            res.append(Chem.MolToSmiles(scaff))
        return pd.Series(res, index=props[self.idProp])

    def __str__(self):
        return "BemisMurckoRDKit"


class BemisMurcko(Scaffold):
    """Extension of rdkit's BM-like scaffold to make it more true to the paper.
    In BM's paper, exo bonds on linkers or on rings get cutoff but two
    electrons remain.

    In the rdkit implementation, both atoms in the exo bond get included.
    This means for BM C1CCC1=N and C1CCC1=O are the same, for rdkit they are
    different.

    When flattening the BM scaffold using MakeScaffoldGeneric() this leads to
    distinct scaffolds, as C1CCC1=O is flattened to C1CCC1C and not C1CCC1.

    In this approach, the two electrons are represented as SMILES "=*". This
    is to make sure the automatic oxidation state assignment of sulfur does
    not flatten C1CS1(=*)(=*) into C1CS1 when explicit hydrogen count is
    provided.

    Ref.:

    Bemis, G. W., & Murcko, M. A. (1996). "The properties of known drugs. 1.
    Molecular frameworks." Journal of medicinal chemistry, 39(15), 2887-2893.

    Related RDKit issue: https://github.com/rdkit/rdkit/discussions/6844

    Credit: Original code provided by Wim Dehaen (@dehaenw)
    """

    def __init__(
        self,
        real_bemismurcko: bool = True,
        use_csk: bool = False,
        id_prop: bool | None = None,
    ):
        """
        Initialize the scaffold generator.

        Args:
            real_bemismurcko (bool): Use guidelines from Bemis murcko paper.
                otherwise, use native rdkit implementation.
            use_csk (bool): Make scaffold generic (convert all bonds to single
                and all atoms to carbon). If real_bemismurcko is on, also
                remove all flattened exo bonds.
        """
        super().__init__(id_prop=id_prop)
        self.realBemisMurcko = real_bemismurcko
        self.useCSK = use_csk

    @staticmethod
    def findTerminalAtoms(mol):
        res = []
        for a in mol.GetAtoms():
            if len(a.GetBonds()) == 1:
                res.append(a)
        return res

    def __call__(self, mols, props, *args, **kwargs):
        res = []
        for mol in mols:
            mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
            Chem.RemoveStereochemistry(mol)  # important for canonization !
            scaff = MurckoScaffold.GetScaffoldForMol(mol)

            if self.realBemisMurcko:
                scaff = ReplaceSubstructs(
                    scaff,
                    Chem.MolFromSmarts("[$([D1]=[*])]"),
                    Chem.MolFromSmarts("[*]"),
                    replaceAll=True,
                )[0]

            if self.useCSK:
                scaff = MurckoScaffold.MakeScaffoldGeneric(scaff)
                if self.realBemisMurcko:
                    scaff = MurckoScaffold.GetScaffoldForMol(scaff)
            Chem.SanitizeMol(scaff)
            res.append(Chem.MolToSmiles(scaff))
        return pd.Series(res, index=props[self.idProp])

    def __str__(self):
        return "Bemis-Murcko"
