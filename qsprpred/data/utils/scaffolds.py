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
    Extension of rdkit's BM-like scaffold to make it more true to the paper.
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

    Ref.: Bemis, G. W., & Murcko, M. A. (1996). "The properties of known drugs. 1.
    Molecular frameworks." Journal of medicinal chemistry, 39(15), 2887-2893.

    """
    def __init__(
        self, real_bemismurcko=True, use_csk=False
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
        self.realBemisMurcko = real_bemismurcko
        self.useCSK = use_csk


    def __call__(self, mol):
        """
        Calculate the Bemis-Murcko scaffold for a molecule.

        Args:
            mol: SMILES as `str` or an instance of `Mol`

        Returns:
            SMILES of the Bemis-Murcko scaffold as `str`
        """
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit.Chem.AllChem import ReplaceSubstructs

        mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
        Chem.RemoveStereochemistry(mol) #important for canonization !
        scaff=MurckoScaffold.GetScaffoldForMol(mol)
        
        if self.realBemisMurcko:
            scaff=ReplaceSubstructs(scaff,Chem.MolFromSmarts("[$([D1]=[*])]"),Chem.MolFromSmarts("[*]"),replaceAll=True)[0]
                                            
        if self.useCSK:
            scaff=MurckoScaffold.MakeScaffoldGeneric(scaff)
            if self.realBemisMurcko:
                scaff=MurckoScaffold.GetScaffoldForMol(scaff)
        Chem.SanitizeMol(scaff)
        return Chem.MolToSmiles(scaff)

    def __str__(self):
        return "Bemis-Murcko"
