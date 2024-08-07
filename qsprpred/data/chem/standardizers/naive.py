from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from .base import ChemStandardizer


def standardize_mol(mol):
    """
    Standardizes SMILES and removes fragments
    Arguments:
        mols (lst)                : list of rdkit-molecules
    Returns:
        smiles (set)              : set of SMILES
    """

    charger = rdMolStandardize.Uncharger()
    chooser = rdMolStandardize.LargestFragmentChooser()
    disconnector = rdMolStandardize.MetalDisconnector()
    normalizer = rdMolStandardize.Normalizer()
    carbon = Chem.MolFromSmarts("[#6]")
    salts = Chem.MolFromSmarts("[Na,Zn]")
    try:
        mol = disconnector.Disconnect(mol)
        mol = normalizer.normalize(mol)
        mol = chooser.choose(mol)
        mol = charger.uncharge(mol)
        mol = disconnector.Disconnect(mol)
        mol = normalizer.normalize(mol)
        smileR = Chem.MolToSmiles(mol, 0)
        # remove SMILES that do not contain carbon
        if len(mol.GetSubstructMatches(carbon)) == 0:
            return None
        # remove SMILES that still contain salts
        if len(mol.GetSubstructMatches(salts)) > 0:
            return None
        return Chem.CanonSmiles(smileR)
    except:
        print("Parsing Error:", Chem.MolToSmiles(mol))

    return None


class NaiveStandardizer(ChemStandardizer):
    def convert_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return standardize_mol(mol)

    @property
    def settings(self):
        return {}

    def get_id(self):
        return "NaiveStandardizer"

    @classmethod
    def from_settings(cls, settings: dict):
        return cls()
