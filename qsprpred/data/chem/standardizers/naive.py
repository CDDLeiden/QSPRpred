from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from .base import ChemStandardizer


def standardize_mol(mol) -> str | None:
    """Standardizes SMILES and removes fragments
    
    Standardizes SMILES using RDKit MolStandardize to 
        disconnect metals, normalize, remove salts (largest fragment), and uncharge.
        Followed by a second round of disconnecting metals and normalizing.
        Finally, the SMILES is canonicalized.

    Args:
        mol: RDKit molecule object
        
    Returns:
        str: Standardized SMILES or None if SMILES could not be standardized or
            if SMILES does not contain carbon or contains salts after standardization
    """

    disconnector = rdMolStandardize.MetalDisconnector()
    normalizer = rdMolStandardize.Normalizer()
    chooser = rdMolStandardize.LargestFragmentChooser()
    charger = rdMolStandardize.Uncharger()
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
        # TODO: log error
        print("Parsing Error:", Chem.MolToSmiles(mol))

    return None


class NaiveStandardizer(ChemStandardizer):
    """Naive standardizer
    
       Briefly, the standardization process involves disconnecting metals, normalizing,
       removing salts (largest fragment) and charges. See 
       `qsprpred.data.chem.standardizers.naive.standardize_mol` for more details.
    """
    
    def convert_smiles(self, smiles: str) -> tuple[str | None, str]:
        """Standardize SMILES using `standardize_mol`.
        
        Args:
            smiles (str): SMILES to be standardized
        
        Returns:
            (tuple[str | None, str]): a tuple where the first element is the 
                standardized SMILES and the second element is the original SMILES
        """
        mol = Chem.MolFromSmiles(smiles)
        return standardize_mol(mol), smiles

    @property
    def settings(self) -> dict:
        """Settings of the standardizer.
        
        Returns:
            dict: settings of the standardizer
        """
        return {}

    def get_id(self) -> str:
        """Get the ID of the standardizer.
        
        Returns:
            str: ID of the standardizer
        """
        return "NaiveStandardizer"

    @classmethod
    def from_settings(cls, settings: dict) -> "NaiveStandardizer":
        """Create a naive standardizer from settings.
        
        Args:
            settings (dict): settings of the standardizer
            
        Returns:
            NaiveStandardizer: a naive standardizer
        """
        return cls()
