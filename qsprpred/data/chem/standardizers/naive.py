from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from .base import ChemStandardizer


def standardize_mol(mol) -> str | None:
    """Standardizes SMILES and removes fragments

    Standardizes SMILES using RDKit MolStandardize to disconnect metals,
    normalize, remove salts (largest fragment), and uncharge. Followed by a second
    round of disconnecting metals and normalizing. Finally, the SMILES is canonicalized.

    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object

    Returns:
        (str | None):
            Standardized SMILES or None if SMILES could not be standardized or
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
    except Exception:
        # TODO: log error
        print("Parsing Error:", Chem.MolToSmiles(mol))

    return None


class NaiveStandardizer(ChemStandardizer):
    """Naive standardizer

    Briefly, the standardization process involves disconnecting metals, normalizing,
    removing salts (largest fragment) and charges. See
    `qsprpred.data.chem.standardizers.naive.standardize_mol` for more details.
    """
    def convertSMILES(self, smiles: str) -> str | None:
        """Standardize SMILES using `standardize_mol`.

        Args:
            smiles (str): SMILES to be standardized

        Returns:
            str | None: standardized SMILES or `None` if SMILES could not be standardized
        """
        mol = Chem.MolFromSmiles(smiles)
        return standardize_mol(mol)

    @property
    def settings(self) -> dict:
        """Settings of the standardizer. They are empty in this case."""
        return {}

    def getID(self) -> str:
        """Return the unique identifier of the standardizer, which
        in this case is "NaiveStandardizer" without any settings.
        """
        return "NaiveStandardizer"

    @classmethod
    def fromSettings(cls, settings: dict) -> "NaiveStandardizer":
        """Create a naive standardizer from settings.
        In this case, the settings are ignored.

        Args:
            settings (dict): settings of the standardizer

        Returns:
            NaiveStandardizer: a naive standardizer
        """
        return cls()
