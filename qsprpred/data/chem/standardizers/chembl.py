from chembl_structure_pipeline import standardizer as chembl_stand
from rdkit import Chem

from qsprpred.logs import logger
from .base import ChemStandardizer


def chembl_smi_standardizer(
        smi: str, isomeric_smiles: bool = True, sanitize: bool = True
) -> str | None:
    """Standardize SMILES using ChEMBL standardizer.

    Args:
        smi (str): SMILES string to be standardized.
        isomeric_smiles (bool): return the isomeric smiles. Defaults to True.
        sanitize (bool): 
            applies sanitization using the ChEMBL standardizer. Defaults to True.

    Returns:
        (str): standardized SMILES string or `None` if standardization failed.
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            raise ValueError(f"Failed to parse SMILES: {smi}")
        standard_mol = chembl_stand.standardize_mol(mol, sanitize=sanitize)
        standard_smiles = Chem.MolToSmiles(
            standard_mol,
            kekuleSmiles=False,
            canonical=True,
            isomericSmiles=isomeric_smiles,
        )
        return standard_smiles
    except Exception as exp:  # E722
        logger.warning(f"Could not standardize SMILES: {smi} due to: {exp}.")
        return None


class ChemblStandardizer(ChemStandardizer):
    """Standardizer using the ChEMBL standardizer.

    Attributes:
        isomericSmiles (bool): return the isomeric smiles.
        sanitize (bool): sanitize SMILES before standardization.
    """

    def __init__(
            self,
            isomeric_smiles: bool = True,
            sanitize: bool = True,
    ):
        """Initialize the ChEMBL standardizer.

        Args:
            isomeric_smiles (bool): return the isomeric smiles. Defaults to True.
            sanitize (bool): sanitize SMILES before standardization. Defaults to True.
        """
        self.isomericSmiles = isomeric_smiles
        self.sanitize = sanitize

    def convertSMILES(self, smiles: str) -> str | None:
        """Standardize SMILES using the ChEMBL standardizer.
        
        Args:
            smiles (str): SMILES to be standardized
        
        Returns:
            (str): standardized SMILES string or `None` if standardization failed.
        """
        return chembl_smi_standardizer(
            smiles, isomeric_smiles=self.isomericSmiles, sanitize=self.sanitize
        )

    @property
    def settings(self) -> dict:
        return {
            "isomeric_smiles": self.isomericSmiles,
            "sanitize": self.sanitize,
        }

    def getID(self) -> str:
        """Return the unique identifier of the standardizer.

        In this case, the identifier starts with "ChEMBLStandardizer" followed by
        the settings of the standardizer concatenated with "~".
        
        Returns:
            (str): unique identifier of the standardizer
        """
        return ("ChEMBLStandardizer"
                f"~isomeric_smiles={self.isomericSmiles}"
                f"~sanitize={self.sanitize}")

    @classmethod
    def fromSettings(cls, settings: dict) -> "ChemblStandardizer":
        """Create a standardizer from settings.	
        
        Args:
            settings (dict): Settings of the standardizer
            
        Returns:
            (ChemblStandardizer): The standardizer created from settings
        """
        return cls(
            isomeric_smiles=settings["isomeric_smiles"],
            sanitize=settings["sanitize"],
        )
