from chembl_structure_pipeline import standardizer as chembl_stand
from rdkit import Chem

from qsprpred.logs import logger
from .base import ChemStandardizer


def chembl_smi_standardizer(
        smi: str, isomeric_smiles: bool = True, sanitize: bool = True
) -> str:
    """Standardize SMILES using ChEMBL standardizer.

    Args:
        smi: SMILES string to be standardized.
        isomeric_smiles: return the isomeric smiles. Defaults to True.
        sanitize: applies sanitization using the ChEMBL standardizer. Defaults to True.

    Returns:
        smiles (str): standardized SMILES string or `None` if standardization failed.
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
        isomericSmiles: return the isomeric smiles. Defaults to True.
        sanitize: sanitize SMILES before standardization. Defaults to True.
    """

    def __init__(
            self,
            isomeric_smiles: bool = True,
            sanitize: bool = True,
    ):
        """Initialize the ChEMBL standardizer.

        Args:
            isomeric_smiles: return the isomeric smiles. Defaults to True.
            sanitize: sanitize SMILES before standardization. Defaults to True.
        """
        self.isomericSmiles = isomeric_smiles
        self.sanitize = sanitize

    def convert_smiles(self, smiles):
        return chembl_smi_standardizer(
            smiles, isomeric_smiles=self.isomericSmiles, sanitize=self.sanitize
        ), smiles

    @property
    def settings(self):
        return {
            "isomeric_smiles": self.isomericSmiles,
            "sanitize": self.sanitize,
        }

    def get_id(self):
        return ("ChEMBLStandardizer"
                "~isomeric_smiles={self.isomeric_smiles}"
                "~sanitize={self.sanitize}")

    @classmethod
    def from_settings(cls, settings: dict):
        return cls(
            isomeric_smiles=settings["isomeric_smiles"],
            sanitize=settings["sanitize"],
        )
