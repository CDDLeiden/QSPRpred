from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol

from qsprpred.data.chem.standardizers import ChemStandardizer
from qsprpred.data.processing.mol_processor import MolProcessorWithID
from qsprpred.data.storage.interfaces.stored_mol import StoredMol


class CheckSmilesValid(MolProcessorWithID):
    """Processor to check the validity of the SMILES."""

    def __call__(
            self, mols: list[StoredMol | str | Mol], props: dict | None = None, *args,
            **kwargs
    ) -> Any:
        """Check the validity of the SMILES.
        
        Args:
            mols (list[StoredMol | str | Mol]): List of molecules to be checked.
            props (dict, optional): Dictionary of properties. Defaults to None.
            args: Additional arguments (not used).
            kwargs: 
                Additional keyword arguments (used to set the throw flag, if 
                kwargs["throw"] is True,
            
        Returns:
            Any: A pandas Series where the index is the molecule ID and the value is 
                True if the molecule is valid, False otherwise.
                
        Raises:
            ValueError: If the molecule is invalid and the throw flag is set to True
        """
        throw = kwargs.get("throw", False)
        ret = []
        ret_ids = []
        for idx, mol in enumerate(mols):
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
                mol_id = props[self.idProp][idx]
            elif isinstance(mol, StoredMol):
                mol_id = mol.id
                mol = mol.as_rd_mol()
            else:
                mol = mol
                mol_id = props[self.idProp][idx]
            is_valid = True
            exception = None
            if not mol:
                is_valid = False
                exception = ValueError(f"Empty molecule: {mol}")
            try:
                Chem.SanitizeMol(mol)
            except Exception as exp:
                is_valid = False
                exception = exp
            if exception and throw:
                raise exception
            else:
                ret.append(is_valid)
                ret_ids.append(mol_id)
        ret = pd.Series(ret, index=ret_ids)
        return ret

    @property
    def supportsParallel(self) -> bool:
        """Return True if the processor supports parallel processing."""
        return True


class ValidationStandardizer(ChemStandardizer):
    """Standardizer that checks the validity of the SMILES by
    attempting to sanitize the molecule using RDKit.
    
    Attributes:
        checker (CheckSmilesValid): Processor to check the validity of the SMILES
    """

    def __init__(self):
        """Initialize the standardizer.
        
        Raises:
            ValueError: If the SMILES is invalid
        """
        super().__init__()
        self.checker = CheckSmilesValid(id_prop="index")

    def convertSMILES(self, smiles: str) -> str | None:
        """Check the validity of the SMILES.
        
        Args:
            smiles (str): SMILES to be checked
        
        Returns:
            str | None:
                the standardized SMILES
        """
        checks = self.checker([smiles], {"index": [0]})
        if not checks[0]:
            raise ValueError(f"Invalid SMILES found: {smiles}")
        return smiles

    @property
    def settings(self):
        """Settings of the standardizer.
        Empty in this case since there is nothing to set except the default settings.
        """
        return {}

    def getID(self):
        """Return the unique identifier of the standardizer. In this case, it is
        just "ValidationStandardizer". There are no settings to consider.
        """
        return "ValidationStandardizer"

    @classmethod
    def fromSettings(cls, settings: dict) -> "ValidationStandardizer":
        """Create a standardizer from settings. In this case, the settings are ignored.
        
        Args:
            settings (dict): Settings of the standardizer
            
        Returns:
            ValidationStandardizer: The standardizer created from settings
        """
        return cls()
