from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol

from qsprpred.data.chem.standardizers import ChemStandardizer
from qsprpred.data.processing.mol_processor import MolProcessorWithID
from qsprpred.data.storage.interfaces.stored_mol import StoredMol


class CheckSmilesValid(MolProcessorWithID):
    """Processor to check the validity of the SMILES.
    
    Attributes:
        idProp (str): Property name of the molecule ID.
    """
    
    def __call__(
            self, mols: list[StoredMol | str | Mol], props: dict | None = None, *args,
            **kwargs
    ) -> Any:
        """Check the validity of the SMILES.
        
        Args:
            mols (list[StoredMol | str | Mol]): List of molecules to be checked.
            props (dict, optional): Dictionary of properties. Defaults to None.
            args: Additional arguments (not used).
            kwargs: Additional keyword arguments 
                (used to set the throw flag, if kwargs["throw"] is True,
            
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
    """Standardizer that checks the validity of the SMILES."""

    def __init__(self):
        """Initialize the standardizer.
        
        Raises:
            ValueError: If the SMILES is invalid
        """
        super().__init__()
        self.checker = CheckSmilesValid(id_prop="index")

    def convert_smiles(self, smiles: str) -> tuple[str | None, str]:
        """Check the validity of the SMILES.
        
        Args:
            smiles (str): SMILES to be checked
        
        Returns:
            (tuple[str | None, str]): a tuple where the first element is the 
                standardized SMILES and the second element is the original SMILES
        """
        checks = self.checker([smiles], {"index": [0]})
        if not checks[0]:
            raise ValueError(f"Invalid SMILES found: {smiles}")
        return smiles, smiles

    @property
    def settings(self):
        """Settings of the standardizer."""
        return {}

    def get_id(self):
        """Return the unique identifier of the standardizer."""
        return "ValidationStandardizer"

    @classmethod
    def from_settings(cls, settings: dict) -> "ValidationStandardizer":
        """Create a standardizer from settings.
        
        Args:
            settings (dict): Settings of the standardizer
            
        Returns:
            ValidationStandardizer: The standardizer created from settings
        """
        return cls()
