from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol

from qsprpred.data.chem.standardizers import ChemStandardizer
from qsprpred.data.processing.mol_processor import MolProcessorWithID
from qsprpred.data.storage.interfaces.stored_mol import StoredMol


class CheckSmilesValid(MolProcessorWithID):
    def __call__(
            self, mols: list[StoredMol | str | Mol], props: dict | None = None, *args,
            **kwargs
    ) -> Any:
        throw = kwargs.get("throw", False)
        ret = []
        ret_ids = []
        for idx, mol in enumerate(mols):
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
                mol_id = props[self.idProp][idx]
            elif isinstance(mol, StoredMol):
                mol = mol.as_rd_mol()
                mol_id = mol.id
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
        return True


class ValidationStandardizer(ChemStandardizer):

    def __init__(self):
        super().__init__()
        self.checker = CheckSmilesValid(id_prop="index")

    def convert_smiles(self, smiles):
        checks = self.checker([smiles], {"index": [0]})
        if not checks[0]:
            raise ValueError(f"Invalid SMILES found: {smiles}")
        return smiles

    @property
    def settings(self):
        return {}

    def get_id(self):
        return "ValidationStandardizer"

    @classmethod
    def from_settings(cls, settings: dict):
        return cls()
