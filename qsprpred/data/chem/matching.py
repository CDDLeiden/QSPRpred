from typing import Literal, Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol

from qsprpred.data.processing.mol_processor import MolProcessorWithID
from qsprpred.data.storage.interfaces.stored_mol import StoredMol


def match_mol_to_smarts(
        mol: Chem.Mol | str,
        smarts: list[str],
        operator: Literal["or", "and"] = "or",
        use_chirality: bool = False
) -> bool:
    """
    Check if a molecule matches a SMARTS pattern.

    Args:
        mol: Molecule to check.
        smarts: SMARTS patterns to check.
        operator: Whether to use an "or" or "and" operator on patterns. Defaults to "or".
        use_chirality: Whether to use chirality in the search.

    Returns:
        (bool): True if the molecule matches the pattern, False otherwise.
    """
    mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
    ret = False
    for smart in [Chem.MolFromSmarts(smart) for smart in smarts]:
        ret = mol.HasSubstructMatch(smart, useChirality=use_chirality)
        if operator == "or":
            if ret:
                return True
        elif operator == "and":
            if ret:
                ret = True
            else:
                return False
    return ret


class SMARTSMatchProcessor(MolProcessorWithID):
    def __call__(
            self,
            mols: list[str | Mol | StoredMol],
            *args,
            props: dict[str, list] | None = None,
            **kwargs
    ) -> Any:
        if len(mols) == 0:
            return pd.DataFrame(index=pd.Index([], name=self.idProp))
        if isinstance(mols[0], StoredMol):
            ids = [mol.id for mol in mols]
            mols = [mol.as_rd_mol() for mol in mols]
        else:
            mols = [
                mol if isinstance(mol, Mol)
                else Chem.MolFromSmiles(mol)
                for mol in mols
            ]
            ids = props[self.idProp] if props is not None else list(range(len(mols)))
        res = []
        for mol in mols:
            res.append(
                match_mol_to_smarts(mol, *args, **kwargs)
            )
        return pd.DataFrame(
            {"match": res},
            index=pd.Index(ids, name=self.idProp)
        )

    @property
    def supportsParallel(self) -> bool:
        return True
