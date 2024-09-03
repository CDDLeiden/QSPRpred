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
    """Check if a molecule matches a SMARTS pattern.

    Args:
        mol (Chem.Mol or str): Molecule to check.
        smarts (list[str]): List of SMARTS patterns to check.
        operator (literal["or", "and"], optional):
            Whether to use an "or" or "and" operator on patterns. Defaults to "or".
            use_chirality: Whether to use chirality in the search.
        use_chirality (bool, optional): Whether to use chirality in the search.

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
    """Processor that checks if a molecule matches a SMARTS pattern.
    
    Attributes:
        supportsParallel (bool): Whether the processor supports parallel processing
    """
    
    def __call__(
            self,
            mols: list[str | Mol | StoredMol],
            *args,
            props: dict[str, list] | None = None,
            **kwargs
    ) -> pd.DataFrame:
        """Check if a molecule matches a SMARTS pattern.	
        
        Args:
            mols (list[str or Mol or StoredMol]): Molecules to check.
            props (dict[str, list], optional): Dictionary of properties.
            args: SMARTS patterns to check.
            kwargs: Additional arguments to pass to `match_mol_to_smarts`.
        
        Returns:
            pd.DataFrame: DataFrame with the results.
        """
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
        """Check if the processor supports parallel processing."""
        return True
