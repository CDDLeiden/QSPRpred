from typing import Literal

from rdkit import Chem


def match_mol_to_smarts(
        mol: Chem.Mol | str,
        smarts: list[str],
        operator: Literal["or", "and"] = "or",
        use_chirality: bool = False
):
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
    for smart in smarts:
        ret = mol.HasSubstructMatch(Chem.MolFromSmarts(smart),
                                    useChirality=use_chirality)
        if operator == "or":
            if ret:
                return True
        elif operator == "and":
            if ret:
                ret = True
            else:
                return False
    return ret

