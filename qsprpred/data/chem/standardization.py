"""Functions to pre-process SMILES for QSPR modelling."""

import re
from typing import Any

import pandas as pd
from chembl_structure_pipeline import standardizer as chembl_stand
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.SaltRemover import SaltRemover

from ..processing.mol_processor import MolProcessorWithID
from ...logs import logger


class CheckSmilesValid(MolProcessorWithID):
    def __call__(
        self, mols: list[str | Mol], props: dict[str, list[Any]], *args, **kwargs
    ) -> Any:
        throw = kwargs.get("throw", False)
        ret = []
        for mol in mols:
            is_valid = True
            exception = None
            if not mol:
                is_valid = False
                exception = ValueError(f"Empty molecule: {mol}")
            try:
                mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
                if not mol:
                    raise ValueError(f"Invalid molecule: {mol}")
                Chem.SanitizeMol(mol)
            except Exception as exp:
                is_valid = False
                exception = exp
            if exception and throw:
                raise exception
            else:
                ret.append(is_valid)
        ret = pd.Series(ret, index=props[self.idProp])
        return ret

    @property
    def supportsParallel(self) -> bool:
        return True


def neutralize_atoms(mol):
    """Neutralize charged molecules by atom.

    From https://www.rdkit.org/docs/Cookbook.html, adapted from
    https://baoilleach.blogspot.com/2019/12/no-charge-simple-approach-to.html

    Arguments:
        mol: rdkit molecule to be neutralized

    Returns:
        mol: neutralized rdkit mol
    """
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


def chembl_smi_standardizer(
    smi: str, isomericSmiles: bool = True, sanitize: bool = True
) -> str:
    """Standardize SMILES using ChEMBL standardizer.

    Args:
        smi: SMILES string to be standardized.
        isomericSmiles: return the isomeric smiles. Defaults to True.
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
            isomericSmiles=isomericSmiles,
        )
        return standard_smiles
    except Exception as exp:  # E722
        logger.warning(f"Could not standardize SMILES: {smi} due to: {exp}.")
        return None


def old_standardize_sanitize(smi: str) -> str:
    """Adaptation of the old QSPRpred molecule standardization/sanitization.

    Standardize the rdkit mol object and gets parent molecule using
    chembl_structure_pipeline, and applies some sanitization steps.
    Using this function is not recommended and it will be deprecated within
    next releases.

    Arguments:
        smi: single SMILES string to be sanitized.

    Returns:
        sanitized SMILES string.
    """
    mol = Chem.MolFromSmiles(smi)
    standard_mol = chembl_stand.standardize_mol(mol)
    result = chembl_stand.get_parent_mol(
        standard_mol
    )  # Tuple with molecule in #0 and Boolean in #1
    # Boolean states whether there was an exclusion flag. For more details, check:
    # https://github.com/chembl/ChEMBL_Structure_Pipeline/wiki/Exclusion-Flag
    parent_mol = result[0]
    parent_smi = Chem.MolToSmiles(
        parent_mol, kekuleSmiles=False, canonical=True, isomericSmiles=True
    )
    salts = re.compile(r"\..?Cl|\..?Br|\..?Ca|\..?K|\..?Na|\..?Li|\..?Zn|/\..?Gd")
    s_acid_remover = re.compile(r"\.OS\(\=O\)\(\=O\)O")
    boron_pattern = re.compile(r"B")
    remover = SaltRemover(defnData="[Cl,Br,Ca,K,Na,Zn]")
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    mol = Chem.MolFromSmiles(parent_smi)
    # Removing sulfuric acid (smiles = .OS(=O)(=O)O)
    if s_acid_remover.findall(parent_smi):
        parent_smi = re.sub(s_acid_remover, "", parent_smi)
        try:
            Chem.MolFromSmiles(parent_smi)
        except:  # noqa: E722
            print(f"{parent_smi} could not be parsed after removing sulfuric acids!")
            return None
    # Removing external molecules by splitting on . and picking the largest smiles
    if "." in parent_smi:
        parent_smi = max(parent_smi.split("."), key=len)
        try:
            mol = Chem.MolFromSmiles(parent_smi)
        except:  # noqa: E722
            print(f"Compound, ({parent_smi}) could not be parsed!!")
            return None
    # Trying to remove the salts
    if salts.findall(parent_smi):
        res, deleted = remover.StripMolWithDeleted(mol)
        # avoid neutralizing smiles with boron atoms
        if all([res is not None, not boron_pattern.findall(parent_smi)]):
            neutralize_atoms(res)
            # If it didn't remove, let's continue
            if salts.findall(Chem.MolToSmiles(res)):
                print(f"Unable to remove salts from compound {parent_smi}")
                return None
            else:
                parent_smi = Chem.MolToSmiles(res)
                mol = Chem.MolFromSmiles(parent_smi)
    # Are the molecules charged according to the "pattern" variable?
    if mol.GetSubstructMatches(pattern):
        res, deleted = remover.StripMolWithDeleted(mol)
        # avoid neutralizing smiles with boron atoms
        if all([res is not None, not boron_pattern.findall(parent_smi)]):
            neutralize_atoms(res)
        if salts.findall(Chem.MolToSmiles(res)):
            print(
                f"Unable to remove salts from compound {parent_smi} after neutralizing"
            )
            return None
        else:
            parent_smi = Chem.MolToSmiles(res)
    return parent_smi
