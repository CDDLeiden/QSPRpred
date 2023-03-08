"""Functions to pre-process SMILES for QSPR modelling."""

import re

from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from chembl_structure_pipeline import standardizer as chembl_stand


# TODO: add wrapper as `old_standardization`. And deprecation warning
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


def chembl_smi_standardizer(smi: str, isomericSmiles:bool=True, sanitize:bool=True) -> str:
    try:
        mol = Chem.MolFromSmiles(smi)
    except:  # noqa E722
        print('Could not parse smiles: ', smi)
        return None
    standard_mol = chembl_stand.standardize_mol(mol, sanitize=sanitize)
    standard_smiles = Chem.MolToSmiles(
        standard_mol, kekuleSmiles=False, canonical=True, isomericSmiles=isomericSmiles
    )
    return standard_smiles


def sanitize_smiles(smi: str) -> str:
    """Sanitize a SMILES string.

    Removes sulfurs, extermal molecules and salts, neutralizes charges
    using the function `neutralize_atoms` and returns the resulting smiles.
    
    Arguments:
        smi: single SMILES string to be sanitized.

    Returns:
        sanitized SMILES string.
    """
    salts = re.compile(r"\..?Cl|\..?Br|\..?Ca|\..?K|\..?Na|\..?Li|\..?Zn|/\..?Gd")
    s_acid_remover = re.compile(r"\.OS\(\=O\)\(\=O\)O")
    boron_pattern = re.compile(r"B")
    remover = SaltRemover(defnData="[Cl,Br,Ca,K,Na,Zn]")
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    mol = Chem.MolFromSmiles(smi)
    # Removing sulfuric acid (smiles = .OS(=O)(=O)O)
    if s_acid_remover.findall(smi):
        smi = re.sub(s_acid_remover, "", smi)
        try:
            Chem.MolFromSmiles(smi)
        except:
            print(f"{smi} could not be parsed after removing sulfuric acids!")
            return None
    # Removing external molecules by splitting on . and picking the largest smiles
    if "." in smi:
        smi = max(smi.split("."), key=len)
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            print(f"Compound, ({smi}) could not be parsed!!")
            return None
    # Trying to remove the salts
    if salts.findall(smi):
        res, deleted = remover.StripMolWithDeleted(mol)
        # avoid neutralizing smiles with boron atoms
        if all([res is not None, not boron_pattern.findall(smi)]):
            neutralize_atoms(res)
            # If it didn't remove, let's continue
            if salts.findall(Chem.MolToSmiles(res)):
                print(f"Unable to remove salts from compound {smi}")
                return None
            else:
                smi = Chem.MolToSmiles(res)
                mol = Chem.MolFromSmiles(smi)
    # Are the molecules charged according to the "pattern" variable?
    if mol.GetSubstructMatches(pattern):
        res, deleted = remover.StripMolWithDeleted(mol)
        # avoid neutralizing smiles with boron atoms
        if all([res is not None, not boron_pattern.findall(smi)]):
            neutralize_atoms(res)
        if salts.findall(Chem.MolToSmiles(res)):
            print(f"Unable to remove salts from compound {smi} after neutralizing")
            return None
        else:
            smi = Chem.MolToSmiles(res)
    return smi
