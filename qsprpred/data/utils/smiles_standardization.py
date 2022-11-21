"""Functions to pre-process SMILES for QSPR modelling."""

import re

from chembl_structure_pipeline import standardizer
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover


def chembl_smi_standardizer(smi: str) -> tuple:
    """Standardize a SMILES string.
    
    Returns a tuple containing 'parent smiles' and a 'bool',
    which defaults to if the standardizer failed and false when
    there were no errors.

    Arguments:
        smi: smiles string to be standardized with the 'chembl_structure_pipeline'.
    
    Returns:
        Tuple containing the parent smiles and a bool indicating if the standardizer
        failed. If True -> standardizer failed..
    """
    mol = Chem.MolFromSmiles(smi)
    standard_mol = standardizer.standardize_mol(mol)
    result = standardizer.get_parent_mol(
        standard_mol
    )  # Tuple with molecule in #0 and Boolean in #1
    # Boolean states whether there was an exclusion flag. For more details, check:
    # https://github.com/chembl/ChEMBL_Structure_Pipeline/wiki/Exclusion-Flag
    parent_mol = result[0]
    parent_smi = Chem.MolToSmiles(
        parent_mol, kekuleSmiles=False, canonical=True, isomericSmiles=True
    )
    if result[1]:
        return parent_smi, True
    else:
        return parent_smi, False


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
        if res is not None:
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
        if res is not None:
            neutralize_atoms(res)
        if salts.findall(Chem.MolToSmiles(res)):
            print(f"Unable to remove salts from compound {smi} after neutralizing")
            return None
        else:
            smi = Chem.MolToSmiles(res)
    return smi
