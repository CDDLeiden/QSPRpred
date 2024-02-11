from typing import Callable

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from qsprpred.data.tables.base import MoleculeDataTable

standard_grid = Draw.MolsToGridImage


def interactive_grid(mols, *args, molsPerRow=5, **kwargs):
    """
    install mols2grid with pip to use
    """

    import mols2grid

    return mols2grid.display(mols, *args, n_cols=molsPerRow, **kwargs)


def smiles_to_grid(
    smiles, *args, mols_per_row=5, impl: Callable = standard_grid, **kwargs
):
    mols = []
    for smile in smiles:
        try:
            m = Chem.MolFromSmiles(smile)
            if m:
                AllChem.Compute2DCoords(m)
                mols.append(m)
            else:
                raise Exception(f"Molecule empty for SMILES: {smile}")
        except Exception as exp:
            pass

    return impl(mols, *args, molsPerRow=mols_per_row, **kwargs)


def table_to_grid(
    table: MoleculeDataTable,
    mols_per_row: int = 5,
    impl: Callable = standard_grid,
    *args,
    **kwargs,
):
    return smiles_to_grid(
        table.smiles, *args, mols_per_row=mols_per_row, impl=impl, **kwargs
    )
