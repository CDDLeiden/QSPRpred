import os

import pytest
from dotenv import load_dotenv

from qsprpred.data.storage.postgres import PostgresChemStore
from qsprpred.data.tables.mol import MoleculeTable

load_dotenv()


@pytest.fixture()
def postgres_store():
    dsn = os.getenv("QSPR_POSTGRES_DSN")
    if not dsn:
        pytest.skip("QSPR_POSTGRES_DSN is not set")

    store = PostgresChemStore(
        name="molecule_table_test",
        connection_string=dsn,
        table_name="test_molecule_table",
        schema=os.getenv("QSPR_POSTGRES_SCHEMA", "public"),
        use_rdkit_cartridge=True,
        create=True,
    )
    store.clear()
    return store


def test_molecule_table_from_smiles_with_postgres_storage(postgres_store):
    table = MoleculeTable.fromSMILES(
        name="molecule_table_test",
        path=".",
        smiles=["CCO", "CCN", "c1ccccc1"],
        storage=postgres_store,
    )

    assert table.storage.getMolCount() == 3

    df = table.getDF()
    assert len(df) == 3
    assert "SMILES" in df.columns
    assert set(df["SMILES"]) == {"CCO", "CCN", "c1ccccc1"}


def test_molecule_table_postgres_smarts_search(postgres_store):
    table = MoleculeTable.fromSMILES(
        name="molecule_table_test",
        path=".",
        smiles=["CCO", "CCN", "c1ccccc1", "Cc1ccccc1"],
        storage=postgres_store,
    )

    aromatic_store = table.storage.searchWithSMARTS(["c1ccccc1"])
    aromatic_df = aromatic_store.getDF()

    assert len(aromatic_df) == 2
    assert set(aromatic_df["SMILES"]) == {"c1ccccc1", "Cc1ccccc1"}