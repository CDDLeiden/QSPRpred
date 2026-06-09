"""Integration tests for the PostgreSQL/RDKit ChemStore backend.

These tests require a real PostgreSQL database with the RDKit extension enabled.
They are skipped automatically when QSPR_POSTGRES_DSN is not configured.
"""

import os

import pytest
from dotenv import load_dotenv

from qsprpred.data.storage.postgres import PostgresChemStore


load_dotenv()


def _get_store(table_name: str = "test_molecules_pytest") -> PostgresChemStore:
    dsn = os.getenv("QSPR_POSTGRES_DSN")
    if not dsn:
        pytest.skip("QSPR_POSTGRES_DSN is not configured")

    return PostgresChemStore(
        name="test_chemstore",
        connection_string=dsn,
        table_name=table_name,
        schema=os.getenv("QSPR_POSTGRES_SCHEMA", "public"),
        use_rdkit_cartridge=os.getenv("QSPR_USE_RDKIT", "true").lower() == "true",
        create=True,
    )


def test_postgres_chemstore_basic_workflow():
    store = _get_store("test_molecules_pytest_basic")
    store.clear()

    store.addMols(
        smiles=["CCO", "CCN", "c1ccccc1"],
        props={
            "chem_name": ["ethanol", "ethylamine", "benzene"],
            "source": ["pytest", "pytest", "pytest"],
            "score": [1.1, 2.2, 3.3],
        },
        raise_on_existing=False,
    )

    assert store.getMolCount() == 3
    assert store.getMolIDs() == ("00001", "00002", "00003")
    assert list(store.getProperty("chem_name")) == ["ethanol", "ethylamine", "benzene"]
    assert len(store.getDF()) == 3

    store.removeMol("00001")
    assert store.getMolCount() == 2


def test_postgres_chemstore_smarts_search():
    store = _get_store("test_molecules_pytest_smarts")
    store.clear()

    store.addMols(
        smiles=["CCO", "CCN", "c1ccccc1", "Cc1ccccc1"],
        props={
            "chem_name": ["ethanol", "ethylamine", "benzene", "toluene"],
        },
        raise_on_existing=False,
    )

    aromatic = store.searchWithSMARTS(["c1ccccc1"], name="test_molecules_pytest_smarts_aromatic")
    assert aromatic.getMolCount() == 2
    assert set(aromatic.getProperty("chem_name")) == {"benzene", "toluene"}

    nitrogen_or_aromatic = store.searchWithSMARTS(
        ["N", "c1ccccc1"],
        operator="or",
        name="test_molecules_pytest_smarts_or",
    )
    assert nitrogen_or_aromatic.getMolCount() == 3

    nitrogen_and_aromatic = store.searchWithSMARTS(
        ["N", "c1ccccc1"],
        operator="and",
        name="test_molecules_pytest_smarts_and",
    )
    assert nitrogen_and_aromatic.getMolCount() == 0
