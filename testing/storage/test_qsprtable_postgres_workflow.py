import os

import pandas as pd
import pytest
from dotenv import load_dotenv

from qsprpred.data.descriptors.fingerprints import MorganFP
from qsprpred.data.storage.postgres import PostgresChemStore
from qsprpred.data.tables.qspr import QSPRTable
from qsprpred.tasks import TargetProperty, TargetTasks


load_dotenv()


@pytest.fixture()
def postgres_store():
    dsn = os.getenv("QSPR_POSTGRES_DSN")
    if not dsn:
        pytest.skip("QSPR_POSTGRES_DSN is not set")

    store = PostgresChemStore(
        name="qsprtable_workflow_test",
        connection_string=dsn,
        table_name="test_qsprtable_workflow",
        schema=os.getenv("QSPR_POSTGRES_SCHEMA", "public"),
        use_rdkit_cartridge=True,
        create=True,
    )
    store.clear()
    return store


def test_qsprtable_prepare_dataset_with_postgres_storage(postgres_store, tmp_path):
    df = pd.DataFrame(
        {
            "SMILES": [
                "CCO",
                "CCN",
                "c1ccccc1",
                "Cc1ccccc1",
                "CC(=O)O",
            ],
            "activity": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    dataset = QSPRTable.fromDF(
        name="qsprtable_postgres_workflow",
        df=df,
        path=str(tmp_path),
        smiles_col="SMILES",
        target_props=[
            TargetProperty(
                name="activity",
                task=TargetTasks.REGRESSION,
            )
        ],
        storage=postgres_store,
    )

    assert dataset.storage is postgres_store
    assert dataset.storage.getMolCount() == 5

    dataset.prepareDataset(
        feature_calculators=[
            MorganFP(radius=2, nBits=64),
        ],
        recalculate_features=True,
    )

    assert dataset.X is not None
    assert dataset.y is not None
    assert len(dataset.X) == 5
    assert len(dataset.y) == 5
    assert dataset.X.shape[1] > 0

    df_storage = postgres_store.getDF()
    assert "activity" in df_storage.columns
    assert set(df_storage["SMILES"]) == set(df["SMILES"])