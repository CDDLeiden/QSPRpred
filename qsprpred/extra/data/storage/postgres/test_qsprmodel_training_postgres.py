import os
import uuid
from datetime import datetime, timezone

import pandas as pd
import pytest
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from qsprpred import TargetSpec, TargetTasks
from qsprpred.data.descriptors.fingerprints import MorganFP
from qsprpred.data.tables.qspr import QSPRTable
from .chem_store import PostgresChemStore

load_dotenv()


def create_test_run(dsn, schema, run_id, test_name, table_name):
    """Create or migrate PostgreSQL test-run registry tables.

    The registry tables may already exist from older test versions. Therefore this
    function is intentionally idempotent: it creates minimal tables if needed, then
    adds all required columns with ADD COLUMN IF NOT EXISTS.

    Registry purpose:
    - chemstore_test_runs stores metadata about each test/workflow run.
    - chemstore_test_tables stores tables created by a specific run.

    This makes it possible to inspect old test runs and decide which generated
    tables should be kept or removed later.
    """
    import psycopg

    started_at = datetime.now(timezone.utc)
    table_prefix = run_id
    note = "QSPRModel training workflow test with PostgreSQL ChemStore"

    with psycopg.connect(dsn, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            # Create minimal registry tables if they do not exist yet.
            # Existing older tables are preserved and migrated below.
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {schema}.chemstore_test_runs (
                    run_id TEXT PRIMARY KEY
                );
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {schema}.chemstore_test_tables (
                    id BIGSERIAL PRIMARY KEY
                );
                """
            )

            # Migrate/complete chemstore_test_runs schema.
            # Some previous versions created table_prefix as NOT NULL, so we must
            # always provide it during INSERT.
            cur.execute(
                f"""
                ALTER TABLE {schema}.chemstore_test_runs
                ADD COLUMN IF NOT EXISTS test_name TEXT;
                """
            )
            cur.execute(
                f"""
                ALTER TABLE {schema}.chemstore_test_runs
                ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ;
                """
            )
            cur.execute(
                f"""
                ALTER TABLE {schema}.chemstore_test_runs
                ADD COLUMN IF NOT EXISTS backend TEXT;
                """
            )
            cur.execute(
                f"""
                ALTER TABLE {schema}.chemstore_test_runs
                ADD COLUMN IF NOT EXISTS table_prefix TEXT;
                """
            )
            cur.execute(
                f"""
                ALTER TABLE {schema}.chemstore_test_runs
                ADD COLUMN IF NOT EXISTS note TEXT;
                """
            )

            # Backfill old rows if they exist. This is safe even if there are no rows.
            cur.execute(
                f"""
                UPDATE {schema}.chemstore_test_runs
                SET table_prefix = COALESCE(table_prefix, run_id)
                WHERE table_prefix IS NULL;
                """
            )
            cur.execute(
                f"""
                UPDATE {schema}.chemstore_test_runs
                SET backend = COALESCE(backend, 'postgres')
                WHERE backend IS NULL;
                """
            )
            cur.execute(
                f"""
                UPDATE {schema}.chemstore_test_runs
                SET started_at = COALESCE(started_at, now())
                WHERE started_at IS NULL;
                """
            )

            # Migrate/complete chemstore_test_tables schema.
            cur.execute(
                f"""
                ALTER TABLE {schema}.chemstore_test_tables
                ADD COLUMN IF NOT EXISTS run_id TEXT;
                """
            )
            cur.execute(
                f"""
                ALTER TABLE {schema}.chemstore_test_tables
                ADD COLUMN IF NOT EXISTS table_name TEXT;
                """
            )
            cur.execute(
                f"""
                ALTER TABLE {schema}.chemstore_test_tables
                ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now();
                """
            )
            cur.execute(
                f"""
                ALTER TABLE {schema}.chemstore_test_tables
                ADD COLUMN IF NOT EXISTS note TEXT;
                """
            )

            # Insert current test run and table metadata.
            cur.execute(
                f"""
                INSERT INTO {schema}.chemstore_test_runs
                    (run_id, test_name, started_at, backend, table_prefix, note)
                VALUES
                    (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    test_name = EXCLUDED.test_name,
                    started_at = EXCLUDED.started_at,
                    backend = EXCLUDED.backend,
                    table_prefix = EXCLUDED.table_prefix,
                    note = EXCLUDED.note;
                """,
                (
                    run_id,
                    test_name,
                    started_at,
                    "postgres",
                    table_prefix,
                    note,
                ),
            )

            cur.execute(
                f"""
                INSERT INTO {schema}.chemstore_test_tables
                    (run_id, table_name, created_at, note)
                VALUES
                    (%s, %s, %s, %s);
                """,
                (
                    run_id,
                    table_name,
                    started_at,
                    "Main molecule table for QSPRModel training workflow test",
                ),
            )

        conn.commit()


@pytest.fixture()
def postgres_store():
    dsn = os.getenv("QSPR_POSTGRES_DSN")
    if not dsn:
        pytest.skip("QSPR_POSTGRES_DSN is not set")

    schema = os.getenv("QSPR_POSTGRES_SCHEMA", "public")

    run_id = f"qsprmodel_training_{uuid.uuid4().hex[:8]}"
    table_name = "cst_qsprmodel_training"

    create_test_run(
        dsn=dsn,
        schema=schema,
        run_id=run_id,
        test_name="test_qsprmodel_training_with_postgres_storage",
        table_name=table_name,
    )

    store = PostgresChemStore(
        name=run_id,
        connection_string=dsn,
        table_name=table_name,
        schema=schema,
        run_id=run_id,
        use_rdkit_cartridge=True,
        create=True,
    )
    store.clear()

    return store


def test_qsprmodel_training_with_postgres_storage(postgres_store, tmp_path):
    df = pd.DataFrame(
        {
            "SMILES": [
                "CCO",
                "CCN",
                "CCC",
                "CCCC",
                "CCCO",
                "CCCN",
                "c1ccccc1",
                "Cc1ccccc1",
                "CC(=O)O",
                "CC(C)O",
            ],
            "activity": [1.0, 1.2, 1.5, 2.0, 2.2, 2.4, 3.0, 3.3, 1.8, 2.1],
        }
    )

    dataset = QSPRTable.fromDF(
        name="qsprmodel_postgres_training",
        df=df,
        path=str(tmp_path),
        smiles_col="SMILES",
        target_props=[
            TargetSpec(
                name="activity",
                task=TargetTasks.REGRESSION,
            )
        ],
        storage=postgres_store,
    )
    dataset.addDescriptors([MorganFP(radius=2, nBits=128)], recalculate=True)

    assert dataset is not None
    assert dataset.getTargets() is not None
    assert len(dataset.getDescriptors()) == len(df)
    assert len(dataset.getTargets()) == len(df)

    model = RandomForestRegressor(
        n_estimators=20,
        random_state=42,
    )

    model.fit(dataset.getDescriptors(), dataset.getTargets().values.ravel())
    preds = model.predict(dataset.getDescriptors())

    assert len(preds) == len(df)
    assert r2_score(dataset.getTargets().values.ravel(), preds) > 0.5

    stored_df = postgres_store.getDF()
    assert len(stored_df) == len(df)
    assert "activity" in stored_df.columns
