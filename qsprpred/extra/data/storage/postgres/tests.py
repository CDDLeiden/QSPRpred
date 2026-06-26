import os
import sys
import uuid
from unittest import TestCase

import pandas as pd
import psycopg
import pytest
from dotenv import load_dotenv

from qsprpred.data.chem.identifiers import InchiIdentifier
from qsprpred.data.chem.standardizers.check_smiles import CheckSmilesValid
from qsprpred.data.chem.standardizers.papyrus import PapyrusStandardizer
from qsprpred.data.storage.tabular.simple import PandasChemStore
from qsprpred.data.storage.tests import StorageTest
from .chem_store import PostgresChemStore


def _storage_test_help():
    """Print a short backend usage note during test collection."""
    print(
        "\n[QSPRpred storage tests]\n"
        "Default command runs the original Pandas storage tests:\n"
        "  python -m pytest qsprpred/data/storage/tests.py\n\n"
        "PostgreSQL/RDKit backend tests are available explicitly with:\n"
        "  python -m pytest qsprpred/data/storage/tests.py -k PostgresTabularStorageTest\n",
        flush=True,
    )


_storage_test_help()


def _postgres_tests_requested() -> bool:
    """Return True only when the PostgreSQL test class is explicitly selected."""
    argv = " ".join(sys.argv)
    return "PostgresTabularStorageTest" in argv


def _safe_pg_identifier(value: str) -> str:
    """Make a conservative PostgreSQL identifier from a test name."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in value.lower())
    if not safe or safe[0].isdigit():
        safe = f"t_{safe}"
    return safe[:55]


def _postgres_test_config():
    load_dotenv()
    dsn = os.getenv("QSPR_POSTGRES_DSN")
    if not dsn:
        pytest.skip("QSPR_POSTGRES_DSN is not set; skipping PostgreSQL storage tests.")

    return {
        "dsn": dsn,
        "schema": os.getenv("QSPR_POSTGRES_SCHEMA", "public"),
        "use_rdkit": os.getenv("QSPR_USE_RDKIT", "true").lower() == "true",
    }


def _register_postgres_test_run(
        dsn: str,
        schema: str,
        run_id: str,
        test_name: str,
        table_name: str,
        table_prefix: str,
):
    """Register a PostgreSQL test run and the stable table used by it.

    The registry schema is migrated idempotently because local/cloud databases
    may already contain older versions created by previous test iterations.
    """
    with psycopg.connect(dsn, connect_timeout=10) as conn:
        with conn.cursor() as cur:
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

            for column_sql in [
                "started_at TIMESTAMPTZ NOT NULL DEFAULT now()",
                "backend TEXT NOT NULL DEFAULT 'postgres'",
                "test_name TEXT",
                "test_group TEXT",
                "table_prefix TEXT NOT NULL DEFAULT 'unknown'",
                "note TEXT",
            ]:
                cur.execute(
                    f"ALTER TABLE {schema}.chemstore_test_runs ADD COLUMN IF NOT EXISTS {column_sql};"
                )

            for column_sql in [
                "run_id TEXT",
                "table_name TEXT",
                "created_at TIMESTAMPTZ NOT NULL DEFAULT now()",
            ]:
                cur.execute(
                    f"ALTER TABLE {schema}.chemstore_test_tables ADD COLUMN IF NOT EXISTS {column_sql};"
                )

            cur.execute(
                f"""
                INSERT INTO {schema}.chemstore_test_runs
                    (run_id, backend, test_name, test_group, table_prefix, note)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    backend = EXCLUDED.backend,
                    test_name = EXCLUDED.test_name,
                    test_group = EXCLUDED.test_group,
                    table_prefix = EXCLUDED.table_prefix,
                    note = EXCLUDED.note;
                """,
                (
                    run_id,
                    "postgres",
                    test_name,
                    "PostgresTabularStorageTest",
                    table_prefix,
                    "Legacy storage test executed on PostgreSQL/RDKit backend.",
                ),
            )
            cur.execute(
                f"""
                INSERT INTO {schema}.chemstore_test_tables
                    (run_id, table_name)
                VALUES (%s, %s);
                """,
                (run_id, table_name),
            )
        conn.commit()


class PostgresTabularStorageTest(StorageTest, TestCase):
    """Run the tabular ChemStore tests against PostgreSQL + RDKit Cartridge.

    These tests are opt-in. The normal command

        python -m pytest qsprpred/data/storage/tests.py

    keeps running the original Pandas tests. To run this PostgreSQL backend test
    class explicitly, use:

        python -m pytest qsprpred/data/storage/tests.py -k PostgresTabularStorageTest
    """

    def setUp(self):
        if not _postgres_tests_requested():
            pytest.skip(
                "PostgreSQL backend tests are opt-in. Run with: "
                "python -m pytest qsprpred/data/storage/tests.py -k PostgresTabularStorageTest"
            )

        super().setUp()
        cfg = _postgres_test_config()
        self.postgres_dsn = cfg["dsn"]
        self.postgres_schema = cfg["schema"]
        self.postgres_use_rdkit = cfg["use_rdkit"]
        self.postgres_run_id = f"pgtest_{uuid.uuid4().hex[:10]}"
        table_prefixes = {
            "testAddMols": "cst_addmols",
            "testMolProcess": "cst_molprocess",
            "testSubsetting": "cst_subset",
            "testSearch": "cst_search",
        }
        self.postgres_table = table_prefixes.get(
            self._testMethodName,
            _safe_pg_identifier(f"cst_{self._testMethodName}"),
        )
        _register_postgres_test_run(
            self.postgres_dsn,
            self.postgres_schema,
            self.postgres_run_id,
            self._testMethodName,
            self.postgres_table,
            self.postgres_table,
        )

    def _prepare_reference_dataframe(self) -> pd.DataFrame:
        """Use the original Pandas pipeline to prepare the same test molecules.

        The CSV test fixtures require PapyrusStandardizer before InChI identifiers
        are generated. We therefore reuse the original preparation path and then
        copy the resulting tabular data into PostgreSQL.
        """
        temp_store = (
            PandasChemStore(
                f"{self.__class__.__name__}_{self._testMethodName}_reference",
                self.outputPath,
                pd.read_csv(self.exampleFileBasic),
                standardizer=PapyrusStandardizer(),
                identifier=InchiIdentifier(),
            ))
        temp_store.addLibrary(
            f"{temp_store.name}_2",
            pd.read_csv(self.exampleFileIndex),
            smiles_col="smiles",
        )
        return temp_store.getDF().copy()

    def _load_dataframe_into_postgres(self, store: PostgresChemStore, df: pd.DataFrame):
        smiles_col = store.smilesProp
        if smiles_col not in df.columns:
            smiles_col = "SMILES" if "SMILES" in df.columns else "smiles"

        props = {}
        for col in df.columns:
            if col == smiles_col:
                continue
            values = df[col].tolist()
            props[col] = values

        store.addMols(
            df[smiles_col].tolist(),
            props=props,
            raise_on_existing=False,
        )

    def getStorage(self) -> PostgresChemStore:
        store = PostgresChemStore(
            f"{self.__class__.__name__}_{self._testMethodName}",
            connection_string=self.postgres_dsn,
            table_name=self.postgres_table,
            schema=self.postgres_schema,
            run_id=self.postgres_run_id,
            use_rdkit_cartridge=self.postgres_use_rdkit,
            create=True,
            chunk_size=1,
        )
        store.clear()
        self._load_dataframe_into_postgres(store, self._prepare_reference_dataframe())
        return store

    def checkSerialization(self, store):
        """PostgreSQL persistence is database-backed, not file-backed."""
        self.assertEqual(len(store), len(store.getDF()))
        store.save()
        store.reload()
        self.assertEqual(len(store), len(store.getDF()))

    def testAddMols(self):
        store = self.getStorage()
        len_before = len(store)

        added = store.addMols(["O=C(OCCN(CC)CC)c1ccc(N)cc1"])
        self.assertEqual(len(added), 1)
        self.assertEqual(len(store), len_before + 1)
        self.checkSerialization(store)

        mols = store.addMols(
            [
                "O=C(OC(CCC)CN(CC)CC)c1ccc(N)cc1",
                "O=C(OC(CCC)CN(CC)CC)c1ccc(N)cc1C",
            ],
            props={"new_prop": [1, 2]},
        )
        self.assertEqual(len(mols), 2)
        for idx, mol in enumerate(mols):
            self.assertIn("new_prop", mol.props)
            self.assertEqual(mol.props["new_prop"], idx + 1)

        for mol in mols:
            loaded = store.getMol(mol.id)
            self.assertIn("new_prop", loaded.props)

        self.assertEqual(len(store), len_before + 3)
        self.checkSerialization(store)

    def testMolProcess(self):
        store = self.getStorage()
        result = pd.concat(list(store.processMols(CheckSmilesValid())))
        self.assertEqual(len(result), len(store))
        self.assertTrue(all(result))
        for idx in result.index:
            self.assertTrue(idx in store)

        store.nJobs = 2
        result = list(store.processMols(CheckSmilesValid()))
        result = pd.concat(result)
        self.assertEqual(len(result), len(store))
        self.assertTrue(all(result))

    def testSubsetting(self):
        store = self.getStorage()
        mol_1 = next(iter(store))
        mol_2 = list(store)[1]

        subset = store.getSubset(["TestProp1", "ExtraIndexColumn"])
        self.assertEqual(len(subset), len(store))
        for mol in [mol_1, mol_2]:
            self.assertIn(mol.id, subset)
            self.assertIn("TestProp1", subset[mol.id].props)
            self.assertIn("ExtraIndexColumn", subset[mol.id].props)

        subset = store.getSubset(
            ["TestProp1", "ExtraIndexColumn"], [mol_2.id, mol_1.id]
        )
        self.assertEqual(len(subset), 2)
        for mol in [mol_1, mol_2]:
            self.assertIn(mol.id, subset)
            self.assertIn("TestProp1", subset[mol.id].props)
            self.assertIn("ExtraIndexColumn", subset[mol.id].props)

    def testSearch(self):
        store = self.getStorage()

        result = store.searchOnProperty("TestProp1", [1.0])
        result_prop = result.getProperty("TestProp1")
        self.assertTrue(all(result_prop == 1.0))

        result = store.searchOnProperty("ExtraIndexColumn", ["Molecule8"])
        result_prop = result.getProperty("ExtraIndexColumn")
        self.assertTrue(all(result_prop == "Molecule8"))
        self.assertEqual(len(result_prop), 1)

        result = store.searchOnProperty("ExtraIndexColumn", ["MoleculeX"])
        self.assertEqual(len(result), 0)

        result = store.searchWithSMARTS(
            ["N[C@H]"],
            name=f"{self.postgres_table}_smarts",
            use_chirality=False,
        )
        self.assertGreaterEqual(len(result), 1)

        result = store.searchWithSMARTS(
            ["N[C@H]", "C1CCC1"],
            name=f"{self.postgres_table}_smarts_and",
            operator="and",
            use_chirality=False,
        )
        self.assertGreaterEqual(len(result), 0)
