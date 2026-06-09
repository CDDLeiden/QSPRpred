import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from unittest import TestCase

import pandas as pd
import pytest
import psycopg
from dotenv import load_dotenv
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs

from qsprpred.data.chem.identifiers import InchiIdentifier
from qsprpred.data.chem.standardizers.check_smiles import CheckSmilesValid
from qsprpred.data.chem.standardizers.papyrus import PapyrusStandardizer
from qsprpred.data.storage.interfaces.chem_store import ChemStore
from qsprpred.data.storage.tabular.hierarchical import PandasRepresentationStore, \
    RepresentationMol
from qsprpred.data.storage.tabular.simple import PandasChemStore
from qsprpred.data.storage.postgres import PostgresChemStore


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


def _register_postgres_test_run(dsn: str, schema: str, run_id: str, test_name: str, table_name: str):
    """Register a PostgreSQL test run and its generated table.

    The tables are intentionally not dropped by default. This makes it possible
    to inspect the database after a test run and see which tables belong to which
    test invocation.
    """
    with psycopg.connect(dsn, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {schema}.chemstore_test_runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    backend TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    table_prefix TEXT NOT NULL
                );
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {schema}.chemstore_test_tables (
                    id BIGSERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                """
            )
            cur.execute(
                f"""
                INSERT INTO {schema}.chemstore_test_runs
                    (run_id, backend, test_name, table_prefix)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_id) DO NOTHING;
                """,
                (run_id, "postgres", test_name, table_name),
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


class StorageTest(ABC):
    def setUp(self):
        self.testDir = os.path.join(os.path.dirname(__file__), "test_files")
        self.outputPath = os.path.join(self.testDir, "output")
        if os.path.exists(self.outputPath):
            shutil.rmtree(self.outputPath)
        os.makedirs(self.outputPath, exist_ok=True)
        self.exampleFileBasic = os.path.join(self.testDir, "example_table_default.csv")
        self.exampleFileIndex = os.path.join(self.testDir, "example_table_index.csv")

    def tearDown(self):
        if os.path.exists(self.outputPath):
            shutil.rmtree(self.outputPath)

    @abstractmethod
    def getStorage(self) -> ChemStore:
        pass


class TabularStorageTest(StorageTest, TestCase):
    def getStorage(self) -> PandasChemStore:
        store = PandasChemStore(
            f"{self.__class__.__name__}_test_basic",
            self.outputPath,
            pd.read_csv(self.exampleFileBasic),
            standardizer=PapyrusStandardizer(),
            identifier=InchiIdentifier(),
        )
        store.addLibrary(
            f"{store.name}_2",
            pd.read_csv(self.exampleFileIndex),
            smiles_col="smiles",
        )
        return store

    def checkSerialization(self, store):
        store.save()
        # create new and check consistency
        store2 = PandasChemStore(store.name, self.outputPath)
        self.assertEqual(store2.nLibs, store.nLibs)
        self.assertEqual(len(store2), len(store))
        self.assertListEqual(list(store2.smiles), list(store.smiles))
        # create from meta file and check consistency
        store2 = PandasChemStore.fromFile(store.metaFile)
        self.assertEqual(store2.nLibs, store.nLibs)
        self.assertEqual(len(store2), len(store))
        self.assertListEqual(list(store2.smiles), list(store.smiles))
        # add a new library and check consistency after reload
        len_before = len(store)
        added = store.addMols(
            ["CN1[C@H]2CC[C@@H]1[C@@H](C(OC)=O)[C@@H](OC(C3=CC=CC=C3)=O)C2"],
        )
        self.assertEqual(len(added), 1)
        self.assertEqual(len(store), len_before + 1)
        store.reload()
        self.assertEqual(store.nLibs, store2.nLibs)
        self.assertEqual(len(store), len(store2))
        self.assertListEqual(list(store.smiles), list(store2.smiles))
        self.assertEqual(len(store), len(store.getDF()))
        self.assertEqual(len(store2), len(store2.getDF()))

    def testInitsAndSaves(self):
        # test default
        store_default = PandasChemStore(
            f"{self.__class__.__name__}_test_basic",
            self.outputPath,
            pd.read_csv(self.exampleFileBasic),
            standardizer=PapyrusStandardizer(),
            identifier=InchiIdentifier(),
        )
        self.assertEqual(store_default.nLibs, 1)
        self.assertEqual(len(store_default), 2)
        self.checkSerialization(store_default)
        # try to add store with the same name
        self.assertRaises(
            ValueError,
            lambda: store_default.addLibrary(
                f"{store_default.name}_library",
                pd.read_csv(self.exampleFileBasic),
            ),
        )
        # add a library with duplicated molecules
        store_default.addLibrary(
            f"{store_default.name}_2",
            pd.read_csv(self.exampleFileBasic),
        )
        self.assertEqual(store_default.nLibs, 2)
        self.assertEqual(len(store_default), 2)
        self.checkSerialization(store_default)
        # add a new library with additional compounds
        store_default.addLibrary(
            f"{store_default.name}_3",
            pd.read_csv(self.exampleFileIndex),
            smiles_col="smiles",
        )
        self.assertEqual(store_default.nLibs, 3)
        self.assertEqual(len(store_default), 3)
        self.checkSerialization(store_default)
        # test empty init
        store_empty = PandasChemStore(
            f"{self.__class__.__name__}_test_empty",
            self.outputPath,
            standardizer=PapyrusStandardizer(),
            identifier=InchiIdentifier(),
        )
        self.assertEqual(store_empty.nLibs, 1)
        self.assertEqual(len(store_empty), 0)
        self.checkSerialization(store_empty)
        # test with defaults
        df = pd.read_csv(self.exampleFileIndex)
        store_default = PandasChemStore(
            f"{self.__class__.__name__}_test_default",
            self.outputPath,
            df,
            smiles_col="smiles",
        )
        self.assertEqual(store_default.nLibs, 1)
        self.assertEqual(len(store_default), len(df))
        self.checkSerialization(store_default)
        # try from DF
        df = pd.read_csv(self.exampleFileIndex)
        PandasChemStore.fromDF(
            df,
            name=f"{self.__class__.__name__}_test_default_df",
            path=self.outputPath,
            smiles_col="smiles",
        )
        self.assertEqual(store_default.nLibs, 1)
        self.assertEqual(len(store_default), len(df))
        self.checkSerialization(store_default)

    def testAddMols(self):
        store = self.getStorage()
        len_before = len(store)
        added = store.addMols(["O=C(OCCN(CC)CC)c1ccc(N)cc1"], )
        self.assertEqual(len(added), 1)
        self.assertEqual(len(store), len_before + 1)
        self.checkSerialization(store)
        # add to a new library
        store.addMols(
            ["O=C(OC(C)CN(CC)CC)c1ccc(N)cc1"],
            library=f"{store.name}_2",
        )
        self.assertEqual(store.nLibs, 2)
        self.assertEqual(len(store), len_before + 2)
        self.checkSerialization(store)
        # add with new properties
        mols = store.addMols(
            ["O=C(OC(CCC)CN(CC)CC)c1ccc(N)cc1", "O=C(OC(CCC)CN(CC)CC)c1ccc(N)cc1C"],
            props={"new_prop": [1, 2]},
        )
        self.assertEqual(len(mols), 2)
        for idx, mol in enumerate(mols):
            self.assertIn("new_prop", mol.props)
            self.assertEqual(mol.props["new_prop"], idx + 1)
        for mol in store:
            self.assertIn("new_prop", mol.props)
        self.assertEqual(len(store), len_before + 4)
        self.checkSerialization(store)
        # add with existing properties
        mols = store.addMols(
            [
                "O=C(OC(CCC)CN(CC)CC)c1ccc(N)cc1C(C)C",
                "O=C(OC(CCC)CN(C(C)C)CC)c1ccc(N)cc1C",
            ],
            props={
                "TestProp1": [3, 4],
                "TestProp2": [5, 6]
            },
        )
        self.assertEqual(len(mols), 2)
        for mol in store:
            self.assertIn("TestProp1", mol.props)
            self.assertIn("TestProp2", mol.props)
        self.assertEqual(len(store), len_before + 6)
        self.checkSerialization(store)
        # add with existing properties and new ones
        mols = store.addMols(
            [
                "O=C(OC(C(O)C)CN(CC)CC)c1ccc(N)cc1C(C)C",
                "O=C(OC(CC(N)C)CN(C(C)C)CC)c1ccc(N)cc1C",
            ],
            props={
                "TestProp1": [3, 4],
                "TestProp2": [5, 6],
                "new_prop": [7, 8]
            },
        )
        self.assertEqual(len(mols), 2)
        for mol in store:
            self.assertIn("TestProp1", mol.props)
            self.assertIn("TestProp2", mol.props)
            self.assertIn("new_prop", mol.props)
        self.assertEqual(len(store), len_before + 8)
        self.checkSerialization(store)

    def testMolProcess(self):
        store = self.getStorage()
        result = pd.concat(list(store.processMols(CheckSmilesValid())))
        self.assertEqual(len(result), len(store))
        self.assertTrue(all(result))
        for idx in result.index:
            self.assertTrue(idx in store)
        # test with parallel
        store.nJobs = 2
        result = list(store.processMols(CheckSmilesValid()))
        self.assertEqual(len(result), 3)
        result = pd.concat(result)
        self.assertEqual(len(result), len(store))
        self.assertTrue(all(result))
        for idx in result.index:
            self.assertTrue(idx in store)

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
        # by property
        store = self.getStorage()
        result = store.searchOnProperty("TestProp1", [1.0])
        result = result.getProperty("TestProp1")
        self.assertTrue(all(result == 1.0))
        # using a string
        result = store.searchOnProperty("ExtraIndexColumn", ["Molecule8"])
        result = result.getProperty("ExtraIndexColumn")
        self.assertTrue(all(result == "Molecule8"))
        self.assertEqual(len(result), 1)
        # find non-existing
        result = store.searchOnProperty("ExtraIndexColumn", ["MoleculeX"])
        self.assertEqual(len(result), 0)
        # by SMARTS
        result = store.searchWithSMARTS(
            ["N[C@H]"],
            name="test_smarts",
            use_chirality=True,
        )
        self.assertEqual(len(result), len(store) - 1)
        # using non-chiral match
        result = store.searchWithSMARTS(
            ["N[C@H]"],
            name="test_smarts",
            use_chirality=False,
        )
        self.assertEqual(len(result), len(store))
        # using multiple patterns
        result = store.searchWithSMARTS(
            ["N[C@H]", "C1CCC1"],
            name="test_smarts",
            operator="and",
            use_chirality=True,
        )
        self.assertEqual(len(result), 0)
        result = store.searchWithSMARTS(
            ["N[C@H]", "C1CCC1"],
            name="test_smarts",
            operator="and",
            use_chirality=False,
        )
        self.assertEqual(len(result), 1)
        # get single molecule and check that is has all the props
        result_mol = next(iter(result))
        result_mol = store.getMol(result_mol.id)
        for prop in store.getProperties():
            self.assertIn(prop, result_mol.props)
        # drop it and check that it is not there
        result.removeMol(result_mol.id)
        self.assertNotIn(result_mol.id, result)
        self.assertEqual(len(result), 0)


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
        self.postgres_run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.postgres_table = _safe_pg_identifier(
            f"chemstore_{self.__class__.__name__}_{self._testMethodName}_{self.postgres_run_id}"
        )
        _register_postgres_test_run(
            self.postgres_dsn,
            self.postgres_schema,
            self.postgres_run_id,
            self._testMethodName,
            self.postgres_table,
        )

    def _prepare_reference_dataframe(self) -> pd.DataFrame:
        """Use the original Pandas pipeline to prepare the same test molecules.

        The CSV test fixtures require PapyrusStandardizer before InChI identifiers
        are generated. We therefore reuse the original preparation path and then
        copy the resulting tabular data into PostgreSQL.
        """
        temp_store = PandasChemStore(
            f"{self.__class__.__name__}_{self._testMethodName}_reference",
            self.outputPath,
            pd.read_csv(self.exampleFileBasic),
            standardizer=PapyrusStandardizer(),
            identifier=InchiIdentifier(),
        )
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



class TabularRepresentationStorageTest(StorageTest, TestCase):
    def setUp(self):
        super().setUp()
        store = PandasChemStore(
            f"{self.__class__.__name__}_test_basic_main",
            self.outputPath,
            pd.read_csv(self.exampleFileBasic),
            standardizer=PapyrusStandardizer(),
            identifier=InchiIdentifier(),
        )
        store.addLibrary(
            f"{store.name}_2",
            pd.read_csv(self.exampleFileIndex),
            smiles_col="smiles",
        )
        self.main = store

    def getStorage(self) -> PandasRepresentationStore:
        return PandasRepresentationStore(
            f"{self.__class__.__name__}_test_basic",
            path=self.outputPath,
            chem_store=self.main,
        )

    def addConformers(self, store):
        # generate conformers for each molecule in main store
        parent_ids = []
        sdfs = []
        smiles = []
        for mol in self.main:
            rd_mol = mol.as_rd_mol()
            rd_mol = Chem.AddHs(rd_mol)
            EmbedMultipleConfs(
                rd_mol,
                numConfs=3,
                randomSeed=42,
                pruneRmsThresh=0.5,
            )
            # get SDFs for each conformer
            for conf in rd_mol.GetConformers():
                parent_ids.append(mol.id)
                smiles.append(mol.smiles)
                sdfs.append(Chem.MolToMolBlock(rd_mol, confId=conf.GetId()))
        # add them as representations
        store.addMols(smiles, {
            "parent_id": parent_ids,
            "sdf": sdfs,
        })
        self.assertEqual(len(store.representations), len(parent_ids))

    def testAddConformers(self):
        store = self.getStorage()
        self.addConformers(store)
        for mol in store:
            self.assertTrue(mol.representations)
            for rep in mol.representations:
                mol = rep.as_rd_mol()
                self.assertTrue(mol)
                self.assertFalse(rep.representations)

    @staticmethod
    def check_representations(mols):
        ret = []
        for mol in mols:
            reps = mol.representations
            for rep in reps:
                assert isinstance(rep, RepresentationMol)
                ret.append(rep.as_rd_mol())
        return ret

    def testParallel(self):
        store = self.getStorage()
        self.addConformers(store)
        # iterate in parallel and check that all conformers are valid rdkit molecules
        store.nJobs = 2
        for result in store.apply(self.check_representations):
            for mol in result:
                self.assertIsInstance(mol, Chem.Mol)
                self.assertTrue(mol)
