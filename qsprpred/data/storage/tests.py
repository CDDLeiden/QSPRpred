import os
import shutil
from abc import ABC, abstractmethod
from unittest import TestCase

import pandas as pd

from qsprpred.data.chem.identifiers import InchiIdentifier
from qsprpred.data.chem.standardizers.check_smiles import CheckSmilesValid
from qsprpred.data.chem.standardizers.papyrus import PapyrusStandardizer
from qsprpred.data.storage.interfaces.chem_store import ChemStore
from qsprpred.data.storage.tabular.basic_storage import TabularStorageBasic


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

    def getStorage(self) -> TabularStorageBasic:
        store = TabularStorageBasic(
            f"{self.__class__.__name__}_test_basic",
            self.outputPath,
            pd.read_csv(self.exampleFileBasic),
            standardizer=PapyrusStandardizer(),
            identifier=InchiIdentifier(),
        )
        store.add_library(
            f"{store.name}_2",
            pd.read_csv(self.exampleFileIndex),
            smiles_col="smiles",
        )
        return store

    def checkSerialization(self, store):
        store.save()
        # create new and check consistency
        store2 = TabularStorageBasic(store.name, self.outputPath)
        self.assertEqual(store2.nLibs, store.nLibs)
        self.assertEqual(len(store2), len(store))
        self.assertListEqual(list(store2.smiles), list(store.smiles))
        # create from meta file and check consistency
        store2 = TabularStorageBasic.fromFile(store.metaFile)
        self.assertEqual(store2.nLibs, store.nLibs)
        self.assertEqual(len(store2), len(store))
        self.assertListEqual(list(store2.smiles), list(store.smiles))
        # add a new library and check consistency after reload
        len_before = len(store)
        added = store.add_mols(
            ["CN1[C@H]2CC[C@@H]1[C@@H](C(OC)=O)[C@@H](OC(C3=CC=CC=C3)=O)C2"],
        )
        self.assertEqual(len(added), 1)
        self.assertEqual(len(store), len_before + 1)
        store.reload()
        self.assertEqual(store.nLibs, store2.nLibs)
        self.assertEqual(len(store), len(store2))
        self.assertListEqual(list(store.smiles), list(store2.smiles))

    def testInitsAndSaves(self):
        # test default
        store_default = TabularStorageBasic(
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
        self.assertRaises(ValueError, lambda: store_default.add_library(
            store_default.name,
            pd.read_csv(self.exampleFileBasic),
        ))
        # add a library with duplicated molecules
        store_default.add_library(
            f"{store_default.name}_2",
            pd.read_csv(self.exampleFileBasic),
        )
        self.assertEqual(store_default.nLibs, 1)
        self.assertEqual(len(store_default), 2)
        self.checkSerialization(store_default)
        # add a new library with additional compounds
        store_default.add_library(
            f"{store_default.name}_3",
            pd.read_csv(self.exampleFileIndex),
            smiles_col="smiles",
        )
        self.assertEqual(store_default.nLibs, 2)
        self.assertEqual(len(store_default), 3)
        self.checkSerialization(store_default)
        # test empty init
        store_empty = TabularStorageBasic(
            f"{self.__class__.__name__}_test_empty",
            self.outputPath,
            standardizer=PapyrusStandardizer(),
            identifier=InchiIdentifier(),
        )
        self.assertEqual(store_empty.nLibs, 0)
        self.assertEqual(len(store_empty), 0)
        self.checkSerialization(store_empty)
        # test with defaults
        df = pd.read_csv(self.exampleFileIndex)
        store_default = TabularStorageBasic(
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
        TabularStorageBasic.fromDF(
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
        added = store.add_mols(
            ["O=C(OCCN(CC)CC)c1ccc(N)cc1"],
        )
        self.assertEqual(len(added), 1)
        self.assertEqual(len(store), len_before + 1)
        self.checkSerialization(store)
        # add to a new library
        store.add_mols(
            ["O=C(OC(C)CN(CC)CC)c1ccc(N)cc1"],
            library=f"{store.name}_2",
        )
        self.assertEqual(store.nLibs, 2)
        self.assertEqual(len(store), len_before + 2)
        self.checkSerialization(store)
        # add with new properties
        mols = store.add_mols(
            [
                "O=C(OC(CCC)CN(CC)CC)c1ccc(N)cc1",
                "O=C(OC(CCC)CN(CC)CC)c1ccc(N)cc1C"
            ],
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
        mols = store.add_mols(
            [
                "O=C(OC(CCC)CN(CC)CC)c1ccc(N)cc1C(C)C",
                "O=C(OC(CCC)CN(C(C)C)CC)c1ccc(N)cc1C"
            ],
            props={"TestProp1": [3, 4], "TestProp2": [5, 6]},
        )
        self.assertEqual(len(mols), 2)
        for mol in store:
            self.assertIn("TestProp1", mol.props)
            self.assertIn("TestProp2", mol.props)
        self.assertEqual(len(store), len_before + 6)
        self.checkSerialization(store)
        # add with existing properties and new ones
        mols = store.add_mols(
            [
                "O=C(OC(C(O)C)CN(CC)CC)c1ccc(N)cc1C(C)C",
                "O=C(OC(CC(N)C)CN(C(C)C)CC)c1ccc(N)cc1C"
            ],
            props={"TestProp1": [3, 4], "TestProp2": [5, 6], "new_prop": [7, 8]},
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
        self.assertEqual(len(result), 2)
        result = pd.concat(result)
        self.assertEqual(len(result), len(store))
        self.assertTrue(all(result))
        for idx in result.index:
            self.assertTrue(idx in store)

    def testSubsetting(self):
        store = self.getStorage()
        mol_1 = [x for x in store][0]
        mol_2 = [x for x in store][1]
        subset = store.getSubset(["TestProp1", "ExtraIndexColumn"])
        self.assertEqual(len(subset), len(store))
        for mol in [mol_1, mol_2]:
            self.assertIn(mol.id, subset)
            self.assertIn("TestProp1", subset[mol.id].props)
            self.assertIn("ExtraIndexColumn", subset[mol.id].props)
        subset = store.getSubset(
            ["TestProp1", "ExtraIndexColumn"],
            [mol_2.id, mol_1.id]
        )
        self.assertEqual(len(subset), 2)
        for mol in [mol_1, mol_2]:
            self.assertIn(mol.id, subset)
            self.assertIn("TestProp1", subset[mol.id].props)
            self.assertIn("ExtraIndexColumn", subset[mol.id].props)
