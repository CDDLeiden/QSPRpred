import os
import shutil
import tempfile
from copy import deepcopy

import numpy as np
import pandas as pd
from parameterized import parameterized
from sklearn.model_selection import KFold, ShuffleSplit

from qsprpred.data.descriptors.sets import DrugExPhyschem
from .interfaces.qspr_data_set import QSPRDataSet
from .mol import MoleculeTable
from ..chem.standardizers.papyrus import PapyrusStandardizer
from ..descriptors.fingerprints import MorganFP
from ..processing.data_filters import CategoryFilter, NaNFilter
from ..processing.pipeline import DatasetPipeline
from ..processing.step import Shuffle, DummyStep
from ... import TargetSpec, TargetTasks
from ...data.storage.tabular.simple import PandasChemStore
from ...data.tables.qspr import QSPRTable
from ...utils.stopwatch import StopWatch
from ...utils.testing.base import QSPRTestCase
from ...utils.testing.check_mixins import DataPrepCheckMixIn
from ...utils.testing.path_mixins import DataSetsPathMixIn, PathMixIn


class TestMolTable(DataSetsPathMixIn, QSPRTestCase):
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def getStorage(self):
        df = self.getSmallDF()
        return PandasChemStore(
            "test",
            self.generatedDataPath,
            df,
            standardizer=PapyrusStandardizer(),
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )

    def getTable(self):
        storage = self.getStorage()
        return MoleculeTable(storage, path=self.generatedDataPath)

    def testTableCreation(self):
        """Test the creation of a table from a data set."""
        storage = self.getStorage()
        mt = MoleculeTable(storage, path=self.generatedDataPath)
        self.assertEqual(len(mt), len(storage))
        # from SMILES
        mt = MoleculeTable.fromSMILES(
            f"{mt.name}_from_smiles",
            list(storage.smiles),
            path=self.generatedDataPath,
            standardizer=PapyrusStandardizer(),
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        self.assertEqual(len(mt), len(storage))
        # from table file
        mt = MoleculeTable.fromTableFile(
            f"{mt.name}_from_file",
            f"{self.inputDataPath}/test_data.tsv",
            path=self.generatedDataPath,
        )
        self.assertEqual(len(mt), len(storage))

    def testTableSerialization(self):
        mt = self.getTable()
        mt.save()
        self.assertTrue(os.path.exists(mt.metaFile))
        mt_new = MoleculeTable.fromFile(mt.metaFile)
        self.assertEqual(len(mt), len(mt_new))
        self.assertListEqual(list(mt.smiles), list(mt_new.smiles))
        # see if we can still reload even if we move the files to a different location
        random_new_folder = tempfile.mkdtemp()
        shutil.move(self.generatedDataPath, random_new_folder)
        mt_moved = MoleculeTable.fromFile(
            os.path.join(random_new_folder, "datasets", mt.name, "meta.json")
        )
        self.assertEqual(len(mt), len(mt_moved))
        self.assertListEqual(list(mt.smiles), list(mt_moved.smiles))

    @staticmethod
    def getDescriptorSets():
        return [MorganFP(radius=2, nBits=128), DrugExPhyschem()]

    def testDescriptors(self):
        # add descriptors
        mt = self.getTable()
        mt.addDescriptors(self.getDescriptorSets())
        self.assertEqual(len(mt.descriptors), len(self.getDescriptorSets()))
        mt.save()
        mt_new = MoleculeTable.fromFile(mt.metaFile)
        self.assertEqual(len(mt_new.descriptors), len(self.getDescriptorSets()))
        # move the files and check if we can still reload
        random_new_folder = tempfile.mkdtemp()
        shutil.move(self.generatedDataPath, random_new_folder)
        mt_moved = MoleculeTable.fromFile(
            os.path.join(random_new_folder, "datasets", mt.name, "meta.json")
        )
        self.assertEqual(len(mt_moved.descriptors), len(mt.descriptors))
        # drop descriptors
        df_descriptors = mt.getDescriptors()
        old_shape = df_descriptors.shape
        all_descriptors = mt.getDescriptorNames()
        mt_moved.dropDescriptors([all_descriptors[0], all_descriptors[-1]])
        self.assertEqual(mt_moved.getDescriptors().shape[0], len(mt_moved))
        self.assertEqual(
            mt_moved.getDescriptors().shape[1], len(mt_moved.getDescriptorNames())
        )
        new_shape = mt_moved.getDescriptors().shape
        self.assertEqual(new_shape[1], old_shape[1] - 2)
        self.assertTrue(new_shape[0] == old_shape[0])
        self.assertTrue(all_descriptors[0] not in mt_moved.getDescriptorNames())
        self.assertTrue(all_descriptors[-1] not in mt_moved.getDescriptorNames())
        # drop a descriptor set
        old_shape = new_shape
        ds = mt_moved.descriptorSets[0]
        n_removed = len(ds.descriptors)
        mt_moved.dropDescriptorSets([ds])
        new_shape = mt_moved.getDescriptors().shape
        self.assertEqual(new_shape[1], old_shape[1] - n_removed)
        self.assertEqual(new_shape[0], old_shape[0])
        self.assertTrue(ds in mt_moved.descriptorSets)
        self.assertEqual(len(ds.descriptors), 0)
        mt_moved.restoreDescriptorSets([ds])
        new_shape = mt_moved.getDescriptors().shape
        self.assertEqual(new_shape[0], old_shape[0])
        self.assertEqual(new_shape[1], old_shape[1] + 1)
        mt_moved.dropDescriptorSets([ds])
        new_shape = mt_moved.getDescriptors().shape
        # try save and reload
        mt_moved.save()
        mt_moved = MoleculeTable.fromFile(mt_moved.metaFile)
        self.assertEqual(mt_moved.getDescriptors().shape[1], new_shape[1])
        self.assertEqual(len(mt_moved.descriptorSets), len(mt.descriptorSets))
        self.assertEqual(len(ds.descriptors), 0)
        # drop completely
        mt_moved.dropDescriptorSets([ds], full_removal=True)
        self.assertEqual(len(mt_moved.descriptorSets), len(mt.descriptorSets) - 1)
        # try save and reload
        mt_moved.save()
        mt_moved = MoleculeTable.fromFile(mt_moved.metaFile)
        self.assertEqual(len(mt_moved.descriptorSets), len(mt.descriptorSets) - 1)
        # try restore
        self.assertRaises(ValueError, lambda: mt_moved.restoreDescriptorSets([ds]))
        # drop a descriptor and restore with reload
        old_shape = mt_moved.getDescriptors().shape
        mt_moved.dropDescriptors(mt_moved.getDescriptorNames()[0:2])
        new_shape = mt_moved.getDescriptors().shape
        self.assertEqual(new_shape[1], old_shape[1] - 2)
        mt_moved.reload()
        self.assertEqual(mt_moved.getDescriptors().shape[1], old_shape[1])

    def testSubsetting(self):
        mt = self.getTable()
        mt.addDescriptors(self.getDescriptorSets())
        # get subset for all ids
        mt_sub = mt.getSubset(mt.getProperties(), path=self.generatedDataPath)
        self.assertEqual(len(mt), len(mt_sub))
        self.assertListEqual(list(mt.smiles), list(mt_sub.smiles))
        self.assertListEqual(list(mt.getProperties()), list(mt_sub.getProperties()))
        self.assertEqual(mt.getDescriptors().shape, mt_sub.getDescriptors().shape)
        mt_sub.save()
        self.assertTrue(os.path.exists(mt_sub.metaFile))
        self.assertTrue(os.path.exists(mt_sub.storage.metaFile))
        new = MoleculeTable.fromFile(mt_sub.metaFile)
        self.assertEqual(len(mt_sub), len(new))
        self.assertListEqual(list(mt_sub.smiles), list(new.smiles))
        self.assertListEqual(list(mt_sub.getProperties()), list(new.getProperties()))
        # move the files and check if we can still reload
        random_new_folder = tempfile.mkdtemp()
        shutil.move(self.generatedDataPath, random_new_folder)
        mt_moved = MoleculeTable.fromFile(
            os.path.join(random_new_folder, "datasets", mt_sub.name, "meta.json")
        )
        self.assertEqual(len(mt_sub), len(mt_moved))
        self.assertListEqual(list(mt_sub.smiles), list(mt_moved.smiles))
        self.assertListEqual(
            list(mt_sub.getProperties()), list(mt_moved.getProperties())
        )
        # check sampling
        mt_sample = mt.sample(5)
        self.assertEqual(len(mt_sample), 5)
        # drop entries
        mt_sample.dropEntries(mt_sample.getProperty(mt_sample.idProp)[0:2])
        self.assertEqual(len(mt_sample), 3)
        self.assertEqual(mt_sample.getDescriptors().shape[0], len(mt_sample))
        self.assertEqual(
            mt_sample.getDescriptors().shape[1], len(mt_sample.getDescriptorNames())
        )


class TestQSPRTable(DataSetsPathMixIn, QSPRTestCase):
    """Simple tests for dataset creation and serialization under different conditions
    and error states."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def checkConsistency(self, ds: QSPRDataSet):
        self.assertNotIn("Notes", ds.getProperties())
        self.assertNotIn("HBD", ds.getProperties())
        self.assertTrue(len(self.getSmallDF()) - 1 == len(ds))
        self.assertEqual(ds.targetProperties[0].task, TargetTasks.REGRESSION)
        self.assertTrue(ds.hasProperty("CL"))
        self.assertEqual(ds.targetProperties[0].name, "CL")
        self.assertEqual(len(ds.getDescriptors()), len(ds))
        self.assertEqual(len(ds.getTargets()), len(ds))
        self.assertEqual(ds.getDescriptors().shape[1], 128)

    def checkConsistencyMulticlass(self, ds):
        self.assertTrue(ds.isMultiTask)
        self.assertEqual(ds.nTargetProperties, 2)
        self.assertEqual(len(ds.targetProperties), 2)
        self.assertEqual(ds.targetProperties[0].name, "CL")
        self.assertEqual(ds.targetProperties[1].name, "fu")
        self.assertEqual(ds.targetProperties[0].task, TargetTasks.REGRESSION)
        self.assertEqual(ds.targetProperties[1].task, TargetTasks.REGRESSION)
        self.assertEqual(len(ds.getDescriptors()), len(ds))
        self.assertEqual(len(ds.getTargets()), len(ds))
        self.assertEqual(len(ds.getTargets().columns), 2)
        self.assertEqual(ds.getTargets().columns[0], "CL")
        self.assertEqual(ds.getTargets().columns[1], "fu")

    def checkConsistencySingleclass(self, ds):
        self.assertFalse(ds.isMultiTask)
        self.assertEqual(ds.nTargetProperties, 1)
        self.assertEqual(len(ds.targetProperties), 1)
        self.assertEqual(ds.targetProperties[0].name, "CL")
        self.assertEqual(ds.targetProperties[0].task, TargetTasks.REGRESSION)
        self.assertEqual(len(ds.getDescriptors()), len(ds))
        self.assertEqual(len(ds.getTargets()), len(ds))
        self.assertEqual(len(ds.getTargets().columns), 1)
        self.assertEqual(ds.getTargets().columns[0], "CL")

    def checkBadInit(self, ds):
        ds_copy = deepcopy(ds)
        with self.assertRaises(AssertionError):
            ds_copy.makeClassification("CL", [])
        with self.assertRaises(AssertionError):
            ds_copy.makeClassification("CL", th=6.5)
        with self.assertRaises(AssertionError):
            ds_copy.makeClassification("CL", th=[0, 2, 3])
        self.assertEqual(len(ds_copy.targetProperties), len(ds.targetProperties))
        for tp, tp_copy in zip(ds.targetProperties, ds_copy.targetProperties):
            self.assertEqual(tp.name, tp_copy.name)
            self.assertEqual(tp.task, tp_copy.task)
            if tp.task.isClassification():
                self.assertEqual(tp.th, tp_copy.th)
                self.assertEqual(tp.nClasses, tp_copy.nClasses)

    def checkClassification(self, ds, target_names, ths):
        # Test that the dataset properties are correctly initialized
        self.assertTrue(len(ds.targetProperties) == len(target_names) == len(ths))
        for idx, target_prop in enumerate(ds.targetProperties):
            if ths[idx] is None:
                if target_prop.task == TargetTasks.MULTICLASS:
                    self.assertGreaterEqual(target_prop.nClasses, 3)
                else:
                    self.assertEqual(target_prop.nClasses, 2)
            elif len(ths[idx]) == 1:
                self.assertEqual(target_prop.task, TargetTasks.SINGLECLASS)
            else:
                self.assertEqual(target_prop.task, TargetTasks.MULTICLASS)
            self.assertEqual(target_prop.name, target_names[idx])
            y = ds.getTargets()
            self.assertTrue(y.columns[idx] == target_prop.name)
            if target_prop.task == TargetTasks.SINGLECLASS:
                self.assertEqual(y[target_prop.name].unique().shape[0], 2)
            elif ths[idx] != None:
                self.assertEqual(
                    y[target_prop.name].unique().shape[0], (len(ths[idx]) - 1)
                )
            self.assertEqual(target_prop.th, ths[idx])

    def checkRegression(self, ds, target_names):
        self.assertTrue(len(ds.targetProperties) == len(target_names))
        for idx, target_prop in enumerate(ds.targetProperties):
            self.assertEqual(target_prop.task, TargetTasks.REGRESSION)
            self.assertTrue(ds.hasProperty(target_names[idx]))
            self.assertEqual(target_prop.name, target_names[idx])

    def testDefaults(self):
        """Test basic dataset creation and serialization with mostly default options."""
        # create a basic regression data set
        storage = self.getStorage(
            self.getSmallDF(),
            "test_defaults_storage",
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        dataset = QSPRTable(
            storage,
            "test_defaults",
            [{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
            path=self.generatedDataPath,
        )
        dataset.addDescriptors([MorganFP(radius=2, nBits=128)])
        self.assertIn("HBD", dataset.getProperties())
        dataset.removeProperty("HBD")
        self.assertNotIn("HBD", dataset.getProperties())
        stopwatch = StopWatch()
        dataset.save()
        stopwatch.stop("Saving took: ")
        self.assertTrue(os.path.exists(dataset.metaFile))
        # load the data set again and check if everything is consistent after loading
        # creation from file
        stopwatch.reset()
        dataset_new = QSPRTable.fromFile(dataset.metaFile)
        stopwatch.stop("Loading from file took: ")
        self.checkConsistency(dataset_new)
        # creation by reinitialization
        stopwatch.reset()
        dataset_new = QSPRTable.fromFile(dataset.metaFile)
        stopwatch.stop("Reinitialization took: ")
        self.checkConsistency(dataset_new)
        # creation from a table file
        stopwatch.reset()
        dataset_new = QSPRTable.fromTableFile(
            "test_defaults",
            f"{self.inputDataPath}/test_data.tsv",
            path=self.generatedDataPath,
            target_props=[{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
        )
        stopwatch.stop("Loading from table file took: ")
        self.assertTrue(isinstance(dataset_new, QSPRTable))
        self.checkConsistency(dataset_new)
        # creation from a table file with a new name
        dataset_new = QSPRTable.fromTableFile(
            "test_defaults_new",  # new name implies HBD below should exist again
            f"{self.inputDataPath}/test_data.tsv",
            target_props=[{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
            path=self.generatedDataPath,
        )
        self.assertTrue(isinstance(dataset_new, QSPRTable))
        self.assertIn("HBD", dataset_new.getProperties())
        dataset_new.removeProperty("HBD")
        self.assertEqual(dataset_new.getDescriptors().shape[1], 0)
        dataset_new.addDescriptors([MorganFP(radius=2, nBits=128)])
        self.checkConsistency(dataset_new)
        # test subset creation
        descriptors = dataset_new.getDescriptors()
        targets = dataset_new.getTargets()
        subset = dataset_new.getSubset(["CL"], path=self.generatedDataPath)
        props = subset.getProperties()
        self.assertIn("CL", props)
        self.assertNotIn("HBD", props)
        self.assertNotIn("Notes", props)
        self.assertEqual(len(subset), len(dataset_new))
        descriptors_new, targets_new = subset.getDescriptors(), subset.getTargets()
        self.assertTrue(np.allclose(descriptors_new, descriptors))
        self.assertTrue(np.allclose(targets_new, targets))
        # subset only first two ids
        subset = dataset_new.getSubset(
            ["CL"], ids=list(dataset_new.getProperty(dataset_new.idProp)[0:2])
        )
        self.assertEqual(len(subset), 2)
        descriptors_subset = subset.getDescriptors()
        targets_subset = subset.getTargets()
        self.assertEqual(descriptors_subset.shape[0], 2)
        self.assertEqual(targets_subset.shape[0], 2)
        self.assertListEqual(
            list(descriptors_subset.index),
            list(descriptors.iloc[0:2, :].index),
        )
        self.assertListEqual(
            list(targets_subset.index),
            list(targets.iloc[0:2, :].index)
        )
        self.assertTrue(
            np.allclose(
                descriptors_subset,
                descriptors.iloc[0:2, :],
            )
        )
        self.assertTrue(
            np.allclose(targets_subset, targets.iloc[0:2, :])
        )

    def testMultitask(self):
        """Test multi-task dataset creation and functionality."""
        storage = self.getStorage(
            self.getSmallDF(),
            "testMultitask_storage",
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        dataset = QSPRTable(
            storage,
            "testMultitask",
            [
                {
                    "name": "CL",
                    "task": TargetTasks.REGRESSION
                },
                {
                    "name": "fu",
                    "task": TargetTasks.REGRESSION
                },
            ],
            path=self.generatedDataPath,
        )
        # Check that the dataset is correctly initialized
        self.checkConsistencyMulticlass(dataset)
        # Check the dataset after dropping a task
        dataset.unsetTargetProperty("fu")
        self.checkConsistencySingleclass(dataset)
        with self.assertRaises(AssertionError):
            dataset.unsetTargetProperty("fu")
        with self.assertRaises(AssertionError):
            dataset.unsetTargetProperty("CL")
        # Check the dataset after adding a task
        dataset.addTargetProperty({"name": "fu", "task": TargetTasks.REGRESSION})
        self.checkConsistencyMulticlass(dataset)

    def testTargetProperty(self):
        """Test target property creation and serialization
        in the context of a dataset.
        """
        storage = self.getStorage(
            self.getSmallDF(),
            "test_targets_storage",
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        dataset = QSPRTable(
            storage,
            "testTargetProperty",
            [
                {
                    "name": "CL",
                    "task": TargetTasks.REGRESSION
                },
                {
                    "name": "fu",
                    "task": TargetTasks.REGRESSION
                },
            ],
            path=self.generatedDataPath,
        )
        # Check that the make classification method works as expected
        self.checkBadInit(dataset)
        dataset.makeClassification("CL", th=[6.5])
        dataset.makeClassification("fu", th=[0.3])
        self.checkClassification(dataset, ["CL", "fu"], [[6.5], [0.3]])
        dataset.makeClassification("CL", th=[0, 15, 30, 60])
        self.checkClassification(dataset, ["CL", "fu"], [[0, 15, 30, 60], [0.3]])
        dataset.save()
        # check precomputed threshold setting
        df_new = storage.getDF().copy()
        del df_new["CL_original"]
        storage = self.getStorage(
            df_new,
            "test_targets_storage_precomputed",
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        dataset = QSPRTable(
            storage,
            "testTargetProperty-precomputed",
            [{
                "name": "CL",
                "task": TargetTasks.MULTICLASS,
                "th": None,
                "n_classes": 3
            }],
            path=self.generatedDataPath,
        )
        self.assertEqual(len(dataset.targetProperties), 1)
        self.assertEqual(dataset.targetProperties[0].task, TargetTasks.MULTICLASS)
        self.assertEqual(dataset.targetProperties[0].name, "CL")
        self.assertEqual(dataset.targetProperties[0].nClasses, 3)
        self.assertEqual(dataset.targetProperties[0].th, None)
        # Check that the dataset is correctly loaded from file for classification
        dataset.save()
        dataset_new = QSPRTable.fromFile(dataset.metaFile)
        self.checkBadInit(dataset_new)
        self.checkClassification(dataset_new, ["CL"], [None])
        # Check that the make regression method works as expected
        dataset_new.makeRegression(target_property="CL")
        # Check that the dataset is correctly loaded from file for regression
        self.checkRegression(dataset_new, ["CL"])
        dataset_new.save()
        dataset_new = QSPRTable.fromFile(dataset.metaFile)
        self.checkRegression(dataset_new, ["CL"])

    def testRandomStateSplit(self):
        # create and save the data set
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors(
            [MorganFP(radius=2, nBits=128)],
            featurize=False,
        )
        dataset.save()
        # shuffle and split
        split = ShuffleSplit(1, test_size=0.5, random_state=dataset.randomState)
        dataset.addSplit(split, "shufflesplit")
        _, _, _, _ = next(dataset.iterSplit("shufflesplit", as_type="numpy"))
        _, _ = next(dataset.iterSplit("shufflesplit", as_type="QSPRTable"))
        train, test = next(dataset.iterSplit("shufflesplit", as_type="ids"))
        # reload and check if orders are the same if we redo the split
        # with the same random state
        dataset = QSPRTable.fromFile(dataset.metaFile)
        split = ShuffleSplit(1, test_size=0.5, random_state=dataset.randomState)
        dataset.addSplit(split, "shufflesplit2")
        train2, test2 = next(dataset.iterSplit("shufflesplit2", as_type="ids"))
        self.assertListEqual(train, train2)
        self.assertListEqual(test, test2)

    def testRandomStateFolds(self):
        # create and save the data set (fixes the seed)
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors(
            [MorganFP(radius=2, nBits=128)],
            featurize=False,
        )
        dataset.save()
        # calculate descriptors and iterate over folds
        order_train = dataset.getDescriptors().index.tolist()
        order_folds = []
        split = KFold(5, shuffle=True, random_state=dataset.randomState)
        dataset.addSplit(split, "kfold")
        for ids in dataset.getSplit("kfold", as_type="ids"):
            order_folds.append(ids)
        # reload and check if orders are the same if we redo the folds from saved data
        dataset = QSPRTable.fromFile(dataset.metaFile)
        self.assertListEqual(dataset.getDescriptors().index.tolist(), order_train)
        split = KFold(5, shuffle=True, random_state=dataset.randomState)
        dataset.addSplit(split, "kfold2")
        for i, (train_index, test_index) in enumerate(
                dataset.getSplit("kfold2", as_type="ids")):
            self.assertListEqual(train_index, order_folds[i][0])
            self.assertListEqual(test_index, order_folds[i][1])

    def testFilter(self):
        """Test removing entries from the dataset using a DataFilter."""
        dataset = self.createLargeTestDataSet()
        remove_cation = CategoryFilter(
            prop="moka_ionState7.4",
            values=["cationic"],
            data_set=dataset
        )
        self.assertTrue((dataset.getDF()["moka_ionState7.4"] == "cationic").sum() > 0)
        dataset.filter([remove_cation])
        self.assertEqual(len(dataset.getDF()), len(dataset.getDescriptors()))
        self.assertTrue((dataset.getDF()["moka_ionState7.4"] == "cationic").sum() == 0)


class TestSearchFeatures(DataSetsPathMixIn, QSPRTestCase):
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def validateSearch(self, dataset: QSPRDataSet, result: QSPRDataSet, name: str):
        """Validate the results of a search."""
        self.assertTrue(len(result) < len(dataset))
        self.assertTrue(isinstance(result, type(dataset)))
        self.assertEqual(result.name, name)
        self.assertListEqual(dataset.getProperties(), result.getProperties())
        self.assertListEqual(dataset.getDescriptorNames(), result.getDescriptorNames())
        self.assertListEqual(dataset.getTargetPropertiesNames(),
                             result.getTargetPropertiesNames())
        self.assertEqual(len(dataset.descriptorSets), len(result.descriptorSets))
        self.assertEqual(len(dataset.targetProperties), len(result.targetProperties))

    def testSMARTS(self):
        dataset = self.createLargeTestDataSet()
        search_name = "search_name"
        results_and = dataset.searchWithSMARTS(
            ["c1ccccc1", "S(=O)(=O)"],
            operator="and",
            name=search_name,
        )
        self.assertTrue(all("S" in x for x in results_and.smiles))
        self.validateSearch(dataset, results_and, search_name)
        results_or = dataset.searchWithSMARTS(
            ["c1ccccc1", "S"],
            operator="or",
            name=search_name,
        )
        self.validateSearch(dataset, results_or, search_name)
        self.assertFalse(all("S" in x for x in results_or.smiles))
        self.assertTrue(any("S" in x for x in results_or.smiles))
        self.assertTrue(len(results_and) < len(results_or))

    def testPropSearch(self):
        dataset = self.createLargeTestDataSet()
        search_name = "search_name"
        results = dataset.searchOnProperty(
            "moka_ionState7.4",
            ["cationic"],
            name=search_name,
            exact=True,
        )
        self.validateSearch(dataset, results, search_name)
        self.assertTrue(
            all(x == "cationic" for x in results.getProperty("moka_ionState7.4"))
        )
        results = dataset.searchOnProperty(
            "Reference",
            ["Cook"],
            name=search_name,
            exact=False,
        )
        self.validateSearch(dataset, results, search_name)
        self.assertTrue(all("Cook" in x for x in results.getProperty("Reference")))
        results = dataset.searchOnProperty(
            "Reference",
            ["Cook"],
            name=search_name,
            exact=True,
        )
        self.assertTrue(len(results) == 0)


class TestTargetSpec(QSPRTestCase):
    """Test the TargetSpec class."""

    def checkTargetSpec(self, target_spec, name, task, th, n_classes=None):
        # Check the target spec creation consistency
        self.assertEqual(target_spec.name, name)
        self.assertEqual(target_spec.task, task)
        if task.isClassification():
            self.assertTrue(target_spec.task.isClassification())
            self.assertEqual(target_spec.th, th)
            self.assertEqual(target_spec.nClasses, n_classes)

    def testInit(self):
        """Check the TargetSpec class on target spec creation."""
        # Check the different task types
        target_spec = TargetSpec("CL", TargetTasks.REGRESSION)
        self.checkTargetSpec(target_spec, "CL", TargetTasks.REGRESSION, None)
        target_spec = TargetSpec("CL", TargetTasks.MULTICLASS, th=[0, 1, 10, 1200])
        self.checkTargetSpec(
            target_spec, "CL", TargetTasks.MULTICLASS, [0, 1, 10, 1200], 3
        )
        target_spec = TargetSpec("CL", TargetTasks.SINGLECLASS, th=[5])
        self.checkTargetSpec(target_spec, "CL", TargetTasks.SINGLECLASS, [5], 2)
        target_spec = TargetSpec("CL", TargetTasks.SINGLECLASS, n_classes=2)
        self.checkTargetSpec(target_spec, "CL", TargetTasks.SINGLECLASS, None, 2)
        # check if incorrect task raises an error
        with self.assertRaises(AssertionError):
            TargetSpec("CL", TargetTasks.SINGLECLASS, th=[5], n_classes=2)
        with self.assertRaises(AssertionError):
            TargetSpec("CL", TargetTasks.SINGLECLASS, th=5)
        with self.assertRaises(AssertionError):
            TargetSpec("CL", TargetTasks.SINGLECLASS, th=[])
        with self.assertRaises(AssertionError):
            TargetSpec("CL", TargetTasks.SINGLECLASS, th=[5, 6])
        with self.assertRaises(AssertionError):
            TargetSpec("CL", TargetTasks.MULTICLASS, th=[5, 6])
        with self.assertRaises(AssertionError):
            TargetSpec("CL", TargetTasks.SINGLECLASS, th=[0, 1, 10, 1200])

        # Check from dictionary creation
        targetprop = TargetSpec.fromDict(
            {
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }
        )
        self.checkTargetSpec(targetprop, "CL", TargetTasks.REGRESSION, None)
        targetprop = TargetSpec.fromDict(
            {
                "name": "CL",
                "task": TargetTasks.MULTICLASS,
                "th": [0, 1, 10, 1200]
            }
        )
        self.checkTargetSpec(
            targetprop, "CL", TargetTasks.MULTICLASS, [0, 1, 10, 1200], 3
        )
        # Check from list creation, selection and serialization support functions
        targetprops = TargetSpec.fromList(
            [
                {
                    "name": "CL",
                    "task": TargetTasks.REGRESSION
                },
                {
                    "name": "fu",
                    "task": TargetTasks.REGRESSION
                },
            ]
        )
        self.checkTargetSpec(targetprops[0], "CL", TargetTasks.REGRESSION, None)
        self.checkTargetSpec(targetprops[1], "fu", TargetTasks.REGRESSION, None)
        self.assertListEqual(TargetSpec.getNames(targetprops), ["CL", "fu"])
        targetprops = TargetSpec.toList(targetprops)
        self.assertIsInstance(targetprops, list)
        self.assertIsInstance(targetprops[0], dict)
        self.assertEqual(targetprops[0]["name"], "CL")
        self.assertEqual(targetprops[0]["task"], TargetTasks.REGRESSION)

    @parameterized.expand(
        [
            (TargetTasks.REGRESSION, "CL", None),
            (TargetTasks.MULTICLASS, "CL", [0, 1, 10, 1200]),
        ]
    )
    def testSerialization(self, task, name, th):
        spec = TargetSpec(name, task, th=th)
        json_form = spec.toJSON()
        spec_from_json = TargetSpec.fromJSON(json_form)
        self.assertEqual(spec_from_json.name, spec.name)
        self.assertEqual(spec_from_json.task, spec.task)
        if task.isClassification():
            self.assertEqual(spec_from_json.th, spec.th)
            self.assertEqual(spec_from_json.nClasses, spec.nClasses)


class TestDataSetPreProcessing(DataSetsPathMixIn, DataPrepCheckMixIn, QSPRTestCase):
    """Test as many possible combinations of data sets and their preparation
    settings. These can run potentially for a long time so use the ``skip`` decorator
    if you want to skip all these tests to speed things up during development."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    @parameterized.expand(DataSetsPathMixIn.getPrepCombos())
    def testPrepCombos(
            self,
            _,
            name,
            feature_calculators,
            split,
            feature_standardizer,
            feature_filter,
            data_filter,
            applicability_domain,
    ):
        """Tests one combination of a data set and its preparation settings.

        This generates a large number of parameterized tests. Use the ``skip`` decorator
        if you want to skip all these tests. Note that the combinations are not
        exhaustive, but defined by `DataSetsPathMixIn.getPrepCombos()`."""
        dataset = self.createLargeTestDataSet(name=name)
        pipeline = DatasetPipeline(
            feature_calculators=feature_calculators,
            steps={
                "shuffle": Shuffle(),
                "feature_standardizer": feature_standardizer if feature_standardizer else DummyStep(),
                "feature_filter": feature_filter if feature_filter else DummyStep(),
                "data_filter": data_filter if data_filter else DummyStep(),
                # The outlierfilter cannot handle NaN values, so a NaN filter has to be added
                # as the RDKit descriptors sometimes have NaN values
                "NaNFilter": NaNFilter() if applicability_domain else DummyStep(),
                "outlier_filter": applicability_domain if applicability_domain else DummyStep(),
            }
        )
        self.checkPrep(
            dataset,
            pipeline,
            split,
        )


class TestTargetImputation(PathMixIn, QSPRTestCase):
    """Small tests to only check if the target imputation works on its own."""

    def setUp(self):
        """Set up the test Dataframe."""
        super().setUp()
        self.setUpPaths()
        self.descriptors = [
            "Descriptor_F1",
            "Descriptor_F2",
            "Descriptor_F3",
            "Descriptor_F4",
            "Descriptor_F5",
        ]
        self.df = pd.DataFrame(
            data=np.array(
                [
                    ["C", 1, 4, 2, 6, 2, 1, 2],
                    ["C", 1, 8, 4, 2, 4, 1, 2],
                    ["C", 1, 4, 3, 2, 5, 1, np.NaN],
                    ["C", 1, 8, 4, 9, 8, 2, 2],
                    ["C", 1, 4, 2, 3, 9, 2, 2],
                    ["C", 1, 8, 4, 7, 12, 2, 2],
                ]
            ),
            columns=["SMILES", *self.descriptors, "y", "z"],
        )


class TestApply(DataSetsPathMixIn, QSPRTestCase):
    """Tests the apply method of the data set."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    @staticmethod
    def regularFunc(props, *args, **kwargs):
        df = pd.DataFrame(props)
        for idx, arg in enumerate(args):
            df[f"arg_{idx}"] = arg
        for key, value in kwargs.items():
            df[key] = value
        return df

    @parameterized.expand([(1, None), (2, None), (1, 25), (2, 25)])
    def testRegular(self, n_jobs, chunk_size):
        dataset = self.createLargeTestDataSet()
        dataset.nJobs = n_jobs
        dataset.chunkSize = chunk_size
        result = dataset.apply(
            self.regularFunc,
            func_args=[1, 2, 3],
            func_kwargs={
                "A_col": "A",
                "B_col": "B"
            },
            chunk_type="df",
        )
        for item in result:
            self.assertIsInstance(item, pd.DataFrame)
            self.assertTrue("CL" in item.columns)
            self.assertTrue("fu" in item.columns)
            self.assertTrue("A_col" in item.columns)
            self.assertTrue("B_col" in item.columns)
            self.assertTrue("arg_0" in item.columns)
            self.assertTrue("arg_1" in item.columns)
            self.assertTrue("arg_2" in item.columns)
            self.assertTrue(all(item["arg_0"] == 1))
            self.assertTrue(all(item["arg_1"] == 2))
            self.assertTrue(all(item["arg_2"] == 3))
            self.assertTrue(all(item["A_col"] == "A"))
            self.assertTrue(all(item["B_col"] == "B"))
