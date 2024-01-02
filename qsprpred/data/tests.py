"""This module holds the test for functions regarding QSPR data preparation."""

import copy
import os

import numpy as np
import pandas as pd
from parameterized import parameterized
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .chem.clustering import (
    FPSimilarityLeaderPickerClusters,
    FPSimilarityMaxMinClusters,
)
from .chem.scaffolds import BemisMurcko, Murcko
from .tables.qspr import QSPRDataset
from ..data.descriptors.calculators import (
    CustomDescriptorsCalculator,
    MoleculeDescriptorsCalculator,
)
from ..data.descriptors.sets import (
    DataFrameDescriptorSet,
    DrugExPhyschem,
    FingerprintSet,
    PredictorDesc,
    RDKitDescs,
    SmilesDesc,
    TanimotoDistances,
)
from ..data.processing.data_filters import CategoryFilter, RepeatsFilter
from ..data.processing.feature_filters import (
    BorutaFilter,
    HighCorrelationFilter,
    LowVarianceFilter,
)
from ..data.processing.feature_standardizers import SKLearnStandardizer
from ..data.sampling.folds import FoldsFromDataSplit
from ..data.sampling.splits import (
    BootstrapSplit,
    ClusterSplit,
    ManualSplit,
    RandomSplit,
    ScaffoldSplit,
    TemporalSplit,
)
from ..models.scikit_learn import SklearnModel
from ..tasks import TargetProperty, TargetTasks
from ..utils.stopwatch import StopWatch
from ..utils.testing.base import QSPRTestCase
from ..utils.testing.check_mixins import DataPrepCheckMixIn, DescriptorInDataCheckMixIn
from ..utils.testing.path_mixins import PathMixIn, DataSetsPathMixIn


# self.nCPU = 2
# self.chunkSize = 20
# TIME_SPLIT_YEAR = 2000
# logging.basicConfig(level=logging.DEBUG)


class TestDataSetCreationAndSerialization(DataSetsPathMixIn, QSPRTestCase):
    """Simple tests for dataset creation and serialization under different conditions
    and error states."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def checkConsistency(self, ds: QSPRDataset):
        self.assertNotIn("Notes", ds.getProperties())
        self.assertNotIn("HBD", ds.getProperties())
        self.assertTrue(len(self.getSmallDF()) - 1 == len(ds))
        self.assertEqual(ds.targetProperties[0].task, TargetTasks.REGRESSION)
        self.assertTrue(ds.hasProperty("CL"))
        self.assertEqual(ds.targetProperties[0].name, "CL")
        self.assertEqual(len(ds.X), len(ds))
        self.assertEqual(len(ds.X_ind), 0)
        self.assertEqual(len(ds.y), len(ds))
        self.assertEqual(len(ds.y_ind), 0)

    def checkConsistencyMulticlass(self, ds):
        self.assertTrue(ds.isMultiTask)
        self.assertEqual(ds.nTargetProperties, 2)
        self.assertEqual(len(ds.targetProperties), 2)
        self.assertEqual(ds.targetProperties[0].name, "CL")
        self.assertEqual(ds.targetProperties[1].name, "fu")
        self.assertEqual(ds.targetProperties[0].task, TargetTasks.REGRESSION)
        self.assertEqual(ds.targetProperties[1].task, TargetTasks.REGRESSION)
        self.assertEqual(len(ds.X), len(ds))
        self.assertEqual(len(ds.y), len(ds))
        self.assertEqual(len(ds.y.columns), 2)
        self.assertEqual(ds.y.columns[0], "CL")
        self.assertEqual(ds.y.columns[1], "fu")

    def checkConsistencySingleclass(self, ds):
        self.assertFalse(ds.isMultiTask)
        self.assertEqual(ds.nTargetProperties, 1)
        self.assertEqual(len(ds.targetProperties), 1)
        self.assertEqual(ds.targetProperties[0].name, "CL")
        self.assertEqual(ds.targetProperties[0].task, TargetTasks.REGRESSION)
        self.assertEqual(len(ds.X), len(ds))
        self.assertEqual(len(ds.y), len(ds))
        self.assertEqual(len(ds.y.columns), 1)
        self.assertEqual(ds.y.columns[0], "CL")

    def checkBadInit(self, ds):
        with self.assertRaises(AssertionError):
            ds.makeClassification("CL", [])
        with self.assertRaises(AssertionError):
            ds.makeClassification("CL", th=6.5)
        with self.assertRaises(AssertionError):
            ds.makeClassification("CL", th=[0, 2, 3])
        with self.assertRaises(AssertionError):
            ds.makeClassification("CL", th=[0, 2, 3])

    def checkClassification(self, ds, target_names, ths):
        # Test that the dataset properties are correctly initialized
        self.assertTrue(len(ds.targetProperties) == len(target_names) == len(ths))
        for idx, target_prop in enumerate(ds.targetProperties):
            if len(ths[idx]) == 1:
                self.assertEqual(target_prop.task, TargetTasks.SINGLECLASS)
            else:
                self.assertEqual(target_prop.task, TargetTasks.MULTICLASS)
            self.assertEqual(target_prop.name, target_names[idx])
            y = ds.getTargetPropertiesValues(concat=True)
            self.assertTrue(y.columns[idx] == target_prop.name)
            if target_prop.task == TargetTasks.SINGLECLASS:
                self.assertEqual(y[target_prop.name].unique().shape[0], 2)
            elif ths[idx] != "precomputed":
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
            ds.getTargetPropertiesValues(concat=True)

    def testDefaults(self):
        """Test basic dataset creation and serialization with mostly default options."""
        # create a basic regression data set
        dataset = QSPRDataset(
            "test_defaults",
            [{"name": "CL", "task": TargetTasks.REGRESSION}],
            df=self.getSmallDF(),
            store_dir=self.generatedDataPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        self.assertIn("HBD", dataset.getProperties())
        dataset.removeProperty("HBD")
        self.assertNotIn("HBD", dataset.getProperties())
        stopwatch = StopWatch()
        dataset.save()
        stopwatch.stop("Saving took: ")
        self.assertTrue(os.path.exists(dataset.storePath))
        # load the data set again and check if everything is consistent after loading
        # creation from file
        stopwatch.reset()
        dataset_new = QSPRDataset.fromFile(dataset.metaFile)
        stopwatch.stop("Loading from file took: ")
        self.checkConsistency(dataset_new)
        # creation by reinitialization
        stopwatch.reset()
        dataset_new = QSPRDataset(
            "test_defaults",
            [{"name": "CL", "task": TargetTasks.REGRESSION}],
            store_dir=self.generatedDataPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        stopwatch.stop("Reinitialization took: ")
        self.checkConsistency(dataset_new)
        # creation from a table file
        stopwatch.reset()
        dataset_new = QSPRDataset.fromTableFile(
            "test_defaults",
            f"{self.inputDataPath}/test_data.tsv",
            target_props=[{"name": "CL", "task": TargetTasks.REGRESSION}],
            store_dir=self.generatedDataPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        stopwatch.stop("Loading from table file took: ")
        self.assertTrue(isinstance(dataset_new, QSPRDataset))
        self.checkConsistency(dataset_new)
        # creation from a table file with a new name
        dataset_new = QSPRDataset.fromTableFile(
            "test_defaults_new",  # new name implies HBD below should exist again
            f"{self.inputDataPath}/test_data.tsv",
            target_props=[{"name": "CL", "task": TargetTasks.REGRESSION}],
            store_dir=self.generatedDataPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        self.assertTrue(isinstance(dataset_new, QSPRDataset))
        self.assertIn("HBD", dataset_new.getProperties())
        dataset_new.removeProperty("HBD")
        self.checkConsistency(dataset_new)

    def testMultitask(self):
        """Test multi-task dataset creation and functionality."""
        dataset = QSPRDataset(
            "testMultitask",
            [
                {"name": "CL", "task": TargetTasks.REGRESSION},
                {"name": "fu", "task": TargetTasks.REGRESSION},
            ],
            df=self.getSmallDF(),
            store_dir=self.generatedDataPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
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
        dataset.setTargetProperty({"name": "fu", "task": TargetTasks.REGRESSION})
        self.checkConsistencyMulticlass(dataset)

    def testTargetProperty(self):
        """Test target property creation and serialization
        in the context of a dataset.
        """
        dataset = QSPRDataset(
            "testTargetProperty",
            [
                {"name": "CL", "task": TargetTasks.REGRESSION},
                {"name": "fu", "task": TargetTasks.REGRESSION},
            ],
            df=self.getSmallDF(),
            store_dir=self.generatedDataPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
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
        df_new = dataset.df.copy()
        del df_new["CL_original"]
        dataset = QSPRDataset(
            "testTargetProperty-precomputed",
            [{"name": "CL", "task": TargetTasks.MULTICLASS, "th": "precomputed"}],
            df=df_new,
            store_dir=self.generatedDataPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        self.assertEqual(len(dataset.targetProperties), 1)
        self.assertEqual(dataset.targetProperties[0].task, TargetTasks.MULTICLASS)
        self.assertEqual(dataset.targetProperties[0].name, "CL")
        self.assertEqual(dataset.targetProperties[0].nClasses, 3)
        self.assertEqual(dataset.targetProperties[0].th, "precomputed")
        # Check that the dataset is correctly loaded from file for classification
        dataset.save()
        dataset_new = QSPRDataset.fromFile(dataset.metaFile)
        self.checkBadInit(dataset_new)
        self.checkClassification(dataset_new, ["CL"], ["precomputed"])
        # Check that the make regression method works as expected
        dataset_new.makeRegression(target_property="CL")
        # Check that the dataset is correctly loaded from file for regression
        self.checkRegression(dataset_new, ["CL"])
        dataset_new.save()
        dataset_new = QSPRDataset.fromFile(dataset.metaFile)
        self.checkRegression(dataset_new, ["CL"])

    def testIndexing(self):
        # default index
        QSPRDataset(
            "testTargetProperty",
            [{"name": "CL", "task": TargetTasks.REGRESSION}],
            df=self.getSmallDF(),
            store_dir=self.generatedDataPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        # set index to SMILES column
        QSPRDataset(
            "testTargetProperty",
            [{"name": "CL", "task": TargetTasks.REGRESSION}],
            df=self.getSmallDF(),
            store_dir=self.generatedDataPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
            index_cols=["SMILES"],
        )
        # multiindex
        QSPRDataset(
            "testTargetProperty",
            [{"name": "CL", "task": TargetTasks.REGRESSION}],
            df=self.getSmallDF(),
            store_dir=self.generatedDataPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
            index_cols=["SMILES", "Name"],
        )
        # index with duplicates
        self.assertRaises(
            ValueError,
            lambda: QSPRDataset(
                "testTargetProperty",
                [{"name": "CL", "task": TargetTasks.REGRESSION}],
                df=self.getSmallDF(),
                store_dir=self.generatedDataPath,
                n_jobs=self.nCPU,
                chunk_size=self.chunkSize,
                index_cols=["moka_ionState7.4"],
            ),
        )
        # index has nans
        self.assertRaises(
            ValueError,
            lambda: QSPRDataset(
                "testTargetProperty",
                [{"name": "CL", "task": TargetTasks.REGRESSION}],
                df=self.getSmallDF(),
                store_dir=self.generatedDataPath,
                n_jobs=self.nCPU,
                chunk_size=self.chunkSize,
                index_cols=["fu"],
            ),
        )

    @parameterized.expand([(1,), (2,)])  # use one or two CPUs
    def testInvalidsDetection(self, n_cpu):
        df = self.getBigDF()
        all_mols = len(df)
        dataset = QSPRDataset(
            "testInvalidsDetection",
            [{"name": "CL", "task": TargetTasks.REGRESSION}],
            df=df,
            store_dir=self.generatedDataPath,
            drop_invalids=False,
            drop_empty=False,
            n_jobs=n_cpu,
        )
        self.assertEqual(dataset.df.shape[0], df.shape[0])
        self.assertRaises(ValueError, lambda: dataset.checkMols())
        self.assertRaises(
            ValueError,
            lambda: dataset.addDescriptors(
                MoleculeDescriptorsCalculator(
                    [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1024)]
                )
            ),
        )
        invalids = dataset.checkMols(throw=False)
        self.assertEqual(sum(~invalids), 1)
        dataset.dropInvalids()
        self.assertEqual(dataset.df.shape[0], all_mols - 1)

    def testRandomStateShuffle(self):
        dataset = self.createLargeTestDataSet()
        seed = dataset.randomState
        dataset.shuffle()
        order = dataset.getDF().index.tolist()
        dataset.save()
        dataset.shuffle()
        order_next = dataset.getDF().index.tolist()
        # reload and check if seed and order are the same
        dataset = QSPRDataset.fromFile(dataset.metaFile)
        self.assertEqual(dataset.randomState, seed)
        self.assertListEqual(dataset.getDF().index.tolist(), order)
        # shuffle again and check if order is the same as before
        dataset.shuffle()
        self.assertListEqual(dataset.getDF().index.tolist(), order_next)

    def testRandomStateFeaturization(self):
        # create and save the data set
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors(
            MoleculeDescriptorsCalculator(
                [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1024)]
            ),
            featurize=False,
        )
        dataset.save()
        # split and featurize with shuffling
        split = ShuffleSplit(1, test_size=0.5, random_state=dataset.randomState)
        dataset.split(split, featurize=False)
        dataset.featurizeSplits(shuffle=True)
        train, test = dataset.getFeatures()
        train_order = train.index.tolist()
        test_order = test.index.tolist()
        # reload and check if orders are the same if we redo the split
        # and featurization with the same random state
        dataset = QSPRDataset.fromFile(dataset.metaFile)
        split = ShuffleSplit(1, test_size=0.5, random_state=dataset.randomState)
        dataset.split(split, featurize=False)
        dataset.featurizeSplits(shuffle=True)
        train, test = dataset.getFeatures()
        self.assertListEqual(train.index.tolist(), train_order)
        self.assertListEqual(test.index.tolist(), test_order)

    def testRandomStateFolds(self):
        # create and save the data set (fixes the seed)
        dataset = self.createLargeTestDataSet()
        dataset.save()
        # calculate descriptors and iterate over folds
        dataset.prepareDataset(
            feature_calculators=[
                MoleculeDescriptorsCalculator(
                    [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1024)]
                ),
            ]
        )
        train, _ = dataset.getFeatures()
        order_train = train.index.tolist()
        order_folds = []
        split = KFold(5, shuffle=True, random_state=dataset.randomState)
        for _, _, _, _, train_index, test_index in dataset.iterFolds(split):
            order_folds.append(train.iloc[train_index].index.tolist())
        # reload and check if orders are the same if we redo the folds from saved data
        dataset = QSPRDataset.fromFile(dataset.metaFile)
        dataset.prepareDataset(
            feature_calculators=[
                MoleculeDescriptorsCalculator(
                    [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1024)]
                ),
            ]
        )
        train, _ = dataset.getFeatures()
        self.assertListEqual(train.index.tolist(), order_train)
        split = KFold(5, shuffle=True, random_state=dataset.randomState)
        for i, (_, _, _, _, train_index, test_index) in enumerate(
            dataset.iterFolds(split)
        ):
            self.assertListEqual(train.iloc[train_index].index.tolist(), order_folds[i])


class TestSearchFeatures(DataSetsPathMixIn, QSPRTestCase):
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def validateSearch(self, dataset: QSPRDataset, result: QSPRDataset, name: str):
        """Validate the results of a search."""
        self.assertTrue(len(result) < len(dataset))
        self.assertTrue(isinstance(result, type(dataset)))
        self.assertEqual(result.name, name)
        self.assertListEqual(dataset.getProperties(), result.getProperties())
        self.assertListEqual(dataset.getFeatureNames(), result.getFeatureNames())
        self.assertListEqual(dataset.targetPropertyNames, result.targetPropertyNames)
        self.assertEqual(len(dataset.descriptors), len(result.descriptors))
        self.assertEqual(
            len(dataset.descriptorCalculators), len(result.descriptorCalculators)
        )
        self.assertEqual(len(dataset.targetProperties), len(result.targetProperties))
        self.assertEqual(dataset.nTargetProperties, result.nTargetProperties)

    def testSMARTS(self):
        dataset = self.createLargeTestDataSet(
            preparation_settings=self.getDefaultPrep()
        )
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
        dataset = self.createLargeTestDataSet(
            preparation_settings=self.getDefaultPrep()
        )
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


def prop_transform(x):
    return np.log10(x)


class TestTargetProperty(QSPRTestCase):
    """Test the TargetProperty class."""

    def checkTargetProperty(self, target_prop, name, task, th):
        # Check the target property creation consistency
        self.assertEqual(target_prop.name, name)
        self.assertEqual(target_prop.task, task)
        if task.isClassification():
            self.assertTrue(target_prop.task.isClassification())
            self.assertEqual(target_prop.th, th)

    def testInit(self):
        """Check the TargetProperty class on target
        property creation.
        """
        # Check the different task types
        targetprop = TargetProperty("CL", TargetTasks.REGRESSION)
        self.checkTargetProperty(targetprop, "CL", TargetTasks.REGRESSION, None)
        targetprop = TargetProperty("CL", TargetTasks.MULTICLASS, th=[0, 1, 10, 1200])
        self.checkTargetProperty(
            targetprop, "CL", TargetTasks.MULTICLASS, [0, 1, 10, 1200]
        )
        targetprop = TargetProperty("CL", TargetTasks.SINGLECLASS, th=[5])
        self.checkTargetProperty(targetprop, "CL", TargetTasks.SINGLECLASS, [5])
        # check with precomputed values
        targetprop = TargetProperty(
            "CL", TargetTasks.SINGLECLASS, th="precomputed", n_classes=2
        )
        self.checkTargetProperty(
            targetprop, "CL", TargetTasks.SINGLECLASS, "precomputed"
        )
        # Check from dictionary creation
        targetprop = TargetProperty.fromDict(
            {"name": "CL", "task": TargetTasks.REGRESSION}
        )
        self.checkTargetProperty(targetprop, "CL", TargetTasks.REGRESSION, None)
        targetprop = TargetProperty.fromDict(
            {"name": "CL", "task": TargetTasks.MULTICLASS, "th": [0, 1, 10, 1200]}
        )
        self.checkTargetProperty(
            targetprop, "CL", TargetTasks.MULTICLASS, [0, 1, 10, 1200]
        )
        # Check from list creation, selection and serialization support functions
        targetprops = TargetProperty.fromList(
            [
                {"name": "CL", "task": TargetTasks.REGRESSION},
                {"name": "fu", "task": TargetTasks.REGRESSION},
            ]
        )
        self.checkTargetProperty(targetprops[0], "CL", TargetTasks.REGRESSION, None)
        self.checkTargetProperty(targetprops[1], "fu", TargetTasks.REGRESSION, None)
        self.assertListEqual(TargetProperty.getNames(targetprops), ["CL", "fu"])
        targetprops = TargetProperty.toList(targetprops)
        self.assertIsInstance(targetprops, list)
        self.assertIsInstance(targetprops[0], dict)
        self.assertEqual(targetprops[0]["name"], "CL")
        self.assertEqual(targetprops[0]["task"], TargetTasks.REGRESSION)

    @parameterized.expand(
        [
            (TargetTasks.REGRESSION, "CL", None, prop_transform),
            (TargetTasks.MULTICLASS, "CL", [0, 1, 10, 1200], lambda x: x + 1),
            # (TargetTasks.SINGLECLASS, "CL", [5], np.log), FIXME: np.log does not save
        ]
    )
    def testSerialization(self, task, name, th, transformer):
        prop = TargetProperty(name, task, transformer=transformer, th=th)
        json_form = prop.toJSON()
        prop2 = TargetProperty.fromJSON(json_form)
        self.assertEqual(prop2.name, prop.name)
        self.assertEqual(prop2.task, prop.task)
        rnd_number = np.random.rand(10)
        self.assertTrue(
            all(prop2.transformer(rnd_number) == prop.transformer(rnd_number))
        )


class TestDataSplitters(DataSetsPathMixIn, QSPRTestCase):
    """Small tests to only check if the data splitters work on their own.

    The tests here should be used to check for all their specific parameters and edge
    cases."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()
        self.splitYear = 2000

    def testManualSplit(self):
        """Test the manual split function, where the split is done manually."""
        dataset = self.createLargeTestDataSet()
        dataset.nJobs = self.nCPU
        dataset.chunkSize = self.chunkSize
        # Add extra column to the data frame to use for splitting
        dataset.df["split"] = "train"
        dataset.df.loc[dataset.df.sample(frac=0.1).index, "split"] = "test"
        split = ManualSplit(dataset.df["split"], "train", "test")
        dataset.prepareDataset(split=split)
        self.validate_split(dataset)

    @parameterized.expand(
        [
            (False,),
            (True,),
        ]
    )
    def testRandomSplit(self, multitask):
        """Test the random split function."""
        if multitask:
            dataset = self.createLargeMultitaskDataSet()
        else:
            dataset = self.createLargeTestDataSet()
        dataset = self.createLargeTestDataSet()
        dataset.prepareDataset(split=RandomSplit(test_fraction=0.1))
        self.validate_split(dataset)

    @parameterized.expand(
        [
            (False,),
            (True,),
        ]
    )
    def testTemporalSplit(self, multitask):
        """Test the temporal split function, where the split is done based on a time
        property."""
        if multitask:
            dataset = self.createLargeMultitaskDataSet()
        else:
            dataset = self.createLargeTestDataSet()
        split = TemporalSplit(
            timesplit=self.splitYear,
            timeprop="Year of first disclosure",
        )
        # prepare and validate the split
        dataset.prepareDataset(split=split)
        self.validate_split(dataset)
        # test if dates higher than 2000 are in test set
        test_set = dataset.getFeatures()[1]
        years = dataset.getDF().loc[test_set.index, "Year of first disclosure"]
        self.assertTrue(all(years > self.splitYear))
        # test bootstrapping
        if multitask:
            dataset = self.createLargeMultitaskDataSet(
                name="TemporalSplit_bootstrap_mt"
            )
        else:
            dataset = self.createLargeTestDataSet(name="TemporalSplit_bootstrap")
        split = TemporalSplit(
            timesplit=[self.splitYear - 1, self.splitYear, self.splitYear + 1],
            timeprop="Year of first disclosure",
        )
        bootstrap_split = BootstrapSplit(
            split=split,
            n_bootstraps=10,
        )
        for time, fold_info in zip(
            split.timeSplit, list(dataset.iterFolds(bootstrap_split))
        ):
            years = dataset.getDF().loc[fold_info[1].index, "Year of first disclosure"]
            self.assertTrue(all(years > time))

    @parameterized.expand(
        [
            (False, Murcko(), None),
            (False, BemisMurcko(), ["ScaffoldSplit_000", "ScaffoldSplit_001"]),
            (True, Murcko(), None),
        ]
    )
    def testScaffoldSplit(self, multitask, scaffold, custom_test_list):
        """Test the scaffold split function."""
        if multitask:
            dataset = self.createLargeMultitaskDataSet(name="ScaffoldSplit")
        else:
            dataset = self.createLargeTestDataSet(name="ScaffoldSplit")
        split = ScaffoldSplit(
            scaffold=scaffold,
            custom_test_list=custom_test_list,
        )
        dataset.prepareDataset(split=split)
        self.validate_split(dataset)
        # check that smiles in custom_test_list are in the test set
        if custom_test_list:
            self.assertTrue(
                all(mol_id in dataset.X_ind.index for mol_id in custom_test_list)
            )
        # check folding by scaffold
        if multitask:
            dataset = self.createLargeMultitaskDataSet(name="ScaffoldSplit_folding_mt")
        else:
            dataset = self.createLargeTestDataSet(name="ScaffoldSplit_folding")
        n_folds = 5
        split = ScaffoldSplit(
            scaffold=scaffold,
            custom_test_list=custom_test_list,
            n_folds=n_folds,
        )
        test_index_all = []
        for k, (X_train, X_test, y_train, y_test, train_index, test_index) in enumerate(
            dataset.iterFolds(split)
        ):
            self.assertTrue(all(x not in test_index_all for x in test_index))
            self.assertTrue(len(X_train) > len(X_test))
            test_index_all.extend(X_test.index.tolist())
        self.assertEqual(k, n_folds - 1)
        self.assertEqual(len(test_index_all), len(dataset.getFeatures(concat=True)))

    @parameterized.expand(
        [
            (False, FPSimilarityLeaderPickerClusters(), None),
            (
                False,
                FPSimilarityMaxMinClusters(),
                ["ClusterSplit_000", "ClusterSplit_001"],
            ),
            (True, FPSimilarityMaxMinClusters(), None),
            (
                True,
                FPSimilarityLeaderPickerClusters(),
                ["ClusterSplit_000", "ClusterSplit_001"],
            ),
        ]
    )
    def testClusterSplit(self, multitask, clustering_algorithm, custom_test_list):
        """Test the cluster split function."""
        if multitask:
            dataset = self.createLargeMultitaskDataSet(name="ClusterSplit")
        else:
            dataset = self.createLargeTestDataSet(name="ClusterSplit")
        split = ClusterSplit(
            clustering=clustering_algorithm,
            custom_test_list=custom_test_list,
            time_limit_seconds=10,
        )
        dataset.prepareDataset(split=split)
        self.validate_split(dataset)
        # check that smiles in custom_test_list are in the test set
        if custom_test_list:
            self.assertTrue(
                all(mol_id in dataset.X_ind.index for mol_id in custom_test_list)
            )

    def testSerialization(self):
        """Test the serialization of dataset with datasplit."""
        dataset = self.createLargeTestDataSet()
        split = ScaffoldSplit()
        N_BITS = 1024
        calculator = MoleculeDescriptorsCalculator(
            [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=N_BITS)]
        )
        dataset.prepareDataset(
            split=split,
            feature_calculators=[calculator],
            feature_standardizer=StandardScaler(),
        )
        self.validate_split(dataset)
        test_ids = dataset.X_ind.index.values
        train_ids = dataset.y_ind.index.values
        dataset.save()
        dataset_new = QSPRDataset.fromFile(dataset.metaFile)
        self.validate_split(dataset_new)
        self.assertTrue(dataset_new.descriptorCalculators)
        self.assertTrue(dataset_new.feature_standardizer)
        self.assertTrue(len(dataset_new.featureNames) == N_BITS)
        self.assertTrue(all(mol_id in dataset_new.X_ind.index for mol_id in test_ids))
        self.assertTrue(all(mol_id in dataset_new.y_ind.index for mol_id in train_ids))
        dataset_new.clearFiles()


class TestFoldSplitters(DataSetsPathMixIn, QSPRTestCase):
    """Small tests to only check if the fold splitters work on their own.

    The tests here should be used to check for all their specific parameters and
    edge cases."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def validateFolds(self, folds, more=None):
        """Check if the folds have the data they should have after splitting."""
        k = 0
        tested_indices = []
        for (
            X_train,
            X_test,
            y_train,
            y_test,
            train_index,
            test_index,
        ) in folds:
            k += 1
            self.assertEqual(len(X_train), len(y_train))
            self.assertEqual(len(X_test), len(y_test))
            self.assertEqual(len(train_index), len(y_train))
            self.assertEqual(len(test_index), len(y_test))
            tested_indices.extend(X_test.index.tolist())
            if more:
                more(X_train, X_test, y_train, y_test, train_index, test_index)
        return k, tested_indices

    def testStandardFolds(self):
        """Test the default fold generator, which is a 5-fold cross validation."""
        # test default settings with regression
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors(
            MoleculeDescriptorsCalculator(
                [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1024)]
            )
        )
        fold = KFold(5, shuffle=True, random_state=dataset.randomState)
        generator = FoldsFromDataSplit(fold)
        k, indices = self.validateFolds(generator.iterFolds(dataset))
        self.assertEqual(k, 5)
        self.assertFalse(set(dataset.df.index) - set(indices))
        # test directly on data set
        k, indices = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        self.assertFalse(set(dataset.df.index) - set(indices))
        # test default settings with classification
        dataset.makeClassification("CL", th=[20])
        fold = StratifiedKFold(5, shuffle=True, random_state=dataset.randomState)
        k, indices = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        self.assertFalse(set(dataset.df.index) - set(indices))
        # test with a standarizer
        MAX_VAL = 2
        MIN_VAL = 1
        scaler = MinMaxScaler(feature_range=(MIN_VAL, MAX_VAL))
        dataset.prepareDataset(feature_standardizer=scaler)
        k, indices = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        self.assertFalse(set(dataset.df.index) - set(indices))

        def check_min_max(X_train, X_test, *args, **kwargs):
            self.assertTrue(np.max(X_train.values) == MAX_VAL)
            self.assertTrue(np.min(X_train.values) == MIN_VAL)
            self.assertTrue(np.max(X_test.values) == MAX_VAL)
            self.assertTrue(np.min(X_test.values) == MIN_VAL)

        self.validateFolds(dataset.iterFolds(fold), more=check_min_max)
        k, indices = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        self.assertFalse(set(dataset.df.index) - set(indices))

        # try with a split data set
        dataset.split(RandomSplit(test_fraction=0.1))
        k, indices = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        self.assertFalse(set(dataset.X.index) - set(indices))

    def testBootstrappedFold(self):
        dataset = self.createLargeTestDataSet(random_state=None)
        dataset.addDescriptors(
            MoleculeDescriptorsCalculator(
                [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1024)]
            )
        )
        split = RandomSplit(0.2)
        fold = BootstrapSplit(split, n_bootstraps=5)
        k, indices = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        # check if the indices are the same if we do the same split again
        split = RandomSplit(0.2)
        fold = BootstrapSplit(split, n_bootstraps=5, seed=dataset.randomState)
        k, indices_second = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        self.assertListEqual(indices, indices_second)
        # check if the indices are different if we do a different split
        split = RandomSplit(0.2)
        fold = BootstrapSplit(split, n_bootstraps=5, seed=42)
        k, indices_third = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        self.assertEqual(split.getSeed(), None)
        self.assertNotEqual(indices, indices_third)


class TestDataFilters(DataSetsPathMixIn, QSPRTestCase):
    """Small tests to only check if the data filters work on their own.

    The tests here should be used to check for all their specific parameters and
    edge cases."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def testCategoryFilter(self):
        """Test the category filter, which drops specific values from dataset
        properties."""
        remove_cation = CategoryFilter(name="moka_ionState7.4", values=["cationic"])
        df_anion = remove_cation(self.getBigDF())
        self.assertTrue((df_anion["moka_ionState7.4"] == "cationic").sum() == 0)

        only_cation = CategoryFilter(
            name="moka_ionState7.4", values=["cationic"], keep=True
        )
        df_cation = only_cation(self.getBigDF())
        self.assertTrue((df_cation["moka_ionState7.4"] != "cationic").sum() == 0)

    def testRepeatsFilter(self):
        """Test the duplicate filter, which drops rows with identical descriptors
        from dataset."""
        descriptor_names = [f"Descriptor_{i}" for i in range(3)]
        df = pd.DataFrame(
            data=np.array(
                [
                    ["C", 1, 2, 1, 1],
                    ["CC", 1, 2, 2, 2],
                    ["CCC", 1, 2, 3, 3],
                    ["C", 1, 2, 1, 4],
                    ["C", 1, 2, 1, 5],
                    ["CC", 1, 2, 2, 6],  # 3rd "descriptor" is length of SMILES
                ]
            ),
            columns=["SMILES", *descriptor_names, "Year"],
        )
        # only warnings
        df_copy = copy.deepcopy(df)
        dup_filter1 = RepeatsFilter(keep=True)
        df_copy = dup_filter1(df_copy, df_copy[descriptor_names])
        self.assertEqual(len(df_copy), len(df))
        self.assertTrue(df_copy.equals(df))
        # drop duplicates
        df_copy = copy.deepcopy(df)
        dup_filter2 = RepeatsFilter(keep=False)
        df_copy = dup_filter2(df_copy, df_copy[descriptor_names])
        self.assertEqual(len(df_copy), 1)  # only CCC has one occurence
        self.assertTrue(df_copy.equals(df.iloc[[2]]))
        # keep first, by year
        df_copy = copy.deepcopy(df)
        dup_filter3 = RepeatsFilter(keep="first", timecol="Year")
        df_copy = dup_filter3(df_copy, df_copy[descriptor_names])
        self.assertEqual(len(df_copy), 3)  # three unique SMILES
        self.assertTrue(df_copy.equals(df.iloc[[0, 1, 2]]))  # keep first by year

        # check with additional columns
        df_copy = copy.deepcopy(df)
        df_copy["proteinid"] = ["A", "B", "B", "B", "B", "B"]
        dup_filter4 = RepeatsFilter(additional_cols=["proteinid"])
        df_copy = dup_filter4(df_copy, df_copy[descriptor_names])
        self.assertEqual(len(df_copy), 2)  # C (protein A, idx 0) and CCC are unique,
        # but C (protein B, idx 3) is a duplicate
        # of C (protein B, idx 4) and is dropped

    def testConsistency(self):
        dataset = self.createLargeTestDataSet()
        remove_cation = CategoryFilter(name="moka_ionState7.4", values=["cationic"])
        self.assertTrue((dataset.getDF()["moka_ionState7.4"] == "cationic").sum() > 0)
        dataset.filter([remove_cation])
        self.assertEqual(len(dataset.getDF()), len(dataset.getFeatures(concat=True)))
        self.assertTrue((dataset.getDF()["moka_ionState7.4"] == "cationic").sum() == 0)


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

    def testImputation(self):
        """Test the imputation of missing values in the target properties."""
        self.dataset = QSPRDataset(
            "TestImputation",
            target_props=[
                {
                    "name": "y",
                    "task": TargetTasks.REGRESSION,
                    "imputer": SimpleImputer(strategy="mean"),
                },
                {
                    "name": "z",
                    "task": TargetTasks.REGRESSION,
                    "imputer": SimpleImputer(strategy="mean"),
                },
            ],
            df=self.df,
            store_dir=self.generatedPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        self.assertEqual(self.dataset.targetProperties[0].name, "y")
        self.assertEqual(self.dataset.targetProperties[1].name, "z")
        self.assertTrue("y_before_impute" in self.dataset.df.columns)
        self.assertTrue("z_before_impute" in self.dataset.df.columns)
        self.assertEqual(self.dataset.df["y"].isna().sum(), 0)
        self.assertEqual(self.dataset.df["z"].isna().sum(), 0)


class TestFeatureFilters(PathMixIn, QSPRTestCase):
    """Tests to check if the feature filters work on their own."""

    def setUp(self):
        """Set up the small test Dataframe."""
        super().setUp()
        self.setUpPaths()
        descriptors = [
            "Descriptor_F1",
            "Descriptor_F2",
            "Descriptor_F3",
            "Descriptor_F4",
            "Descriptor_F5",
        ]
        self.df_descriptors = pd.DataFrame(
            data=np.array(
                [
                    [1, 4, 2, 6, 2],
                    [1, 8, 4, 2, 4],
                    [1, 4, 3, 2, 5],
                    [1, 8, 4, 9, 8],
                    [1, 4, 2, 3, 9],
                    [1, 8, 4, 7, 12],
                ]
            ),
            columns=descriptors,
        )
        self.df = pd.DataFrame(
            data=np.array([["C", 1], ["C", 2], ["C", 3], ["C", 4], ["C", 5], ["C", 6]]),
            columns=["SMILES", "y"],
        )
        self.dataset = QSPRDataset(
            "TestFeatureFilters",
            target_props=[{"name": "y", "task": TargetTasks.REGRESSION}],
            df=self.df,
            store_dir=self.generatedPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        self.df_descriptors.index = self.dataset.df.index
        calculator = CustomDescriptorsCalculator(
            [DataFrameDescriptorSet(self.df_descriptors)]
        )
        self.dataset.addCustomDescriptors(calculator)
        self.descriptors = self.dataset.featureNames

    def testLowVarianceFilter(self):
        """Test the low variance filter, which drops features with a variance below
        a threshold."""
        self.dataset.filterFeatures([LowVarianceFilter(0.01)])
        # check if correct columns selected and values still original
        self.assertListEqual(list(self.dataset.featureNames), self.descriptors[1:])
        self.assertListEqual(list(self.dataset.X.columns), self.descriptors[1:])

    def testHighCorrelationFilter(self):
        """Test the high correlation filter, which drops features with a correlation
        above a threshold."""
        self.dataset.filterFeatures([HighCorrelationFilter(0.8)])
        # check if correct columns selected and values still original
        self.descriptors.pop(2)
        self.assertListEqual(list(self.dataset.featureNames), self.descriptors)
        self.assertListEqual(list(self.dataset.X.columns), self.descriptors)

    def testBorutaFilter(self):
        """Test the Boruta filter, which removes the features which are statistically as
        relevant as random features."""
        self.dataset.filterFeatures([BorutaFilter()])
        # check if correct columns selected and values still original
        self.assertListEqual(list(self.dataset.featureNames), self.descriptors[-1:])
        self.assertListEqual(list(self.dataset.X.columns), self.descriptors[-1:])


class TestDescriptorCalculation(DataSetsPathMixIn, QSPRTestCase):
    """Test the calculation of descriptors."""

    def setUp(self):
        """Set up the test Dataframe."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createLargeTestDataSet(self.__class__.__name__)

    def testSwitching(self):
        """Test if the feature calculator can be switched to a new dataset."""
        feature_calculator = MoleculeDescriptorsCalculator(
            desc_sets=[
                FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=2048),
                DrugExPhyschem(),
            ]
        )
        split = RandomSplit(test_fraction=0.1)
        lv = LowVarianceFilter(0.05)
        hc = HighCorrelationFilter(0.9)
        self.dataset.prepareDataset(
            split=split,
            feature_calculators=[feature_calculator],
            feature_filters=[lv, hc],
            recalculate_features=True,
            feature_fill_value=np.nan,
        )
        # create new dataset with different feature calculator
        dataset_next = self.createLargeTestDataSet(self.__class__.__name__)
        dataset_next.prepareDataset(
            split=split,
            feature_calculators=[feature_calculator],
            feature_filters=[lv, hc],
            recalculate_features=True,
            feature_fill_value=np.nan,
        )


class TestDescriptorSets(DataSetsPathMixIn, QSPRTestCase):
    """Test the descriptor sets."""

    def setUp(self):
        """Create the test Dataframe."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)
        self.dataset.shuffle()

    def testPredictorDescriptor(self):
        """Test the PredictorDesc descriptor set."""
        # give path to saved model parameters
        meta_path = (
            f"{self.inputBasePath}/test_predictor/"
            f"qspr/models/RFC_SINGLECLASS/RFC_SINGLECLASS_meta.json"
        )
        model = SklearnModel.fromFile(meta_path)
        desc_calc = MoleculeDescriptorsCalculator([PredictorDesc(model)])
        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 1))
        self.assertTrue(self.dataset.X.any().any())
        # test from file instantiation
        desc_calc.toFile(f"{self.generatedDataPath}/test_calc.json")
        desc_calc_file = MoleculeDescriptorsCalculator.fromFile(
            f"{self.generatedDataPath}/test_calc.json"
        )
        self.dataset.addDescriptors(desc_calc_file, recalculate=True)
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 1))
        self.assertTrue(self.dataset.X.any().any())

    def testFingerprintSet(self):
        """Test the fingerprint set descriptor calculator."""
        desc_calc = MoleculeDescriptorsCalculator(
            [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1000)]
        )
        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 1000))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    def testTanimotoDistances(self):
        """Test the Tanimoto distances descriptor calculator, which calculates the
        Tanimoto distances between a list of SMILES."""
        list_of_smiles = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC"]
        desc_calc = MoleculeDescriptorsCalculator(
            [
                TanimotoDistances(
                    list_of_smiles=list_of_smiles,
                    fingerprint_type="MorganFP",
                    radius=3,
                    nBits=1000,
                )
            ]
        )
        self.dataset.addDescriptors(desc_calc)

    def testDrugExPhyschem(self):
        """Test the DrugExPhyschem descriptor calculator."""
        desc_calc = MoleculeDescriptorsCalculator([DrugExPhyschem()])
        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 19))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    def testRDKitDescs(self):
        """Test the rdkit descriptors calculator."""
        desc_calc = MoleculeDescriptorsCalculator([RDKitDescs()])
        self.dataset.addDescriptors(desc_calc)
        rdkit_desc_count = len(set(Descriptors._descList))
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), rdkit_desc_count))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)
        # with 3D
        desc_calc = MoleculeDescriptorsCalculator([RDKitDescs(include_3d=True)])
        self.dataset.addDescriptors(desc_calc, recalculate=True)
        self.assertEqual(
            self.dataset.X.shape, (len(self.dataset), rdkit_desc_count + 10)
        )

    def testSmilesDesc(self):
        """Test the smiles descriptors calculator."""
        desc_calc = MoleculeDescriptorsCalculator([SmilesDesc()])
        self.dataset.addDescriptors(desc_calc)

        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 1))
        self.assertTrue(self.dataset.X.any().any())

    def testConsistency(self):
        """Test if the descriptor calculator is consistent with the dataset."""
        len_prev = len(self.dataset)
        desc_calc = MoleculeDescriptorsCalculator(
            [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1000)]
        )
        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(len_prev, len(self.dataset))
        self.assertEqual(len_prev, len(self.dataset.getDescriptors()))
        self.assertEqual(len_prev, len(self.dataset.X))
        self.assertEqual(1000, self.dataset.getDescriptors().shape[1])
        self.assertEqual(1000, self.dataset.X.shape[1])
        self.assertEqual(1000, self.dataset.X_ind.shape[1])
        self.assertEqual(1000, self.dataset.getFeatures(concat=True).shape[1])
        self.assertEqual(len_prev, self.dataset.getFeatures(concat=True).shape[0])


class TestScaffolds(DataSetsPathMixIn, QSPRTestCase):
    """Test calculation of scaffolds."""

    def setUp(self):
        """Create a small dataset."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)

    def testScaffoldAdd(self):
        """Test the adding and getting of scaffolds."""
        self.dataset.addScaffolds([Murcko()])
        scaffs = self.dataset.getScaffolds()
        self.assertEqual(scaffs.shape, (len(self.dataset), 1))
        self.dataset.addScaffolds([Murcko()], add_rdkit_scaffold=True, recalculate=True)
        scaffs = self.dataset.getScaffolds(includeMols=True)
        self.assertEqual(scaffs.shape, (len(self.dataset), 2))
        for mol in scaffs[f"Scaffold_{Murcko()}_RDMol"]:
            self.assertTrue(isinstance(mol, Chem.rdchem.Mol))


class TestFeatureStandardizer(DataSetsPathMixIn, QSPRTestCase):
    """Test the feature standardizer."""

    def setUp(self):
        """Create a small test dataset with MorganFP descriptors."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)
        self.dataset.addDescriptors(
            MoleculeDescriptorsCalculator(
                [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1000)]
            )
        )

    def testFeaturesStandardizer(self):
        """Test the feature standardizer fitting, transforming and serialization."""
        scaler = SKLearnStandardizer.fromFit(self.dataset.X, StandardScaler())
        scaled_features = scaler(self.dataset.X)
        scaler.toFile(f"{self.generatedPath}/test_scaler.json")
        scaler_fromfile = SKLearnStandardizer.fromFile(
            f"{self.generatedPath}/test_scaler.json"
        )
        scaled_features_fromfile = scaler_fromfile(self.dataset.X)
        self.assertIsInstance(scaled_features, np.ndarray)
        self.assertEqual(scaled_features.shape, (len(self.dataset), 1000))
        self.assertEqual(
            np.array_equal(scaled_features, scaled_features_fromfile), True
        )


class TestStandardizers(DataSetsPathMixIn, QSPRTestCase):
    """Test the standardizers."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def testInvalidFilter(self):
        """Test the invalid filter."""
        df = self.getSmallDF()
        orig_len = len(df)
        mask = [False] * orig_len
        mask[0] = True
        df.loc[mask, "SMILES"] = "C(C)(C)(C)(C)(C)(C)(C)(C)(C)"  # bad valence example
        dataset = QSPRDataset(
            "standardization_test_invalid_filter",
            df=df,
            target_props=[{"name": "CL", "task": TargetTasks.REGRESSION}],
            drop_invalids=False,
            drop_empty=False,
        )
        dataset.standardizeSmiles("chembl", drop_invalid=False)
        self.assertEqual(len(dataset), len(df))
        dataset.standardizeSmiles("chembl", drop_invalid=True)
        self.assertEqual(len(dataset), orig_len - 1)


class TestDataSetPreparation(DataSetsPathMixIn, DataPrepCheckMixIn, QSPRTestCase):
    """Test as many possible combinations of data sets and their preparation
    settings. These can run potentially for a long time so use the `skip` decorator
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
    ):
        """Tests one combination of a data set and its preparation settings.

        This generates a large number of parameterized tests. Use the `skip` decorator
        if you want to skip all these tests. Note that the combinations are not
        exhaustive, but defined by `DataSetsPathMixIn.getPrepCombos()`."""
        np.random.seed(42)
        dataset = self.createLargeTestDataSet(name=name)
        self.checkPrep(
            dataset,
            feature_calculators,
            split,
            feature_standardizer,
            feature_filter,
            data_filter,
            ["CL"],
        )


class TestDescriptorsAll(DataSetsPathMixIn, DescriptorInDataCheckMixIn, QSPRTestCase):
    """Test all descriptor sets in all data sets."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    @parameterized.expand(
        [
            (
                f"{desc_set}_{TargetTasks.MULTICLASS}",
                desc_set,
                [
                    {
                        "name": "CL",
                        "task": TargetTasks.MULTICLASS,
                        "th": [0, 1, 10, 1200],
                    }
                ],
            )
            for desc_set in DataSetsPathMixIn.getAllDescriptors()
        ]
        + [
            (
                f"{desc_set}_{TargetTasks.REGRESSION}",
                desc_set,
                [{"name": "CL", "task": TargetTasks.REGRESSION}],
            )
            for desc_set in DataSetsPathMixIn.getAllDescriptors()
        ]
        + [
            (
                f"{desc_set}_Multitask",
                desc_set,
                [
                    {"name": "CL", "task": TargetTasks.REGRESSION},
                    {"name": "fu", "task": TargetTasks.SINGLECLASS, "th": [0.3]},
                ],
            )
            for desc_set in DataSetsPathMixIn.getAllDescriptors()
        ]
    )
    def testDescriptorsAll(self, _, desc_set, target_props):
        """Tests all available descriptor sets.

        Note that they are not checked with all possible settings and all possible
        preparations, but only with the default settings provided by
        `DataSetsPathMixIn.getDefaultPrep()`. The list itself is defined and configured by
        `DataSetsPathMixIn.getAllDescriptors()`, so if you need a specific descriptor
        tested, add it there.
        """
        np.random.seed(42)
        dataset = self.createLargeTestDataSet(
            name=self.getDatSetName(desc_set, target_props), target_props=target_props
        )
        self.checkDataSetContainsDescriptorSet(
            dataset, desc_set, self.getDefaultPrep(), target_props
        )
