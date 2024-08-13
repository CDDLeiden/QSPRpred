import os

import numpy as np
import pandas as pd
from parameterized import parameterized
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, ShuffleSplit

from ..descriptors.fingerprints import MorganFP
from ... import TargetProperty, TargetTasks
from ...data import QSPRDataset
from ...utils.stopwatch import StopWatch
from ...utils.testing.base import QSPRTestCase
from ...utils.testing.check_mixins import DataPrepCheckMixIn
from ...utils.testing.path_mixins import DataSetsPathMixIn, PathMixIn


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
            [{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
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
            [{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
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
            target_props=[{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
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
            target_props=[{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
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
                {
                    "name": "CL",
                    "task": TargetTasks.REGRESSION
                },
                {
                    "name": "fu",
                    "task": TargetTasks.REGRESSION
                },
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
                {
                    "name": "CL",
                    "task": TargetTasks.REGRESSION
                },
                {
                    "name": "fu",
                    "task": TargetTasks.REGRESSION
                },
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
            [{
                "name": "CL",
                "task": TargetTasks.MULTICLASS,
                "th": "precomputed"
            }],
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
            [{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
            df=self.getSmallDF(),
            store_dir=self.generatedDataPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
        )
        # set index to SMILES column
        QSPRDataset(
            "testTargetProperty",
            [{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
            df=self.getSmallDF(),
            store_dir=self.generatedDataPath,
            n_jobs=self.nCPU,
            chunk_size=self.chunkSize,
            index_cols=["SMILES"],
        )
        # multiindex
        QSPRDataset(
            "testTargetProperty",
            [{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
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
                [{
                    "name": "CL",
                    "task": TargetTasks.REGRESSION
                }],
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
                [{
                    "name": "CL",
                    "task": TargetTasks.REGRESSION
                }],
                df=self.getSmallDF(),
                store_dir=self.generatedDataPath,
                n_jobs=self.nCPU,
                chunk_size=self.chunkSize,
                index_cols=["fu"],
            ),
        )

    @parameterized.expand([(1, ), (2, )])  # use one or two CPUs
    def testInvalidsDetection(self, n_cpu):
        df = self.getBigDF()
        all_mols = len(df)
        dataset = QSPRDataset(
            "testInvalidsDetection",
            [{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
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
            lambda: dataset.addDescriptors([MorganFP(radius=2, nBits=128)]),
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
            [MorganFP(radius=2, nBits=128)],
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
        dataset.prepareDataset(feature_calculators=[MorganFP(radius=2, nBits=128)])
        train, _ = dataset.getFeatures()
        order_train = train.index.tolist()
        order_folds = []
        split = KFold(5, shuffle=True, random_state=dataset.randomState)
        for _, _, _, _, train_index, test_index in dataset.iterFolds(split):
            order_folds.append(train.iloc[train_index].index.tolist())
        # reload and check if orders are the same if we redo the folds from saved data
        dataset = QSPRDataset.fromFile(dataset.metaFile)
        dataset.prepareDataset(feature_calculators=[MorganFP(radius=2, nBits=128)])
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
        self.assertEqual(len(dataset.descriptorSets), len(result.descriptorSets))
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
            {
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }
        )
        self.checkTargetProperty(targetprop, "CL", TargetTasks.REGRESSION, None)
        targetprop = TargetProperty.fromDict(
            {
                "name": "CL",
                "task": TargetTasks.MULTICLASS,
                "th": [0, 1, 10, 1200]
            }
        )
        self.checkTargetProperty(
            targetprop, "CL", TargetTasks.MULTICLASS, [0, 1, 10, 1200]
        )
        # Check from list creation, selection and serialization support functions
        targetprops = TargetProperty.fromList(
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


class TestDataSetPreparation(DataSetsPathMixIn, DataPrepCheckMixIn, QSPRTestCase):
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
        np.random.seed(42)
        dataset = self.createLargeTestDataSet(name=name)
        self.checkPrep(
            dataset,
            feature_calculators,
            split,
            feature_standardizer,
            feature_filter,
            data_filter,
            applicability_domain,
            ["CL"],
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


class TestTargetTransformation(DataSetsPathMixIn, QSPRTestCase):
    """Tests the transformation of target properties."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def prop_transform(self, x):
        return np.log10(x)

    def testTransformation(self):
        dataset = self.createLargeTestDataSet(
            target_props=[
                {
                    "name": "CL",
                    "task": TargetTasks.REGRESSION,
                    "transformer": prop_transform,
                },
            ]
        )
        self.assertTrue(all(dataset.df["CL"] == np.log10(dataset.df["CL_before_transform"])))


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

    @parameterized.expand([(None, None), (2, None), (None, 50), (2, 50)])
    def testRegular(self, n_jobs, chunk_size):
        dataset = self.createLargeTestDataSet()
        dataset.nJobs = n_jobs
        dataset.chunkSize = chunk_size
        result = dataset.apply(
            self.regularFunc,
            on_props=["CL", "fu"],
            func_args=[1, 2, 3],
            func_kwargs={"A_col": "A", "B_col": "B"},
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
