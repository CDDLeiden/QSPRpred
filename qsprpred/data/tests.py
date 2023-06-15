"""This module holds the test for functions regarding QSPR data preparation."""

import copy
import glob
import itertools
import logging
import os
import shutil
from collections.abc import Iterable
from unittest import TestCase

import numpy as np
import pandas as pd
from parameterized import parameterized
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .utils.datafilters import CategoryFilter
from .utils.datasplitters import (
    ManualSplit,
    RandomSplit,
    ScaffoldSplit,
    TemporalSplit,
)
from .utils.descriptorcalculator import (
    CustomDescriptorsCalculator,
    DescriptorsCalculator,
    MoleculeDescriptorsCalculator,
)
from .utils.descriptorsets import (
    DataFrameDescriptorSet,
    DescriptorSet,
    DrugExPhyschem,
    FingerprintSet,
    PredictorDesc,
    RDKitDescs,
    TanimotoDistances,
)
from .utils.feature_standardization import SKLearnStandardizer
from .utils.featurefilters import (
    BorutaFilter,
    HighCorrelationFilter,
    LowVarianceFilter,
)
from .utils.scaffolds import BemisMurcko, Murcko
from ..logs.stopwatch import StopWatch
from ..models.models import QSPRsklearn
from ..models.tasks import TargetTasks
from .data import QSPRDataset, TargetProperty

N_CPU = 2
CHUNK_SIZE = 100
TIME_SPLIT_YEAR = 2000
logging.basicConfig(level=logging.DEBUG)


class PathMixIn:
    """Mix-in class that provides paths to test files and directories and handles their
    creation and deletion."""

    datapath = f"{os.path.dirname(__file__)}/test_files/data"
    qsprdatapath = f"{os.path.dirname(__file__)}/test_files/qspr/data"

    def setUp(self):
        """Create the directories that are used for testing."""
        self.tearDown()
        if not os.path.exists(self.qsprdatapath):
            os.makedirs(self.qsprdatapath)

    def tearDown(self):
        """Remove all files and directories that are used for testing."""
        self.clean_directories()
        if os.path.exists(self.qsprdatapath):
            shutil.rmtree(self.qsprdatapath)
        for extension in ["log", "pkg", "json"]:
            globs = glob.glob(f"{self.datapath}/*.{extension}")
            for path in globs:
                os.remove(path)

    @classmethod
    def clean_directories(cls):
        """Remove the directories that are used for testing."""
        if os.path.exists(cls.qsprdatapath):
            shutil.rmtree(cls.qsprdatapath)


class DataSetsMixIn(PathMixIn):
    """Mix-in class that provides a small and large testing data set and some common
    preparation settings to use in tests."""
    @staticmethod
    def get_default_prep():
        """Return a dictionary with default preparation settings."""
        return {
            "feature_calculators":
                [
                    MoleculeDescriptorsCalculator(
                        [
                            FingerprintSet(
                                fingerprint_type="MorganFP", radius=3, nBits=1024
                            )
                        ]
                    )
                ],
            "split": RandomSplit(0.1),
            "feature_standardizer": StandardScaler(),
            "feature_filters": [LowVarianceFilter(0.05),
                                HighCorrelationFilter(0.8)],
        }

    @classmethod
    def get_all_descriptors(cls):
        """Return a list of all available descriptor sets. Might still not be a complete
        list, though.

        TODO: would be nice to create the list automatically by dynamic import.

        Returns:
            list: `list` of `DescriptorCalculator` objects
        """
        descriptor_sets = [
            RDKitDescs(),
            DrugExPhyschem(),
            PredictorDesc(
                QSPRsklearn.fromFile(
                    f"{os.path.dirname(__file__)}/test_files/test_predictor/qspr/models/SVC_MULTICLASS/SVC_MULTICLASS_meta.json"
                )
            ),
            TanimotoDistances(
                list_of_smiles=["C", "CC", "CCC"],
                fingerprint_type="MorganFP",
                radius=3,
                nBits=1000,
            ),
            FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=2048),
        ]

        return descriptor_sets

    @classmethod
    def get_desc_calculators(cls):
        feature_sets = [
            FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1024),
            RDKitDescs(),
        ]
        mol_descriptor_calculators = [
            [MoleculeDescriptorsCalculator(combo)]
            for combo in itertools.combinations(feature_sets, 1)
        ] + [
            [MoleculeDescriptorsCalculator(combo)]
            for combo in itertools.combinations(feature_sets, 2)
        ]
        return mol_descriptor_calculators

    @classmethod
    def get_prep_grid(cls):
        """Return a list of many possible combinations of descriptor calculators,
        splits, feature standardizers, feature filters and data filters. Again, this is
        not exhaustive, but should cover a lot of cases.

        Returns:
            grid: a generator that yields tuples of all possible combinations as stated
            above, each tuple is defined as:
                (descriptor_calculator, split, feature_standardizer, feature_filters,
                 data_filters)`
        """
        # get the feature calculators
        descriptor_calculators = cls.get_desc_calculators()

        # lists with common preparation settings
        splits = [None, RandomSplit(0.1)]
        feature_standardizers = [None, StandardScaler()]
        feature_filters = [None, HighCorrelationFilter(0.9)]
        data_filters = [
            None,
            # FIXME: this needs to be made more general and not specific to one dataset
            # CategoryFilter(
            #     name="moka_ionState7.4",
            #     values=["cationic"]
            # ),
        ]

        # All combinations of the above preparation settings (passed to prepareDataset)
        return (
            # deep copy to avoid conflicts cayed by operating on one instance twice
            copy.deepcopy(combo) for combo in itertools.product(
                descriptor_calculators,
                splits,
                feature_standardizers,
                feature_filters,
                data_filters,
            )
        )

    @staticmethod
    def get_prep_combinations():
        """Make a list of all possible combinations of preparation settings."""
        def get_name(thing):
            return (
                str(None) if thing is None else thing.__class__.__name__ if
                (not isinstance(thing, (DescriptorsCalculator,
                                        SKLearnStandardizer))) else str(thing)
            )

        def get_name_list(thing):
            if isinstance(thing, Iterable):
                return "_".join([get_name_list(i) for i in thing])
            else:
                return get_name(thing)

        ret = [2 * [get_name_list(x)] + list(x) for x in DataSetsMixIn.get_prep_grid()]
        return ret

    def getBigDF(self):
        """Get a large data frame for testing purposes.

        Returns:
            pd.DataFrame: a `pandas.DataFrame` containing the dataset
        """
        return pd.read_csv(f"{self.datapath}/test_data_large.tsv", sep="\t")

    def getSmallDF(self):
        """Get a small data frame for testing purposes.

        Returns:
            pd.DataFrame: a `pandas.DataFrame` containing the dataset
        """
        return pd.read_csv(f"{self.datapath}/test_data.tsv", sep="\t").sample(10)

    def create_large_dataset(
        self,
        name="QSPRDataset_test",
        target_props=[{
            "name": "CL",
            "task": TargetTasks.REGRESSION
        }],
        target_imputer=None,
        preparation_settings=None,
    ):
        """Create a large dataset for testing purposes.

        Args:
            name (str): name of the dataset
            target_props (List of dicts or TargetProperty): list of target properties
            preparation_settings (dict): dictionary containing preparation settings

        Returns:
            QSPRDataset: a `QSPRDataset` object
        """
        return self.create_dataset(
            self.getBigDF(),
            name=name,
            target_props=target_props,
            target_imputer=target_imputer,
            prep=preparation_settings,
        )

    def create_small_dataset(
        self,
        name="QSPRDataset_test",
        target_props=[{
            "name": "CL",
            "task": TargetTasks.REGRESSION
        }],
        target_imputer=None,
        preparation_settings=None,
    ):
        """Create a small dataset for testing purposes.

        Args:
            name (str): name of the dataset
            target_props (List of dicts or TargetProperty): list of target properties
            preparation_settings (dict): dictionary containing preparation settings

        Returns:
            QSPRDataset: a `QSPRDataset` object
        """
        return self.create_dataset(
            self.getSmallDF(),
            name=name,
            target_props=target_props,
            target_imputer=target_imputer,
            prep=preparation_settings,
        )

    def create_dataset(
        self,
        df,
        name="QSPRDataset_test",
        target_props=[{
            "name": "CL",
            "task": TargetTasks.REGRESSION
        }],
        target_imputer=None,
        prep=None,
    ):
        """Create a dataset for testing purposes from the given data frame.

        Args:
            df (pd.DataFrame): data frame containing the dataset
            name (str): name of the dataset
            target_props (List of dicts or TargetProperty): list of target properties
            prep (dict): dictionary containing preparation settings

        Returns:
            QSPRDataset: a `QSPRDataset` object
        """
        ret = QSPRDataset(
            name,
            target_props=target_props,
            df=df,
            store_dir=self.qsprdatapath,
            target_imputer=target_imputer,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )
        if prep:
            ret.prepareDataset(**prep)
        return ret

    def validate_split(self, dataset):
        """Check if the split has the data it should have after splitting."""
        self.assertTrue(dataset.X is not None)
        self.assertTrue(dataset.X_ind is not None)
        self.assertTrue(dataset.y is not None)
        self.assertTrue(dataset.y_ind is not None)


class DescriptorCheckMixIn:
    def feature_consistency_checks(self, ds, expected_length):
        """Check if the feature names and the feature matrix of a data set is consistent
        with expected number of variables.

        Args:
            ds (QSPRDataset): The data set to check.
            expected_length (int): The expected number of features.
        """
        self.assertEqual(len(ds.featureNames), expected_length)
        self.assertEqual(len(ds.getFeatureNames()), expected_length)
        if expected_length > 0:
            features = ds.getFeatures(concat=True)
            self.assertEqual(features.shape[0], len(ds))
            self.assertEqual(features.shape[1], expected_length)
            self.assertEqual(ds.X.shape[1], expected_length)
            self.assertEqual(ds.X_ind.shape[1], expected_length)
            for fold in ds.createFolds():
                self.assertIsInstance(fold, tuple)
                self.assertEqual(fold[0].shape[1], expected_length)
                self.assertEqual(fold[1].shape[1], expected_length)
        else:
            self.assertEqual(ds.X.shape[1], expected_length)
            self.assertEqual(ds.X_ind.shape[1], expected_length)
            self.assertRaises(ValueError, ds.getFeatures, concat=True)
            self.assertRaises(ValueError, ds.createFolds)

    def desc_consistency_check(self, dataset, target_props):
        # test some basic consistency rules on the resulting features
        expected_length = 0
        for calc in dataset.descriptorCalculators:
            for descset in calc.descSets:
                expected_length += len(descset.descriptors)
        self.feature_consistency_checks(dataset, expected_length)

        # save to file, check if it can be loaded, and if the features are consistent
        dataset.save()
        ds_loaded = QSPRDataset.fromFile(
            dataset.storePath, n_jobs=N_CPU, chunk_size=CHUNK_SIZE
        )
        for ds_loaded_prop, target_prop in zip(
            ds_loaded.targetProperties, target_props
        ):
            if ds_loaded_prop.task.isClassification():
                self.assertEqual(ds_loaded_prop.name, f"{target_prop['name']}_class")
                self.assertEqual(ds_loaded_prop.task, target_prop["task"])
        self.assertTrue(ds_loaded.descriptorCalculators)
        for calc in ds_loaded.descriptorCalculators:
            self.assertTrue(isinstance(calc, DescriptorsCalculator))
            for descset in calc.descSets:
                self.assertTrue(isinstance(descset, DescriptorSet))
        self.feature_consistency_checks(dataset, expected_length)


class TestDataSetCreationSerialization(DataSetsMixIn, TestCase):
    """Simple tests for dataset creation and serialization under different conditions
    and error states."""
    def test_defaults(self):
        """Test default dataset creation and serialization."""
        # creation from data frame
        dataset = QSPRDataset(
            "test_defaults",
            [{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
            df=self.getSmallDF(),
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )
        self.assertIn("HBD", dataset.getProperties())
        dataset.removeProperty("HBD")
        self.assertNotIn("HBD", dataset.getProperties())
        stopwatch = StopWatch()
        dataset.save()
        stopwatch.stop("Saving took: ")
        self.assertTrue(os.path.exists(dataset.storePath))

        def check_consistency(dataset_to_check):
            self.assertNotIn("Notes", dataset_to_check.getProperties())
            self.assertNotIn("HBD", dataset_to_check.getProperties())
            self.assertTrue(len(self.getSmallDF()) - 1 == len(dataset_to_check))
            self.assertEqual(
                dataset_to_check.targetProperties[0].task, TargetTasks.REGRESSION
            )
            self.assertTrue(dataset_to_check.hasProperty("CL"))
            self.assertEqual(dataset_to_check.targetProperties[0].name, "CL")
            self.assertEqual(dataset_to_check.targetProperties[0].originalName, "CL")
            self.assertEqual(len(dataset_to_check.X), len(dataset_to_check))
            self.assertEqual(len(dataset_to_check.X_ind), 0)
            self.assertEqual(len(dataset_to_check.y), len(dataset_to_check))
            self.assertEqual(len(dataset_to_check.y_ind), 0)

        # creation from file
        stopwatch.reset()
        dataset_new = QSPRDataset.fromFile(dataset.storePath)
        stopwatch.stop("Loading from file took: ")
        check_consistency(dataset_new)

        # creation by reinitialization
        stopwatch.reset()
        dataset_new = QSPRDataset(
            "test_defaults",
            [{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )
        stopwatch.stop("Reinitialization took: ")
        check_consistency(dataset_new)

        # creation from a table file
        stopwatch.reset()
        dataset_new = QSPRDataset.fromTableFile(
            "test_defaults",
            f"{os.path.dirname(__file__)}/test_files/data/test_data.tsv",
            target_props=[{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )
        stopwatch.stop("Loading from table file took: ")
        self.assertTrue(isinstance(dataset_new, QSPRDataset))
        check_consistency(dataset_new)

        dataset_new = QSPRDataset.fromTableFile(
            "test_defaults_new",  # new name implies HBD below should exist again
            f"{os.path.dirname(__file__)}/test_files/data/test_data.tsv",
            target_props=[{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )
        self.assertTrue(isinstance(dataset_new, QSPRDataset))
        self.assertIn("HBD", dataset_new.getProperties())
        dataset_new.removeProperty("HBD")
        check_consistency(dataset_new)

    def test_multi_task(self):
        """Test multi-task dataset creation and functionality."""
        dataset = QSPRDataset(
            "test_multi_task",
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
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )

        def check_multiclass(dataset_to_check):
            self.assertTrue(dataset_to_check.isMultiTask)
            self.assertEqual(dataset_to_check.nTasks, 2)
            self.assertEqual(len(dataset_to_check.targetProperties), 2)
            self.assertEqual(dataset_to_check.targetProperties[0].name, "CL")
            self.assertEqual(dataset_to_check.targetProperties[1].name, "fu")
            self.assertEqual(
                dataset_to_check.targetProperties[0].task, TargetTasks.REGRESSION
            )
            self.assertEqual(
                dataset_to_check.targetProperties[1].task, TargetTasks.REGRESSION
            )
            self.assertEqual(len(dataset_to_check.X), len(dataset_to_check))
            self.assertEqual(len(dataset_to_check.y), len(dataset_to_check))
            self.assertEqual(len(dataset_to_check.y.columns), 2)
            self.assertEqual(dataset_to_check.y.columns[0], "CL")
            self.assertEqual(dataset_to_check.y.columns[1], "fu")

        # Check that the dataset is correctly initialized
        check_multiclass(dataset)

        # Check the dataset after dropping a task
        dataset.dropTask("fu")

        def check_singleclass(dataset_to_check):
            self.assertFalse(dataset_to_check.isMultiTask)
            self.assertEqual(dataset_to_check.nTasks, 1)
            self.assertEqual(len(dataset_to_check.targetProperties), 1)
            self.assertEqual(dataset_to_check.targetProperties[0].name, "CL")
            self.assertEqual(
                dataset_to_check.targetProperties[0].task, TargetTasks.REGRESSION
            )
            self.assertEqual(len(dataset_to_check.X), len(dataset_to_check))
            self.assertEqual(len(dataset_to_check.y), len(dataset_to_check))
            self.assertEqual(len(dataset_to_check.y.columns), 1)
            self.assertEqual(dataset_to_check.y.columns[0], "CL")

        check_singleclass(dataset)

        with self.assertRaises(AssertionError):
            dataset.dropTask("fu")

        with self.assertRaises(AssertionError):
            dataset.dropTask("CL")

        # Check the dataset after adding a task
        dataset.addTask({"name": "fu", "task": TargetTasks.REGRESSION})
        check_multiclass(dataset)

    def test_target_property(self):
        """Test target property creation and serialization in the context of a
        dataset."""
        dataset = QSPRDataset(
            "test_target_property",
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
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )

        def test_bad_init(dataset_to_test):
            with self.assertRaises(AssertionError):
                dataset_to_test.makeClassification("CL", [])
            with self.assertRaises(AssertionError):
                dataset_to_test.makeClassification("CL", th=6.5)
            with self.assertRaises(AssertionError):
                dataset_to_test.makeClassification("CL", th=[0, 2, 3])
            with self.assertRaises(AssertionError):
                dataset_to_test.makeClassification("CL", th=[0, 2, 3])

        def test_classification(dataset_to_test, target_names, ths):
            # Test that the dataset properties are correctly initialized
            for idx, target_prop in enumerate(dataset_to_test.targetProperties):
                if len(ths[idx]) == 1:
                    self.assertEqual(target_prop.task, TargetTasks.SINGLECLASS)
                else:
                    self.assertEqual(target_prop.task, TargetTasks.MULTICLASS)
                self.assertEqual(target_prop.name, f"{target_names[idx]}_class")
                self.assertEqual(target_prop.originalName, f"{target_names[idx]}")
                y = dataset_to_test.getTargetPropertiesValues(concat=True)
                self.assertTrue(y.columns[idx] == target_prop.name)
                if target_prop.task == TargetTasks.SINGLECLASS:
                    self.assertEqual(y[target_prop.name].unique().shape[0], 2)
                else:
                    self.assertEqual(
                        y[target_prop.name].unique().shape[0], (len(ths[idx]) - 1)
                    )
                self.assertEqual(target_prop.th, ths[idx])

        # Check that the make classification method works as expected
        test_bad_init(dataset)
        dataset.makeClassification("CL", th=[6.5])
        dataset.makeClassification("fu", th=[0.3])
        test_classification(dataset, ["CL", "fu"], [[6.5], [0.3]])
        dataset.makeClassification("CL", th=[0, 15, 30, 60])
        test_classification(dataset, ["CL", "fu"], [[0, 15, 30, 60], [0.3]])
        dataset.save()

        # check precomputed th
        dataset = QSPRDataset(
            "test_target_property",
            [{
                "name": "CL_class",
                "task": TargetTasks.MULTICLASS,
                "th": "precomputed"
            }],
            df=dataset.df,
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )
        self.assertEqual(dataset.targetProperties[0].task, TargetTasks.MULTICLASS)
        self.assertEqual(dataset.targetProperties[0].name, "CL_class_class")
        self.assertEqual(dataset.targetProperties[0].nClasses, 3)
        self.assertEqual(dataset.targetProperties[0].th, "precomputed")

        # Check that the dataset is correctly loaded from file for classification
        dataset_new = QSPRDataset.fromFile(dataset.storePath)
        test_bad_init(dataset_new)
        test_classification(dataset_new, ["CL", "fu"], [[0, 15, 30, 60], [0.3]])

        # Check that the make regression method works as expected
        dataset_new.makeRegression(target_property="CL")
        dataset_new.makeRegression(target_property="fu")

        def check_regression(dataset_to_check, target_names):
            for idx, target_prop in enumerate(dataset_to_check.targetProperties):
                self.assertEqual(target_prop.task, TargetTasks.REGRESSION)
                self.assertTrue(dataset_to_check.hasProperty(target_names[idx]))
                self.assertEqual(target_prop.name, target_names[idx])
                self.assertEqual(target_prop.originalName, target_names[idx])
                dataset_to_check.getTargetPropertiesValues(concat=True)

        # Check that the dataset is correctly loaded from file for regression
        check_regression(dataset_new, ["CL", "fu"])
        dataset_new.save()
        dataset_new = QSPRDataset.fromFile(dataset.storePath)
        check_regression(dataset_new, ["CL", "fu"])

    def test_indexing(self):
        # default index
        QSPRDataset(
            "test_target_property",
            [{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
            df=self.getSmallDF(),
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )

        # set index to SMILES column
        QSPRDataset(
            "test_target_property",
            [{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
            df=self.getSmallDF(),
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
            index_cols=["SMILES"],
        )

        # multiindex
        QSPRDataset(
            "test_target_property",
            [{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
            df=self.getSmallDF(),
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
            index_cols=["SMILES", "Name"],
        )

        # index with duplicates
        self.assertRaises(
            ValueError,
            lambda: QSPRDataset(
                "test_target_property",
                [{
                    "name": "CL",
                    "task": TargetTasks.REGRESSION
                }],
                df=self.getSmallDF(),
                store_dir=self.qsprdatapath,
                n_jobs=N_CPU,
                chunk_size=CHUNK_SIZE,
                index_cols=["moka_ionState7.4"],
            ),
        )

        # index has nans
        self.assertRaises(
            ValueError,
            lambda: QSPRDataset(
                "test_target_property",
                [{
                    "name": "CL",
                    "task": TargetTasks.REGRESSION
                }],
                df=self.getSmallDF(),
                store_dir=self.qsprdatapath,
                n_jobs=N_CPU,
                chunk_size=CHUNK_SIZE,
                index_cols=["fu"],
            ),
        )

    @parameterized.expand([(1, ), (N_CPU, )])
    def test_invalids_detection(self, n_cpu):
        df = self.getBigDF()
        all_mols = len(df)
        dataset = QSPRDataset(
            "test_invalids_detection",
            [{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
            df=df,
            store_dir=self.qsprdatapath,
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


class TestTargetProperty(TestCase):
    """Test the TargetProperty class."""
    def test_target_property(self):
        """Check the TargetProperty class on target property creation and serialization.
        """
        def check_target_property(targetprop, name, task, original_name, th):
            # Check the target property creation consistency
            self.assertEqual(targetprop.name, name)
            self.assertEqual(targetprop.task, task)
            self.assertEqual(targetprop.originalName, original_name)
            if task.isClassification():
                self.assertTrue(targetprop.task.isClassification())
                self.assertEqual(targetprop.th, th)

        # Check the different task types
        targetprop = TargetProperty("CL", TargetTasks.REGRESSION)
        check_target_property(targetprop, "CL", TargetTasks.REGRESSION, "CL", None)

        targetprop = TargetProperty("CL", TargetTasks.MULTICLASS, th=[0, 1, 10, 1200])
        check_target_property(
            targetprop, "CL", TargetTasks.MULTICLASS, "CL", [0, 1, 10, 1200]
        )

        targetprop = TargetProperty("CL", TargetTasks.SINGLECLASS, th=[5])
        check_target_property(targetprop, "CL", TargetTasks.SINGLECLASS, "CL", [5])

        # check with precomputed values
        targetprop = TargetProperty(
            "CL", TargetTasks.SINGLECLASS, th="precomputed", n_classes=2
        )
        check_target_property(
            targetprop, "CL", TargetTasks.SINGLECLASS, "CL", "precomputed"
        )

        # Check from dictionary creation
        targetprop = TargetProperty.fromDict(
            {
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }
        )
        check_target_property(targetprop, "CL", TargetTasks.REGRESSION, "CL", None)

        targetprop = TargetProperty.fromDict(
            {
                "name": "CL",
                "task": TargetTasks.MULTICLASS,
                "th": [0, 1, 10, 1200]
            }
        )
        check_target_property(
            targetprop, "CL", TargetTasks.MULTICLASS, "CL", [0, 1, 10, 1200]
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
        check_target_property(targetprops[0], "CL", TargetTasks.REGRESSION, "CL", None)
        check_target_property(targetprops[1], "fu", TargetTasks.REGRESSION, "fu", None)
        self.assertEqual(
            TargetProperty.selectFromList(targetprops, "CL")[0], targetprops[0]
        )
        self.assertListEqual(TargetProperty.getNames(targetprops), ["CL", "fu"])

        targetprops = TargetProperty.toList(targetprops)
        self.assertIsInstance(targetprops, list)
        self.assertIsInstance(targetprops[0], dict)
        self.assertEqual(targetprops[0]["name"], "CL")
        self.assertEqual(targetprops[0]["task"], TargetTasks.REGRESSION)


class TestDataSplitters(DataSetsMixIn, TestCase):
    """Small tests to only check if the data splitters work on their own.

    The tests here should be used to check for all their specific parameters and edge
    cases."""
    def test_manualsplit(self):
        """Test the manual split function, where the split is done manually."""
        dataset = self.create_large_dataset()

        # Add extra column to the data frame to use for splitting
        dataset.df["split"] = "train"
        dataset.df.loc[dataset.df.sample(frac=0.1).index, "split"] = "test"

        split = ManualSplit(dataset.df["split"], "train", "test")
        dataset.prepareDataset(split=split)
        self.validate_split(dataset)

    def test_randomsplit(self):
        """Test the random split function, where the split is done randomly."""
        dataset = self.create_large_dataset()
        dataset.prepareDataset(split=RandomSplit(0.1))
        self.validate_split(dataset)

    def test_temporalsplit(self):
        """Test the temporal split function, where the split is done based on a time
        property."""
        dataset = self.create_large_dataset()
        split = TemporalSplit(
            dataset=dataset,
            timesplit=TIME_SPLIT_YEAR,
            timeprop="Year of first disclosure"
        )

        dataset.prepareDataset(split=split)
        self.validate_split(dataset)

        # test if dates higher than 2000 are in test set
        self.assertTrue(
            sum(dataset.X_ind["Year of first disclosure"] > TIME_SPLIT_YEAR) ==
            len(dataset.X_ind)
        )

    @parameterized.expand(
        [
            (Murcko(), True, None),
            (BemisMurcko(), False, ["ScaffoldSplit_0", "ScaffoldSplit_1"]),
        ]
    )
    def test_ScaffoldSplit(self, scaffold, shuffle, custom_test_list):
        dataset = self.create_large_dataset(name="ScaffoldSplit")
        split = ScaffoldSplit(scaffold, 0.1, shuffle, custom_test_list)
        dataset.prepareDataset(split=split)
        self.validate_split(dataset)

        # check that smiles in custom_test_list are in the test set
        if custom_test_list:
            self.assertTrue(
                all(mol_id in dataset.X_ind.index for mol_id in custom_test_list)
            )

    def test_serialization(self):
        """Test the serialization of dataset with datasplit."""
        dataset = self.create_large_dataset()
        split = ScaffoldSplit(Murcko(), 0.1)
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

        dataset_new = QSPRDataset.fromFile(dataset.storePath)
        self.validate_split(dataset_new)
        self.assertTrue(dataset_new.descriptorCalculators)
        self.assertTrue(dataset_new.feature_standardizer)
        self.assertTrue(dataset_new.fold_generator.featureStandardizer)
        self.assertTrue(len(dataset_new.featureNames) == N_BITS)
        self.assertTrue(all(mol_id in dataset_new.X_ind.index for mol_id in test_ids))
        self.assertTrue(all(mol_id in dataset_new.y_ind.index for mol_id in train_ids))

        dataset_new.clearFiles()


class TestFoldSplitters(DataSetsMixIn, TestCase):
    """Small tests to only check if the fold splitters work on their own.

    The tests here should be used to check for all their specific parameters and
    edge cases."""
    def validate_folds(self, dataset, more=None):
        """Check if the folds have the data they should have after splitting."""
        k = 0
        for (
            X_train,
            X_test,
            y_train,
            y_test,
            train_index,
            test_index,
        ) in dataset.createFolds():
            k += 1
            self.assertEqual(len(X_train), len(y_train))
            self.assertEqual(len(X_test), len(y_test))
            self.assertEqual(len(train_index), len(y_train))
            self.assertEqual(len(test_index), len(y_test))

            if more:
                more(X_train, X_test, y_train, y_test, train_index, test_index)

        self.assertEqual(k, 5)

    def test_defaults(self):
        """Test the default fold generator, which is a 5-fold cross validation."""
        # test default settings with regression
        dataset = self.create_large_dataset()
        dataset.addDescriptors(
            MoleculeDescriptorsCalculator(
                [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1024)]
            )
        )
        self.validate_folds(dataset)

        # test default settings with classification
        dataset.makeClassification("CL", th=[20])
        self.validate_folds(dataset)

        # test with a standarizer
        MAX_VAL = 2
        MIN_VAL = 1
        scaler = MinMaxScaler(feature_range=(MIN_VAL, MAX_VAL))
        dataset.prepareDataset(feature_standardizer=scaler)

        def check_min_max(X_train, X_test, y_train, y_test, train_index, test_index):
            self.assertTrue(np.max(X_train) == MAX_VAL)
            self.assertTrue(np.min(X_train) == MIN_VAL)
            self.assertTrue(np.max(X_test) == MAX_VAL)
            self.assertTrue(np.min(X_test) == MIN_VAL)

        self.validate_folds(dataset, more=check_min_max)


class TestDataFilters(DataSetsMixIn, TestCase):
    """Small tests to only check if the data filters work on their own.

    The tests here should be used to check for all their specific parameters and
    edge cases."""
    def test_Categoryfilter(self):
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


class TestTargetImputation(PathMixIn, TestCase):
    """Small tests to only check if the target imputation works on its own."""
    def setUp(self):
        """Set up the test Dataframe."""
        super().setUp()
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

    def test_imputation(self):
        """Test the imputation of missing values in the target properties."""
        self.dataset = QSPRDataset(
            "TestImputation",
            target_props=[
                {
                    "name": "y",
                    "task": TargetTasks.REGRESSION
                },
                {
                    "name": "z",
                    "task": TargetTasks.REGRESSION
                },
            ],
            df=self.df,
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
            target_imputer=SimpleImputer(strategy="mean"),
        )
        self.assertEqual(self.dataset.targetProperties[0].originalName, "y")
        self.assertEqual(self.dataset.targetProperties[1].originalName, "z")
        self.assertEqual(self.dataset.targetProperties[0].name, "y_imputed")
        self.assertEqual(self.dataset.targetProperties[1].name, "z_imputed")
        self.assertEqual(self.dataset.df["z_imputed"].isna().sum(), 0)


class TestFeatureFilters(PathMixIn, TestCase):
    """Tests to check if the feature filters work on their own."""
    def setUp(self):
        """Set up the small test Dataframe."""
        super().setUp()
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
            target_props=[{
                "name": "y",
                "task": TargetTasks.REGRESSION
            }],
            df=self.df,
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )
        self.df_descriptors.index = self.dataset.df.index
        calculator = CustomDescriptorsCalculator(
            [DataFrameDescriptorSet(self.df_descriptors)]
        )
        self.dataset.addCustomDescriptors(calculator)
        self.descriptors = self.dataset.featureNames

    def test_lowVarianceFilter(self):
        """Test the low variance filter, which drops features with a variance below
        a threshold."""
        self.dataset.filterFeatures([LowVarianceFilter(0.01)])

        # check if correct columns selected and values still original
        self.assertListEqual(list(self.dataset.featureNames), self.descriptors[1:])
        self.assertListEqual(list(self.dataset.X.columns), self.descriptors[1:])

    def test_highCorrelationFilter(self):
        """Test the high correlation filter, which drops features with a correlation
        above a threshold."""
        self.dataset.filterFeatures([HighCorrelationFilter(0.8)])

        # check if correct columns selected and values still original
        self.descriptors.pop(2)
        self.assertListEqual(list(self.dataset.featureNames), self.descriptors)
        self.assertListEqual(list(self.dataset.X.columns), self.descriptors)

    def test_BorutaFilter(self):
        """Test the Boruta filter, which removes the features which are statistically as
        relevant as random features."""
        self.dataset.filterFeatures([BorutaFilter()])

        # check if correct columns selected and values still original
        self.assertListEqual(list(self.dataset.featureNames), self.descriptors[-1:])
        self.assertListEqual(list(self.dataset.X.columns), self.descriptors[-1:])


class TestDescriptorCalculation(DataSetsMixIn, TestCase):
    """Test the calculation of descriptors."""
    def setUp(self):
        """Set up the test Dataframe."""
        super().setUp()
        self.dataset = self.create_large_dataset(self.__class__.__name__)

    def test_switching(self):
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
        dataset_next = self.create_large_dataset(self.__class__.__name__)
        dataset_next.prepareDataset(
            split=split,
            feature_calculators=[feature_calculator],
            feature_filters=[lv, hc],
            recalculate_features=True,
            feature_fill_value=np.nan,
        )


class TestDescriptorsets(DataSetsMixIn, TestCase):
    """Test the descriptor sets."""
    def setUp(self):
        """Create the test Dataframe."""
        super().setUp()
        self.dataset = self.create_small_dataset(self.__class__.__name__)
        self.dataset.shuffle()

    def test_PredictorDesc(self):
        """Test the PredictorDesc descriptor set."""
        # give path to saved model parameters
        meta_path = (
            f"{os.path.dirname(__file__)}/test_files/test_predictor/"
            "qspr/models/SVC_MULTICLASS/SVC_MULTICLASS_meta.json"
        )
        from qsprpred.models.models import QSPRsklearn

        model = QSPRsklearn.fromFile(meta_path)
        desc_calc = MoleculeDescriptorsCalculator([PredictorDesc(model)])

        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 1))
        self.assertTrue(self.dataset.X.any().any())

        # test from file instantiation
        desc_calc.toFile(f"{self.qsprdatapath}/test_calc.json")
        desc_calc_file = MoleculeDescriptorsCalculator.fromFile(
            f"{self.qsprdatapath}/test_calc.json"
        )
        self.dataset.addDescriptors(desc_calc_file, recalculate=True)
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 1))
        self.assertTrue(self.dataset.X.any().any())

    def test_fingerprintSet(self):
        """Test the fingerprint set descriptor calculator."""
        desc_calc = MoleculeDescriptorsCalculator(
            [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1000)]
        )
        self.dataset.addDescriptors(desc_calc)

        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 1000))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    def test_TanimotoDistances(self):
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

    def test_DrugExPhyschem(self):
        """Test the DrugExPhyschem descriptor calculator."""
        desc_calc = MoleculeDescriptorsCalculator([DrugExPhyschem()])
        self.dataset.addDescriptors(desc_calc)

        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 19))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    def test_RDKitDescs(self):
        """Test the rdkit descriptors calculator."""
        desc_calc = MoleculeDescriptorsCalculator([RDKitDescs()])
        self.dataset.addDescriptors(desc_calc)

        self.assertEqual(
            self.dataset.X.shape, (len(self.dataset), len(Descriptors._descList))
        )
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

        # with 3D
        desc_calc = MoleculeDescriptorsCalculator([RDKitDescs(compute_3Drdkit=True)])
        self.dataset.addDescriptors(desc_calc, recalculate=True)
        self.assertEqual(
            self.dataset.X.shape, (len(self.dataset), len(Descriptors._descList) + 10)
        )

    def test_consistency(self):
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


class TestScaffolds(DataSetsMixIn, TestCase):
    """Test calculation of scaffolds."""
    def setUp(self):
        """Create a small dataset."""
        super().setUp()
        self.dataset = self.create_small_dataset(self.__class__.__name__)

    def test_scaffold_add(self):
        """Test the adding and getting of scaffolds."""
        self.dataset.addScaffolds([Murcko()])
        scaffs = self.dataset.getScaffolds()
        self.assertEqual(scaffs.shape, (len(self.dataset), 1))

        self.dataset.addScaffolds([Murcko()], add_rdkit_scaffold=True, recalculate=True)
        scaffs = self.dataset.getScaffolds(includeMols=True)
        self.assertEqual(scaffs.shape, (len(self.dataset), 2))
        for mol in scaffs[f"Scaffold_{Murcko()}_RDMol"]:
            self.assertTrue(isinstance(mol, Chem.rdchem.Mol))


class TestFeatureStandardizer(DataSetsMixIn, TestCase):
    """Test the feature standardizer."""
    def setUp(self):
        """Create a small test dataset with MorganFP descriptors."""
        super().setUp()
        self.dataset = self.create_small_dataset(self.__class__.__name__)
        self.dataset.addDescriptors(
            MoleculeDescriptorsCalculator(
                [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1000)]
            )
        )

    def test_featurestandarizer(self):
        """Test the feature standardizer fitting, transforming and serialization."""
        scaler = SKLearnStandardizer.fromFit(self.dataset.X, StandardScaler())
        scaled_features = scaler(self.dataset.X)
        scaler.toFile(
            f"{os.path.dirname(__file__)}/test_files/qspr/data/test_scaler.json"
        )
        scaler_fromfile = SKLearnStandardizer.fromFile(
            f"{os.path.dirname(__file__)}/test_files/qspr/data/test_scaler.json"
        )
        scaled_features_fromfile = scaler_fromfile(self.dataset.X)
        self.assertIsInstance(scaled_features, np.ndarray)
        self.assertEqual(scaled_features.shape, (len(self.dataset), 1000))
        self.assertEqual(
            np.array_equal(scaled_features, scaled_features_fromfile), True
        )


class TestStandardizers(DataSetsMixIn, TestCase):
    """Test the standardizers."""
    def test_invalid_filter(self):
        """Test the invalid filter."""
        df = self.getSmallDF()
        orig_len = len(df)
        mask = [False] * orig_len
        mask[0] = True
        df.loc[mask, "SMILES"
              ] = "C(C)(C)(C)(C)(C)(C)(C)(C)(C)"  # molecule with bad valence as example

        dataset = QSPRDataset(
            "standardization_test_invalid_filter",
            df=df,
            target_props=[{
                "name": "CL",
                "task": TargetTasks.REGRESSION
            }],
            drop_invalids=False,
            drop_empty=False,
        )
        dataset.standardizeSmiles("chembl", drop_invalid=False)
        self.assertEqual(len(dataset), len(df))
        dataset.standardizeSmiles("chembl", drop_invalid=True)
        self.assertEqual(len(dataset), orig_len - 1)


class DataPrepTestMixIn(DescriptorCheckMixIn):
    """Mixin for testing data preparation."""
    def check_prep(
        self,
        dataset,
        feature_calculators,
        split,
        feature_standardizer,
        feature_filter,
        data_filter,
        expected_target_props,
    ):
        """Check the consistency of the dataset after preparation."""
        name = dataset.name

        # if a split needs a dataset, give it one
        if split and hasattr(split, "setDataSet"):
            split.setDataSet(None)
            self.assertRaises(ValueError, split.getDataSet)
            split.setDataSet(dataset)
            self.assertEquals(dataset, split.getDataSet())

        # prepare the dataset and check consistency
        dataset.prepareDataset(
            feature_calculators=feature_calculators,
            split=split if split else None,
            feature_standardizer=feature_standardizer if feature_standardizer else None,
            feature_filters=[feature_filter] if feature_filter else None,
            datafilters=[data_filter] if data_filter else None,
        )
        expected_feature_count = len(dataset.featureNames)
        self.feature_consistency_checks(dataset, expected_feature_count)

        # save the dataset
        dataset.save()

        # reload the dataset and check consistency again
        dataset = QSPRDataset.fromFile(dataset.storePath)
        self.assertEqual(dataset.name, name)
        self.assertEqual(dataset.targetProperties[0].task, TargetTasks.REGRESSION)
        for idx, prop in enumerate(expected_target_props):
            self.assertEqual(dataset.targetProperties[idx].name, prop)
        for calc in dataset.descriptorCalculators:
            self.assertIsInstance(calc, DescriptorsCalculator)
        if feature_standardizer is not None:
            self.assertIsInstance(dataset.feature_standardizer, SKLearnStandardizer)
        else:
            self.assertIsNone(dataset.feature_standardizer)
        self.feature_consistency_checks(dataset, expected_feature_count)


class TestDataSetPreparation(DataSetsMixIn, DataPrepTestMixIn, TestCase):
    """Test as many possible combinations of data sets and their preparation
    settings."""
    @parameterized.expand(
        DataSetsMixIn.get_prep_combinations()
    )  # add @skip("Not now...") below this line to skip these tests
    def test_prep_combinations(
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
        exhaustive, but defined by `DataSetsMixIn.get_prep_combinations()`."""
        np.random.seed(42)
        dataset = self.create_large_dataset(name=name)
        self.check_prep(
            dataset,
            feature_calculators,
            split,
            feature_standardizer,
            feature_filter,
            data_filter,
            ["CL"],
        )


class TestDescriptorInDataMixIn(DescriptorCheckMixIn):
    """Mixin for testing descriptor sets in data sets."""
    def get_ds_name(self, desc_set, target_props):
        """Get a unique name for a data set."""
        target_props_id = [
            f"{target_prop['name']}_{target_prop['task']}"
            for target_prop in target_props
        ]
        return f"{desc_set}_{target_props_id}"

    def get_calculators(self, desc_sets):
        """Get the calculators for a descriptor set."""
        return [MoleculeDescriptorsCalculator(desc_sets)]

    def check_desc_in_dataset(self, dataset, desc_set, prep_combo, target_props):
        """Check if a descriptor set is in a data set."""
        # run the preparation
        logging.debug(f"Testing descriptor set: {desc_set} in data set: {dataset.name}")
        descriptor_sets = [desc_set]
        preparation = {}
        preparation.update(prep_combo)
        preparation["feature_calculators"] = self.get_calculators(descriptor_sets)
        dataset.prepareDataset(**preparation)

        # test consistency
        self.desc_consistency_check(dataset, target_props)


class TestDescriptorsAll(DataSetsMixIn, TestDescriptorInDataMixIn, TestCase):
    """Test all descriptor sets in all data sets."""
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
            ) for desc_set in DataSetsMixIn.get_all_descriptors()
        ] + [
            (
                f"{desc_set}_{TargetTasks.REGRESSION}",
                desc_set,
                [{
                    "name": "CL",
                    "task": TargetTasks.REGRESSION
                }],
            ) for desc_set in DataSetsMixIn.get_all_descriptors()
        ] + [
            (
                f"{desc_set}_Multitask",
                desc_set,
                [
                    {
                        "name": "CL",
                        "task": TargetTasks.REGRESSION
                    },
                    {
                        "name": "fu",
                        "task": TargetTasks.SINGLECLASS,
                        "th": [0.3]
                    },
                ],
            ) for desc_set in DataSetsMixIn.get_all_descriptors()
        ]
    )
    def test_descriptors_all(self, _, desc_set, target_props):
        """Tests all available descriptor sets.

        Note that they are not checked with all possible settings and all possible
        preparations, but only with the default settings provided by
        `DataSetsMixIn.get_default_prep()`. The list itself is defined and configured by
        `DataSetsMixIn.get_all_descriptors()`, so if you need a specific descriptor
        tested, add it there.
        """
        np.random.seed(42)

        dataset = self.create_large_dataset(
            name=self.get_ds_name(desc_set, target_props), target_props=target_props
        )

        self.check_desc_in_dataset(
            dataset, desc_set, self.get_default_prep(), target_props
        )
