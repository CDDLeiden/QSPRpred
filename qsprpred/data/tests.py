"""This module holds the test for functions regarding QSPR data preparation."""
import copy
import glob
import itertools
import logging
import os
import platform
import shutil
from unittest import TestCase, skipIf, skip

import mordred
import numpy as np
import pandas as pd
from mordred import descriptors as mordreddescriptors
from parameterized import parameterized

from qsprpred.data.data import QSPRDataset
from qsprpred.data.utils.datafilters import CategoryFilter
from qsprpred.data.utils.datasplitters import randomsplit, scaffoldsplit, temporalsplit
from qsprpred.data.utils.descriptorcalculator import DescriptorsCalculator
from qsprpred.data.utils.descriptorsets import (
    DrugExPhyschem,
    FingerprintSet,
    Mordred,
    PredictorDesc,
    TanimotoDistances,
    rdkit_descs,
    Mold2,
    PaDEL, DescriptorSet,
)
from qsprpred.data.utils.feature_standardization import SKLearnStandardizer
from qsprpred.data.utils.featurefilters import (
    BorutaFilter,
    highCorrelationFilter,
    lowVarianceFilter,
)
from qsprpred.data.utils.scaffolds import Murcko, BemisMurcko
from qsprpred.logs.stopwatch import StopWatch
from qsprpred.models.models import QSPRsklearn
from qsprpred.models.tasks import ModelTasks
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

N_CPU = 2
CHUNK_SIZE = 100
logging.basicConfig(level=logging.DEBUG)

class PathMixIn:
    """
    Mix-in class that provides paths to test files and directories and handles their creation and deletion.
    """

    datapath = f'{os.path.dirname(__file__)}/test_files/data'
    qsprdatapath = f'{os.path.dirname(__file__)}/test_files/qspr/data'

    def setUp(self):
        self.tearDown()
        if not os.path.exists(self.qsprdatapath):
            os.makedirs(self.qsprdatapath)

    def tearDown(self):
        self.clean_directories()
        if os.path.exists(self.qsprdatapath):
            shutil.rmtree(self.qsprdatapath)
        for extension in ['log', 'pkg', 'json']:
            globs = glob.glob(f'{self.datapath}/*.{extension}')
            for path in globs:
                os.remove(path)

    @classmethod
    def clean_directories(cls):
        if os.path.exists(cls.qsprdatapath):
            shutil.rmtree(cls.qsprdatapath)


class DataSetsMixIn(PathMixIn):
    """
    Mix-in class that provides a small and large testing data set and some common preparation
    settings to use in tests.

    """

    @staticmethod
    def get_default_prep():
        return {
            "feature_calculator": DescriptorsCalculator(
                [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1024)]),
            "split": randomsplit(0.1),
            "feature_standardizer": StandardScaler(),
            "feature_filters": [
                lowVarianceFilter(0.05),
                highCorrelationFilter(0.8)
            ],
        }

    @staticmethod
    def get_all_descriptors():
        """
        Returns a list of all available descriptor calculators. Might still not be a complete list, though. TODO: would be nice to create the list automatically.

        Returns:
            list: `list` of `DescriptorCalculator` objects
        """

        descriptor_sets = [
            rdkit_descs(),
            DrugExPhyschem(),
            PredictorDesc(
                QSPRsklearn.fromFile(
                    f'{os.path.dirname(__file__)}/test_files/test_predictor/qspr/models/SVC_CLASSIFICATION/SVC_CLASSIFICATION_meta.json')
            ),
            TanimotoDistances(list_of_smiles=["C", "CC", "CCC"], fingerprint_type="MorganFP", radius=3, nBits=1000),
            FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=2048),
            Mordred(),
            Mold2(),
        ]
        if platform.system() != "Linux":
            # FIXME: Java-based descriptors do not run on Linux
            descriptor_sets.extend([
                FingerprintSet(fingerprint_type="CDKFP", searchDepth=7, size=2048),
                FingerprintSet(fingerprint_type="CDKExtendedFP", searchDepth=7, size=2048),
                FingerprintSet(fingerprint_type="CDKEStatedFP"),
                FingerprintSet(fingerprint_type="CDKGraphOnlyFP", searchDepth=7, size=2048),
                FingerprintSet(fingerprint_type="CDKMACCSFP"),
                FingerprintSet(fingerprint_type="CDKPubchemFP"),
                FingerprintSet(fingerprint_type="CDKSubstructureFP", useCounts=False),
                FingerprintSet(fingerprint_type="CDKKlekotaRothFP", useCounts=True),
                FingerprintSet(fingerprint_type="CDKAtomPairs2DFP", useCounts=False),
                FingerprintSet(fingerprint_type="CDKSubstructureFP", useCounts=True),
                FingerprintSet(fingerprint_type="CDKKlekotaRothFP", useCounts=False),
                FingerprintSet(fingerprint_type="CDKAtomPairs2DFP", useCounts=True),
                PaDEL(),
            ])

        return descriptor_sets

    @staticmethod
    def get_prep_grid():
        """
        Returns a list of many possible combinations of descriptor calculators, splits, feature standardizers, feature filters and data filters. Again, this is not exhaustive, but should cover a lot of cases.

        Returns:
            grid: a generator that yields tuples of all possible combinations as stated above, each tuple is defined as: `(descriptor_calculator, split, feature_standardizer, feature_filters, data_filters)
        """

        # feature sets that can be used to create combinations of descriptor calculators
        feature_sets = [
            FingerprintSet(
                fingerprint_type="MorganFP",
                radius=3,
                nBits=1024
            ),
            rdkit_descs(),
            DrugExPhyschem()
        ]
        # interesting feature set combinations as descriptor calculators (either 1 or 2 sets at the same time)
        descriptor_calculators = [
        DescriptorsCalculator(combo) for combo in itertools.combinations(
            feature_sets, 1
        )] + [
        DescriptorsCalculator(combo) for combo in itertools.combinations(
            feature_sets, 2
        )]

        # lists with common preparation settings
        splits = [
            None,
            randomsplit(0.1),
            temporalsplit(timesplit=2000,timeprop="Year of first disclosure"),
            scaffoldsplit(test_fraction=0.1, scaffold=Murcko()),
            scaffoldsplit(test_fraction=0.1, scaffold=BemisMurcko()),
        ]
        feature_standardizers = [
            None,
            StandardScaler(),
            SKLearnStandardizer(scaler=MinMaxScaler())
        ]
        feature_filters = [
            None,
            BorutaFilter(max_iter=3, alpha=0.5),
            lowVarianceFilter(0.05),
            highCorrelationFilter(0.8)
        ]
        data_filters = [
            None,
            CategoryFilter(
                name="moka_ionState7.4",
                values=["cationic"]
            ),
        ]

        # grid of all combinations of the above preparation settings (passed to prepareDataset)
        return (
        copy.deepcopy(combo) for combo in itertools.product( # deep copy to avoid conflicts resulting from operating on one instance twice
            descriptor_calculators,
            splits,
            feature_standardizers,
            feature_filters,
            data_filters
        ))

    @staticmethod
    def get_prep_combinations():
        def get_name(thing):
            return str(None) if thing is None else thing.__class__.__name__ if not type(thing) in [
                DescriptorsCalculator, SKLearnStandardizer] else str(thing)

        return [
            2 * ["_".join(get_name(i) for i in x)] + list(x)
            for x in DataSetsMixIn.get_prep_grid()
        ]

    def getBigDF(self):
        """
        Get a large data frame for testing purposes.

        Returns:
            pd.DataFrame: a `pandas.DataFrame` containing the dataset
        """
        return pd.read_csv(
            f'{self.datapath}/test_data_large.tsv',
            sep='\t')

    def getSmallDF(self):
        """
        Get a small data frame for testing purposes.

        Returns:
            pd.DataFrame: a `pandas.DataFrame` containing the dataset
        """
        return pd.read_csv(
            f'{self.datapath}/test_data.tsv',
            sep='\t').sample(10)

    def create_large_dataset(self, name="QSPRDataset_test", task=ModelTasks.REGRESSION, target_prop='CL', th=None,
                             preparation_settings=None):
        """
        Create a large dataset for testing purposes.

        Args:
            name (str): name of the dataset
            task (ModelTasks): task of the dataset
            target_prop (str): name of the target property
            th (list): threshold for classification tasks
            preparation_settings (dict): dictionary containing preparation settings

        Returns:
            QSPRDataset: a `QSPRDataset` object
        """

        return self.create_dataset(
            self.getBigDF(),
            name=name,
            task=task,
            target_prop=target_prop,
            th=th,
            prep=preparation_settings
        )

    def create_small_dataset(self, name="QSPRDataset_test", task=ModelTasks.REGRESSION, target_prop='CL', th=None,
                             preparation_settings=None):
        """
        Create a small dataset for testing purposes.

        Args:
            name (str): name of the dataset
            task (ModelTasks): task of the dataset
            target_prop (str): name of the target property
            th (list): threshold for classification tasks
            preparation_settings (dict): dictionary containing preparation settings

        Returns:
            QSPRDataset: a `QSPRDataset` object
        """

        return self.create_dataset(
            self.getSmallDF(),
            name=name,
            task=task,
            target_prop=target_prop,
            th=th,
            prep=preparation_settings
        )

    def create_dataset(self, df, name="QSPRDataset_test", task=ModelTasks.REGRESSION, target_prop='CL', th=None, prep=None):
        """
        Create a dataset for testing purposes from the given data frame.

        Args:
            df (pd.DataFrame): data frame containing the dataset
            name (str): name of the dataset
            task (ModelTasks): task of the dataset
            target_prop (str): name of the target property
            th (list): threshold for classification tasks
            prep (dict): dictionary containing preparation settings

        Returns:
            QSPRDataset: a `QSPRDataset` object
        """

        ret = QSPRDataset(
            name, target_prop=target_prop, task=task, df=df,
            store_dir=self.qsprdatapath, n_jobs=N_CPU, chunk_size=CHUNK_SIZE, th=th)
        if prep:
            ret.prepareDataset(**prep)
        return ret


class TestDataSetCreationSerialization(DataSetsMixIn, TestCase):
    """
    Simple tests for dataset creation and serialization under different conditions and error states.
    """

    def test_defaults(self):
        # creation from data frame
        dataset = QSPRDataset(
            "test_defaults",
            "CL",
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
        stopwatch.stop('Saving took: ')
        self.assertTrue(os.path.exists(dataset.storePath))

        def check_consistency(dataset_to_check):
            self.assertNotIn("Notes", dataset_to_check.getProperties())
            self.assertNotIn("HBD", dataset_to_check.getProperties())
            self.assertTrue(len(self.getSmallDF()) - 1 == len(dataset_to_check))
            self.assertEqual(dataset_to_check.task, ModelTasks.REGRESSION)
            self.assertTrue(dataset_to_check.hasProperty("CL"))
            self.assertEqual(dataset_to_check.targetProperty, "CL")
            self.assertEqual(dataset_to_check.originalTargetProperty, "CL")
            self.assertEqual(len(dataset_to_check.X), len(dataset_to_check))
            self.assertEqual(len(dataset_to_check.X_ind), 0)
            self.assertEqual(len(dataset_to_check.y), len(dataset_to_check))
            self.assertEqual(len(dataset_to_check.y_ind), 0)

        # creation from file
        stopwatch.reset()
        dataset_new = QSPRDataset.fromFile(dataset.storePath)
        stopwatch.stop('Loading from file took: ')
        check_consistency(dataset_new)

        # creation by reinitialization
        stopwatch.reset()
        dataset_new = QSPRDataset(
            "test_defaults",
            "CL",
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )
        stopwatch.stop('Reinitialization took: ')
        check_consistency(dataset_new)

        # creation from a table file
        stopwatch.reset()
        dataset_new = QSPRDataset.fromTableFile(
            "test_defaults",
            f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv',
            target_prop="CL",
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )
        stopwatch.stop('Loading from table file took: ')
        self.assertTrue(isinstance(dataset_new, QSPRDataset))
        check_consistency(dataset_new)

        dataset_new = QSPRDataset.fromTableFile(
            "test_defaults_new",  # new name implies HBD below should exist again
            f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv',
            target_prop="CL",
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )
        self.assertTrue(isinstance(dataset_new, QSPRDataset))
        self.assertIn("HBD", dataset_new.getProperties())
        dataset_new.removeProperty("HBD")
        check_consistency(dataset_new)

    def test_target_property(self):
        dataset = QSPRDataset(
            "test_target_property",
            "CL",
            df=self.getSmallDF(),
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )

        def test_bad_init(dataset_to_test):
            with self.assertRaises(AssertionError):
                dataset_to_test.makeClassification([])
            with self.assertRaises(TypeError):
                dataset_to_test.makeClassification(th=6.5)
            with self.assertRaises(AssertionError):
                dataset_to_test.makeClassification(th=[0, 2, 3])
            with self.assertRaises(AssertionError):
                dataset_to_test.makeClassification(th=[0, 2, 3])

        def test_classification(dataset_to_test):
            self.assertEqual(dataset_to_test.task, ModelTasks.CLASSIFICATION)
            self.assertEqual(dataset_to_test.targetProperty, "CL_class")
            self.assertEqual(dataset_to_test.originalTargetProperty, "CL")
            y = dataset_to_test.getTargetProperties(concat=True)
            self.assertTrue(y.columns[0] == dataset_to_test.targetProperty)
            if len(th) == 1:
                self.assertEqual(y[dataset_to_test.targetProperty].unique().shape[0], 2)
            else:
                self.assertEqual(y[dataset_to_test.targetProperty].unique().shape[0], (len(th) - 1))
            self.assertEqual(dataset_to_test.th, th)

        th = [6.5]
        test_bad_init(dataset)
        dataset.makeClassification(th=th)
        test_classification(dataset)
        th = [0, 15, 30, 60]
        dataset.makeClassification(th=th)
        test_classification(dataset)
        dataset.save()

        dataset_new = QSPRDataset.fromFile(dataset.storePath)
        test_bad_init(dataset)
        test_classification(dataset_new)

        dataset_new.makeRegression(target_property="CL")

        def check_regression(dataset_to_check):
            self.assertEqual(dataset_to_check.task, ModelTasks.REGRESSION)
            self.assertTrue(dataset_to_check.hasProperty("CL"))
            self.assertEqual(dataset_to_check.targetProperty, "CL")
            self.assertEqual(dataset_to_check.originalTargetProperty, "CL")
            y = dataset_to_check.getTargetProperties(concat=True)
            self.assertNotEqual(y[dataset_to_check.targetProperty].unique().shape[0], (len(th) - 1))

        check_regression(dataset_new)
        dataset_new.save()
        dataset_new = QSPRDataset.fromFile(dataset.storePath)
        check_regression(dataset_new)

class TestDataSplitters(DataSetsMixIn, TestCase):
    """
    Small tests to only check if the data splitters work on their own. The tests here should be used to check for all their specific parameters and edge cases.
    """

    def validate_split(self, dataset):
        self.assertTrue(dataset.X is not None)
        self.assertTrue(dataset.X_ind is not None)
        self.assertTrue(dataset.y is not None)
        self.assertTrue(dataset.y_ind is not None)

    def test_randomsplit(self):
        dataset = self.create_large_dataset()
        dataset.prepareDataset(split=randomsplit(0.1))
        self.validate_split(dataset)

    def test_temporalsplit(self):
        dataset = self.create_large_dataset()
        split = temporalsplit(
            dataset=dataset,
            timesplit=2000,
            timeprop="Year of first disclosure")

        dataset.prepareDataset(split=split)
        self.validate_split(dataset)

        # test if dates higher than 2000 are in test set
        self.assertTrue(sum(dataset.X_ind['Year of first disclosure'] > 2000) == len(dataset.X_ind))

    def test_scaffoldsplit(self):
        dataset = self.create_large_dataset()
        split = scaffoldsplit(dataset, Murcko(), 0.1)
        dataset.prepareDataset(split=split)
        self.validate_split(dataset)

    def test_serialization(self):
        dataset = self.create_large_dataset()
        split = scaffoldsplit(dataset, Murcko(), 0.1)
        calculator = DescriptorsCalculator([FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1024)])
        dataset.prepareDataset(
            split=split,
            feature_calculator=calculator,
            feature_standardizer=StandardScaler())
        self.validate_split(dataset)
        dataset.save()

        dataset_new = QSPRDataset.fromFile(dataset.storePath)
        self.validate_split(dataset_new)
        self.assertTrue(dataset_new.descriptorCalculator)
        self.assertTrue(dataset_new.feature_standardizer)
        self.assertTrue(dataset_new.fold_generator.featureStandardizer)
        self.assertTrue(len(dataset_new.featureNames) == 1024)

        dataset_new.clearFiles()


class TestFoldSplitters(DataSetsMixIn, TestCase):
    """
    Small tests to only check if the fold splitters work on their own. The tests here should be used to check for all their specific parameters and edge cases.
    """

    def validate_folds(self, dataset, more=None):
        k = 0
        for X_train, X_test, y_train, y_test, train_index, test_index in dataset.createFolds():
            k += 1
            self.assertEqual(len(X_train), len(y_train))
            self.assertEqual(len(X_test), len(y_test))
            self.assertEqual(len(train_index), len(y_train))
            self.assertEqual(len(test_index), len(y_test))

            if more:
                more(X_train, X_test, y_train, y_test, train_index, test_index)

        self.assertEqual(k, 5)

    def test_defaults(self):
        # test default settings with regression
        dataset = self.create_large_dataset()
        dataset.addDescriptors(DescriptorsCalculator(
            [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1024)]))
        self.validate_folds(dataset)

        # test default settings with classification
        dataset.makeClassification(th=[20])
        self.validate_folds(dataset)

        # test with a standarizer
        scaler = MinMaxScaler(feature_range=(1, 2))
        dataset.prepareDataset(feature_standardizer=scaler)

        def check_min_max(X_train, X_test, y_train, y_test, train_index, test_index):
            self.assertTrue(np.max(X_train) == 2)
            self.assertTrue(np.min(X_train) == 1)
            self.assertTrue(np.max(X_test) == 2)
            self.assertTrue(np.min(X_test) == 1)

        self.validate_folds(dataset, more=check_min_max)


class TestDataFilters(DataSetsMixIn, TestCase):
    """
    Small tests to only check if the data filters work on their own. The tests here should be used to check for all their specific parameters and edge cases.
    """

    def test_Categoryfilter(self):
        remove_cation = CategoryFilter(
            name="moka_ionState7.4", values=["cationic"])
        df_anion = remove_cation(self.getBigDF())
        self.assertTrue(
            (df_anion["moka_ionState7.4"] == "cationic").sum() == 0)

        only_cation = CategoryFilter(
            name="moka_ionState7.4",
            values=["cationic"],
            keep=True)
        df_cation = only_cation(self.getBigDF())
        self.assertTrue(
            (df_cation["moka_ionState7.4"] != "cationic").sum() == 0)


class TestFeatureFilters(PathMixIn, TestCase):

    def setUp(self):
        super().setUp()
        self.descriptors = ["Descriptor_F1", "Descriptor_F2", "Descriptor_F3", "Descriptor_F4", "Descriptor_F5"]
        self.df = pd.DataFrame(
            data=np.array(
                [["C", 1, 4, 2, 6, 2, 1],
                 ["C", 1, 8, 4, 2, 4, 2],
                 ["C", 1, 4, 3, 2, 5, 3],
                 ["C", 1, 8, 4, 9, 8, 4],
                 ["C", 1, 4, 2, 3, 9, 5],
                 ["C", 1, 8, 4, 7, 12, 6]]),
            columns=["SMILES"] + self.descriptors + ["y"]
        )
        self.dataset = QSPRDataset(
            "TestFeatureFilters",
            target_prop="y",
            df=self.df,
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE)

    def test_lowVarianceFilter(self):
        self.dataset.filterFeatures([lowVarianceFilter(0.01)])

        # check if correct columns selected and values still original
        self.assertListEqual(list(self.dataset.featureNames), self.descriptors[1:])
        self.assertListEqual(list(self.dataset.X.columns), self.descriptors[1:])

    def test_highCorrelationFilter(self):
        self.dataset.filterFeatures([highCorrelationFilter(0.8)])

        # check if correct columns selected and values still original
        self.descriptors.pop(2)
        self.assertListEqual(list(self.dataset.featureNames), self.descriptors)
        self.assertListEqual(list(self.dataset.X.columns), self.descriptors)

    def test_BorutaFilter(self):
        self.dataset.filterFeatures([BorutaFilter()])

        # check if correct columns selected and values still original
        self.assertListEqual(list(self.dataset.featureNames), self.descriptors[-1:])
        self.assertListEqual(list(self.dataset.X.columns), self.descriptors[-1:])


class TestDescriptorCalculation(DataSetsMixIn, TestCase):

    def setUp(self):
        super().setUp()
        self.dataset = self.create_large_dataset(self.__class__.__name__)

    def test_switching(self):
        feature_calculator = DescriptorsCalculator(
            descsets=[FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=2048), DrugExPhyschem()])
        split = randomsplit(test_fraction=0.1)
        lv = lowVarianceFilter(0.05)
        hc = highCorrelationFilter(0.9)

        self.dataset.prepareDataset(
            split=split,
            feature_calculator=feature_calculator,
            feature_filters=[lv, hc],
            recalculate_features=True,
            fill_value=np.nan
        )

        # create new dataset with different feature calculator
        dataset_next = self.create_large_dataset(self.__class__.__name__)
        dataset_next.prepareDataset(
            split=split,
            feature_calculator=feature_calculator,
            feature_filters=[lv, hc],
            recalculate_features=True,
            fill_value=np.nan
        )


class TestDescriptorsets(DataSetsMixIn, TestCase):

    def setUp(self):
        super().setUp()
        self.dataset = self.create_small_dataset(self.__class__.__name__)

    def test_PredictorDesc(self):
        # give path to saved model parameters
        meta_path = f'{os.path.dirname(__file__)}/test_files/test_predictor/qspr/models/SVC_CLASSIFICATION/SVC_CLASSIFICATION_meta.json'
        from qsprpred.models.models import QSPRsklearn
        model = QSPRsklearn.fromFile(meta_path)
        desc_calc = DescriptorsCalculator([PredictorDesc(model)])

        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 1))
        self.assertTrue(self.dataset.X.any().any())

        # test from file instantiation
        desc_calc.toFile(f"{self.qsprdatapath}/test_calc.json")
        desc_calc_file = DescriptorsCalculator.fromFile(f"{self.qsprdatapath}/test_calc.json")
        self.dataset.addDescriptors(desc_calc_file, recalculate=True)
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 1))
        self.assertTrue(self.dataset.X.any().any())

    def test_fingerprintSet(self):
        desc_calc = DescriptorsCalculator([FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1000)])
        self.dataset.addDescriptors(desc_calc)

        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 1000))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    def test_TanimotoDistances(self):
        list_of_smiles = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC"]
        desc_calc = DescriptorsCalculator(
            [TanimotoDistances(list_of_smiles=list_of_smiles, fingerprint_type="MorganFP", radius=3, nBits=1000)])
        self.dataset.addDescriptors(desc_calc)

    def test_Mordred(self):
        desc_calc = DescriptorsCalculator([Mordred()])
        self.dataset.addDescriptors(desc_calc)

        self.assertEqual(
            self.dataset.X.shape,
            (len(self.dataset), len(mordred.Calculator(mordreddescriptors).descriptors)))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    def test_Mold2(self):
        desc_calc = DescriptorsCalculator([Mold2()])
        self.dataset.addDescriptors(desc_calc)

        self.assertEqual(
            self.dataset.X.shape,
            (len(self.dataset), 777))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    # FIXME: PaDEL descriptors are not available on Linux
    @skipIf(platform.system() == "Linux", "PaDEL descriptors are not available on Linux")
    def test_PaDEL(self):
        desc_calc = DescriptorsCalculator([PaDEL()])
        self.dataset.addDescriptors(desc_calc)

        self.assertEqual(
            self.dataset.X.shape,
            (len(self.dataset), 1444))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    def test_DrugExPhyschem(self):
        desc_calc = DescriptorsCalculator([DrugExPhyschem()])
        self.dataset.addDescriptors(desc_calc)

        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 19))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    def test_rdkit_descs(self):
        desc_calc = DescriptorsCalculator([rdkit_descs()])
        self.dataset.addDescriptors(desc_calc)

        self.assertEqual(self.dataset.X.shape, (len(self.dataset), len(Descriptors._descList)))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

        # with 3D
        desc_calc = DescriptorsCalculator([rdkit_descs(compute_3Drdkit=True)])
        self.dataset.addDescriptors(desc_calc, recalculate=True)
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), len(Descriptors._descList) + 10))


class TestScaffolds(DataSetsMixIn, TestCase):

    def setUp(self):
        super().setUp()
        self.dataset = self.create_small_dataset(self.__class__.__name__)

    def test_scaffold_add(self):
        self.dataset.addScaffolds([Murcko()])
        scaffs = self.dataset.getScaffolds()
        self.assertEqual(scaffs.shape, (len(self.dataset), 1))

        self.dataset.addScaffolds([Murcko()], add_rdkit_scaffold=True, recalculate=True)
        scaffs = self.dataset.getScaffolds(includeMols=True)
        self.assertEqual(scaffs.shape, (len(self.dataset), 2))
        for mol in scaffs[f"Scaffold_{Murcko()}_RDMol"]:
            self.assertTrue(isinstance(mol, Chem.rdchem.Mol))


class TestFeatureStandardizer(DataSetsMixIn, TestCase):

    def setUp(self):
        super().setUp()
        self.dataset = self.create_small_dataset(self.__class__.__name__)
        self.dataset.addDescriptors(DescriptorsCalculator(
            [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1000)]))

    def test_featurestandarizer(self):
        scaler = SKLearnStandardizer.fromFit(self.dataset.X, StandardScaler())
        scaled_features = scaler(self.dataset.X)
        scaler.toFile(
            f'{os.path.dirname(__file__)}/test_files/qspr/data/test_scaler.json')
        scaler_fromfile = SKLearnStandardizer.fromFile(
            f'{os.path.dirname(__file__)}/test_files/qspr/data/test_scaler.json')
        scaled_features_fromfile = scaler_fromfile(self.dataset.X)
        self.assertIsInstance(scaled_features, np.ndarray)
        self.assertEqual(scaled_features.shape, (len(self.dataset), 1000))
        self.assertEqual(
            np.array_equal(
                scaled_features,
                scaled_features_fromfile),
            True)

class TestDataSetPreparation(DataSetsMixIn, TestCase):
    """
    Tests many possible combinations of data sets and their preparation settings.
    """

    def feature_consistency_checks(self, ds, expected_length):
        """
        Checks if the feature names and the feature matrix of a data set is consistent with expected number of variables.

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

    @parameterized.expand(
        DataSetsMixIn.get_prep_combinations()
    ) # add @skip("Not now...") below this line to skip these tests
    def test_prep_combinations(
            self,
            _,
            name,
            feature_calculator,
            split,
            feature_standardizer,
            feature_filter,
            data_filter
        ):
        """
        Tests one combination of a data set and its preparation settings. This generates a large number of parameterized tests. Use the `skip` decorator if you want to skip all these tests.
        Note that the combinations are not exhaustive, but defined by `DataSetsMixIn.get_prep_combinations()`.
        """

        # fetch a new data set
        dataset = self.create_small_dataset(name=name)

        # if a split needs a dataset, give it one
        if split and hasattr(split, "setDataSet"):
            split.setDataSet(None)
            self.assertRaises(ValueError, split.getDataSet)
            split.setDataSet(dataset)
            self.assertEquals(dataset, split.getDataSet())

        # prepare the dataset and check consistency
        dataset.prepareDataset(
            feature_calculator=feature_calculator,
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
        self.assertEqual(dataset.task, ModelTasks.REGRESSION)
        self.assertEqual(dataset.targetProperty, "CL")
        self.assertIsInstance(dataset.descriptorCalculator, feature_calculator.__class__)
        if feature_standardizer is not None:
            self.assertIsInstance(dataset.feature_standardizer, SKLearnStandardizer)
        else:
            self.assertIsNone(dataset.feature_standardizer)
        self.feature_consistency_checks(dataset, expected_feature_count)

    @parameterized.expand(
        [(f"{desc_set}_{ModelTasks.CLASSIFICATION}", desc_set, ModelTasks.CLASSIFICATION) for desc_set in DataSetsMixIn.get_all_descriptors()] +
        [(f"{desc_set}_{ModelTasks.REGRESSION}", desc_set, ModelTasks.REGRESSION) for desc_set in DataSetsMixIn.get_all_descriptors()]
    )
    def test_descriptors_all(self, _, desc_set, task):
        """
        Tests all available descriptor sets. Note that they are not checked with all possible settings and all possible preparations, but only with the default settings provided by `DataSetsMixIn.get_default_prep()`. The list itself is defined and configured by `DataSetsMixIn.get_all_descriptors()` so if you need a specific descriptor tested, add it there.
        """

        np.random.seed(42)

        # get the data set
        ds_name = f"{desc_set}_{task}" # unique name to avoid conflicts
        logging.debug(f"Testing data set: {ds_name}")
        dataset = self.create_large_dataset(
            name=ds_name,
            task=task,
            th=[0, 1, 10, 1200] if task == ModelTasks.CLASSIFICATION else None
        )

        # run the preparation
        descriptor_sets = [desc_set]
        preparation = dict()
        preparation.update(self.get_default_prep())
        preparation['feature_calculator'] = DescriptorsCalculator(descriptor_sets)
        dataset.prepareDataset(**preparation)

        # test some basic consistency rules on the resulting features
        expected_length = sum([len(x.descriptors) for x in descriptor_sets if x in dataset.descriptorCalculator])
        self.feature_consistency_checks(dataset, expected_length)

        # save to file and check if it can be loaded and the features are still there and correct
        dataset.save()
        ds_loaded = QSPRDataset.fromFile(dataset.storePath, n_jobs=N_CPU, chunk_size=CHUNK_SIZE)
        if ds_loaded.task == ModelTasks.CLASSIFICATION:
            self.assertEqual(ds_loaded.targetProperty, "CL_class")
        self.assertTrue(ds_loaded.task == task)
        self.assertTrue(ds_loaded.descriptorCalculator)
        self.assertTrue(
            isinstance(
                ds_loaded.descriptorCalculator,
                DescriptorsCalculator))
        for descset in ds_loaded.descriptorCalculator.descsets:
            self.assertTrue(isinstance(descset, DescriptorSet))
        self.feature_consistency_checks(dataset, expected_length)
