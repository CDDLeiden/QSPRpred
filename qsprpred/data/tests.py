"""This module holds the test for functions regarding QSPR data preparation."""
import os
import shutil
import time
from unittest import TestCase

import mordred
import numpy as np
import pandas as pd
import sklearn_json as skljson
from mordred import descriptors as mordreddescriptors
from qsprpred.data.data import QSPRDataset, TargetProperty
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
)
from qsprpred.data.utils.feature_standardization import SKLearnStandardizer
from qsprpred.data.utils.featurefilters import (
    BorutaFilter,
    highCorrelationFilter,
    lowVarianceFilter,
)
from qsprpred.data.utils.scaffolds import Murcko
from qsprpred.models.tasks import ModelTasks
from qsprpred.scorers.predictor import Predictor
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles
from sklearn.preprocessing import MinMaxScaler, StandardScaler

N_CPU = 2
CHUNK_SIZE = 100


class PathMixIn:
    datapath = f'{os.path.dirname(__file__)}/test_files/data'
    qsprmodelspath = f'{os.path.dirname(__file__)}/test_files/qspr/models'
    qsprdatapath = f'{os.path.dirname(__file__)}/test_files/qspr/data'

    @classmethod
    def setUpClass(cls):
        cls.clean_directories()
        if not os.path.exists(cls.qsprmodelspath):
            os.makedirs(cls.qsprmodelspath)
        if not os.path.exists(cls.qsprdatapath):
            os.makedirs(cls.qsprdatapath)

    @classmethod
    def tearDownClass(cls):
        cls.clean_directories()

    @classmethod
    def clean_directories(cls):
        if os.path.exists(cls.qsprmodelspath):
            shutil.rmtree(cls.qsprmodelspath)
        if os.path.exists(cls.qsprdatapath):
            shutil.rmtree(cls.qsprdatapath)


class DataSets(PathMixIn):
    df_large = pd.read_csv(
        f'{os.path.dirname(__file__)}/test_files/data/test_data_large.tsv',
        sep='\t')
    df_small = pd.read_csv(
        f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv',
        sep='\t').sample(10)

    def create_dataset(self, df, name="QSPRDataset_test", task=ModelTasks.REGRESSION, target_prop='CL'):
        return QSPRDataset(
            name, target_props=[{"name": target_prop, "task": task}], df=df,
            store_dir=self.qsprdatapath, n_jobs=N_CPU, chunk_size=CHUNK_SIZE)

    def create_small_dataset(self, name="QSPRDataset_test", task=ModelTasks.REGRESSION, target_prop='CL'):
        return self.create_dataset(self.df_small, name, task=task, target_prop=target_prop)

    def create_large_dataset(self, name="QSPRDataset_test", task=ModelTasks.REGRESSION, target_prop='CL'):
        return self.create_dataset(self.df_large, name, task=task, target_prop=target_prop)


class StopWatch:

    def __init__(self):
        self.start = time.perf_counter()

    def reset(self):
        self.start = time.perf_counter()

    def stop(self, msg='Time it took: '):
        ret = time.perf_counter() - self.start
        print(msg + str(ret))
        self.reset()
        return ret


class TestDataSetCreationSerialization(DataSets, TestCase):

    def test_defaults(self):
        # creation from data frame
        dataset = QSPRDataset(
            "test_defaults",
            [{"name": "CL", "task": ModelTasks.REGRESSION}],
            df=self.df_small,
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
            self.assertTrue(len(self.df_small) - 1 == len(dataset_to_check))
            self.assertEqual(dataset_to_check.targetProperties[0].task, ModelTasks.REGRESSION)
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
        stopwatch.stop('Loading from file took: ')
        check_consistency(dataset_new)

        # creation by reinitialization
        stopwatch.reset()
        dataset_new = QSPRDataset(
            "test_defaults",
            [{"name": "CL", "task": ModelTasks.REGRESSION}],
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
            target_props=[{"name": "CL", "task": ModelTasks.REGRESSION}],
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
            target_props=[{"name": "CL", "task": ModelTasks.REGRESSION}],
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )
        self.assertTrue(isinstance(dataset_new, QSPRDataset))
        self.assertIn("HBD", dataset_new.getProperties())
        dataset_new.removeProperty("HBD")
        check_consistency(dataset_new)

    def test_multi_task(self):
        # test multi task
        dataset = QSPRDataset(
            "test_multi_task",
            [{"name": "CL", "task": ModelTasks.REGRESSION},
             {"name": "fu", "task": ModelTasks.REGRESSION}],
            df=self.df_small,
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
            self.assertEqual(dataset_to_check.targetProperties[0].task, ModelTasks.REGRESSION)
            self.assertEqual(dataset_to_check.targetProperties[1].task, ModelTasks.REGRESSION)
            self.assertEqual(len(dataset_to_check.X), len(dataset_to_check))
            self.assertEqual(len(dataset_to_check.y), len(dataset_to_check))
            self.assertEqual(len(dataset_to_check.y.columns), 2)
            self.assertEqual(dataset_to_check.y.columns[0], "CL")
            self.assertEqual(dataset_to_check.y.columns[1], "fu")

        check_multiclass(dataset)

        dataset.dropTask("fu")

        def check_singleclass(dataset_to_check):
            self.assertFalse(dataset_to_check.isMultiTask)
            self.assertEqual(dataset_to_check.nTasks, 1)
            self.assertEqual(len(dataset_to_check.targetProperties), 1)
            self.assertEqual(dataset_to_check.targetProperties[0].name, "CL")
            self.assertEqual(dataset_to_check.targetProperties[0].task, ModelTasks.REGRESSION)
            self.assertEqual(len(dataset_to_check.X), len(dataset_to_check))
            self.assertEqual(len(dataset_to_check.y), len(dataset_to_check))
            self.assertEqual(len(dataset_to_check.y.columns), 1)
            self.assertEqual(dataset_to_check.y.columns[0], "CL")

        check_singleclass(dataset)

        with self.assertRaises(AssertionError):
            dataset.dropTask("fu")

        with self.assertRaises(AssertionError):
            dataset.dropTask("CL")

        dataset.addTask({"name": "fu", "task": ModelTasks.REGRESSION})
        check_multiclass(dataset)

    def test_target_property(self):
        dataset = QSPRDataset(
            "test_target_property",
            [{"name": "CL", "task": ModelTasks.REGRESSION},
             {"name": "fu", "task": ModelTasks.REGRESSION}],
            df=self.df_small,
            store_dir=self.qsprdatapath,
            n_jobs=N_CPU,
            chunk_size=CHUNK_SIZE,
        )

        def test_bad_init(dataset_to_test):
            with self.assertRaises(AssertionError):
                dataset_to_test.makeClassification("CL", [])
            with self.assertRaises(TypeError):
                dataset_to_test.makeClassification("CL", th=6.5)
            with self.assertRaises(AssertionError):
                dataset_to_test.makeClassification("CL", th=[0, 2, 3])
            with self.assertRaises(AssertionError):
                dataset_to_test.makeClassification("CL", th=[0, 2, 3])

        def test_classification(dataset_to_test, target_names, ths):
            for idx, target_prop in enumerate(dataset_to_test.targetProperties):
                if len(ths[idx]) == 1:
                    self.assertEqual(target_prop.task, ModelTasks.SINGLECLASS)
                else:
                    self.assertEqual(target_prop.task, ModelTasks.MULTICLASS)
                self.assertEqual(target_prop.name, f"{target_names[idx]}_class")
                self.assertEqual(target_prop.originalName, f"{target_names[idx]}")
                y = dataset_to_test.getTargetPropertiesValues(concat=True)
                self.assertTrue(y.columns[idx] == target_prop.name)
                if target_prop.task == ModelTasks.SINGLECLASS:
                    self.assertEqual(y[target_prop.name].unique().shape[0], 2)
                else:
                    self.assertEqual(y[target_prop.name].unique().shape[0], (len(ths[idx]) - 1))
                self.assertEqual(target_prop.th, ths[idx])

        test_bad_init(dataset)
        dataset.makeClassification("CL", th=[6.5])
        dataset.makeClassification("fu", th=[0.3])
        test_classification(dataset, ["CL", "fu"], [[6.5], [0.3]])
        dataset.makeClassification("CL", th=[0, 15, 30, 60])
        test_classification(dataset, ["CL", "fu"], [[0, 15, 30, 60], [0.3]])
        dataset.save()

        dataset_new = QSPRDataset.fromFile(dataset.storePath)
        test_bad_init(dataset)
        test_classification(dataset_new, ["CL", "fu"], [[0, 15, 30, 60], [0.3]])

        dataset_new.makeRegression(target_property="CL")
        dataset_new.makeRegression(target_property="fu")

        def check_regression(dataset_to_check, target_names, ths):
            for idx, target_prop in enumerate(dataset_to_check.targetProperties):
                self.assertEqual(target_prop.task, ModelTasks.REGRESSION)
                self.assertTrue(dataset_to_check.hasProperty(target_names[idx]))
                self.assertEqual(target_prop.name, target_names[idx])
                self.assertEqual(target_prop.originalName, target_names[idx])
                y = dataset_to_check.getTargetPropertiesValues(concat=True)
                self.assertNotEqual(y[target_prop.name].unique().shape[0], (len(ths[idx]) - 1))

        check_regression(dataset_new, ["CL", "fu"], [[0, 15, 30, 60], [0.3]])
        dataset_new.save()
        dataset_new = QSPRDataset.fromFile(dataset.storePath)
        check_regression(dataset_new, ["CL", "fu"], [[0, 15, 30, 60], [0.3]])


class TestTargetProperty(TestCase):

    def test_target_property(self):
        def check_target_property(targetprop, name, task, original_name, th):
            self.assertEqual(targetprop.name, name)
            self.assertEqual(targetprop.task, task)
            self.assertEqual(targetprop.originalName, original_name)
            if task.isClassification():
                self.assertTrue(targetprop.task.isClassification())
                self.assertEqual(targetprop.th, th)

        targetprop = TargetProperty("CL", ModelTasks.REGRESSION)
        check_target_property(targetprop, "CL", ModelTasks.REGRESSION, "CL", None)

        targetprop = TargetProperty("CL", ModelTasks.MULTICLASS, th=[0, 1, 10, 1200])
        check_target_property(targetprop, "CL", ModelTasks.MULTICLASS, "CL", [0, 1, 10, 1200])

        targetprop = TargetProperty("CL", ModelTasks.SINGLECLASS, th=[5])
        check_target_property(targetprop, "CL", ModelTasks.SINGLECLASS, "CL", [5])

        targetprop = TargetProperty.fromDict({"name": "CL", "task": ModelTasks.REGRESSION})
        check_target_property(targetprop, "CL", ModelTasks.REGRESSION, "CL", None)

        targetprop = TargetProperty.fromDict({"name": "CL", "task": ModelTasks.MULTICLASS, "th": [0, 1, 10, 1200]})
        check_target_property(targetprop, "CL", ModelTasks.MULTICLASS, "CL", [0, 1, 10, 1200])

        targetprops = TargetProperty.fromList(
            [{"name": "CL", "task": ModelTasks.REGRESSION},
             {"name": "fu", "task": ModelTasks.REGRESSION}])
        check_target_property(targetprops[0], "CL", ModelTasks.REGRESSION, "CL", None)
        check_target_property(targetprops[1], "fu", ModelTasks.REGRESSION, "fu", None)
        self.assertEqual(TargetProperty.selectFromList(targetprops, "CL")[0], targetprops[0])
        self.assertListEqual(TargetProperty.getNames(targetprops), ["CL", "fu"])

        targetprops = TargetProperty.toList(targetprops)
        self.assertIsInstance(targetprops, list)
        self.assertIsInstance(targetprops[0], dict)
        self.assertEqual(targetprops[0]["name"], "CL")
        self.assertEqual(targetprops[0]["task"], ModelTasks.REGRESSION)


class TestDataSetPreparation(DataSets, TestCase):

    def test_preparation(self):
        paths = []
        targetprops_list = [[{"name": "CL", "task": ModelTasks.REGRESSION}],
                            [{"name": "CL", "task": ModelTasks.MULTICLASS, "th": [0, 1, 10, 1200]}],
                            [{"name": "fu", "task": ModelTasks.REGRESSION},
                             {"name": "CL", "task": ModelTasks.MULTICLASS, "th": [0, 1, 10, 1200]}]]

        for targetprops in targetprops_list:
            targetprops = TargetProperty.fromList(targetprops)
            targetprops_id = "_".join([f"{targetprop.name}_{targetprop.task.name}" for targetprop in targetprops])
            dataset = QSPRDataset(f"test_create_prep_{targetprops_id}", targetprops, df=self.df_large,
                                  store_dir=self.qsprdatapath, n_jobs=N_CPU, chunk_size=CHUNK_SIZE)
            np.random.seed(42)
            descriptor_sets = [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=2048),
                               Mordred(),
                               rdkit_descs(),
                               DrugExPhyschem(),
                               PredictorDesc(
                f'{os.path.dirname(__file__)}/test_files/test_predictor/qspr/models/RF_CLS_fu_class.json',
                f'{os.path.dirname(__file__)}/test_files/test_predictor/qspr/data/fu_CLS_QSPRdata_meta.json'),
                TanimotoDistances(list_of_smiles=["C", "CC", "CCC"], fingerprint_type="MorganFP", radius=3, nBits=1000)
            ]
            expected_length = sum([len(x.descriptors) for x in descriptor_sets])
            dataset.prepareDataset(
                feature_calculator=DescriptorsCalculator(descriptor_sets),
                split=randomsplit(0.1),
                datafilters=[
                    CategoryFilter(
                        name="moka_ionState7.4",
                        values=["cationic"])],
                feature_filters=[lowVarianceFilter(0.05),
                                 highCorrelationFilter(0.8)])

            # test some basics
            descriptors = dataset.getDescriptors()
            descriptor_names = dataset.getDescriptorNames()
            self.assertEqual(len(descriptor_names), expected_length)
            self.assertEqual(descriptors.shape[0], len(dataset))
            self.assertEqual(descriptors.shape[1], expected_length)

            # save to file
            dataset.save()
            paths.append(dataset.storePath)

        for path, targetprops in zip(paths, targetprops_list):
            targetprops = TargetProperty.fromList(targetprops)
            ds = QSPRDataset.fromFile(path, n_jobs=N_CPU, chunk_size=CHUNK_SIZE)
            for targetprop in targetprops:
                dstargetprop = ds.getTargetProperties(targetprop.originalName, original_names=True)[0]
                if targetprop.task.isClassification():
                    self.assertEqual(dstargetprop.name, f"{targetprop.originalName}_class")
                self.assertTrue(dstargetprop.task == targetprop.task)
            self.assertTrue(ds.descriptorCalculator)
            self.assertTrue(
                isinstance(
                    ds.descriptorCalculator,
                    DescriptorsCalculator))


class TestDataSplitters(DataSets, TestCase):

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
        standardizers = [StandardScaler()]
        dataset.prepareDataset(
            split=split,
            feature_calculator=calculator,
            feature_standardizers=standardizers)
        self.validate_split(dataset)
        dataset.save()

        dataset_new = QSPRDataset.fromFile(dataset.storePath)
        self.validate_split(dataset_new)
        self.assertTrue(dataset_new.descriptorCalculator)
        self.assertTrue(len(dataset_new.feature_standardizers) == 1)
        self.assertTrue(len(dataset_new.fold_generator.featureStandardizers) == 1)
        self.assertTrue(len(dataset_new.featureNames) == 1024)

        dataset_new.clearFiles()


class TestFoldSplitters(DataSets, TestCase):

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
        dataset.makeClassification("CL", th=[20])
        self.validate_folds(dataset)

        # test with a standarizer
        scaler = MinMaxScaler(feature_range=(1, 2))
        dataset.prepareDataset(feature_standardizers=[scaler])

        def check_min_max(X_train, X_test, y_train, y_test, train_index, test_index):
            self.assertTrue(np.max(X_train) == 2)
            self.assertTrue(np.min(X_train) == 1)
            self.assertTrue(np.max(X_test) == 2)
            self.assertTrue(np.min(X_test) == 1)

        self.validate_folds(dataset, more=check_min_max)


class TestDataFilters(DataSets, TestCase):

    def test_Categoryfilter(self):
        remove_cation = CategoryFilter(
            name="moka_ionState7.4", values=["cationic"])
        df_anion = remove_cation(self.df_large)
        self.assertTrue(
            (df_anion["moka_ionState7.4"] == "cationic").sum() == 0)

        only_cation = CategoryFilter(
            name="moka_ionState7.4",
            values=["cationic"],
            keep=True)
        df_cation = only_cation(self.df_large)
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
            target_props=[{"name": "y", "task": ModelTasks.REGRESSION}],
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


class TestDescriptorCalculation(DataSets, TestCase):

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
            fill_value=None
        )

        # create new dataset with different feature calculator
        dataset_next = self.create_large_dataset(self.__class__.__name__)
        dataset_next.prepareDataset(
            split=split,
            feature_calculator=feature_calculator,
            feature_filters=[lv, hc],
            recalculate_features=True,
            fill_value=None
        )


class TestDescriptorsets(DataSets, TestCase):

    def setUp(self):
        super().setUp()
        self.dataset = self.create_small_dataset(self.__class__.__name__)

    def test_PredictorDesc(self):
        # give path to saved model parameters
        model_path = f'{os.path.dirname(__file__)}/test_files/test_predictor/qspr/models/RF_CLS_fu_class.json'
        meta_path = f'{os.path.dirname(__file__)}/test_files/test_predictor/qspr/data/fu_CLS_QSPRdata_meta.json'
        desc_calc = DescriptorsCalculator([PredictorDesc(model_path, meta_path)])

        self.dataset.addDescriptors(desc_calc)
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


class TestScaffolds(DataSets, TestCase):

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


class TestFeatureStandardizer(DataSets, TestCase):

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
