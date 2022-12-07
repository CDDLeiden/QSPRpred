"""This module holds the test for functions regarding QSPR data preparation."""
import os
import shutil
import time
from unittest import TestCase

import mordred
import numpy as np
import pandas as pd
from mordred import descriptors as mordreddescriptors
from qsprpred.data.data import QSPRDataset
from qsprpred.data.utils.datafilters import CategoryFilter
from qsprpred.data.utils.datasplitters import randomsplit, scaffoldsplit, temporalsplit
from qsprpred.data.utils.descriptorcalculator import DescriptorsCalculator
from qsprpred.data.utils.descriptorsets import (
    DrugExPhyschem,
    Mordred,
    MorganFP,
    rdkit_descs,
)
from qsprpred.data.utils.feature_standardization import SKLearnStandardizer
from qsprpred.data.utils.featurefilters import (
    BorutaFilter,
    highCorrelationFilter,
    lowVarianceFilter,
)
from qsprpred.models.tasks import ModelTasks
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles
from sklearn.preprocessing import StandardScaler


class PathMixIn:
    datapath = f'{os.path.dirname(__file__)}/test_files/data'
    qsprmodelspath = f'{os.path.dirname(__file__)}/test_files/qspr/models'
    qsprdatapath = f'{os.path.dirname(__file__)}/test_files/qspr/data'

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.qsprmodelspath):
            os.mkdir(cls.qsprmodelspath)
        if not os.path.exists(cls.qsprdatapath):
            os.mkdir(cls.qsprdatapath)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.qsprmodelspath)
        shutil.rmtree(cls.qsprdatapath)


class DataSets:
    df_large = pd.read_csv(
        f'{os.path.dirname(__file__)}/test_files/data/test_data_large.tsv',
        sep='\t')
    df_small = pd.read_csv(
        f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv',
        sep='\t')


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


class TestData(PathMixIn, DataSets, TestCase):

    def test_creation_preparation(self):
        # regular creation
        dataset = QSPRDataset(
            "test_create",
            "CL",
            df=self.df_large,
            store_dir=self.qsprdatapath)
        self.assertIn("Notes", dataset.getProperties())
        dataset.removeProperty("Notes")
        self.assertNotIn("Notes", dataset.getProperties())
        stopwatch = StopWatch()
        dataset.save()
        stopwatch.stop('Saving took: ')

        # from file creation
        stopwatch.reset()
        dataset_new = QSPRDataset.fromFile(dataset.storePath)
        stopwatch.stop('Loading took: ')
        self.assertNotIn("Notes", dataset_new.getProperties())

        # test default settings
        self.assertEqual(dataset_new.task, ModelTasks.REGRESSION)
        self.assertTrue(dataset_new.hasProperty("CL"))

        # test switch to classification
        with self.assertRaises(AssertionError):
            dataset_new.makeClassification([])
        with self.assertRaises(TypeError):
            dataset_new.makeClassification(th=6.5)
        with self.assertRaises(AssertionError):
            dataset_new.makeClassification(th=[0, 2, 3])
        with self.assertRaises(AssertionError):
            dataset_new.makeClassification(th=[0, 2, 3])

        dataset_new.makeClassification(th=[0, 1, 10, 1200])
        self.assertEqual(dataset_new.task, ModelTasks.CLASSIFICATION)

        paths = []
        tasks = [ModelTasks.REGRESSION, ModelTasks.CLASSIFICATION]
        for task in tasks:
            dataset = QSPRDataset(
                f"test_create_{task.name}", "CL", df=self.df_large,
                store_dir=self.qsprdatapath, task=task, th=[0, 1, 10, 1200]
                if task == ModelTasks.CLASSIFICATION else None)
            np.random.seed(42)
            dataset.prepareDataset(
                feature_calculator=DescriptorsCalculator(
                    [MorganFP(3, 1000)]),
                datafilters=[
                    CategoryFilter(
                        name="moka_ionState7.4",
                        values=["cationic"])],
                feature_filters=[lowVarianceFilter(0.05),
                                 highCorrelationFilter(0.8)])

            # test some basics
            descriptors = dataset.getDescriptors()
            descriptor_names = dataset.getDescriptorNames()
            self.assertEqual(len(descriptor_names), 1000)
            self.assertEqual(descriptors.shape[0], len(dataset))
            self.assertEqual(descriptors.shape[1], 1000)

            # save to file
            dataset.save()
            paths.append(dataset.storePath)

        for path, task in zip(paths, tasks):
            ds = QSPRDataset.fromFile(path)
            if ds.task == ModelTasks.CLASSIFICATION:
                self.assertEqual(ds.targetProperty, "CL_class")
            self.assertTrue(ds.task == task)
            self.assertTrue(ds.descriptorCalculator)
            self.assertTrue(
                isinstance(
                    ds.descriptorCalculator,
                    DescriptorsCalculator))


class TestDataSplitters(PathMixIn, TestCase):
    df = pd.read_csv(
        f'{os.path.dirname(__file__)}/test_files/data/test_data_large.tsv',
        sep='\t')

    def test_randomsplit(self):
        split = randomsplit()
        split(self.df, "SMILES", "CL")

    def test_temporalsplit(self):
        split = temporalsplit(
            timesplit=2015,
            timecol="Year of first disclosure")
        split(self.df, "SMILES", "CL")

    def test_scaffoldsplit(self):
        split = scaffoldsplit()
        split(self.df, "SMILES", "CL")


class TestDataFilters(PathMixIn, TestCase):
    df = pd.read_csv(
        f'{os.path.dirname(__file__)}/test_files/data/test_data_large.tsv',
        sep='\t')

    def test_Categoryfilter(self):
        remove_cation = CategoryFilter(
            name="moka_ionState7.4", values=["cationic"])
        df_anion = remove_cation(self.df)
        self.assertTrue(
            (df_anion["moka_ionState7.4"] == "cationic").sum() == 0)

        only_cation = CategoryFilter(
            name="moka_ionState7.4",
            values=["cationic"],
            keep=True)
        df_cation = only_cation(self.df)
        self.assertTrue(
            (df_cation["moka_ionState7.4"] != "cationic").sum() == 0)


class TestFeatureFilters(PathMixIn, TestCase):
    df = pd.DataFrame(
        data=np.array(
            [[1, 4, 2, 6, 2, 1],
             [1, 8, 4, 2, 4, 2],
             [1, 4, 3, 2, 5, 3],
             [1, 8, 4, 9, 8, 4],
             [1, 4, 2, 3, 9, 5],
             [1, 8, 4, 7, 12, 6]]),
        columns=["F1", "F2", "F3", "F4", "F5", "y"])

    def test_lowVarianceFilter(self):
        filter = lowVarianceFilter(0.01)
        X = filter(self.df[["F1", "F2", "F3", "F4", "F5"]])

        # check if correct columns selected and values still original
        self.assertListEqual(list(X.columns), ["F2", "F3", "F4", "F5"])
        self.assertTrue(self.df[X.columns].equals(X))

    def test_highCorrelationFilter(self):
        filter = highCorrelationFilter(0.8)
        X = filter(self.df[["F1", "F2", "F3", "F4", "F5"]])

        # check if correct columns selected and values still original
        self.assertListEqual(list(X.columns), ["F1", "F2", "F4", "F5"])
        self.assertTrue(self.df[X.columns].equals(X))

    def test_BorutaFilter(self):
        filter = BorutaFilter()
        X = filter(
            features=self.df[["F1", "F2", "F3", "F4", "F5"]],
            y_col=self.df["y"])

        # check if correct columns selected and values still original
        self.assertListEqual(list(X.columns), ["F5"])
        self.assertTrue(self.df[X.columns].equals(X))


class TestDescriptorsets(PathMixIn, TestCase):
    def prep_testdata(self):
        df = pd.read_csv(
            f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv',
            sep='\t')
        mols = [MolFromSmiles(smiles) for smiles in df.SMILES]
        return mols

    def test_MorganFP(self):
        mols = self.prep_testdata()
        desc_calc = MorganFP(3, nBits=1000)
        descriptors = desc_calc(mols[2])
        self.assertIsInstance(descriptors, pd.DataFrame)
        self.assertEqual(descriptors.shape, (1, 1000))
        self.assertTrue(descriptors.any().any())
        self.assertTrue(descriptors.any().sum() > 1)

    def test_Mordred(self):
        mols = self.prep_testdata()
        desc_calc = Mordred()
        descriptors = desc_calc(mols[1])
        self.assertIsInstance(descriptors, pd.DataFrame)
        self.assertEqual(
            descriptors.shape,
            (1, len(mordred.Calculator(mordreddescriptors).descriptors)))
        self.assertTrue(descriptors.any().any())
        self.assertTrue(descriptors.any().sum() > 1)

    def test_DrugExPhyschem(self):
        mols = self.prep_testdata()
        desc_calc = DrugExPhyschem()
        descriptors = desc_calc(mols[1])
        self.assertIsInstance(descriptors, pd.DataFrame)
        self.assertEqual(descriptors.shape, (1, 19))
        self.assertTrue(descriptors.any().any())
        self.assertTrue(descriptors.any().sum() > 1)

    def test_rdkit_descs(self):
        mols = self.prep_testdata()
        desc_calc = rdkit_descs()
        descriptors = desc_calc(mols[1])
        self.assertIsInstance(descriptors, pd.DataFrame)
        self.assertEqual(descriptors.shape, (1, len(Descriptors._descList)))
        self.assertTrue(descriptors.any().any())
        self.assertTrue(descriptors.any().sum() > 1)

        # with 3D
        desc_calc = rdkit_descs(compute_3Drdkit=True)
        descriptors = desc_calc(mols[1])
        self.assertIsInstance(descriptors, pd.DataFrame)
        self.assertEqual(descriptors.shape,
                         (1, (len(Descriptors._descList) + 10)))
        self.assertTrue(descriptors.any().any())
        self.assertTrue(descriptors.any().sum() > 1)


class TestDescriptorCalculator(PathMixIn, TestCase):
    def prep_testdata(self):
        df = pd.read_csv(
            f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv',
            sep='\t')
        mols = [MolFromSmiles(smiles) for smiles in df.SMILES]
        return mols

    def test_descriptorcalculator(self):
        from mordred import ABCIndex
        mols = self.prep_testdata()
        desc_calc = DescriptorsCalculator(
            [MorganFP(2, 10), DrugExPhyschem(), Mordred(ABCIndex)])
        mols.append(None)
        descriptors = desc_calc(mols)
        self.assertIsInstance(descriptors, pd.DataFrame)
        self.assertEqual(descriptors.shape, (11, 31))
        self.assertTrue(descriptors.any().any())
        self.assertEqual(desc_calc.get_len(), 31)

        filter = highCorrelationFilter(0.99)
        descriptors = filter(descriptors)
        desc_calc.keepDescriptors(descriptors)
        desc_calc.toFile(
            f"{os.path.dirname(__file__)}/test_files/qspr/data/test_calc.json")
        DescriptorsCalculator.fromFile(
            f"{os.path.dirname(__file__)}/test_files/qspr/data/test_calc.json")


class TestFeatureStandardizer(PathMixIn, TestCase):
    def prep_testdata(self):
        df = pd.read_csv(
            f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv',
            sep='\t')
        mols = [MolFromSmiles(smiles) for smiles in df.SMILES]
        features = [
            AllChem.GetMorganFingerprintAsBitVect(
                x, 3, 1000) for x in mols]
        return features

    def test_featurestandarizer(self):
        features = self.prep_testdata()
        scaler = SKLearnStandardizer.fromFit(features, StandardScaler())
        scaled_features = scaler(features)
        scaler.toFile(
            f'{os.path.dirname(__file__)}/test_files/qspr/data/test_scaler.json')
        scaler_fromfile = SKLearnStandardizer.fromFile(
            f'{os.path.dirname(__file__)}/test_files/qspr/data/test_scaler.json')
        scaled_features_fromfile = scaler_fromfile(features)
        self.assertIsInstance(scaled_features, np.ndarray)
        self.assertEqual(scaled_features.shape, (10, 1000))
        self.assertEqual(
            np.array_equal(
                scaled_features,
                scaled_features_fromfile),
            True)
