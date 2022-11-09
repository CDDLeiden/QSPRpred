import os
import shutil
from unittest import TestCase

import mordred
import numpy as np
import pandas as pd
from mordred import descriptors as mordreddescriptors
from qsprpred.data.data import QSPRDataset
from qsprpred.data.utils.datafilters import CategoryFilter, papyrusLowQualityFilter
from qsprpred.data.utils.datasplitters import randomsplit, scaffoldsplit, temporalsplit
from qsprpred.data.utils.descriptorcalculator import descriptorsCalculator
from qsprpred.data.utils.descriptorsets import (
    DrugExPhyschem,
    Mordred,
    MorganFP,
    rdkit_descs,
)
from qsprpred.data.utils.featurefilters import (
    BorutaFilter,
    highCorrelationFilter,
    lowVarianceFilter,
)
from qsprpred.logs import logger
from rdkit.Chem import Descriptors, MolFromSmiles


class PathMixIn:
    datapath = f'{os.path.dirname(__file__)}/test_files/data'
    qsprmodelspath = f'{os.path.dirname(__file__)}/test_files/qsprmodels'

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.qsprmodelspath):
            os.mkdir(cls.qsprmodelspath)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.qsprmodelspath)

class TestDataSplitters(PathMixIn, TestCase):
    df = pd.read_csv(f'{os.path.dirname(__file__)}/test_files/data/test_data_large.tsv', sep='\t')

    def test_randomsplit(self):
        split = randomsplit()
        split(self.df, "SMILES", "CL")

    def test_temporalsplit(self):
        split = temporalsplit(timesplit=2015, timecol="Year of first disclosure")
        split(self.df, "SMILES", "CL")

    def test_scaffoldsplit(self):
        split = scaffoldsplit()
        split(self.df, "SMILES", "CL")

class TestDataFilters(PathMixIn, TestCase):
    df = pd.read_csv(f'{os.path.dirname(__file__)}/test_files/data/test_data_large.tsv', sep='\t')

    def test_Categoryfilter(self):
        remove_cation = CategoryFilter(name="moka_ionState7.4", values=["cationic"])
        df_anion = remove_cation(self.df)
        self.assertTrue((df_anion["moka_ionState7.4"] == "cationic").sum() == 0)

        only_cation = CategoryFilter(name="moka_ionState7.4", values=["cationic"], keep=True)
        df_cation = only_cation(self.df)
        self.assertTrue((df_cation["moka_ionState7.4"] != "cationic").sum() == 0)

class TestFeatureFilters(PathMixIn, TestCase):
    df = pd.DataFrame(data = np.array([[1, 4, 2, 6, 2 ,1],
                                        [1, 8, 4, 2, 4 ,2],
                                        [1, 4, 3, 2, 5 ,3],
                                        [1, 8, 4, 9, 8 ,4],
                                        [1, 4, 2, 3, 9 ,5],
                                        [1, 8, 4, 7, 12,6]]), columns=["F1", "F2", "F3", "F4", "F5", "y"])

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
        X = filter(features = self.df[["F1", "F2", "F3", "F4", "F5"]], y=self.df["y"])

        # check if correct columns selected and values still original
        self.assertListEqual(list(X.columns), ["F5"])
        self.assertTrue(self.df[X.columns].equals(X))

class TestDescriptorsets(PathMixIn, TestCase):
    def prep_testdata(self):
        df = pd.read_csv(f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv', sep='\t')
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
        self.assertEqual(descriptors.shape, (1, len(mordred.Calculator(mordreddescriptors).descriptors)))
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

        #with 3D
        desc_calc = rdkit_descs(compute_3Drdkit=True)
        descriptors = desc_calc(mols[1])
        self.assertIsInstance(descriptors, pd.DataFrame)
        self.assertEqual(descriptors.shape, (1, (len(Descriptors._descList) + 10)))
        self.assertTrue(descriptors.any().any())
        self.assertTrue(descriptors.any().sum() > 1)


class TestDescriptorCalculator(PathMixIn, TestCase):
    def prep_testdata(self):
        df = pd.read_csv(f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv', sep='\t')
        mols = [MolFromSmiles(smiles) for smiles in df.SMILES]
        return mols
    
    def test_descriptorcalculator(self):
        mols = self.prep_testdata()
        desc_calc = descriptorsCalculator([MorganFP(3, 1000), DrugExPhyschem()])
        mols.append(None)
        descriptors = desc_calc(mols)
        self.assertIsInstance(descriptors, pd.DataFrame)
        self.assertEqual(descriptors.shape, (11,1019))
        self.assertEqual(descriptors.columns[0], 'MorganFP_0')
        self.assertEqual(descriptors.columns[1018], 'DrugExPhyschem_MR')
        self.assertTrue(descriptors.any().any())
        self.assertEqual(desc_calc.get_len(), 1019)

class TestData(PathMixIn, TestCase):

    def test_data(self):
        for reg in [True, False]:
            reg_abbr = 'REG' if reg else 'CLS'
            df = pd.read_csv(f'{os.path.dirname(__file__)}/test_files/data/test_data_large.tsv', sep='\t')
            dataset = QSPRDataset(df=df, property="CL", reg=reg, th=[0,1,10,1200])
            self.assertIsInstance(dataset, QSPRDataset)

            dataset.prepareDataset(f'{os.path.dirname(__file__)}/test_files/qsprmodels/CL_{reg_abbr}.tsv',
                                feature_calculators=descriptorsCalculator([MorganFP(3, 1000)]),
                                datafilters=[CategoryFilter(name="moka_ionState7.4", values=["cationic"])],
                                featurefilters=[lowVarianceFilter(0.05), highCorrelationFilter(0.8)])

