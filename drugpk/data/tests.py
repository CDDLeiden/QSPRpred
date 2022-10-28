from unittest import TestCase
import shutil
import os
from os.path import exists

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from drugpk.logs import logger
from drugpk.data.data import QSPRDataset
from drugpk.data.utils.datasplitters import scaffoldsplit, randomsplit, temporalsplit
from drugpk.data.utils.datafilters import CategoryFilter, papyrusLowQualityFilter
from drugpk.data.utils.featurefilters import lowVarianceFilter, highCorrelationFilter, BorutaFilter

class PathMixIn:
    datapath = f'{os.path.dirname(__file__)}/test_files/data'
    envspath = f'{os.path.dirname(__file__)}/test_files/envs'

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.envspath):
            os.mkdir(cls.envspath)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.envspath)

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


class TestData(TestCase):

    def test_data(self):
        for reg in [True, False]:
            df = pd.read_csv(f'{os.path.dirname(__file__)}/test_files/data/test_data_large.tsv', sep='\t')
            dataset = QSPRDataset(df=df, property="CL", reg=reg, th=[0,1,10,1200])
            self.assertIsInstance(dataset, QSPRDataset)

            dataset.prepareDataset(datafilters=[CategoryFilter(name="moka_ionState7.4", values=["cationic"])],
                                featurefilters=[lowVarianceFilter(0.05), highCorrelationFilter(0.8)])


