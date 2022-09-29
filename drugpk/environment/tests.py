import shutil
from unittest import TestCase
import os
from os.path import exists
from drugpk.environment.dataprep_utils.datasplitters import scaffoldsplit, randomsplit, temporalsplit
from drugpk.environment.dataprep_utils.datafilters import CategoryFilter, papyrusLowQualityFilter
from drugpk.logs import logger
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from drugpk.environment.neural_network import STFullyConnected
from drugpk.environment.data import QSKRDataset
from drugpk.environment.models import QSKRsklearn, QSKRDNN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from xgboost import XGBRegressor, XGBClassifier

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

class TestData(TestCase):

    def test_data(self):
        df = pd.read_csv(f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv', sep='\t')
        dataset = QSKRDataset(input_df=df, valuecol="CL")
        self.assertIsInstance(dataset, QSKRDataset)

        dataset.splitDataset()
        self.assertIsInstance(dataset.X, np.ndarray)
        self.assertIsInstance(dataset.X_ind, np.ndarray)
        self.assertIsInstance(dataset.y, np.ndarray)
        self.assertIsInstance(dataset.y_ind, np.ndarray)

        self.assertEqual(dataset.X.shape, (len(dataset.y), 19 + 2048)) # 19 (no. of physchem desc) +  2048 (fp bit len)
        self.assertEqual(dataset.X_ind.shape, (len(dataset.y_ind), 19 + 2048))

        # default case
        self.assertEqual(dataset.X.shape[0] + dataset.X_ind.shape[0], 9) # 1 of 10 datapoints removed, Nan
        self.assertEqual(dataset.X_ind.shape[0], 1) # test_size 0.1, should be 1 test sample
        self.assertEqual(dataset.X.shape[0], 8) # test_size 0.1, should be 1 test sample
        
        # regression is true
        self.assertEqual(np.min(np.concatenate((dataset.y, dataset.y_ind))), 0.36)
        self.assertEqual(np.max(np.concatenate((dataset.y, dataset.y_ind))), 46.58)

        # with test size is 3
        dataset = QSKRDataset(input_df=df, valuecol="CL", test_size=3)
        dataset.splitDataset()
        self.assertEqual(dataset.X_ind.shape[0], 3) # test size of 3
        self.assertEqual(dataset.X.shape[0], 6) # 9 - 3 is 6

        # with timesplit on 2000
        dataset = QSKRDataset(input_df=df, valuecol="CL", test_size=3, timesplit=2000)
        dataset.splitDataset()
        self.assertEqual(dataset.X_ind.shape[0], 2) # two sample year > 2000
        self.assertEqual(dataset.X.shape[0], 7) # 9 - 2 is 7

        # with classification
        dataset = QSKRDataset(input_df=df, valuecol="CL", test_size=3, reg=False, th=7)
        dataset.splitDataset()
        self.assertTrue(np.min(np.concatenate((dataset.y, dataset.y_ind))) == 0)
        self.assertTrue(np.max(np.concatenate((dataset.y, dataset.y_ind))) == 1)
        self.assertEqual(np.sum(np.concatenate((dataset.y, dataset.y_ind)) < 1), 4) # only 4 value below threshold of 7


class NeuralNet(PathMixIn, TestCase):

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        os.remove(f'{cls.datapath}/testmodel.log')
        os.remove(f'{cls.datapath}/testmodel.pkg')

    def prep_testdata(self, reg=True):

        # prepare test dataset
        df = pd.read_csv(f'{self.datapath}/test_data_large.tsv', sep='\t')
        data = QSKRDataset(input_df=df, valuecol="CL", reg=reg)
        data.splitDataset()
        data.X, data.X_ind = data.dataStandardization(data.X, data.X_ind)

        # prepare data for torch DNN
        y = data.y.reshape(-1,1)
        y_ind = data.y_ind.reshape(-1,1)
        trainloader = DataLoader(TensorDataset(torch.Tensor(data.X), torch.Tensor(y)), batch_size=100)
        testloader = DataLoader(TensorDataset(torch.Tensor(data.X_ind), torch.Tensor(y_ind)), batch_size=100)

        return data.X.shape[1], trainloader, testloader

    def test_STFullyConnected(self):
        # prepare test dataset
        no_features, trainloader, testloader = self.prep_testdata(reg=True)

        # fit model with default settings
        model = STFullyConnected(n_dim = no_features)
        model.fit(trainloader, testloader, out=f'{self.datapath}/testmodel', patience = 3)

        # fit model with non-default epochs and learning rate and tolerance
        model = STFullyConnected(n_dim = no_features, n_epochs = 50, lr = 0.5)
        model.fit(trainloader, testloader, out=f'{self.datapath}/testmodel', patience = 3, tol=0.01)

        # fit model with non-default settings for model construction
        model = STFullyConnected(n_dim = no_features, neurons_h1=2000, neurons_hx=500, extra_layer=True)
        model.fit(trainloader, testloader, out=f'{self.datapath}/testmodel', patience = 3)

        # prepare classification test dataset
        no_features, trainloader, testloader = self.prep_testdata(reg=False)

        # fit model with regression is false
        model = STFullyConnected(n_dim = no_features, is_reg=False)
        model.fit(trainloader, testloader, out=f'{self.datapath}/testmodel', patience = 3)


class TestModels(PathMixIn, TestCase):

    def prep_testdata(self, reg=True):
        
        # prepare test dataset
        df = pd.read_csv(f'{os.path.dirname(__file__)}/test_files/data/test_data_large.tsv', sep='\t')
        data = QSKRDataset(input_df=df, valuecol="CL", reg=reg)
        data.splitDataset()
        data.X, data.X_ind = data.dataStandardization(data.X, data.X_ind)
        
        return data

    def QSKRsklearn_models_test(self, alg, alg_name, reg):
        #intialize dataset and model
        data = self.prep_testdata(reg=reg)
        themodel = QSKRsklearn(base_dir = f'{os.path.dirname(__file__)}/test_files/',
                               data=data, alg = alg, alg_name=alg_name)
        
        # train the model on all data
        themodel.fit()
        regid = 'REG' if reg else 'CLS'
        self.assertTrue(exists(f'{os.path.dirname(__file__)}/test_files/envs/{alg_name}_{regid}_{data.valuecol}.pkg'))

        # perform crossvalidation
        themodel.evaluate()
        self.assertTrue(exists(f'{os.path.dirname(__file__)}/test_files/envs/{alg_name}_{regid}_{data.valuecol}.ind.tsv'))
        self.assertTrue(exists(f'{os.path.dirname(__file__)}/test_files/envs/{alg_name}_{regid}_{data.valuecol}.cv.tsv'))
        
        # perform bayes optimization
        fname = f'{os.path.dirname(__file__)}/test_files/search_space_test.json'
        grid_params = QSKRsklearn.loadParamsGrid(fname, "bayes", alg_name)
        search_space_bs = grid_params[grid_params[:,0] == alg_name,1][0]
        themodel.bayesOptimization(search_space_bs=search_space_bs, n_trials=1, save_m=False)
        self.assertTrue(exists(f'{os.path.dirname(__file__)}/test_files/envs/{alg_name}_{regid}_{data.valuecol}_params.json'))

        # perform grid search
        os.remove(f'{os.path.dirname(__file__)}/test_files/envs/{alg_name}_{regid}_{data.valuecol}_params.json')
        grid_params = QSKRsklearn.loadParamsGrid(fname, "grid", alg_name)
        search_space_gs = grid_params[grid_params[:,0] == alg_name,1][0]
        themodel.gridSearch(search_space_gs=search_space_gs, save_m=False)
        self.assertTrue(exists(f'{os.path.dirname(__file__)}/test_files/envs/{alg_name}_{regid}_{data.valuecol}_params.json'))


    def testRF(self):
        alg_name = "RF"
        #test regression
        alg = RandomForestRegressor()
        self.QSKRsklearn_models_test(alg, alg_name, reg=True)

        #test classifier
        alg = RandomForestClassifier()
        self.QSKRsklearn_models_test(alg, alg_name, reg=False)

    def testKNN(self):
        alg_name = "KNN"
        #test regression
        alg = KNeighborsRegressor()
        self.QSKRsklearn_models_test(alg, alg_name, reg=True)

        #test classifier
        alg = KNeighborsClassifier()
        self.QSKRsklearn_models_test(alg, alg_name, reg=False)

    def testXGB(self):
        alg_name = "XGB"
        #test regression
        alg = XGBRegressor(objective='reg:squarederror')
        self.QSKRsklearn_models_test(alg, alg_name, reg=True)

        #test classifier
        alg = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
        self.QSKRsklearn_models_test(alg, alg_name, reg=False)

    def testSVM(self):
        alg_name = "SVM"
        #test regression
        alg = SVR()
        self.QSKRsklearn_models_test(alg, alg_name, reg=True)

        #test classifier
        alg = SVC(probability=True)
        self.QSKRsklearn_models_test(alg, alg_name, reg=False)

    def testPLS(self):
        alg_name = "PLS"
        #test regression
        alg = PLSRegression()
        self.QSKRsklearn_models_test(alg, alg_name, reg=True)

    def testNB(self):
        alg_name = "NB"
        #test classfier
        alg = GaussianNB()
        self.QSKRsklearn_models_test(alg, alg_name, reg=False)

    def test_QSKRDNN(self):
        #intialize model
        for reg in [True, False]:
            data = self.prep_testdata(reg=reg)
            themodel = QSKRDNN(base_dir = f'{os.path.dirname(__file__)}/test_files/', data=data, gpus=[3,2], patience=3, tol=0.02)
            
            #fit and cross-validation
            themodel.evaluate()
            themodel.fit()

            #optimization
            fname = f'{os.path.dirname(__file__)}/test_files/search_space_test.json'

            # grid search
            grid_params = QSKRDNN.loadParamsGrid(fname, "grid", "DNN")
            search_space_gs = grid_params[grid_params[:,0] == "DNN",1][0]
            themodel.gridSearch(search_space_gs=search_space_gs)
  
            # bayesian optimization
            bayes_params = QSKRDNN.loadParamsGrid(fname, "bayes", "DNN")
            search_space_bs = grid_params[bayes_params[:,0] == "DNN",1][0]
            themodel.bayesOptimization(search_space_bs=search_space_bs, n_trials=5)