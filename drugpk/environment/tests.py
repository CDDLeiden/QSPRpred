import shutil
from unittest import TestCase
import os
from os.path import exists
from drugpk.environment.dataprep_utils.datasplitters import scaffoldsplit, randomsplit, temporalsplit
from drugpk.environment.dataprep_utils.datafilters import CategoryFilter, papyrusLowQualityFilter
from drugpk.environment.dataprep_utils.featurefilters import lowVarianceFilter, highCorrelationFilter, BorutaFilter
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
            df = pd.read_csv(f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv', sep='\t')
            dataset = QSKRDataset(df=df, property="CL", reg=reg, th=[0,1,100,1000])
            self.assertIsInstance(dataset, QSKRDataset)

            dataset.prepareDataset(datafilters=[CategoryFilter(name="moka_ionState7.4", values=["cationic"])],
                                featurefilters=[lowVarianceFilter(0.05), highCorrelationFilter(0.8)])


class NeuralNet(PathMixIn, TestCase):

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        os.remove(f'{cls.datapath}/testmodel.log')
        os.remove(f'{cls.datapath}/testmodel.pkg')

    def prep_testdata(self, reg=True, th=[]):

        # prepare test dataset
        df = pd.read_csv(f'{self.datapath}/test_data_large.tsv', sep='\t')
        data = QSKRDataset(df=df, property="CL", reg=reg, th=th)
        data.prepareDataset()
        data.X, data.X_ind = data.dataStandardization(data.X, data.X_ind)

        # prepare data for torch DNN
        y = data.y.reshape(-1,1)
        y_ind = data.y_ind.reshape(-1,1)
        trainloader = DataLoader(TensorDataset(torch.Tensor(data.X), torch.Tensor(y)), batch_size=100)
        testloader = DataLoader(TensorDataset(torch.Tensor(data.X_ind), torch.Tensor(y_ind)), batch_size=100)

        return data.X.shape[1], trainloader, testloader

    def test_STFullyConnected(self):
        ## prepare test regression dataset
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

        ## prepare classification test dataset
        no_features, trainloader, testloader = self.prep_testdata(reg=False)

        # fit model with regression is false
        model = STFullyConnected(n_dim = no_features, is_reg=False, th=[6.5])
        model.fit(trainloader, testloader, out=f'{self.datapath}/testmodel', patience = 3)

        ## prepare multi-classification test dataset
        no_features, trainloader, testloader = self.prep_testdata(reg=False, th=[0, 1, 100])

        # fit model with regression is false
        model = STFullyConnected(n_dim = no_features, is_reg=False)
        model.fit(trainloader, testloader, out=f'{self.datapath}/testmodel', patience = 3)


class TestModels(PathMixIn, TestCase):

    def prep_testdata(self, reg=True, th=[]):
        
        # prepare test dataset
        df = pd.read_csv(f'{os.path.dirname(__file__)}/test_files/data/test_data_large.tsv', sep='\t')
        data = QSKRDataset(df=df, property="CL", reg=reg, th=th)
        data.prepareDataset()
        data.X, data.X_ind = data.dataStandardization(data.X, data.X_ind)
        
        return data

    def QSKRsklearn_models_test(self, alg, alg_name, reg, th=[]):
        #intialize dataset and model
        data = self.prep_testdata(reg=reg, th=th)
        themodel = QSKRsklearn(base_dir = f'{os.path.dirname(__file__)}/test_files/',
                               data=data, alg = alg, alg_name=alg_name)
        
        # train the model on all data
        themodel.fit()
        regid = 'REG' if reg else 'CLS'
        self.assertTrue(exists(f'{os.path.dirname(__file__)}/test_files/envs/{alg_name}_{regid}_{data.property}.pkg'))

        # perform crossvalidation
        themodel.evaluate()
        self.assertTrue(exists(f'{os.path.dirname(__file__)}/test_files/envs/{alg_name}_{regid}_{data.property}.ind.tsv'))
        self.assertTrue(exists(f'{os.path.dirname(__file__)}/test_files/envs/{alg_name}_{regid}_{data.property}.cv.tsv'))
        
        # perform bayes optimization
        fname = f'{os.path.dirname(__file__)}/test_files/search_space_test.json'
        grid_params = QSKRsklearn.loadParamsGrid(fname, "bayes", alg_name)
        search_space_bs = grid_params[grid_params[:,0] == alg_name,1][0]
        themodel.bayesOptimization(search_space_bs=search_space_bs, n_trials=1)
        self.assertTrue(exists(f'{os.path.dirname(__file__)}/test_files/envs/{alg_name}_{regid}_{data.property}_params.json'))

        # perform grid search
        os.remove(f'{os.path.dirname(__file__)}/test_files/envs/{alg_name}_{regid}_{data.property}_params.json')
        grid_params = QSKRsklearn.loadParamsGrid(fname, "grid", alg_name)
        search_space_gs = grid_params[grid_params[:,0] == alg_name,1][0]
        themodel.gridSearch(search_space_gs=search_space_gs)
        self.assertTrue(exists(f'{os.path.dirname(__file__)}/test_files/envs/{alg_name}_{regid}_{data.property}_params.json'))


    def testRF(self):
        alg_name = "RF"
        #test regression
        alg = RandomForestRegressor()
        self.QSKRsklearn_models_test(alg, alg_name, reg=True)

        #test classifier
        alg = RandomForestClassifier()
        self.QSKRsklearn_models_test(alg, alg_name, reg=False, th=[6.5])
        
        #test multi-classifier
        alg = RandomForestClassifier()
        self.QSKRsklearn_models_test(alg, alg_name, reg=False, th=[0, 1, 10, 1100])

    def testKNN(self):
        alg_name = "KNN"
        # test regression
        alg = KNeighborsRegressor()
        self.QSKRsklearn_models_test(alg, alg_name, reg=True)

        # test classifier
        alg = KNeighborsClassifier()
        self.QSKRsklearn_models_test(alg, alg_name, reg=False, th=[6.5])

        # test multiclass
        self.QSKRsklearn_models_test(alg, alg_name, reg=False, th=[0, 1, 10, 1100])

    def testXGB(self):
        alg_name = "XGB"
        #test regression
        alg = XGBRegressor(objective='reg:squarederror')
        self.QSKRsklearn_models_test(alg, alg_name, reg=True)

        #test classifier
        alg = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
        self.QSKRsklearn_models_test(alg, alg_name, reg=False, th=[6.5])

        #test multiclass
        self.QSKRsklearn_models_test(alg, alg_name, reg=False, th=[0, 1, 10, 1100])

    def testSVM(self):
        alg_name = "SVM"
        #test regression
        alg = SVR()
        self.QSKRsklearn_models_test(alg, alg_name, reg=True)

        #test classifier
        alg = SVC(probability=True)
        self.QSKRsklearn_models_test(alg, alg_name, reg=False, th=[6.5])

        #test multiclass
        self.QSKRsklearn_models_test(alg, alg_name, reg=False, th=[0, 1, 10, 1100])

    def testPLS(self):
        alg_name = "PLS"
        #test regression
        alg = PLSRegression()
        self.QSKRsklearn_models_test(alg, alg_name, reg=True)

    def testNB(self):
        alg_name = "NB"
        #test classfier
        alg = GaussianNB()
        self.QSKRsklearn_models_test(alg, alg_name, reg=False, th=[6.5])

        #test multiclass
        self.QSKRsklearn_models_test(alg, alg_name, reg=False, th=[0, 1, 10, 1100])

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