"""This module holds the tests for functions regarding QSPR modelling."""
import glob
import json
import os
import random
import shutil
from os.path import exists
from unittest import TestCase, skip

import numpy as np
import pandas as pd
import sklearn_json as skljson
import torch
from qsprpred.data.data import QSPRDataset
from qsprpred.data.utils.datasplitters import randomsplit
from qsprpred.data.utils.descriptorcalculator import DescriptorsCalculator
from qsprpred.data.utils.descriptorsets import MorganFP
from qsprpred.data.utils.feature_standardization import SKLearnStandardizer
from qsprpred.models.models import QSPRDNN, QSPRsklearn
from qsprpred.models.neural_network import STFullyConnected
from qsprpred.models.tasks import ModelTasks
from qsprpred.scorers.predictor import Predictor
from rdkit.Chem import MolFromSmiles
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier, XGBRegressor


class PathMixIn:
    datapath = f'{os.path.dirname(__file__)}/test_files/data'
    qsprdatapath = f'{os.path.dirname(__file__)}/test_files/qspr/data'
    qsprmodelspath = f'{os.path.dirname(__file__)}/test_files/qspr/models'

    @classmethod
    def setUpClass(cls):
        cls.tearDownClass()
        if not os.path.exists(cls.qsprmodelspath):
            os.makedirs(cls.qsprmodelspath)
        if not os.path.exists(cls.qsprdatapath):
            os.mkdir(cls.qsprdatapath)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.qsprmodelspath):
            shutil.rmtree(cls.qsprmodelspath)
        if os.path.exists(cls.qsprdatapath):
            shutil.rmtree(cls.qsprdatapath)
        for extension in ['log', 'pkg', 'json']:
            globs = glob.glob(f'{cls.datapath}/*.{extension}')
            for path in globs:
                os.remove(path)


class NeuralNet(PathMixIn, TestCase):

    def prep_testdata(self, task=ModelTasks.REGRESSION, th=None):

        # prepare test dataset
        df = pd.read_csv(f'{self.datapath}/test_data_large.tsv', sep='\t')
        data = QSPRDataset(
            name="testmodel",
            df=df,
            target_prop="CL",
            task=task,
            th=th,
            store_dir=self.qsprmodelspath)
        data.prepareDataset(
            feature_calculator=DescriptorsCalculator([MorganFP(3, 1000)]),
            split=randomsplit(0.1),
            feature_standardizers=[StandardScaler()]
        )
        data.save()
        # prepare data for torch DNN
        trainloader = DataLoader(
            TensorDataset(
                torch.Tensor(
                    data.X.values), torch.Tensor(
                    data.y.values)), batch_size=100)
        testloader = DataLoader(
            TensorDataset(
                torch.Tensor(
                    data.X_ind.values), torch.Tensor(
                    data.y_ind.values)), batch_size=100)

        return data.X.shape[1], trainloader, testloader

    def test_STFullyConnected(self):
        # prepare test regression dataset
        no_features, trainloader, testloader = self.prep_testdata()

        # fit model with default settings
        model = STFullyConnected(n_dim=no_features)
        model.fit(
            trainloader,
            testloader,
            out=f'{self.datapath}/testmodel',
            patience=3)

        # fit model with non-default epochs and learning rate and tolerance
        model = STFullyConnected(n_dim=no_features, n_epochs=50, lr=0.5)
        model.fit(
            trainloader,
            testloader,
            out=f'{self.datapath}/testmodel',
            patience=3,
            tol=0.01)

        # fit model with non-default settings for model construction
        model = STFullyConnected(
            n_dim=no_features,
            neurons_h1=2000,
            neurons_hx=500,
            extra_layer=True)
        model.fit(
            trainloader,
            testloader,
            out=f'{self.datapath}/testmodel',
            patience=3)

        # prepare classification test dataset
        no_features, trainloader, testloader = self.prep_testdata(
            task=ModelTasks.CLASSIFICATION, th=[6.5])

        # fit model with regression is false
        model = STFullyConnected(n_dim=no_features, is_reg=False)
        model.fit(
            trainloader,
            testloader,
            out=f'{self.datapath}/testmodel',
            patience=3)

        # prepare multi-classification test dataset
        no_features, trainloader, testloader = self.prep_testdata(
            task=ModelTasks.CLASSIFICATION, th=[0, 1, 10, 1200])

        # fit model with regression is false
        model = STFullyConnected(n_dim=no_features, n_class=3, is_reg=False)
        model.fit(
            trainloader,
            testloader,
            out=f'{self.datapath}/testmodel',
            patience=3)


class TestModels(PathMixIn, TestCase):
    GPUS = [idx for idx in range(torch.cuda.device_count())]

    def prep_testdata(self, reg=True, th=None):
        task = ModelTasks.REGRESSION if reg else ModelTasks.CLASSIFICATION

        reg_abbr = 'REG' if reg else 'CLS'
        random.seed(42)

        # prepare test dataset
        df = pd.read_csv(f'{self.datapath}/test_data_large.tsv', sep='\t')
        data = QSPRDataset(
            name=f"test_data_large_{task.name}",
            df=df,
            target_prop="CL",
            task=task,
            th=th,
            store_dir=self.qsprmodelspath)
        feature_calculators = DescriptorsCalculator([MorganFP(3, 1000)])
        scaler = StandardScaler()
        data.prepareDataset(
            feature_calculator=feature_calculators,
            split=randomsplit(0.1),
            feature_standardizers=[scaler]
        )
        data.save()

        return data, feature_calculators, SKLearnStandardizer(scaler)

    def QSPRsklearn_models_test(self, alg, alg_name, reg, th=None, n_jobs=8):
        # intialize dataset and model
        data, _, _ = self.prep_testdata(reg=reg, th=th)

        if not alg_name in ["NB", "PLS", "SVM"]:
            parameters = {"n_jobs": n_jobs}
        else:
            parameters = {}
        themodel = QSPRsklearn(
            base_dir=f'{os.path.dirname(__file__)}/test_files/',
            data=data,
            alg=alg,
            alg_name=alg_name,
            parameters=parameters)

        # train the model on all data
        themodel.fit()
        regid = 'REG' if reg else 'CLS'
        self.assertTrue(
            exists(
                f'{os.path.dirname(__file__)}/test_files/qspr/models/{alg_name}_{regid}_{data.targetProperty}.json'))

        # perform crossvalidation
        themodel.evaluate()
        self.assertTrue(
            exists(
                f'{os.path.dirname(__file__)}/test_files/qspr/models/{alg_name}_{regid}_{data.targetProperty}.ind.tsv'))
        self.assertTrue(
            exists(
                f'{os.path.dirname(__file__)}/test_files/qspr/models/{alg_name}_{regid}_{data.targetProperty}.cv.tsv'))

        # perform bayes optimization
        fname = f'{os.path.dirname(__file__)}/test_files/search_space_test.json'
        grid_params = QSPRsklearn.loadParamsGrid(fname, "bayes", alg_name)
        search_space_bs = grid_params[grid_params[:, 0] == alg_name, 1][0]
        themodel.bayesOptimization(search_space_bs=search_space_bs, n_trials=1)
        self.assertTrue(
            exists(f'{os.path.dirname(__file__)}/test_files/qspr/models/{alg_name}_{regid}_{data.targetProperty}_params.json'))

        # perform grid search
        os.remove(
            f'{os.path.dirname(__file__)}/test_files/qspr/models/{alg_name}_{regid}_{data.targetProperty}_params.json')
        grid_params = QSPRsklearn.loadParamsGrid(fname, "grid", alg_name)
        search_space_gs = grid_params[grid_params[:, 0] == alg_name, 1][0]
        themodel.gridSearch(search_space_gs=search_space_gs)
        self.assertTrue(
            exists(f'{os.path.dirname(__file__)}/test_files/qspr/models/{alg_name}_{regid}_{data.targetProperty}_params.json'))

    def predictor_test(self, alg_name, reg, th=None):
        # intialize dataset and model
        data, feature_calculators, scaler = self.prep_testdata(reg=reg, th=th)
        regid = 'REG' if reg else 'CLS'
        if alg_name == 'DNN':
            path = f'{os.path.dirname(__file__)}/test_files/qspr/models/DNN_{regid}_{data.targetProperty}.json'
            with open(path) as f:
                themodel_params = json.load(f)
            themodel = STFullyConnected(**themodel_params)
            themodel.load_state_dict(torch.load(f"{path[:-5]}_weights.pkg"))
        else:
            themodel = skljson.from_json(
                f'{os.path.dirname(__file__)}/test_files/qspr/models/{alg_name}_{regid}_{data.targetProperty}.json')

        # initialize predictor
        predictor = Predictor(
            themodel,
            feature_calculators,
            [scaler],
            type=regid,
            th=th,
            name=None,
            modifier=None)

        # load molecules to predict
        df = pd.read_csv(
            f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv',
            sep='\t')
        mols = [MolFromSmiles(smiles) for smiles in df.SMILES]

        # predict property
        predictions = predictor.getScores(mols)
        self.assertEqual(predictions.shape, (len(mols),))
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(predictions[0], np.floating)

        return predictions

    def testRF(self):
        alg_name = "RF"
        # test regression
        alg = RandomForestRegressor()
        self.QSPRsklearn_models_test(alg, alg_name, reg=True)
        self.predictor_test(alg_name, reg=True)

        # test classifier
        alg = RandomForestClassifier()
        self.QSPRsklearn_models_test(alg, alg_name, reg=False, th=[6.5])
        self.predictor_test(alg_name, reg=False, th=[6.5])

        # test multi-classifier
        alg = RandomForestClassifier()
        self.QSPRsklearn_models_test(
            alg, alg_name, reg=False, th=[
                0, 1, 10, 1100])
        self.predictor_test(alg_name, reg=False, th=[0, 1, 10, 1100])

    def testKNN(self):
        alg_name = "KNN"
        # test regression
        alg = KNeighborsRegressor()
        self.QSPRsklearn_models_test(alg, alg_name, reg=True)
        self.predictor_test(alg_name, reg=True)

        # test classifier
        alg = KNeighborsClassifier()
        self.QSPRsklearn_models_test(alg, alg_name, reg=False, th=[6.5])
        self.predictor_test(alg_name, reg=False, th=[6.5])

        # test multiclass
        self.QSPRsklearn_models_test(
            alg, alg_name, reg=False, th=[
                0, 1, 10, 1100])
        self.predictor_test(alg_name, reg=False, th=[0, 1, 10, 1100])

    def testXGB(self):
        alg_name = "XGB"
        # test regression
        alg = XGBRegressor(objective='reg:squarederror')
        self.QSPRsklearn_models_test(alg, alg_name, reg=True)
        self.predictor_test(alg_name, reg=True)

        # test classifier
        alg = XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss')
        self.QSPRsklearn_models_test(alg, alg_name, reg=False, th=[6.5])
        self.predictor_test(alg_name, reg=False, th=[6.5])

        # test multiclass
        alg = XGBClassifier(
            objective='multi:softmax',
            use_label_encoder=False,
            eval_metric='logloss')
        self.QSPRsklearn_models_test(
            alg, alg_name, reg=False, th=[
                0, 1, 10, 1100])
        self.predictor_test(alg_name, reg=False, th=[0, 1, 10, 1100])

    def testSVM(self):
        alg_name = "SVM"
        # test regression
        alg = SVR()
        self.QSPRsklearn_models_test(alg, alg_name, reg=True)
        self.predictor_test(alg_name, reg=True)

        # test classifier
        alg = SVC(probability=True)
        self.QSPRsklearn_models_test(alg, alg_name, reg=False, th=[6.5])
        self.predictor_test(alg_name, reg=False, th=[6.5])

        # test multiclass
        self.QSPRsklearn_models_test(
            alg, alg_name, reg=False, th=[
                0, 1, 10, 1100])
        self.predictor_test(alg_name, reg=False, th=[0, 1, 10, 1100])

    def testPLS(self):
        alg_name = "PLS"
        # test regression
        alg = PLSRegression()
        self.QSPRsklearn_models_test(alg, alg_name, reg=True)
        self.predictor_test(alg_name, reg=True)

    def testNB(self):
        alg_name = "NB"
        # test classfier
        alg = GaussianNB()
        self.QSPRsklearn_models_test(alg, alg_name, reg=False, th=[6.5])
        self.predictor_test(alg_name, reg=False, th=[6.5])

        # test multiclass
        self.QSPRsklearn_models_test(
            alg, alg_name, reg=False, th=[
                0, 1, 10, 1100])
        self.predictor_test(alg_name, reg=False, th=[0, 1, 10, 1100])

    def test_QSPRDNN(self):
        # intialize model for single class, multi class and regression DNN's
        for reg in [(False, [6.5]), (False, [0, 1, 10, 1100]), (True, [])]:
            data, feature_calculators, scaler = self.prep_testdata(
                reg=reg[0], th=reg[1])
            themodel = QSPRDNN(
                base_dir=f'{os.path.dirname(__file__)}/test_files',
                data=data,
                gpus=self.GPUS,
                patience=3,
                tol=0.02)

            #fit and cross-validation
            themodel.evaluate()
            themodel.fit()

            # optimization
            fname = f'{os.path.dirname(__file__)}/test_files/search_space_test.json'

            # grid search
            grid_params = QSPRDNN.loadParamsGrid(fname, "grid", "DNN")
            search_space_gs = grid_params[grid_params[:, 0] == "DNN", 1][0]
            themodel.gridSearch(search_space_gs=search_space_gs)

            # bayesian optimization
            bayes_params = QSPRDNN.loadParamsGrid(fname, "bayes", "DNN")
            search_space_bs = bayes_params[bayes_params[:, 0] == "DNN", 1][0]
            themodel.bayesOptimization(
                search_space_bs=search_space_bs, n_trials=5)

            # predictor
            self.predictor_test('DNN', reg=reg[0], th=reg[1])
