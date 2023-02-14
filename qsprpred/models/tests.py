"""This module holds the tests for functions regarding QSPR modelling."""
import numbers
import os
import shutil
import logging
from os.path import exists
from unittest import TestCase
from parameterized import parameterized

import numpy as np
import pandas as pd
import torch
from qsprpred.data.utils.datasplitters import randomsplit
from qsprpred.data.utils.descriptorcalculator import DescriptorsCalculator
from qsprpred.data.utils.descriptorsets import FingerprintSet
from qsprpred.data.utils.featurefilters import lowVarianceFilter, highCorrelationFilter
from qsprpred.models.interfaces import QSPRModel
from qsprpred.models.models import QSPRDNN, QSPRsklearn
from qsprpred.models.neural_network import STFullyConnected
from qsprpred.models.tasks import ModelTasks
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier, XGBRegressor
from qsprpred.data.tests import DataSets

N_CPUS = 2
GPUS = [idx for idx in range(torch.cuda.device_count())]
logging.basicConfig(level=logging.DEBUG)

class ModelDataSets(DataSets):
    qsprmodelspath = f'{os.path.dirname(__file__)}/test_files/qspr/models'
    preparation = {
        "feature_calculator" : DescriptorsCalculator([FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=1000)]),
        "split" : randomsplit(0.1),
        "feature_standardizer" : StandardScaler(),
        "feature_filters" : [
            lowVarianceFilter(0.05),
            highCorrelationFilter(0.8)
        ],
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not os.path.exists(cls.qsprmodelspath):
            os.makedirs(cls.qsprmodelspath)

    @classmethod
    def clean_directories(cls):
        super().clean_directories()
        if os.path.exists(cls.qsprmodelspath):
            shutil.rmtree(cls.qsprmodelspath)

    @staticmethod
    def prepare(dataset):
        dataset.prepareDataset(
            **ModelDataSets.preparation,
        )

        return dataset

    def create_large_dataset(self, name="QSPRDataset_test", task=ModelTasks.REGRESSION, target_prop='CL', th=None):
        dataset = super().create_large_dataset(name=name, task=task, target_prop=target_prop, th=th)
        return self.prepare(dataset)

    def create_small_dataset(self, name="QSPRDataset_test", task=ModelTasks.REGRESSION, target_prop='CL', th=None):
        dataset = super().create_small_dataset(name=name, task=task, target_prop=target_prop, th=th)
        return self.prepare(dataset)

class ModelTestMixIn:
    def fit_test(self, themodel):
        # perform bayes optimization
        fname = f'{os.path.dirname(__file__)}/test_files/search_space_test.json'
        mname = themodel.name.split("_")[0]
        grid_params = themodel.__class__.loadParamsGrid(fname, "bayes", mname)
        search_space_bs = grid_params[grid_params[:, 0] == mname, 1][0]
        themodel.bayesOptimization(search_space_bs=search_space_bs, n_trials=1)
        self.assertTrue(exists(f"{themodel.baseDir}/{themodel.metaInfo['parameters_path']}"))

        # perform grid search
        themodel.cleanFiles()
        grid_params = themodel.__class__.loadParamsGrid(fname, "grid", mname)
        search_space_gs = grid_params[grid_params[:, 0] == mname, 1][0]
        themodel.gridSearch(search_space_gs=search_space_gs)
        self.assertTrue(exists(f"{themodel.baseDir}/{themodel.metaInfo['parameters_path']}"))
        themodel.cleanFiles()

        # perform crossvalidation
        themodel.evaluate()
        self.assertTrue(
            exists(
                f'{themodel.outDir}/{themodel.name}.ind.tsv'))
        self.assertTrue(
            exists(
                f'{themodel.outDir}/{themodel.name}.cv.tsv'))

        # train the model on all data
        themodel.fit()
        self.assertTrue(exists(themodel.metaFile))
        self.assertTrue(exists(f"{themodel.baseDir}/{themodel.metaInfo['model_path']}"))
        self.assertTrue(exists(f"{themodel.baseDir}/{themodel.metaInfo['parameters_path']}"))
        self.assertTrue(exists(f"{themodel.baseDir}/{themodel.metaInfo['feature_calculator_path']}"))
        self.assertTrue(exists(f"{themodel.baseDir}/{themodel.metaInfo['feature_standardizer_path']}"))

    def predictor_test(self, model_name, base_dir, cls : QSPRModel = QSPRsklearn):
        # initialize model as predictor
        predictor = cls(name=model_name, base_dir=base_dir)

        # load molecules to predict
        df = pd.read_csv(
            f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv',
            sep='\t')

        # predict the property
        predictions = predictor.predictMols(df.SMILES.to_list())
        self.assertEqual(predictions.shape, (len(df.SMILES),))
        self.assertIsInstance(predictions, np.ndarray)
        if predictor.task == ModelTasks.REGRESSION:
            self.assertIsInstance(predictions[0], numbers.Real)
        elif predictor.task == ModelTasks.CLASSIFICATION:
            self.assertIsInstance(predictions[0], numbers.Integral)
        else:
            return AssertionError(f"Unknown task: {predictor.task}")

        # test with an invalid smiles
        invalid_smiles = ["C1CCCCC1", "C1CCCCC"]
        predictions = predictor.predictMols(invalid_smiles)
        self.assertEqual(predictions.shape, (len(invalid_smiles),))
        self.assertEqual(predictions[1], None)
        self.assertIsInstance(predictions[0], numbers.Number)

class NeuralNet(ModelDataSets, ModelTestMixIn, TestCase):

    @staticmethod
    def get_model(name, alg=None, dataset=None, parameters=None):
        # intialize dataset and model
        return QSPRDNN(
            base_dir=f'{os.path.dirname(__file__)}/test_files/',
            alg=alg,
            data=dataset,
            name=name,
            parameters=parameters,
            gpus=GPUS,
            patience=3,
            tol=0.02
        )

    def prep_testdata(self, task=ModelTasks.REGRESSION, th=None):

        # prepare test dataset
        data = self.create_large_dataset(task=task, th=th)
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

    @parameterized.expand([
        (f"{alg_name}_{task}", task, alg_name, alg, th)
        for alg, alg_name, task, th in (
                (STFullyConnected, "STFullyConnected", ModelTasks.REGRESSION, None),
                (STFullyConnected, "STFullyConnected", ModelTasks.CLASSIFICATION, [6.5]),
                (STFullyConnected, "STFullyConnected", ModelTasks.CLASSIFICATION, [0, 1, 10, 1200]),
        )
    ])
    def test_base_model(self, _, task, alg_name, alg, th):
        # prepare test regression dataset
        is_reg = True if task == ModelTasks.REGRESSION else False
        no_features, trainloader, testloader = self.prep_testdata(task=task, th=th)

        # fit model with default settings
        model = alg(n_dim=no_features, is_reg=is_reg)
        model.fit(
            trainloader,
            testloader,
            out=f'{self.datapath}/{alg_name}_{task}',
            patience=3)

        # fit model with non-default epochs and learning rate and tolerance
        model = alg(n_dim=no_features, n_epochs=50, lr=0.5, is_reg=is_reg)
        model.fit(
            trainloader,
            testloader,
            out=f'{self.datapath}/{alg_name}_{task}',
            patience=3,
            tol=0.01)

        # fit model with non-default settings for model construction
        model = alg(
            n_dim=no_features,
            neurons_h1=2000,
            neurons_hx=500,
            extra_layer=True,
            is_reg=is_reg
        )
        model.fit(
            trainloader,
            testloader,
            out=f'{self.datapath}/{alg_name}_{task}',
            patience=3)

    @parameterized.expand([
        (f"{alg_name}_{task}", task, alg_name, alg, th)
        for alg, alg_name, task, th in (
                (STFullyConnected, "STFullyConnected", ModelTasks.REGRESSION, None),
                (STFullyConnected, "STFullyConnected", ModelTasks.CLASSIFICATION, [6.5]),
                (STFullyConnected, "STFullyConnected", ModelTasks.CLASSIFICATION, [0, 1, 10, 1100]),
        )
    ])
    def test_qsprpred_model(self, _, task, alg_name, alg, th):
        # initialize dataset
        dataset = self.create_large_dataset(task=task, th=th)

        # initialize model for training from class
        alg_name = f"{alg_name}_{task}_th={th}"
        model = self.get_model(
            name=alg_name,
            alg=alg,
            dataset=dataset
        )
        self.fit_test(model)
        self.predictor_test(alg_name, model.baseDir, QSPRDNN)

class TestQSPRsklearn(ModelDataSets, ModelTestMixIn, TestCase):

    @staticmethod
    def get_model(name, alg=None, dataset=None, parameters=None):
        # intialize dataset and model
        return QSPRsklearn(
            base_dir=f'{os.path.dirname(__file__)}/test_files/',
            alg=alg,
            data=dataset,
            name=name,
            parameters=parameters
        )

    @parameterized.expand([
        (alg_name, ModelTasks.REGRESSION, alg_name, alg)
        for alg, alg_name  in (
        (PLSRegression, "PLSR"),
        (SVR, "SVR"),
        (RandomForestRegressor, "RFR"),
        (XGBRegressor, "XGBR"),
        (KNeighborsRegressor, "KNNR")
    )
    ])
    def test_regression_basic_fit(self, _, task, model_name, model_class):
        if not model_name in ["SVR", "PLSR"]:
            parameters = {"n_jobs": N_CPUS}
        else:
            parameters = {}

        # initialize dataset
        dataset = self.create_large_dataset(task=task)

        # initialize model for training from class
        model = self.get_model(
            name=f"{model_name}_{task}",
            alg=model_class,
            dataset=dataset,
            parameters=parameters
        )
        self.fit_test(model)
        self.predictor_test(f"{model_name}_{task}", model.baseDir)
    @parameterized.expand([
        (alg_name, ModelTasks.CLASSIFICATION, alg_name, alg)
        for alg, alg_name in (
                (SVC, "SVC"),
                (RandomForestClassifier, "RFC"),
                (XGBClassifier, "XGBC"),
                (KNeighborsClassifier, "KNNC"),
                (GaussianNB, "NB")
        )
    ])
    def test_classification_basic_fit(self, _, task, model_name, model_class):
        if not model_name in ["NB", "SVC"]:
            parameters = {"n_jobs": N_CPUS}
        else:
            parameters = {}

        if model_name == "SVC":
            parameters.update({"probability": True})

        # initialize dataset
        dataset = self.create_large_dataset(task=task, th=[0, 1, 10, 1100])

        # test classifier
        # initialize model for training from class
        model = self.get_model(
            name=f"{model_name}_{task}",
            alg=model_class,
            dataset=dataset,
            parameters=parameters
        )
        self.fit_test(model)
        self.predictor_test(f"{model_name}_{task}", model.baseDir)