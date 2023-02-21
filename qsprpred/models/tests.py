"""This module holds the tests for functions regarding QSPR modelling."""
import logging
import numbers
import os
import shutil
import types
from os.path import exists
from unittest import TestCase

import numpy as np
import pandas as pd
import torch
from parameterized import parameterized
from qsprpred.data.tests import DataSetsMixIn
from qsprpred.models.interfaces import QSPRModel
from qsprpred.models.models import QSPRDNN, QSPRsklearn
from qsprpred.models.neural_network import STFullyConnected
from qsprpred.models.tasks import TargetTasks
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier, XGBRegressor

N_CPUS = 2
GPUS = [idx for idx in range(torch.cuda.device_count())]
logging.basicConfig(level=logging.DEBUG)


class ModelDataSetsMixIn(DataSetsMixIn):
    """This class sets up the datasets for the model tests."""

    qsprmodelspath = f'{os.path.dirname(__file__)}/test_files/qspr/models'

    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        if not os.path.exists(self.qsprmodelspath):
            os.makedirs(self.qsprmodelspath)

    @classmethod
    def clean_directories(cls):
        """Clean the directories."""
        super().clean_directories()
        if os.path.exists(cls.qsprmodelspath):
            shutil.rmtree(cls.qsprmodelspath)


class ModelTestMixIn:
    """This class holds the tests for the QSPRmodel class."""

    def fit_test(self, themodel):
        """Test model fitting, optimization and evaluation."""
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

    def predictor_test(self, model_name, base_dir, cls: QSPRModel = QSPRsklearn, n_tasks=1):
        """Test using a QSPRmodel as predictor."""
        # initialize model as predictor
        predictor = cls(name=model_name, base_dir=base_dir)

        # load molecules to predict
        df = pd.read_csv(
            f'{os.path.dirname(__file__)}/test_files/data/test_data.tsv',
            sep='\t')

        def check_shape(input_smiles):
            if predictor.targetProperties[0].task.isClassification() and use_probas:
                if predictor.isMultiTask:
                    self.assertEqual(len(predictions), len(predictor.targetProperties))
                    self.assertEqual(
                        predictions[0].shape,
                        (len(input_smiles),
                         predictor.targetProperties[0].getNclasses()))
                else:
                    self.assertEqual(
                        predictions.shape,
                        (len(input_smiles),
                         predictor.targetProperties[0].getNclasses()))
            else:
                self.assertEqual(predictions.shape, (len(input_smiles), len(predictor.targetProperties)))

        # predict the property
        for use_probas in [True, False]:
            predictions = predictor.predictMols(df.SMILES.to_list(), use_probas=use_probas)
            check_shape(df.SMILES.to_list())
            if isinstance(predictions, list):
                for prediction in predictions:
                    self.assertIsInstance(prediction, np.ndarray)
            else:
                self.assertIsInstance(predictions, np.ndarray)

            singleoutput = predictions[0][0, 0] if isinstance(predictions, list) else predictions[0, 0]
            if predictor.targetProperties[0].task == TargetTasks.REGRESSION or use_probas:
                self.assertIsInstance(singleoutput, numbers.Real)
            elif predictor.targetProperties[0].task == TargetTasks.MULTICLASS or isinstance(predictor.model, XGBClassifier):
                self.assertIsInstance(singleoutput, numbers.Integral)
            elif predictor.targetProperties[0].task == TargetTasks.SINGLECLASS:
                self.assertIn(singleoutput, [1, 0])
            else:
                return AssertionError(f"Unknown task: {predictor.task}")

            # test with an invalid smiles
            invalid_smiles = ["C1CCCCC1", "C1CCCCC"]
            predictions = predictor.predictMols(invalid_smiles, use_probas=use_probas)
            check_shape(invalid_smiles)
            singleoutput = predictions[0][0, 0] if isinstance(predictions, list) else predictions[0, 0]
            self.assertEqual(predictions[0][1, 0] if isinstance(predictions, list) else predictions[1, 0], None)
            if predictor.targetProperties[0].task == TargetTasks.SINGLECLASS and not isinstance(
                    predictor.model, XGBClassifier) and not use_probas:
                self.assertIn(singleoutput, [0, 1])
            else:
                self.assertIsInstance(singleoutput, numbers.Number)

        # test the same for classification with probabilities
        if predictor.task == TargetTasks.CLASSIFICATION:
            predictions = predictor.predictMols(invalid_smiles, use_probas=True)
            self.assertEqual(predictions.shape, (len(invalid_smiles), predictor.nClasses))
            for cls in range(predictor.nClasses):
                self.assertIsInstance(predictions[0, 1], numbers.Real)
                self.assertTrue(np.isnan(predictions[1, cls]))


class NeuralNet(ModelDataSetsMixIn, ModelTestMixIn, TestCase):
    """This class holds the tests for the QSPRDNN class."""

    @staticmethod
    def get_model(name, alg=None, dataset=None, parameters=None):
        """Intialize dataset and model."""
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

    def prep_testdata(self, task=TargetTasks.REGRESSION, th=None):
        """Prepare test dataset."""
        data = self.create_large_dataset(task=task, th=th, preparation_settings=self.get_default_prep())
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
            (STFullyConnected, "STFullyConnected", TargetTasks.REGRESSION, None),
            (STFullyConnected, "STFullyConnected", TargetTasks.SINGLECLASS, [6.5]),
            (STFullyConnected, "STFullyConnected", TargetTasks.SINGLECLASS, [0, 1, 10, 1200]),
        )
    ])
    def test_base_model(self, _, task, alg_name, alg, th):
        """Test the base DNN model."""
        # prepare test regression dataset
        is_reg = True if task == TargetTasks.REGRESSION else False
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
            (STFullyConnected, "STFullyConnected", TargetTasks.REGRESSION, None),
            (STFullyConnected, "STFullyConnected", TargetTasks.SINGLECLASS, [6.5]),
            (STFullyConnected, "STFullyConnected", TargetTasks.SINGLECLASS, [0, 1, 10, 1100]),
        )
    ])
    def test_qsprpred_model(self, _, task, alg_name, alg, th):
        """Test the QSPRDNN model."""
        # initialize dataset
        dataset = self.create_large_dataset(task=task, th=th, preparation_settings=self.get_default_prep())

        # initialize model for training from class
        alg_name = f"{alg_name}_{task}_th={th}"
        model = self.get_model(
            name=alg_name,
            alg=alg,
            dataset=dataset
        )
        self.fit_test(model)
        self.predictor_test(alg_name, model.baseDir, QSPRDNN)


class TestQSPRsklearn(ModelDataSetsMixIn, ModelTestMixIn, TestCase):
    """This class holds the tests for the QSPRsklearn class."""

    @staticmethod
    def get_model(name, alg=None, dataset=None, parameters=None):
        """intialize dataset and model."""
        return QSPRsklearn(
            base_dir=f'{os.path.dirname(__file__)}/test_files/',
            alg=alg,
            data=dataset,
            name=name,
            parameters=parameters
        )

    @parameterized.expand([
        (alg_name, TargetTasks.REGRESSION, alg_name, alg)
        for alg, alg_name in (
            (PLSRegression, "PLSR"),
            (SVR, "SVR"),
            (RandomForestRegressor, "RFR"),
            (XGBRegressor, "XGBR"),
            (KNeighborsRegressor, "KNNR")
        )
        for alg, alg_name in (
            (PLSRegression, "PLSR"),
            (SVR, "SVR"),
            (RandomForestRegressor, "RFR"),
            (XGBRegressor, "XGBR"),
            (KNeighborsRegressor, "KNNR")
        )
    ])
    def test_regression_basic_fit(self, _, task, model_name, model_class):
        """Test model training for regression models."""
        if not model_name in ["SVR", "PLSR"]:
            parameters = {"n_jobs": N_CPUS}
        else:
            parameters = None

        # initialize dataset
        dataset = self.create_large_dataset(
            target_props=[{"name": 'CL', "task": task}],
            preparation_settings=self.get_default_prep())

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
        (f"{alg_name}_{task}", task, th, alg_name, alg)
        for alg, alg_name in (
            (SVC, "SVC"),
            (RandomForestClassifier, "RFC"),
            (XGBClassifier, "XGBC"),
            (KNeighborsClassifier, "KNNC"),
            (GaussianNB, "NB")
        ) for task, th in
        ((TargetTasks.SINGLECLASS, [6.5]),
         (TargetTasks.MULTICLASS, [0, 1, 10, 1100]))
    ])
    def test_classification_basic_fit(self, _, task, th, model_name, model_class):
        """Test model training for classification models."""
        if not model_name in ["NB", "SVC"]:
            parameters = {"n_jobs": N_CPUS}
        else:
            parameters = None

        if model_name == "SVC":
            if parameters is not None:
                parameters.update({"probability": True})
            else:
                parameters = {"probability": True}

        # initialize dataset
        dataset = self.create_large_dataset(
            target_props=[{"name": 'CL', "task": task, "th": th}],
            preparation_settings=self.get_default_prep())

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

    @parameterized.expand([
        (alg_name, alg_name, alg)
        for alg, alg_name in (
            (RandomForestRegressor, "RFR"),
            (KNeighborsRegressor, "KNNR"),
        )
    ])
    def test_regression_multitask_fit(self, _, model_name, model_class):
        """Test model training for multitask regression models."""
        if not model_name in ["NB", "SVC"]:
            parameters = {"n_jobs": N_CPUS}
        else:
            parameters = {}

        if model_name == "SVC":
            parameters.update({"probability": True})

        # initialize dataset
        dataset = self.create_large_dataset(target_props=[{"name": "fu", "task": TargetTasks.REGRESSION}, {
                                            "name": "CL", "task": TargetTasks.REGRESSION}],
                                            target_imputer=SimpleImputer(strategy='mean'),
                                            preparation_settings=self.get_default_prep())

        # test classifier
        # initialize model for training from class
        model = self.get_model(
            name=f"{model_name}_multitask_regression",
            alg=model_class,
            dataset=dataset,
            parameters=parameters
        )
        self.fit_test(model)
        self.predictor_test(f"{model_name}_multitask_regression", model.baseDir)

    @parameterized.expand([
        (alg_name, alg_name, alg)
        for alg, alg_name in (
            (RandomForestClassifier, "RFC"),
            (KNeighborsClassifier, "KNNC"),
        )
    ])
    def test_classification_multitask_fit(self, _, model_name, model_class):
        """Test model training for multitask classification models."""
        if not model_name in ["NB", "SVC"]:
            parameters = {"n_jobs": N_CPUS}
        else:
            parameters = {}

        if model_name == "SVC":
            parameters.update({"probability": True})

        # initialize dataset
        dataset = self.create_large_dataset(
            target_props=[{"name": "fu", "task": TargetTasks.SINGLECLASS, "th": [0.3]},
                          {"name": "CL", "task": TargetTasks.SINGLECLASS, "th": [6.5]}],
            target_imputer=SimpleImputer(strategy='mean'),
            preparation_settings=self.get_default_prep())

        # test classifier
        # initialize model for training from class
        model = self.get_model(
            name=f"{model_name}_multitask_classification",
            alg=model_class,
            dataset=dataset,
            parameters=parameters
        )
        self.fit_test(model)
        self.predictor_test(f"{model_name}_multitask_classification", model.baseDir)
