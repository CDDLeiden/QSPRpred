"""This module holds the tests for functions regarding QSPR modelling."""
import logging
import numbers
import os
import shutil
from os.path import exists
from unittest import TestCase

import numpy as np
import pandas as pd
from parameterized import parameterized
from qsprpred.data.tests import DataSetsMixIn
from qsprpred.models.hyperparam_optimization import (
    GridSearchOptimization,
    OptunaOptimization,
)
from qsprpred.models.interfaces import QSPRModel
from qsprpred.models.metrics import SklearnMetric
from qsprpred.models.models import QSPRsklearn
from qsprpred.models.tasks import ModelTasks, TargetTasks
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import get_scorer as get_sklearn_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

N_CPUS = 2
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

    @property
    def gridFile(self):
        return f'{os.path.dirname(__file__)}/test_files/search_space_test.json'

    def getParamGrid(self, model, grid):
        mname = model.name.split("_")[0]
        grid_params = model.__class__.loadParamsGrid(self.gridFile, grid, mname)
        return grid_params[grid_params[:, 0] == mname, 1][0]

    def fit_test(self, themodel):
        """Test model fitting, optimization and evaluation."""
        # perform bayes optimization
        search_space_bs = self.getParamGrid(themodel, "bayes")
        bayesoptimizer = OptunaOptimization(scoring = themodel.score_func, param_grid=search_space_bs, n_trials=1)
        best_params = bayesoptimizer.optimize(themodel)
        self.assertTrue(exists(f"{themodel.outDir}/{themodel.name}_params.json"))
        # perform grid search
        search_space_gs = self.getParamGrid(themodel, "grid")
        gridsearcher = GridSearchOptimization(scoring = themodel.score_func, param_grid=search_space_gs)
        best_params = gridsearcher.optimize(themodel)
        self.assertTrue(exists(f"{themodel.outDir}/{themodel.name}_params.json"))
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
        self.assertTrue(exists(f"{themodel.baseDir}/{themodel.metaInfo['estimator_path']}"))
        self.assertTrue(exists(f"{themodel.baseDir}/{themodel.metaInfo['parameters_path']}"))
        self.assertTrue(all(exists(f"{themodel.baseDir}/{x}") for x in themodel.metaInfo['feature_calculator_paths']))
        self.assertTrue(exists(f"{themodel.baseDir}/{themodel.metaInfo['feature_standardizer_path']}"))

    def predictor_test(self, predictor : QSPRModel, **pred_kwargs):
        """Test a model as predictor."""


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
                         predictor.targetProperties[0].nClasses))
                else:
                    self.assertEqual(
                        predictions.shape,
                        (len(input_smiles),
                         predictor.targetProperties[0].nClasses))
            else:
                self.assertEqual(predictions.shape, (len(input_smiles), len(predictor.targetProperties)))

        # predict the property
        for use_probas in [True, False]:
            predictions = predictor.predictMols(df.SMILES.to_list(), use_probas=use_probas, **pred_kwargs)
            check_shape(df.SMILES.to_list())
            if isinstance(predictions, list):
                for prediction in predictions:
                    self.assertIsInstance(prediction, np.ndarray)
            else:
                self.assertIsInstance(predictions, np.ndarray)

            singleoutput = predictions[0][0, 0] if isinstance(predictions, list) else predictions[0, 0]
            if predictor.targetProperties[0].task == TargetTasks.REGRESSION or use_probas:
                self.assertIsInstance(singleoutput, numbers.Real)
            elif predictor.targetProperties[0].task == TargetTasks.MULTICLASS or isinstance(predictor.estimator, XGBClassifier):
                self.assertIsInstance(singleoutput, numbers.Integral)
            elif predictor.targetProperties[0].task == TargetTasks.SINGLECLASS:
                self.assertIn(singleoutput, [1, 0])
            else:
                return AssertionError(f"Unknown task: {predictor.task}")

            # test with an invalid smiles
            invalid_smiles = ["C1CCCCC1", "C1CCCCC"]
            predictions = predictor.predictMols(invalid_smiles, use_probas=use_probas, **pred_kwargs)
            check_shape(invalid_smiles)
            singleoutput = predictions[0][0, 0] if isinstance(predictions, list) else predictions[0, 0]
            self.assertEqual(predictions[0][1, 0] if isinstance(predictions, list) else predictions[1, 0], None)
            if predictor.targetProperties[0].task == TargetTasks.SINGLECLASS and not isinstance(
                    predictor.estimator, XGBClassifier) and not use_probas:
                self.assertIn(singleoutput, [0, 1])
            else:
                self.assertIsInstance(singleoutput, numbers.Number)


class TestQSPRsklearn(ModelDataSetsMixIn, ModelTestMixIn, TestCase):
    """This class holds the tests for the QSPRsklearn class."""

    @staticmethod
    def get_model(name, alg=None, dataset=None, parameters=None):
        """Intialize dataset and model."""
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
        predictor = QSPRsklearn(name=f"{model_name}_{task}", base_dir=model.baseDir)
        self.predictor_test(predictor)

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
        predictor = QSPRsklearn(name=f"{model_name}_{task}", base_dir=model.baseDir)
        self.predictor_test(predictor)

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
        predictor = QSPRsklearn(name=f"{model_name}_multitask_regression", base_dir=model.baseDir)
        self.predictor_test(predictor)

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
        predictor = QSPRsklearn(name=f"{model_name}_multitask_classification", base_dir=model.baseDir)
        self.predictor_test(predictor)


class test_Metrics(TestCase):
    """Test the SklearnMetrics from the metrics module."""

    def checkMetric(self, metric, task, y_true, y_pred, y_pred_proba=None):
        """Check if the metric is correctly implemented."""
        scorer = SklearnMetric.getMetric(metric)
        self.assertEqual(scorer.name, metric)
        self.assertTrue(getattr(scorer, f'supports_{task}'))
        self.assertTrue(scorer.supportsTask(task))

        # lambda function to get the sklearn scoring function from the scorer object
        sklearn_scorer = get_sklearn_scorer(metric)

        def sklearn_func(y_true, y_pred):
            return sklearn_scorer._sign * sklearn_scorer._score_func(y_true, y_pred, **sklearn_scorer._kwargs)

        if y_pred_proba is not None and scorer.needs_proba_to_score:
            self.assertEqual(scorer(y_true, y_pred_proba), sklearn_func(y_true, y_pred_proba))
        else:
            self.assertEqual(scorer(y_true, y_pred), sklearn_func(y_true, y_pred))

    def test_RegressionMetrics(self):
        """Test the regression metrics."""
        y_true = np.array([1.2, 2.2, 3.2, 4.2, 5.2])
        y_pred = np.array([1.2, 2.2, 3.2, 4.2, 5.2])

        for metric in SklearnMetric.regressionMetrics:
            self.checkMetric(metric, ModelTasks.REGRESSION, y_true, y_pred)

    def test_SingleClassMetrics(self):
        """Test the single class metrics."""
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.9, 0.2, 0.8, 0.1, 0.9])

        for metric in SklearnMetric.singleClassMetrics:
            self.checkMetric(metric, ModelTasks.SINGLECLASS, y_true, y_pred, y_pred_proba)

    def test_MultiClassMetrics(self):
        """Test the multi class metrics."""
        y_true = np.array([0, 1, 2, 1, 1])
        y_pred = np.array([0, 1, 2, 1, 1])
        y_pred_proba = np.array([[0.9, 0.1, 0.0],
                                 [0.1, 0.8, 0.1],
                                 [0.0, 0.1, 0.9],
                                 [0.1, 0.8, 0.1],
                                 [0.1, 0.8, 0.1]])

        for metric in SklearnMetric.multiClassMetrics:
            self.checkMetric(metric, ModelTasks.MULTICLASS, y_true, y_pred, y_pred_proba)

    def test_MultiTaskRegressionMetrics(self):
        """Test the multi task regression metrics."""
        y_true = np.array([[1.2, 2.2, 3.2, 4.2, 5.2],
                           [1.2, 2.2, 3.2, 4.2, 5.2]])
        y_pred = np.array([[1.2, 2.2, 3.2, 4.2, 5.2],
                           [1.2, 2.2, 3.2, 4.2, 5.2]])

        for metric in SklearnMetric.multiTaskRegressionMetrics:
            self.checkMetric(metric, ModelTasks.MULTITASK_REGRESSION, y_true, y_pred)

    def test_MultiTaskSingleClassMetrics(self):
        """Test the multi task single class metrics."""
        y_true = np.array([[1, 0],
                           [1, 1],
                           [1, 0],
                           [0, 0],
                           [1, 0]])
        y_pred = np.array([[1, 0],
                           [1, 1],
                           [1, 0],
                           [0, 0],
                           [1, 0]])
        y_pred_proba = np.array([[0.9, 0.6],
                                 [0.5, 0.4],
                                 [0.3, 0.8],
                                 [0.7, 0.1],
                                 [1, 0.4]])

        for metric in SklearnMetric.multiTaskSingleClassMetrics:
            self.checkMetric(metric, ModelTasks.MULTITASK_SINGLECLASS, y_true, y_pred, y_pred_proba)
