"""This module holds the tests for functions regarding QSPR modelling."""

import logging
import numbers
import os
from os.path import exists
from typing import Type
from unittest import TestCase

import numpy as np
import pandas as pd
from parameterized import parameterized
from sklearn import metrics
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import get_scorer as get_sklearn_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from ..data.data import QSPRDataset
from ..data.tests import DataSetsMixIn
from ..models.assessment_methods import CrossValAssessor, TestSetAssessor
from ..models.early_stopping import EarlyStopping, EarlyStoppingMode, early_stopping
from ..models.hyperparam_optimization import GridSearchOptimization, OptunaOptimization
from ..models.interfaces import QSPRModel
from ..models.metrics import SklearnMetric
from ..models.models import QSPRsklearn
from ..models.tasks import ModelTasks, TargetTasks

N_CPUS = 2
logging.basicConfig(level=logging.DEBUG)


class ModelDataSetsMixIn(DataSetsMixIn):
    """This class sets up the datasets for the model tests."""
    def setUp(self):
        """Set up the test environment."""
        super().setUp()
        self.generatedModelsPath = f"{self.generatedPath}/models/"
        if not os.path.exists(self.generatedModelsPath):
            os.makedirs(self.generatedModelsPath)


class ModelTestMixIn:
    """This class holds the tests for the QSPRmodel class."""
    @property
    def gridFile(self):
        return f"{os.path.dirname(__file__)}/test_files/search_space_test.json"

    def getParamGrid(self, model: QSPRModel, grid: str) -> dict:
        """Get the parameter grid for a model.

        Args:
            model (QSPRModel): The model to get the parameter grid for.
            grid (str): The grid type to get the parameter grid for.

        Returns:
            dict: The parameter grid.
        """

        mname = model.name.split("_")[0]
        grid_params = model.__class__.loadParamsGrid(self.gridFile, grid, mname)
        return grid_params[grid_params[:, 0] == mname, 1][0]

    def fitTest(self, model: QSPRModel):
        """Test model fitting, optimization and evaluation.

        Args:
            model (QSPRModel): The model to test.
        """
        # perform bayes optimization
        score_func = SklearnMetric.getDefaultMetric(model.task)
        search_space_bs = self.getParamGrid(model, "bayes")
        bayesoptimizer = OptunaOptimization(
            scoring=score_func,
            param_grid=search_space_bs,
            n_trials=1,
            model_assessor=CrossValAssessor(mode=EarlyStoppingMode.NOT_RECORDING),
        )
        best_params = bayesoptimizer.optimize(model)
        model.saveParams(best_params)
        self.assertTrue(exists(f"{model.outDir}/{model.name}_params.json"))
        # perform grid search
        search_space_gs = self.getParamGrid(model, "grid")
        if model.task.isClassification():
            score_func = SklearnMetric.getMetric("accuracy")
        gridsearcher = GridSearchOptimization(
            scoring=score_func,
            param_grid=search_space_gs,
            score_aggregation=np.median,
            model_assessor=TestSetAssessor(
                use_proba=False, mode=EarlyStoppingMode.NOT_RECORDING
            ),
        )
        best_params = gridsearcher.optimize(model)
        model.saveParams(best_params)
        self.assertTrue(exists(f"{model.outDir}/{model.name}_params.json"))
        model.cleanFiles()
        # perform crossvalidation
        CrossValAssessor(mode=EarlyStoppingMode.RECORDING)(model)
        TestSetAssessor(mode=EarlyStoppingMode.NOT_RECORDING)(model)
        self.assertTrue(exists(f"{model.outDir}/{model.name}.ind.tsv"))
        self.assertTrue(exists(f"{model.outDir}/{model.name}.cv.tsv"))
        # train the model on all data
        model.fitAttached()
        self.assertTrue(exists(model.metaFile))
        self.assertTrue(exists(f"{model.baseDir}/{model.metaInfo['estimator_path']}"))
        self.assertTrue(exists(f"{model.baseDir}/{model.metaInfo['parameters_path']}"))
        self.assertTrue(
            all(
                exists(f"{model.baseDir}/{x}")
                for x in model.metaInfo["feature_calculator_paths"]
            )
        )
        if model.metaInfo["feature_standardizer_path"] is not None:
            self.assertTrue(
                exists(
                    f"{model.baseDir}/{model.metaInfo['feature_standardizer_path']}"
                )
            )

    def predictorTest(
            self,
            predictor: QSPRModel,
            expect_equal_result=True,
            expected_pred_use_probas=None,
            expected_pred_not_use_probas=None,
            **pred_kwargs):
        """Test a model as predictor.

        Args:
            predictor (QSPRModel):
                The model to test.
            **pred_kwargs:
                Extra keyword arguments to pass to the predictor's `predictMols` method.
        """

        # load molecules to predict
        df = pd.read_csv(
            f"{os.path.dirname(__file__)}/test_files/data/test_data.tsv", sep="\t"
        )

        # define checks of the shape of the predictions
        def check_shape(input_smiles):
            if predictor.targetProperties[0].task.isClassification() and use_probas:
                self.assertEqual(len(predictions), len(predictor.targetProperties))
                self.assertEqual(
                    predictions[0].shape,
                    (len(input_smiles), predictor.targetProperties[0].nClasses),
                )
            else:
                self.assertEqual(
                    predictions.shape,
                    (len(input_smiles), len(predictor.targetProperties)),
                )

        # predict the property
        pred = []
        for use_probas in [True, False]:
            predictions = predictor.predictMols(
                df.SMILES.to_list(), use_probas=use_probas, **pred_kwargs
            )
            check_shape(df.SMILES.to_list())
            if isinstance(predictions, list):
                for prediction in predictions:
                    self.assertIsInstance(prediction, np.ndarray)
            else:
                self.assertIsInstance(predictions, np.ndarray)

            singleoutput = (
                predictions[0][0,
                               0] if isinstance(predictions, list) else predictions[0,
                                                                                    0]
            )
            if (
                predictor.targetProperties[0].task == TargetTasks.REGRESSION or
                use_probas
            ):
                self.assertIsInstance(singleoutput, numbers.Real)
            elif predictor.targetProperties[
                0].task == TargetTasks.MULTICLASS or isinstance(
                    predictor.estimator, XGBClassifier
                ):
                self.assertIsInstance(singleoutput, numbers.Integral)
            elif predictor.targetProperties[0].task == TargetTasks.SINGLECLASS:
                self.assertIn(singleoutput, [1, 0])
            else:
                return AssertionError(f"Unknown task: {predictor.task}")
            pred.append(singleoutput)
            # test with an invalid smiles
            invalid_smiles = ["C1CCCCC1", "C1CCCCC"]
            predictions = predictor.predictMols(
                invalid_smiles, use_probas=use_probas, **pred_kwargs
            )
            check_shape(invalid_smiles)
            singleoutput = (
                predictions[0][0,
                               0] if isinstance(predictions, list) else predictions[0,
                                                                                    0]
            )
            self.assertEqual(
                predictions[0][1,
                               0] if isinstance(predictions, list) else predictions[1,
                                                                                    0],
                None,
            )
            if (
                predictor.targetProperties[0].task == TargetTasks.SINGLECLASS and
                not isinstance(predictor.estimator, XGBClassifier) and not use_probas
            ):
                self.assertIn(singleoutput, [0, 1])
            else:
                self.assertIsInstance(singleoutput, numbers.Number)

        if expect_equal_result:
            if expected_pred_use_probas is not None:
                self.assertEqual(pred[0], expected_pred_use_probas)
            if expected_pred_not_use_probas is not None:
                self.assertEqual(pred[1], expected_pred_not_use_probas)
        else:
            if expected_pred_use_probas is not None:
                self.assertNotAlmostEqual(pred[0], expected_pred_use_probas, places=5)
            if expected_pred_not_use_probas is not None:
                self.assertNotAlmostEqual(pred[1], expected_pred_not_use_probas, places=5)


        return pred[0], pred[1]

    # NOTE: below code is taken from CorrelationPlot without the plotting code
    # Would be nice if the summary code without plot was available regardless
    def createCorrelationSummary(self, model):
        cv_path = f"{model.outPrefix}.cv.tsv"
        ind_path = f"{model.outPrefix}.ind.tsv"

        cate = [cv_path, ind_path]
        cate_names = ["cv", "ind"]
        property_name = model.targetProperties[0].name
        summary = {"ModelName": [], "R2": [], "RMSE": [], "Set": []}
        for j, _ in enumerate(["Cross Validation", "Independent Test"]):
            df = pd.read_table(cate[j])
            coef = metrics.r2_score(
                df[f"{property_name}_Label"], df[f"{property_name}_Prediction"]
            )
            rmse = metrics.mean_squared_error(
                df[f"{property_name}_Label"],
                df[f"{property_name}_Prediction"],
                squared=False,
            )
            summary["R2"].append(coef)
            summary["RMSE"].append(rmse)
            summary["Set"].append(cate_names[j])
            summary["ModelName"].append(model.name)
            
        return summary

    # NOTE: below code is taken from MetricsPlot without the plotting code
    # Would be nice if the summary code without plot was available regardless
    def createMetricsSummary(self, model):
        decision_threshold: float = 0.5
        metrics = [
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            accuracy_score,
        ]
        summary = {"Metric": [], "Model": [], "TestSet": [], "Value": []}
        property_name = model.targetProperties[0].name

        cv_path = f"{model.outPrefix}.cv.tsv"
        ind_path = f"{model.outPrefix}.ind.tsv"

        df = pd.read_table(cv_path)

        # cross-validation
        for fold in sorted(df.Fold.unique()):
            y_pred = df[f"{property_name}_ProbabilityClass_1"][df.Fold == fold]
            y_pred_values = [1 if x > decision_threshold else 0 for x in y_pred]
            y_true = df[f"{property_name}_Label"][df.Fold == fold]
            for metric in metrics:
                val = metric(y_true, y_pred_values)
                summary["Metric"].append(metric.__name__)
                summary["Model"].append(model.name)
                summary["TestSet"].append(f"CV{fold + 1}")
                summary["Value"].append(val)

        # independent test set
        df = pd.read_table(ind_path)
        y_pred = df[f"{property_name}_ProbabilityClass_1"]
        th = 0.5
        y_pred_values = [1 if x > th else 0 for x in y_pred]
        y_true = df[f"{property_name}_Label"]
        for metric in metrics:
            val = metric(y_true, y_pred_values)
            summary["Metric"].append(metric.__name__)
            summary["Model"].append(model.name)
            summary["TestSet"].append("IND")
            summary["Value"].append(val)

        return summary

class TestQSPRsklearn(ModelDataSetsMixIn, ModelTestMixIn, TestCase):
    """This class holds the tests for the QSPRsklearn class."""
    def getModel(
        self,
        name: str,
        alg: Type | None = None,
        dataset: QSPRDataset = None,
        parameters: dict | None = None,
        random_state: int | None = None
    ):
        """Create a QSPRsklearn model.

        Args:
            name (str): the name of the model
            alg (Type, optional): the algorithm to use. Defaults to None.
            dataset (QSPRDataset, optional): the dataset to use. Defaults to None.
            parameters (dict, optional): the parameters to use. Defaults to None.

        Returns:
            QSPRsklearn: the model
        """
        return QSPRsklearn(
            base_dir=self.generatedModelsPath,
            alg=alg,
            data=dataset,
            name=name,
            parameters=parameters,
            random_state=random_state
        )

    @parameterized.expand(
        [
            (alg_name, TargetTasks.REGRESSION, alg_name, alg, random_state) for alg, alg_name in (
                (RandomForestRegressor, "RFR"),
                (XGBRegressor, "XGBR"),
            ) for random_state in ([None], [1, 42], [42, 42])
        ] + [
            (alg_name, TargetTasks.REGRESSION, alg_name, alg, [None]) for alg, alg_name in (
                (PLSRegression, "PLSR"),
                (SVR, "SVR"),
                (KNeighborsRegressor, "KNNR"),
            )
        ]
    )
    def testRegressionBasicFit(self, _, task, model_name, model_class, random_state):
        """Test model training for regression models."""
        if model_name not in ["SVR", "PLSR"]:
            parameters = {"n_jobs": N_CPUS}
        else:
            parameters = None
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            target_props=[{
                "name": "CL",
                "task": task
            }],
            preparation_settings=self.getDefaultPrep(),
        )
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_{task}",
            alg=model_class,
            dataset=dataset,
            parameters=parameters,
            random_state=random_state[0]
        )
        self.fitTest(model)
        predictor = QSPRsklearn(name=f"{model_name}_{task}", base_dir=model.baseDir)
        pred_use_probas, pred_not_use_probas = self.predictorTest(predictor)

        if random_state[0] is not None:
            model = self.getModel(
                name=f"{model_name}_{task}",
                alg=model_class,
                dataset=dataset,
                parameters=parameters,
                random_state=random_state[1]
            )
            self.fitTest(model)
            predictor = QSPRsklearn(name=f"{model_name}_{task}", base_dir=model.baseDir)
            self.predictorTest(
                predictor,
                expect_equal_result=random_state[0] == random_state[1],
                expected_pred_use_probas=pred_use_probas,
                expected_pred_not_use_probas=pred_not_use_probas)


    def testPLSRegressionSummaryWithSeed(self):
        """Test model training for regression models."""
        task = TargetTasks.REGRESSION
        model_name = "PLSR"
        model_class = PLSRegression
        parameters = None
        dataset = self.createLargeTestDataSet(
            target_props=[{
                "name": "CL",
                "task": task
            }],
            preparation_settings=self.getDefaultPrep(),
        )
        model = self.getModel(
            name=f"{model_name}_{task}",
            alg=model_class,
            dataset=dataset,
            parameters=parameters,
        )
        self.fitTest(model)
        expected_summary = self.createCorrelationSummary(model)

        #Generate summary again, check that the result is identical
        model = self.getModel(
            name=f"{model_name}_{task}",
            alg=model_class,
            dataset=dataset,
            parameters=parameters,
        )
        self.fitTest(model)
        summary = self.createCorrelationSummary(model)

        self.assertListEqual(summary["ModelName"], expected_summary["ModelName"])
        self.assertListEqual(summary["R2"], expected_summary["R2"])
        self.assertListEqual(summary["RMSE"], expected_summary["RMSE"])
        self.assertListEqual(summary["Set"], expected_summary["Set"])


    @parameterized.expand(
        [
            (f"{alg_name}_{task}", task, th, alg_name, alg, random_state) for alg, alg_name in (
                (SVC, "SVC"),
                (RandomForestClassifier, "RFC"),
                (XGBClassifier, "XGBC"),
            ) for task, th in (
                (TargetTasks.SINGLECLASS, [6.5]),
                (TargetTasks.MULTICLASS, [0, 1, 10, 1100]),
            ) for random_state in (None, 42)
        ] + [
            (f"{alg_name}_{task}", task, th, alg_name, alg, None) for alg, alg_name in (
                (KNeighborsClassifier, "KNNC"),
                (GaussianNB, "NB"),
            ) for task, th in (
                (TargetTasks.SINGLECLASS, [6.5]),
                (TargetTasks.MULTICLASS, [0, 1, 10, 1100]),
            )
        ]
    )
    def testClassificationBasicFit(self, _, task, th, model_name, model_class, random_state):
        """Test model training for classification models."""
        if model_name not in ["NB", "SVC"]:
            parameters = {"n_jobs": N_CPUS}
        else:
            parameters = None
        # special case for SVC
        if model_name == "SVC":
            if parameters is not None:
                parameters.update({"probability": True})
            else:
                parameters = {"probability": True}
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            target_props=[{
                "name": "CL",
                "task": task,
                "th": th
            }],
            preparation_settings=self.getDefaultPrep(),
        )
        # test classifier
        # initialize model for training from class
        model_random_state = random_state if model_name in ["SVC", "RFC", "XGBC"] else None
        model = self.getModel(
            name=f"{model_name}_{task}",
            alg=model_class,
            dataset=dataset,
            parameters=parameters,
            random_state=model_random_state
        )
        self.fitTest(model)
        predictor = QSPRsklearn(name=f"{model_name}_{task}", base_dir=model.baseDir)
        pred_use_probas, pred_not_use_probas = self.predictorTest(predictor)

        if random_state is not None:
            model = self.getModel(
                name=f"{model_name}_{task}",
                alg=model_class,
                dataset=dataset,
                parameters=parameters,
                random_state=model_random_state
            )
            self.fitTest(model)
            predictor = QSPRsklearn(name=f"{model_name}_{task}", base_dir=model.baseDir)
            self.predictorTest(
                predictor,
                expected_pred_use_probas=pred_use_probas,
                expected_pred_not_use_probas=pred_not_use_probas)


    def testRandomForestClassifierFitWithSeed(self):
        parameters = {
            "n_jobs": N_CPUS,
        }
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            target_props=[{
                "name": "CL",
                "task": TargetTasks.SINGLECLASS,
                "th": [6.5]
            }],
            preparation_settings=self.getDefaultPrep(),
        )
        # test classifier
        # initialize model for training from class
        model = self.getModel(
            name=f"RFC_{TargetTasks.SINGLECLASS}",
            alg=RandomForestClassifier,
            dataset=dataset,
            parameters=parameters,
        )
        self.fitTest(model)
        expected_summary = self.createMetricsSummary(model)

        #Generate summary again, check that the result is identical
        model = self.getModel(
            name=f"RFC_{TargetTasks.SINGLECLASS}",
            alg=RandomForestClassifier,
            dataset=dataset,
            parameters=parameters,
        )
        self.fitTest(model)
        summary = self.createMetricsSummary(model)

        self.assertListEqual(summary["Metric"], expected_summary["Metric"])
        self.assertListEqual(summary["Model"], expected_summary["Model"])
        self.assertListEqual(summary["TestSet"], expected_summary["TestSet"])
        self.assertListEqual(summary["Value"], expected_summary["Value"])


    @parameterized.expand(
        [
            (alg_name, alg_name, alg, random_state) for alg, alg_name in (
                (RandomForestRegressor, "RFR"),
            ) for random_state in ([None], [1, 42], [42, 42])
        ] + [
            (alg_name, alg_name, alg, [None]) for alg, alg_name in (
                (KNeighborsRegressor, "KNNR"),
            )
        ]
    )
    def testRegressionMultiTaskFit(self, _, model_name, model_class, random_state):
        """Test model training for multitask regression models."""
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            target_props=[
                {
                    "name": "fu",
                    "task": TargetTasks.REGRESSION
                },
                {
                    "name": "CL",
                    "task": TargetTasks.REGRESSION
                },
            ],
            target_imputer=SimpleImputer(strategy="mean"),
            preparation_settings=self.getDefaultPrep(),
        )
        # test classifier
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_multitask_regression",
            alg=model_class,
            dataset=dataset,
            random_state=random_state[0]
        )
        self.fitTest(model)
        predictor = QSPRsklearn(
            name=f"{model_name}_multitask_regression", base_dir=model.baseDir
        )
        pred_use_probas, pred_not_use_probas = self.predictorTest(predictor)
        if random_state[0] is not None and model_name in ["RFR"]:
            model = self.getModel(
                name=f"{model_name}_multitask_regression",
                alg=model_class,
                dataset=dataset,
                random_state=random_state[1]
            )
            self.fitTest(model)
            predictor = QSPRsklearn(
                name=f"{model_name}_multitask_regression", base_dir=model.baseDir
            )
            self.predictorTest(
                predictor,
                expect_equal_result=random_state[0] == random_state[1],
                expected_pred_use_probas=pred_use_probas,
                expected_pred_not_use_probas=pred_not_use_probas)


    @parameterized.expand(
        [
            (alg_name, alg_name, alg, random_state) for alg, alg_name in (
                (RandomForestClassifier, "RFC"),
            ) for random_state in (None, 42)
        ] + [
            (alg_name, alg_name, alg, None) for alg, alg_name in (
                (KNeighborsClassifier, "KNNC"),
            )
        ]
    )
    def testClassificationMultiTaskFit(self, _, model_name, model_class, random_state):
        """Test model training for multitask classification models."""
        if model_name not in ["NB", "SVC"]:
            parameters = {"n_jobs": N_CPUS}
        else:
            parameters = {}
        # special case for SVC
        if model_name == "SVC":
            parameters.update({"probability": True})
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            target_props=[
                {
                    "name": "fu",
                    "task": TargetTasks.SINGLECLASS,
                    "th": [0.3]
                },
                {
                    "name": "CL",
                    "task": TargetTasks.SINGLECLASS,
                    "th": [6.5]
                },
            ],
            target_imputer=SimpleImputer(strategy="mean"),
            preparation_settings=self.getDefaultPrep(),
        )
        # test classifier
        # initialize model for training from class
        model_random_state = random_state if model_name in ["RFC"] else None
        model = self.getModel(
            name=f"{model_name}_multitask_classification",
            alg=model_class,
            dataset=dataset,
            parameters=parameters,
            random_state=model_random_state
        )
        self.fitTest(model)
        predictor = QSPRsklearn(
            name=f"{model_name}_multitask_classification", base_dir=model.baseDir
        )
        pred_use_probas, pred_not_use_probas = self.predictorTest(predictor)
        if random_state is not None:
            model = self.getModel(
                name=f"{model_name}_multitask_classification",
                alg=model_class,
                dataset=dataset,
                parameters=parameters,
                random_state=model_random_state
            )
            self.fitTest(model)
            predictor = QSPRsklearn(
                name=f"{model_name}_multitask_classification", base_dir=model.baseDir
            )
            self.predictorTest(predictor,
                expected_pred_use_probas=pred_use_probas,
                expected_pred_not_use_probas=pred_not_use_probas)

class TestMetrics(TestCase):
    """Test the SklearnMetrics from the metrics module."""
    def checkMetric(self, metric, task, y_true, y_pred, y_pred_proba=None):
        """Check if the metric is correctly implemented."""
        scorer = SklearnMetric.getMetric(metric)
        self.assertEqual(scorer.name, metric)
        self.assertTrue(scorer.supportsTask(task))
        # lambda function to get the sklearn scoring function from the scorer object
        sklearn_scorer = get_sklearn_scorer(metric)

        def sklearn_func(y_true, y_pred):
            return sklearn_scorer._sign * sklearn_scorer._score_func(
                y_true, y_pred, **sklearn_scorer._kwargs
            )

        # perform the test
        if y_pred_proba is not None and scorer.needsProbasToScore:
            self.assertEqual(
                scorer(y_true, y_pred_proba), sklearn_func(y_true, y_pred_proba)
            )
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
            self.checkMetric(
                metric, ModelTasks.SINGLECLASS, y_true, y_pred, y_pred_proba
            )

    def test_MultiClassMetrics(self):
        """Test the multi class metrics."""
        y_true = np.array([0, 1, 2, 1, 1])
        y_pred = np.array([0, 1, 2, 1, 1])
        y_pred_proba = np.array(
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.8, 0.1],
                [0.0, 0.1, 0.9],
                [0.1, 0.8, 0.1],
                [0.1, 0.8, 0.1],
            ]
        )
        for metric in SklearnMetric.multiClassMetrics:
            self.checkMetric(
                metric, ModelTasks.MULTICLASS, y_true, y_pred, y_pred_proba
            )

    def test_MultiTaskRegressionMetrics(self):
        """Test the multi task regression metrics."""
        y_true = np.array([[1.2, 2.2, 3.2, 4.2, 5.2], [1.2, 2.2, 3.2, 4.2, 5.2]])
        y_pred = np.array([[1.2, 2.2, 3.2, 4.2, 5.2], [1.2, 2.2, 3.2, 4.2, 5.2]])
        for metric in SklearnMetric.multiTaskRegressionMetrics:
            self.checkMetric(metric, ModelTasks.MULTITASK_REGRESSION, y_true, y_pred)

    def test_MultiTaskSingleClassMetrics(self):
        """Test the multi task single class metrics."""
        y_true = np.array([[1, 0], [1, 1], [1, 0], [0, 0], [1, 0]])
        y_pred = np.array([[1, 0], [1, 1], [1, 0], [0, 0], [1, 0]])
        y_pred_proba = np.array(
            [[0.9, 0.6], [0.5, 0.4], [0.3, 0.8], [0.7, 0.1], [1, 0.4]]
        )
        for metric in SklearnMetric.multiTaskSingleClassMetrics:
            self.checkMetric(
                metric, ModelTasks.MULTITASK_SINGLECLASS, y_true, y_pred, y_pred_proba
            )


class TestEarlyStopping(ModelDataSetsMixIn, TestCase):
    def test_earlyStoppingMode(self):
        """Test the early stopping mode enum."""
        # get the enum values
        recording = EarlyStoppingMode.RECORDING
        not_recording = EarlyStoppingMode.NOT_RECORDING
        fixed = EarlyStoppingMode.FIXED
        optimal = EarlyStoppingMode.OPTIMAL

        # check if the enum bools are correct
        self.assertTrue(recording and not_recording)
        self.assertFalse(fixed)
        self.assertFalse(optimal)

        # check if the string representation is correct
        self.assertTrue("RECORDING" == str(recording))
        self.assertTrue("NOT_RECORDING" == str(not_recording))
        self.assertTrue("FIXED" == str(fixed))
        self.assertTrue("OPTIMAL" == str(optimal))

    def test_EarlyStopping(self):
        """Test the early stopping class."""

        earlystopping = EarlyStopping(EarlyStoppingMode.RECORDING)

        # check value error is raised when calling optimal epochs without recording
        with self.assertRaises(ValueError):
            earlystopping.optimalEpochs

        # record some epochs
        earlystopping.recordEpochs(10)
        earlystopping.recordEpochs(20)
        earlystopping.recordEpochs(30)
        earlystopping.recordEpochs(40)
        earlystopping.recordEpochs(60)

        # check if epochs are recorded correctly
        self.assertEqual(earlystopping.trainedEpochs, [10, 20, 30, 40, 60])

        # check if the optimal epochs are correct
        self.assertEqual(earlystopping.optimalEpochs, 32)

        # check if optimal epochs are correct when using a different aggregate function
        earlystopping.aggregateFunc = np.median
        self.assertEqual(earlystopping.optimalEpochs, 30)

        # check set num epochs manually
        earlystopping.numEpochs = 100
        self.assertEqual(earlystopping.numEpochs, 100)

        # check correct epochs are returned when using fixed mode and optimal mode
        earlystopping.mode = EarlyStoppingMode.FIXED
        self.assertEqual(earlystopping.getEpochs(), 100)
        earlystopping.mode = EarlyStoppingMode.OPTIMAL
        self.assertEqual(earlystopping.getEpochs(), 30)

        # check saving
        earlystopping.toFile(
            f"{os.path.dirname(__file__)}/test_files/earlystopping.json"
        )
        self.assertTrue(
            os.path.
            exists(f"{os.path.dirname(__file__)}/test_files/earlystopping.json")
        )

        # check loading
        earlystopping2 = EarlyStopping.fromFile(
            f"{os.path.dirname(__file__)}/test_files/earlystopping.json"
        )
        self.assertEqual(earlystopping2.trainedEpochs, [10, 20, 30, 40, 60])
        self.assertEqual(earlystopping2.mode, EarlyStoppingMode.OPTIMAL)
        self.assertEqual(earlystopping2.aggregateFunc, np.median)
        self.assertEqual(earlystopping2.optimalEpochs, 30)
        self.assertEqual(earlystopping.numEpochs, 100)

    def test_early_stopping_decorator(self):
        """Test the early stopping decorator."""
        class test_class:
            def __init__(self, support=True):
                self.earlyStopping = EarlyStopping(EarlyStoppingMode.RECORDING)
                self.supportsEarlyStopping = support

            @early_stopping
            def test_func(self, X, y, estimator, mode, **kwargs):
                return None, kwargs["best_epoch"]

        test_obj = test_class()
        recording = EarlyStoppingMode.RECORDING
        not_recording = EarlyStoppingMode.NOT_RECORDING
        # epochs are recorded as self.earlyStopping.mode is set to RECORDING
        _ = test_obj.test_func(None, None, None, best_epoch=29)
        # epochs are not recorded as mode is set to NOT_RECORDING in the decorator
        _ = test_obj.test_func(None, None, None, not_recording, best_epoch=49)
        self.assertEqual(test_obj.earlyStopping.mode, not_recording)
        # epochs are recorded as mode is now set to RECORDING in the decorator
        _ = test_obj.test_func(None, None, None, recording, best_epoch=39)
        self.assertEqual(test_obj.earlyStopping.mode, recording)

        # Check if the best epochs are recorded with mode RECORDING using the decorator
        self.assertEqual(test_obj.earlyStopping.optimalEpochs, 35)

        # Check if the best epochs are not recorded with other modes using the decorator
        test_obj.earlyStopping.mode = EarlyStoppingMode.FIXED
        _ = test_obj.test_func(None, None, None, best_epoch=49)
        self.assertEqual(test_obj.earlyStopping.optimalEpochs, 35)
        test_obj.earlyStopping.mode = EarlyStoppingMode.OPTIMAL
        _ = test_obj.test_func(None, None, None, best_epoch=59)
        self.assertEqual(test_obj.earlyStopping.optimalEpochs, 35)
        test_obj.earlyStopping.mode = EarlyStoppingMode.NOT_RECORDING
        _ = test_obj.test_func(None, None, None, best_epoch=69)
        self.assertEqual(test_obj.earlyStopping.optimalEpochs, 35)

        # check decorator raises error when early stopping is not supported
        test_obj = test_class(support=False)
        with self.assertRaises(AssertionError):
            _ = test_obj.test_func(None, None, None, best_epoch=40)
