"""This module holds the tests for functions regarding QSPR modelling."""

import os
from typing import Type
from unittest import TestCase

import numpy as np
from mlchemad.applicability_domains import KNNApplicabilityDomain
from parameterized import parameterized
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    log_loss,
    make_scorer,
    mean_squared_error,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from ..data.processing.applicability_domain import MLChemADWrapper
from ..models.early_stopping import EarlyStopping, EarlyStoppingMode, early_stopping
from ..models.metrics import SklearnMetrics
from ..models.monitors import BaseMonitor, FileMonitor, ListMonitor
from ..models.scikit_learn import SklearnModel
from ..tasks import TargetTasks
from ..utils.testing.base import QSPRTestCase
from ..utils.testing.check_mixins import ModelCheckMixIn, MonitorsCheckMixIn
from ..utils.testing.path_mixins import ModelDataSetsPathMixIn
from .assessment.classification import create_metrics_summary
from .assessment.regression import create_correlation_summary


class SklearnBaseModelTestCase(ModelDataSetsPathMixIn, ModelCheckMixIn, QSPRTestCase):
    """This class holds the tests for the SklearnModel class."""
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def getModel(
        self,
        name: str,
        alg: Type | None = None,
        parameters: dict | None = None,
        random_state: int | None = None,
    ):
        """Create a SklearnModel model.

        Args:
            name (str): the name of the model
            alg (Type, optional): the algorithm to use. Defaults to None.
            dataset (QSPRDataset, optional): the dataset to use. Defaults to None.
            parameters (dict, optional): the parameters to use. Defaults to None.
            random_state(int, optional):
                Random state to use for shuffling and other random operations. Defaults
                to None.

        Returns:
            SklearnModel: the model
        """
        return SklearnModel(
            base_dir=self.generatedModelsPath,
            alg=alg,
            name=name,
            parameters=parameters,
            random_state=random_state,
        )


class TestSklearnRegression(SklearnBaseModelTestCase):
    """Test the SklearnModel class for regression models."""
    @parameterized.expand(
        [
            (alg_name, TargetTasks.REGRESSION, alg_name, alg, random_state)
            for alg, alg_name in (
                (RandomForestRegressor, "RFR"),
                (XGBRegressor, "XGBR"),
            ) for random_state in ([None], [1, 42], [42, 42])
        ] + [
            (alg_name, TargetTasks.REGRESSION, alg_name, alg, [None])
            for alg, alg_name in (
                (PLSRegression, "PLSR"),
                (SVR, "SVR"),
                (KNeighborsRegressor, "KNNR"),
            )
        ]
    )
    def testRegressionBasicFit(self, _, task, model_name, model_class, random_state):
        """Test model training for regression models."""
        if model_name not in ["SVR", "PLSR"]:
            parameters = {"n_jobs": self.nCPU}
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
            parameters=parameters,
            random_state=random_state[0],
        )
        self.fitTest(model, dataset)

        # load in model from file
        predictor = SklearnModel(name=f"{model_name}_{task}", base_dir=model.baseDir)

        # make predictions with the trained model and check if the results are (not)
        # equal if the random state is the (not) same and at the same time
        # check if the output is the same before and after saving and loading
        if random_state[0] is not None:
            comparison_model = self.getModel(
                name=f"{model_name}_{task}",
                alg=model_class,
                parameters=parameters,
                random_state=random_state[1],
            )
            self.fitTest(comparison_model, dataset)
            self.predictorTest(
                predictor, # model loaded from file
                dataset=dataset,
                comparison_model=comparison_model, # comparison model not saved/loaded
                expect_equal_result=random_state[0] == random_state[1],
            )
        else:
            self.predictorTest(predictor, dataset=dataset)

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
            parameters=parameters,
        )
        self.fitTest(model, dataset)
        expected_summary = create_correlation_summary(model)

        # Generate summary again, check that the result is identical
        model = self.getModel(
            name=f"{model_name}_{task}",
            alg=model_class,
            parameters=parameters,
        )
        self.fitTest(model, dataset)
        summary = create_correlation_summary(model)

        self.assertListEqual(summary["ModelName"], expected_summary["ModelName"])
        self.assertListEqual(summary["R2"], expected_summary["R2"])
        self.assertListEqual(summary["RMSE"], expected_summary["RMSE"])
        self.assertListEqual(summary["Set"], expected_summary["Set"])


class TestSklearnRegressionMultiTask(SklearnBaseModelTestCase):
    """Test the SklearnModel class for multi-task regression models."""
    @parameterized.expand(
        [
            (alg_name, alg_name, alg, random_state)
            for alg, alg_name in ((RandomForestRegressor, "RFR"), )
            for random_state in ([None], [1, 42], [42, 42])
        ] + [
            (alg_name, alg_name, alg, [None])
            for alg, alg_name in ((KNeighborsRegressor, "KNNR"), )
        ]
    )
    def testRegressionMultiTaskFit(self, _, model_name, model_class, random_state):
        """Test model training for multitask regression models."""
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            target_props=[
                {
                    "name": "fu",
                    "task": TargetTasks.REGRESSION,
                    "imputer": SimpleImputer(strategy="mean"),
                },
                {
                    "name": "CL",
                    "task": TargetTasks.REGRESSION,
                    "imputer": SimpleImputer(strategy="mean"),
                },
            ],
            preparation_settings=self.getDefaultPrep(),
        )
        # test classifier
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_multitask_regression",
            alg=model_class,
            random_state=random_state[0],
        )
        self.fitTest(model, dataset)

        # load in model from file
        predictor = SklearnModel(
            name=f"{model_name}_multitask_regression", base_dir=model.baseDir
        )

        # make predictions with the trained model and check if the results are (not)
        # equal if the random state is the (not) same and at the same time
        # check if the output is the same before and after saving and loading
        if random_state[0] is not None and model_name in ["RFR"]:
            comparison_model = self.getModel(
                name=f"{model_name}_multitask_regression",
                alg=model_class,
                random_state=random_state[1],
            )
            self.fitTest(comparison_model, dataset)
            self.predictorTest(
                predictor, # model loaded from file
                dataset=dataset,
                comparison_model=comparison_model, # comparison model not saved/loaded
                expect_equal_result=random_state[0] == random_state[1],
            )
        else:
            self.predictorTest(predictor, dataset=dataset)


class TestSklearnSerialization(SklearnBaseModelTestCase):
    def testJSON(self):
        dataset = self.createLargeTestDataSet(
            target_props=[{
                "name": "CL",
                "task": TargetTasks.SINGLECLASS,
                "th": [6.5]
            }],
            preparation_settings=self.getDefaultPrep(),
        )
        model = self.getModel(
            name="TestSerialization",
            alg=RandomForestClassifier,
            parameters={
                "n_jobs": self.nCPU,
                "n_estimators": 10
            },
            random_state=42,
        )
        model.save()
        content = model.toJSON()
        model2 = SklearnModel.fromJSON(content)
        model2.baseDir = model.baseDir
        model3 = SklearnModel.fromFile(model.metaFile)
        model4 = SklearnModel(name=model.name, base_dir=model.baseDir)
        self.assertEqual(model.metaFile, model2.metaFile)
        self.assertEqual(model.metaFile, model3.metaFile)
        self.assertEqual(model.metaFile, model4.metaFile)
        self.assertEqual(model.toJSON(), model2.toJSON())
        self.assertEqual(model.toJSON(), model3.toJSON())
        self.assertEqual(model.toJSON(), model4.toJSON())


class TestSklearnClassification(SklearnBaseModelTestCase):
    """Test the SklearnModel class for classification models."""
    @parameterized.expand(
        [
            (f"{alg_name}_{task}", task, th, alg_name, alg, random_state)
            for alg, alg_name in (
                (RandomForestClassifier, "RFC"),
                (XGBClassifier, "XGBC"),
            ) for task, th in (
                (TargetTasks.SINGLECLASS, [6.5]),
                (TargetTasks.MULTICLASS, [0, 2, 10, 1100]),
            ) for random_state in ([None], [1, 42], [42, 42])
        ] + [
            (f"{alg_name}_{task}", task, th, alg_name, alg, [None])
            for alg, alg_name in (
                (SVC, "SVC"),
                (KNeighborsClassifier, "KNNC"),
                (GaussianNB, "NB"),
            ) for task, th in (
                (TargetTasks.SINGLECLASS, [6.5]),
                (TargetTasks.MULTICLASS, [0, 2, 10, 1100]),
            )
        ]
    )
    def testClassificationBasicFit(
        self, _, task, th, model_name, model_class, random_state
    ):
        """Test model training for classification models."""
        if model_name not in ["NB", "SVC"]:
            parameters = {"n_jobs": self.nCPU}
        else:
            parameters = None
        # special case for SVC
        if model_name == "SVC":
            if parameters is not None:
                parameters.update({"probability": True})
            else:
                parameters = {"probability": True}
        # special case for XGB, set subsample to 0.6 to introduce randomness
        if model_name == "XGBC":
            parameters = {"subsample": 0.3}
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
        model = self.getModel(
            name=f"{model_name}_{task}",
            alg=model_class,
            parameters=parameters,
            random_state=random_state[0],
        )
        self.fitTest(model, dataset)

        # load in model from file
        predictor = SklearnModel(name=f"{model_name}_{task}", base_dir=model.baseDir)

        # make predictions with the trained model and check if the results are (not)
        # equal if the random state is the (not) same and at the same time
        # check if the output is the same before and after saving and loading
        if random_state[0] is not None:
            comparison_model = self.getModel(
                name=f"{model_name}_{task}",
                alg=model_class,
                parameters=parameters,
                random_state=random_state[1],
            )
            self.fitTest(comparison_model, dataset)
            self.predictorTest(
                predictor, # model loaded from file
                dataset=dataset,
                comparison_model=comparison_model, # comparison model not saved/loaded
                expect_equal_result=random_state[0] == random_state[1],
            )
        else:
            self.predictorTest(predictor, dataset=dataset)

    def testRandomForestClassifierFitWithSeed(self):
        parameters = {
            "n_jobs": self.nCPU,
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
            parameters=parameters,
        )
        self.fitTest(model, dataset)
        expected_summary = create_metrics_summary(model)

        # Generate summary again, check that the result is identical
        model = self.getModel(
            name=f"RFC_{TargetTasks.SINGLECLASS}",
            alg=RandomForestClassifier,
            parameters=parameters,
        )
        self.fitTest(model, dataset)
        summary = create_metrics_summary(model)

        self.assertListEqual(summary["Metric"], expected_summary["Metric"])
        self.assertListEqual(summary["Model"], expected_summary["Model"])
        self.assertListEqual(summary["TestSet"], expected_summary["TestSet"])
        self.assertListEqual(summary["Value"], expected_summary["Value"])


class TestSklearnClassificationMultiTask(SklearnBaseModelTestCase):
    """Test the SklearnModel class for multi-task classification models."""
    @parameterized.expand(
        [
            (alg_name, alg_name, alg, random_state)
            for alg, alg_name in ((RandomForestClassifier, "RFC"), )
            for random_state in ([None], [1, 42], [42, 42])
        ] + [
            (alg_name, alg_name, alg, [None])
            for alg, alg_name in ((KNeighborsClassifier, "KNNC"), )
        ]
    )
    def testClassificationMultiTaskFit(self, _, model_name, model_class, random_state):
        """Test model training for multitask classification models."""
        if model_name not in ["NB", "SVC"]:
            parameters = {"n_jobs": self.nCPU}
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
                    "th": [0.3],
                    "imputer": SimpleImputer(strategy="mean"),
                },
                {
                    "name": "CL",
                    "task": TargetTasks.SINGLECLASS,
                    "th": [6.5],
                    "imputer": SimpleImputer(strategy="mean"),
                },
            ],
            preparation_settings=self.getDefaultPrep(),
        )
        # test classifier
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_multitask_classification",
            alg=model_class,
            parameters=parameters,
            random_state=random_state[0],
        )
        self.fitTest(model, dataset)

        # load in model from file
        predictor = SklearnModel(
            name=f"{model_name}_multitask_classification", base_dir=model.baseDir
        )

        # make predictions with the trained model and check if the results are (not)
        # equal if the random state is the (not) same and at the same time
        # check if the output is the same before and after saving and loading
        if random_state[0] is not None:
            comparison_model = self.getModel(
                name=f"{model_name}_multitask_classification",
                alg=model_class,
                parameters=parameters,
                random_state=random_state[1],
            )
            self.fitTest(comparison_model, dataset)
            self.predictorTest(
                predictor, # model loaded from file
                dataset=dataset,
                comparison_model=comparison_model, # comparison model not saved/loaded
                expect_equal_result=random_state[0] == random_state[1],
            )
        else:
            self.predictorTest(predictor, dataset=dataset)


class TestMetrics(TestCase):
    """Test the SklearnMetrics from the metrics module."""
    def test_SklearnMetrics(self):
        """Test the sklearn metrics wrapper."""

        # test regression metrics
        y_true = np.array([1.2, 2.2, 3.2, 4.2, 5.2])
        y_pred = np.array([[2.2], [2.2], [3.2], [4.2], [5.2]])

        ## test explained variance score with scorer from metric
        metric = explained_variance_score
        qsprpred_scorer = SklearnMetrics(make_scorer(metric))
        self.assertEqual(
            qsprpred_scorer(y_true, y_pred),  # 2D np array standard in QSPRpred
            metric(y_true, np.squeeze(y_pred)),  # 1D np array standard in sklearn
        )

        ## test RMSE score with scorer from str (smaller is better)
        qsprpred_scorer = SklearnMetrics("neg_mean_squared_error")
        self.assertEqual(
            qsprpred_scorer(y_true, y_pred),
            -mean_squared_error(y_true, np.squeeze(y_pred)),  # negated
        )

        ## test multitask regression
        y_true = np.array([[1.2, 2.2], [3.2, 4.2], [5.2, 1.2], [2.2, 3.2], [4.2, 5.2]])
        y_pred = np.array([[2.2, 2.2], [3.2, 4.2], [5.2, 1.2], [2.2, 3.2], [4.2, 5.2]])
        qsprpred_scorer = SklearnMetrics("explained_variance")
        self.assertEqual(
            qsprpred_scorer(y_true, y_pred), explained_variance_score(y_true, y_pred)
        )

        # test classification metrics
        ## single class discrete
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([[0], [0], [1], [0], [1]])
        qsprpred_scorer = SklearnMetrics("accuracy")
        self.assertEqual(
            qsprpred_scorer(y_true, y_pred), accuracy_score(y_true, np.squeeze(y_pred))
        )

        ## single class proba
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = [
            np.array([[0.2, 0.8], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]])
        ]  # list of 2D np.arrays
        qsprpred_scorer = SklearnMetrics("neg_log_loss")
        self.assertEqual(
            qsprpred_scorer(y_true, y_pred), -log_loss(y_true, np.squeeze(y_pred[0]))
        )

        ## multi-class with threshold
        y_true = np.array([1, 2, 1, 0, 1])
        y_pred = [
            np.array(
                [
                    [0.9, 0.1, 0.0],
                    [0.1, 0.8, 0.1],
                    [0.0, 0.1, 0.9],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.8, 0.1],
                ]
            )
        ]  # list of 2D np.arrays
        qsprpred_scorer = SklearnMetrics(
            make_scorer(top_k_accuracy_score, needs_threshold=True, k=2)
        )
        self.assertEqual(
            qsprpred_scorer(y_true, y_pred),
            top_k_accuracy_score(y_true, y_pred[0], k=2),
        )

        ## multi-class discrete scorer (_PredictScorer)
        qsprpred_scorer = SklearnMetrics("accuracy")
        self.assertEqual(
            qsprpred_scorer(y_true, y_pred),
            accuracy_score(y_true, np.argmax(y_pred[0], axis=1)),
        )

        ## multi-task single class (same as multi-label in sklearn)
        ### proba
        y_true = np.array([[1, 0], [1, 1], [1, 0], [0, 0], [1, 0]])
        y_pred = [
            np.array([[0.2, 0.8], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]]),
            np.array([[0.2, 0.8], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9], [0.9, 0.1]]),
        ]  # list of 2D np.arrays
        qsprpred_scorer = SklearnMetrics("roc_auc_ovr")
        y_pred_sklearn = np.array([y_pred[0][:, 1], y_pred[1][:, 1]]).T
        self.assertEqual(
            qsprpred_scorer(y_true, y_pred),
            roc_auc_score(y_true, y_pred_sklearn, multi_class="ovr"),
        )

        ### discrete
        y_true = np.array([[1, 0], [1, 1], [1, 0], [0, 0], [1, 0]])
        y_pred = np.array([[0, 0], [1, 1], [0, 0], [0, 0], [1, 0]])
        qsprpred_scorer = SklearnMetrics("accuracy")
        self.assertEqual(
            qsprpred_scorer(y_true, y_pred),
            accuracy_score(y_true, y_pred),
        )


class TestEarlyStopping(ModelDataSetsPathMixIn, TestCase):
    def setUp(self):
        super().setUp()
        self.setUpPaths()

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
            print(earlystopping.optimalEpochs)

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
        earlystopping.toFile(f"{self.inputBasePath}/earlystopping.json")
        self.assertTrue(os.path.exists(f"{self.inputBasePath}/earlystopping.json"))

        # check loading
        earlystopping2 = EarlyStopping.fromFile(
            f"{self.inputBasePath}/earlystopping.json"
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
            def test_func(
                self,
                X,
                y,
                estimator=None,
                mode=EarlyStoppingMode.NOT_RECORDING,
                split=None,
                monitor=None,
                **kwargs,
            ):
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


class TestMonitors(MonitorsCheckMixIn, TestCase):
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def testBaseMonitor(self):
        """Test the base monitor"""
        model = SklearnModel(
            base_dir=self.generatedModelsPath,
            alg=RandomForestRegressor,
            name="RFR",
            random_state=42,
        )
        self.runMonitorTest(
            model,
            self.createLargeTestDataSet(preparation_settings=self.getDefaultPrep()),
            BaseMonitor,
            self.baseMonitorTest,
            False,
        )

    def testFileMonitor(self):
        """Test the file monitor"""
        model = SklearnModel(
            base_dir=self.generatedModelsPath,
            alg=RandomForestRegressor,
            name="RFR",
            random_state=42,
        )
        self.runMonitorTest(
            model,
            self.createLargeTestDataSet(preparation_settings=self.getDefaultPrep()),
            FileMonitor,
            self.fileMonitorTest,
            False,
        )

    def testListMonitor(self):
        """Test the list monitor"""
        model = SklearnModel(
            base_dir=self.generatedModelsPath,
            alg=RandomForestRegressor,
            name="RFR",
            random_state=42,
        )
        self.runMonitorTest(
            model,
            self.createLargeTestDataSet(preparation_settings=self.getDefaultPrep()),
            ListMonitor,
            self.listMonitorTest,
            False,
            [BaseMonitor(), FileMonitor()],
        )


class TestAttachedApplicabilityDomain(ModelDataSetsPathMixIn, QSPRTestCase):
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def testAttachedApplicabilityDomain(self):
        """Test the attached applicability domain class."""

        # initialize test dataset with attached applicability domain
        dataset = self.createLargeTestDataSet(
            target_props=[{
                "name": "CL",
                "task": "REGRESSION"
            }],
            preparation_settings={
                **self.getDefaultPrep(),
                "applicability_domain":
                    KNNApplicabilityDomain(dist="euclidean", alpha=0.9, scaling=None),
            },
        )
        # initialize model for training
        model = SklearnModel(
            base_dir=self.generatedModelsPath,
            alg=RandomForestRegressor,
            name="RFR_with_AD",
            parameters={"n_jobs": self.nCPU},
            random_state=42,
        )

        model.fitDataset(dataset)

        # check if the applicability domain is attached to the model
        self.assertTrue(hasattr(model, "applicabilityDomain"))
        self.assertIsInstance(model.applicabilityDomain, MLChemADWrapper)

        # check if the applicability domain is saved and loaded correctly
        model.save()
        model2 = SklearnModel.fromFile(model.metaFile)
        self.assertTrue(hasattr(model2, "applicabilityDomain"))
        self.assertIsInstance(model2.applicabilityDomain, MLChemADWrapper)

        # make predictions with mlchemad ap on the dataset directly
        comparison_ap = KNNApplicabilityDomain(
            dist="euclidean", alpha=0.9, scaling=None
        )
        features = dataset.getFeatures(
            concat=True, ordered=True, refit_standardizer=False
        )
        comparison_ap.fit(features)
        ap_pred = comparison_ap.contains(features)

        # check if the applicability domain predictions from the dataset are equal to the ones from the model
        _, ap_preds_model = model.predictMols(
            dataset.df["SMILES"], use_applicability_domain=True
        )
        self.assertTrue(np.array_equal(ap_pred.reshape(-1, 1), ap_preds_model))

        # check if the applicability domain predictions arrays are equal after saving and loading
        _, ap_preds_model2 = model2.predictMols(
            dataset.df["SMILES"], use_applicability_domain=True
        )
        self.assertTrue(np.array_equal(ap_pred.reshape(-1, 1), ap_preds_model2))
