import os
from importlib import import_module, util
from typing import Type
from unittest import TestCase

import pytest
import torch
from parameterized import parameterized
from sklearn.impute import SimpleImputer

from ....data.data import QSPRDataset
from ....data.utils.datasplitters import RandomSplit
from ....data.utils.descriptorcalculator import MoleculeDescriptorsCalculator
from ....data.utils.descriptorsets import SmilesDesc
from ....extra.gpu.models.chemprop import ChempropModel
from ....extra.gpu.models.dnn import DNNModel
from ....extra.gpu.models.neural_network import STFullyConnected
from ....models.tasks import ModelTasks, TargetTasks
from ....models.tests import ModelDataSetsMixIn, ModelTestMixIn, TestMonitors
from ....models.monitors import BaseMonitor

GPUS = list(range(torch.cuda.device_count()))


class NeuralNet(ModelDataSetsMixIn, ModelTestMixIn, TestCase):
    """This class holds the tests for the DNNModel class."""
    @property
    def gridFile(self):
        """Return the path to the grid file with test
        search spaces for hyperparameter optimization.
        """
        return f"{os.path.dirname(__file__)}/test_files/search_space_test.json"

    @staticmethod
    def getModel(
        base_dir: str,
        name: str,
        alg: Type | None = None,
        dataset: QSPRDataset = None,
        parameters: dict | None = None,
        random_state: int | None = None,
    ):
        """Initialize model with data set.

        Args:
            base_dir: Base directory for model.
            name: Name of the model.
            alg: Algorithm to use.
            dataset: Data set to use.
            parameters: Parameters to use.
            random_state: Random seed to use for random operations.
        """
        return DNNModel(
            base_dir=base_dir,
            alg=alg,
            data=dataset,
            name=name,
            parameters=parameters,
            gpus=GPUS,
            patience=3,
            tol=0.02,
            random_state=random_state,
        )

    @parameterized.expand(
        [
            (f"{alg_name}_{task}", task, alg_name, alg, th, [None])
            for alg, alg_name, task, th in (
                (STFullyConnected, "STFullyConnected", TargetTasks.REGRESSION, None),
                (STFullyConnected, "STFullyConnected", TargetTasks.SINGLECLASS, [6.5]),
                (
                    STFullyConnected,
                    "STFullyConnected",
                    TargetTasks.MULTICLASS,
                    [0, 1, 10, 1100],
                ),
            )
        ] + [
            (f"{alg_name}_{task}", task, alg_name, alg, th, random_state)
            for alg, alg_name, task, th in
            ((STFullyConnected, "STFullyConnected", TargetTasks.REGRESSION, None), )
            for random_state in ([None], [1, 42], [42, 42])
        ]
    )
    def testSingleTaskModel(
        self,
        _,
        task: TargetTasks,
        alg_name: str,
        alg: Type,
        th: float,
        random_state: int | None,
    ):
        """Test the DNNModel model in one configuration.

        Args:
            task: Task to test.
            alg_name: Name of the algorithm.
            alg: Algorithm to use.
            th: Threshold to use for classification models.
            random_state: Seed to be used for random operations.
        """
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            name=f"{alg_name}_{task}",
            target_props=[{
                "name": "CL",
                "task": task,
                "th": th
            }],
            preparation_settings=self.getDefaultPrep(),
        )

        # initialize model for training from class
        alg_name = f"{alg_name}_{task}_th={th}"
        model = self.getModel(
            base_dir=self.generatedModelsPath,
            name=f"{alg_name}",
            alg=alg,
            dataset=dataset,
            random_state=random_state[0],
        )
        self.fitTest(model)
        predictor = DNNModel(
            name=alg_name, base_dir=model.baseDir, random_state=random_state[0]
        )
        pred_use_probas, pred_not_use_probas = self.predictorTest(predictor)
        if random_state[0] is not None:
            model.cleanFiles()
            model = self.getModel(
                base_dir=self.generatedModelsPath,
                name=f"{alg_name}",
                alg=alg,
                dataset=dataset,
                random_state=random_state[1],
            )
            self.fitTest(model)
            predictor = DNNModel(
                name=alg_name, base_dir=model.baseDir, random_state=random_state[1]
            )
            self.predictorTest(
                predictor,
                expect_equal_result=random_state[0] == random_state[1],
                expected_pred_use_probas=pred_use_probas,
                expected_pred_not_use_probas=pred_not_use_probas,
            )


class ChemProp(ModelDataSetsMixIn, ModelTestMixIn, TestCase):
    """This class holds the tests for the DNNModel class."""
    @property
    def gridFile(self):
        """Return the path to the grid file with test
        search spaces for hyperparameter optimization.
        """
        return f"{os.path.dirname(__file__)}/test_files/search_space_test.json"

    @staticmethod
    def getModel(
        base_dir: str,
        name: str,
        dataset: QSPRDataset = None,
        parameters: dict | None = None,
    ):
        """Initialize model with data set.

        Args:
            base_dir: Base directory for model.
            name: Name of the model.
            alg: Algorithm to use.
            dataset: Data set to use.
            parameters: Parameters to use.
        """
        if parameters is None:
            parameters = {}

        parameters["gpu"] = GPUS[0] if len(GPUS) > 0 else None
        parameters["epochs"] = 2
        return ChempropModel(
            base_dir=base_dir, data=dataset, name=name, parameters=parameters
        )

    @parameterized.expand(
        [
            (f"{alg_name}_{task}", task, alg_name, th) for alg_name, task, th in (
                ("MoleculeModel", TargetTasks.REGRESSION, None),
                ("MoleculeModel", TargetTasks.SINGLECLASS, [6.5]),
                ("MoleculeModel", TargetTasks.MULTICLASS, [0, 1, 10, 1100]),
            )
        ]
    )
    def testSingleTaskModel(self, _, task: TargetTasks, alg_name: str, th: float):
        """Test the DNNModel model in one configuration.

        Args:
            task: Task to test.
            alg_name: Name of the algorithm.
            alg: Algorithm to use.
            th: Threshold to use for classification models.
        """
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            name=f"{alg_name}_{task}",
            target_props=[{
                "name": "CL",
                "task": task,
                "th": th
            }],
            preparation_settings=None,
        )
        dataset.prepareDataset(
            feature_calculators=[
                MoleculeDescriptorsCalculator(desc_sets=[SmilesDesc()])
            ],
            split=RandomSplit(test_fraction=0.1),
        )
        # initialize model for training from class
        alg_name = f"{alg_name}_{task}_th={th}"
        model = self.getModel(
            base_dir=self.generatedModelsPath, name=f"{alg_name}", dataset=dataset
        )
        self.fitTest(model)
        predictor = ChempropModel(name=alg_name, base_dir=model.baseDir)
        self.predictorTest(predictor)

    @parameterized.expand(
        [
            (f"{alg_name}_{task}", task, alg_name) for alg_name, task in (
                ("MoleculeModel", ModelTasks.MULTITASK_REGRESSION),
                ("MoleculeModel", ModelTasks.MULTITASK_SINGLECLASS),
            )
        ]
    )
    def testMultiTaskmodel(self, _, task: TargetTasks, alg_name: str):
        """Test the DNNModel model in one configuration.

        Args:
            task: Task to test.
            alg_name: Name of the algorithm.
            alg: Algorithm to use.
            th: Threshold to use for classification models.
        """
        if task == ModelTasks.MULTITASK_REGRESSION:
            target_props = [
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
            ]
        else:
            target_props = [
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
            ]
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            name=f"{alg_name}_{task}",
            target_props=target_props,
            target_imputer=SimpleImputer(strategy="mean"),
            preparation_settings=None,
        )
        dataset.prepareDataset(
            feature_calculators=[
                MoleculeDescriptorsCalculator(desc_sets=[SmilesDesc()])
            ],
            split=RandomSplit(test_fraction=0.1),
        )
        # initialize model for training from class
        alg_name = f"{alg_name}_{task}"
        model = self.getModel(
            base_dir=self.generatedModelsPath, name=f"{alg_name}", dataset=dataset
        )
        self.fitTest(model)
        predictor = ChempropModel(name=alg_name, base_dir=model.baseDir)
        self.predictorTest(predictor)


@pytest.mark.skipif((spec := util.find_spec("cupy")) is None, reason="requires cupy")
class TestPyBoostModel(ModelDataSetsMixIn, ModelTestMixIn, TestCase):
    """This class holds the tests for the PyBoostModel class."""
    @staticmethod
    def getModel(
        base_dir: str,
        name: str,
        dataset: QSPRDataset = None,
        parameters: dict | None = None,
    ):
        """Create a PyBoostModel model.

        Args:
            name (str): the name of the model
            dataset (QSPRDataset, optional): the dataset to use. Defaults to None.
            parameters (dict, optional): the parameters to use. Defaults to None.

        Returns:
            PyBoostModel the model
        """
        if parameters is None:
            parameters = {}
        parameters["ntrees"] = 10

        return import_module("..pyboost", __name__).PyBoostModel(
            base_dir=base_dir,
            data=dataset,
            name=name,
            parameters=parameters,
        )

    @parameterized.expand(
        [
            ("PyBoost", TargetTasks.REGRESSION, "PyBoost", params) for params in [
                {
                    "loss": "mse",
                    "metric": "r2_score",
                },
                # {
                #     "loss": import_module("..custom_loss", __name__).MSEwithNaNLoss(),
                #     "metric": "r2_score"
                # },
                # {
                #     "loss": "mse",
                #     "metric": import_module("..custom_metrics", __name__).NaNR2Score()
                # },
                # {
                #     "loss": "mse",
                #     "metric":import_module("..custom_metrics",__name__).NaNRMSEScore()
                # },
            ]
        ]
    )
    def testRegressionPyBoostFit(self, _, task, model_name, parameters):
        """Test model training for regression models."""
        parameters["verbose"] = -1
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
            base_dir=self.generatedModelsPath,
            name=f"{model_name}_{task}",
            dataset=dataset,
            parameters=parameters,
        )
        self.fitTest(model)
        predictor = import_module("..pyboost", __name__).PyBoostModel(
            name=f"{model_name}_{task}", base_dir=model.baseDir
        )
        self.predictorTest(predictor)

    @parameterized.expand(
        [
            (f"{'PyBoost'}_{task}", task, th, "PyBoost", params)
            for params, task, th in (
                ({
                    "loss": "bce",
                    "metric": "auc"
                }, TargetTasks.SINGLECLASS, [6.5]),
                ({
                    "loss": "crossentropy"
                }, TargetTasks.MULTICLASS, [0, 1, 10, 1100]),
            )
        ]
    )
    def testClassificationPyBoostFit(self, _, task, th, model_name, parameters):
        """Test model training for classification models."""
        parameters["verbose"] = -1
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
            base_dir=self.generatedModelsPath,
            name=f"{model_name}_{task}",
            dataset=dataset,
            parameters=parameters,
        )
        self.fitTest(model)
        predictor = import_module("..pyboost", __name__).PyBoostModel(
            name=f"{model_name}_{task}", base_dir=model.baseDir
        )
        self.predictorTest(predictor)

    @parameterized.expand(
        [
            ("PyBoost", "PyBoost", params) for params in [
                {
                    "loss": "mse",
                    "metric": "r2_score"
                },
                # {
                #     "loss": import_module("..custom_loss", __name__).MSEwithNaNLoss(),
                #     "metric": "r2_score"
                # },
                # {
                #     "loss": "mse",
                #     "metric": import_module("..custom_metrics",__name__).NaNR2Score()
                # },
            ]
        ]
    )
    def testRegressionMultiTaskFit(self, _, model_name, parameters):
        """Test model training for multitask regression models."""
        parameters["verbose"] = -1
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
            base_dir=self.generatedModelsPath,
            name=f"{model_name}_multitask_regression",
            dataset=dataset,
            parameters=parameters,
        )
        self.fitTest(model)
        predictor = import_module("..pyboost", __name__).PyBoostModel(
            name=f"{model_name}_multitask_regression", base_dir=model.baseDir
        )
        self.predictorTest(predictor)

    # FIXME: This test fails because the PyBoost default auc does not handle
    # mutlitask data and the custom NaN AUC metric is not JSON serializable.

    # @parameterized.expand(
    #     [
    #         ("PyBoost", "PyBoost", params) for params in [
    #             # {
    #             #     "loss": "bce",
    #             #     "metric": "auc"
    #             # },
    #             # {
    #             #     "loss": import_module("..custom_loss", __name__).BCEWithNaNLoss(),
    #             #     "metric": "auc"
    #             # },
    #             {
    #                 "loss": "bce",
    #                 "metric": NaNAucMetric()
    #             },
    #         ]
    #     ]
    # )
    # def testClassificationMultiTaskFit(self, _, model_name, parameters):
    #     """Test model training for multitask classification models."""
    #     parameters["verbose"] = -1

    #     # initialize dataset
    #     dataset = self.createLargeTestDataSet(
    #         target_props=[
    #             {
    #                 "name": "fu",
    #                 "task": TargetTasks.SINGLECLASS,
    #                 "th": [0.3]
    #             },
    #             {
    #                 "name": "CL",
    #                 "task": TargetTasks.SINGLECLASS,
    #                 "th": [6.5]
    #             },
    #         ],
    #         target_imputer=SimpleImputer(strategy="mean"),
    #         preparation_settings=self.getDefaultPrep(),
    #     )
    #     # test classifier
    #     # initialize model for training from class
    #     model = self.getModel(
    #         base_dir=self.generatedModelsPath,
    #         name=f"{model_name}_multitask_classification",
    #         dataset=dataset,
    #         parameters=parameters,
    #     )
    #     self.fitTest(model)
    #     predictor = import_module("..pyboost", __name__).PyBoostModel(
    #         name=f"{model_name}_multitask_classification", base_dir=model.baseDir
    #     )
    #     self.predictorTest(predictor)


class TestNNMonitoring(TestMonitors):
    """This class holds the tests for the monitoring classes."""
    @property
    def gridFile(self):
        """Return the path to the grid file with test
        search spaces for hyperparameter optimization.
        """
        return f"{os.path.dirname(__file__)}/test_files/search_space_test.json"

    def test_BaseMonitor(self):
        model = DNNModel(
            base_dir=self.generatedModelsPath,
            data=self.createLargeTestDataSet(
                preparation_settings=self.getDefaultPrep()
            ),
            name="STFullyConnected",
            gpus=GPUS,
            patience=3,
            tol=0.02,
            random_state=42,
        )

        hyperparam_monitor = BaseMonitor()
        crossval_monitor = BaseMonitor()
        test_monitor = BaseMonitor()
        fit_monitor = BaseMonitor()
        (
            hyperparam_monitor,
            crossval_monitor,
            test_monitor,
            fit_monitor,
        ) = self.trainModelWithMonitoring(
            model, hyperparam_monitor, crossval_monitor, test_monitor, fit_monitor
        )
        self.baseMonitorTest(hyperparam_monitor, "hyperparam", True)
        self.baseMonitorTest(crossval_monitor, "crossval", True)
        self.baseMonitorTest(test_monitor, "test", True)
        self.baseMonitorTest(fit_monitor, "fit", True)
