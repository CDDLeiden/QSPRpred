import os
from importlib import import_module, util
from unittest import TestCase

import pytest
import torch
from parameterized import parameterized
from sklearn.impute import SimpleImputer

from ...data.data import QSPRDataset
from ...models.tasks import TargetTasks
from ...models.tests import ModelDataSetsMixIn, ModelTestMixIn

GPUS = list(range(torch.cuda.device_count()))


@pytest.mark.skipif((spec := util.find_spec("cupy")) is None, reason="requires cupy")
class TestPyBoostModel(ModelDataSetsMixIn, ModelTestMixIn, TestCase):
    """This class holds the tests for the PyBoostModel class."""
    @staticmethod
    def getModel(
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
        return import_module("..models", __name__).PyBoostModel(
            base_dir=f"{os.path.dirname(__file__)}/test_files/qspr/models",
            data=dataset,
            name=name,
            parameters=parameters,
        )

    @parameterized.expand(
        [
            ("PyBoost", TargetTasks.REGRESSION, "PyBoost", params) for params in [
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
        dataset = self.create_large_dataset(
            target_props=[{
                "name": "CL",
                "task": task
            }],
            preparation_settings=self.get_default_prep(),
        )
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_{task}",
            dataset=dataset,
            parameters=parameters,
        )
        self.fitTest(model)
        predictor = import_module("..models", __name__).PyBoostModel(
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
        dataset = self.create_large_dataset(
            target_props=[{
                "name": "CL",
                "task": task,
                "th": th
            }],
            preparation_settings=self.get_default_prep(),
        )
        # test classifier
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_{task}",
            dataset=dataset,
            parameters=parameters,
        )
        self.fitTest(model)
        predictor = import_module("..models", __name__).PyBoostModel(
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
        dataset = self.create_large_dataset(
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
            preparation_settings=self.get_default_prep(),
        )
        # test classifier
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_multitask_regression",
            dataset=dataset,
            parameters=parameters
        )
        self.fitTest(model)
        predictor = import_module("..models", __name__).PyBoostModel(
            name=f"{model_name}_multitask_regression", base_dir=model.baseDir
        )
        self.predictorTest(predictor)

    @parameterized.expand(
        [
            ("PyBoost", "PyBoost", params) for params in [
                {
                    "loss": "bce",
                    "metric": "auc"
                },
                # {
                #     "loss": import_module("..custom_loss", __name__).BCEWithNaNLoss(),
                #     "metric": "auc"
                # },
                # {
                #     "loss": "bce",
                #     "metric":import_module("..custom_metrics",__name__).NaNAucMetric()
                # },
            ]
        ]
    )
    def testClassificationMultiTaskFit(self, _, model_name, parameters):
        """Test model training for multitask classification models."""
        parameters["verbose"] = -1

        # initialize dataset
        dataset = self.create_large_dataset(
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
            preparation_settings=self.get_default_prep(),
        )
        # test classifier
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_multitask_classification",
            dataset=dataset,
            parameters=parameters,
        )
        self.fitTest(model)
        predictor = import_module("..models", __name__).PyBoostModel(
            name=f"{model_name}_multitask_classification", base_dir=model.baseDir
        )
        self.predictorTest(predictor)
