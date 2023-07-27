"""
Test module for testing extra models.

"""

import os
from typing import Type
from unittest import TestCase

from parameterized import parameterized
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from xgboost import XGBClassifier, XGBRegressor

from ...data.data import TargetProperty
from ...models.tasks import TargetTasks
from ...models.tests import N_CPUS, ModelDataSetsMixIn, ModelTestMixIn
from ..data.data import PCMDataSet, QSPRDataset
from ..data.tests import DataSetsMixInExtras
from ..data.utils.descriptor_utils.msa_calculator import ClustalMSA
from ..data.utils.descriptorcalculator import ProteinDescriptorCalculator
from ..data.utils.descriptorsets import ProDec
from ..models.custom_loss import BCEWithNaNLoss, MSEwithNaNLoss
from ..models.custom_metrics import NaNAucMetric, NaNR2Score, NaNRMSEScore
from ..models.models import PyBoostModel
from ..models.pcm import QSPRsklearnPCM


class ModelDataSetsMixInExtras(ModelDataSetsMixIn, DataSetsMixInExtras):
    """This class holds the tests for testing models in extras."""

    qsprModelsPath = f"{os.path.dirname(__file__)}/test_files/qspr/models"


class TestPCM(ModelDataSetsMixInExtras, ModelTestMixIn, TestCase):
    @staticmethod
    def getModel(
        name: str,
        alg: Type | None = None,
        dataset: PCMDataSet | None = None,
        parameters: dict | None = None
    ):
        """Initialize dataset and model.

        Args:
            name (str): Name of the model.
            alg (Type | None): Algorithm class.
            dataset (PCMDataSet | None): Dataset to use.
            parameters (dict | None): Parameters to use.

        Returns:
            QSPRsklearnPCM: Initialized model.
        """
        return QSPRsklearnPCM(
            base_dir=f"{os.path.dirname(__file__)}/test_files/",
            alg=alg,
            data=dataset,
            name=name,
            parameters=parameters,
        )

    @parameterized.expand(
        [
            (
                alg_name,
                [{
                    "name": "pchembl_value_Median",
                    "task": TargetTasks.REGRESSION
                }],
                alg_name,
                alg,
            ) for alg, alg_name in (
                (PLSRegression, "PLSR"),
                (SVR, "SVR"),
                (XGBRegressor, "XGBR"),
            )
        ] + [
            (
                alg_name,
                [
                    {
                        "name": "pchembl_value_Median",
                        "task": TargetTasks.SINGLECLASS,
                        "th": [6.5],
                    }
                ],
                alg_name,
                alg,
            ) for alg, alg_name in (
                (RandomForestClassifier, "RFC"),
                (XGBClassifier, "XGBC"),
            )
        ]
    )
    def testRegressionBasicFitPCM(
        self, _, props: list[TargetProperty | dict], model_name: str, model_class: Type
    ):
        """Test model training for regression models.

        Args:
            _: Name of the test.
            props (list[TargetProperty | dict]): List of target properties.
            model_name (str): Name of the model.
            model_class (Type): Class of the model.

        """
        if model_name not in ["SVR", "PLSR"]:
            parameters = {"n_jobs": N_CPUS}
        else:
            parameters = None
        # initialize dataset
        prep = self.get_default_prep()
        prep["feature_calculators"] = prep["feature_calculators"] + [
            ProteinDescriptorCalculator(
                desc_sets=[ProDec(sets=["Sneath"])],
                msa_provider=ClustalMSA(self.qsprdatapath),
            )
        ]
        dataset = self.createPCMDataSet(
            name=f"{model_name}_{props[0]['task']}_pcm",
            target_props=props,
            preparation_settings=prep,
        )
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_{props[0]['task']}",
            alg=model_class,
            dataset=dataset,
            parameters=parameters,
        )
        self.fitTest(model)
        predictor = QSPRsklearnPCM(
            name=f"{model_name}_{props[0]['task']}", base_dir=model.baseDir
        )
        self.predictorTest(predictor, protein_id=dataset.getDF()["accession"].iloc[0])


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
        return PyBoostModel(
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
                {
                    "loss": MSEwithNaNLoss(),
                    "metric": "r2_score"
                },
                {
                    "loss": "mse",
                    "metric": NaNR2Score()
                },
                {
                    "loss": "mse",
                    "metric": NaNRMSEScore()
                },
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
        predictor = PyBoostModel(name=f"{model_name}_{task}", base_dir=model.baseDir)
        self.predictorTest(predictor)

    @parameterized.expand(
        [
            (f"{'PyBoost'}_{task}", task, th, "PyBoost", params) for params in [
                {
                    "loss": "bce",
                    "metric": "auc"
                },
                {
                    "loss": BCEWithNaNLoss(),
                    "metric": "auc"
                },
                {
                    "loss": "bce",
                    "metric": NaNAucMetric()
                },
            ] for task, th in (
                (TargetTasks.SINGLECLASS, [6.5]),
                (TargetTasks.MULTICLASS, [0, 1, 10, 1100]),
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
        predictor = PyBoostModel(name=f"{model_name}_{task}", base_dir=model.baseDir)
        self.predictorTest(predictor)

    @parameterized.expand(
        [
            ("PyBoost", "PyBoost", params) for params in [
                {
                    "loss": "mse",
                    "metric": "r2_score"
                },
                "PBBaseReg",
                {
                    "loss": MSEwithNaNLoss(),
                    "metric": "r2_score"
                },
                "PBCustomLossReg",
                {
                    "loss": "mse",
                    "metric": NaNR2Score()
                },
                "PBCustomR2",
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
        predictor = PyBoostModel(
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
                "PBBaseCls",
                {
                    "loss": BCEWithNaNLoss(),
                    "metric": "auc"
                },
                "PBCustomLossCls",
                {
                    "loss": "bce",
                    "metric": NaNAucMetric()
                },
                "PBCustomAUC",
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
        predictor = PyBoostModel(
            name=f"{model_name}_multitask_classification", base_dir=model.baseDir
        )
        self.predictorTest(predictor)
