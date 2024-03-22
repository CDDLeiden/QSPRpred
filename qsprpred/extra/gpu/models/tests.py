import os
from importlib import import_module, util
from typing import Type
from unittest import TestCase

import chemprop
import pandas as pd
import pytest
import torch
from parameterized import parameterized
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit

from qsprpred.data.descriptors.sets import SmilesDesc
from qsprpred.data.sampling.splits import RandomSplit
from qsprpred.tasks import ModelTasks, TargetTasks
from ....extra.gpu.models.chemprop import ChempropModel
from ....extra.gpu.models.dnn import DNNModel
from ....extra.gpu.models.neural_network import STFullyConnected
from ....models import CrossValAssessor
from ....models.metrics import SklearnMetrics
from ....models.monitors import BaseMonitor, FileMonitor, ListMonitor
from ....utils.testing.check_mixins import ModelCheckMixIn, MonitorsCheckMixIn
from ....utils.testing.path_mixins import ModelDataSetsPathMixIn

GPUS = list(range(torch.cuda.device_count()))


class NeuralNet(ModelDataSetsPathMixIn, ModelCheckMixIn, TestCase):
    """This class holds the tests for the DNNModel class."""

    def setUp(self):
        """Set up the test case."""
        super().setUp()
        self.setUpPaths()

    @property
    def gridFile(self):
        """Return the path to the grid file with test
        search spaces for hyperparameter optimization.
        """
        return f"{os.path.dirname(__file__)}/test_files/search_space_test.json"

    def getModel(
        self,
        name: str,
        alg: Type | None = None,
        parameters: dict | None = None,
        random_state: int | None = None,
    ):
        """Initialize model with data set.

        Args:
            name: Name of the model.
            alg: Algorithm to use.
            dataset: Data set to use.
            parameters: Parameters to use.
            random_state: Random seed to use for random operations.
        """
        return DNNModel(
            base_dir=self.generatedModelsPath,
            alg=alg,
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
                (STFullyConnected, "STFullyConnected", TargetTasks.SINGLECLASS, [6.5]),
                (
                    STFullyConnected,
                    "STFullyConnected",
                    TargetTasks.MULTICLASS,
                    [0, 1, 10, 1100],
                ),
            )
        ]
        + [
            (
                f"{alg_name}_{task}_{'_'.join(map(str, random_state))}",
                task,
                alg_name,
                alg,
                th,
                random_state,
            )
            for alg, alg_name, task, th in (
                (STFullyConnected, "STFullyConnected", TargetTasks.REGRESSION, None),
            )
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
        random_state: list[int] | None,
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
            target_props=[{"name": "CL", "task": task, "th": th}],
            preparation_settings=self.getDefaultPrep(),
        )

        # initialize model for training from class
        alg_name = f"{alg_name}_{task}_th={th}"
        model = self.getModel(
            name=alg_name,
            alg=alg,
            random_state=random_state[0],
        )
        self.fitTest(model, dataset)
        predictor = DNNModel(
            name=alg_name, base_dir=model.baseDir, random_state=random_state[0]
        )

        # test if the results are (not) equal if the random state is the (not) same
        # and check if the output is the same before and after saving and loading
        if random_state[0] is not None:
            model.cleanFiles()
            comparison_model = self.getModel(
                name=alg_name,
                alg=alg,
                random_state=random_state[1],
            )
            self.fitTest(comparison_model, dataset)
            self.predictorTest(
                predictor, # model loaded from file
                dataset=dataset,
                comparison_model=comparison_model, # model not loaded from file
                expect_equal_result=random_state[0] == random_state[1],
            )
        else:
            self.predictorTest(predictor, dataset=dataset)


class ChemPropTest(ModelDataSetsPathMixIn, ModelCheckMixIn, TestCase):
    """This class holds the tests for the DNNModel class."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    @property
    def gridFile(self):
        """Return the path to the grid file with test
        search spaces for hyperparameter optimization.
        """
        return f"{os.path.dirname(__file__)}/test_files/search_space_test.json"

    def getModel(
        self,
        name: str,
        parameters: dict | None = None,
        random_state: int | None = None,
    ):
        """Initialize model with data set.

        Args:
            name: Name of the model.
            parameters: Parameters to use.
            random_state: Random seed to use for random operations.
        """
        if parameters is None:
            parameters = {}

        if len(GPUS) > 0:
            parameters["gpu"] = GPUS[0]
        parameters["epochs"] = 2
        return ChempropModel(
            base_dir=self.generatedModelsPath,
            name=name,
            parameters=parameters,
            random_state=random_state,
        )

    @parameterized.expand(
        [
            (f"{alg_name}_{task}", task, alg_name, th, [None])
            for alg_name, task, th in (
                ("MoleculeModel", TargetTasks.SINGLECLASS, [6.5]),
                ("MoleculeModel", TargetTasks.MULTICLASS, [0, 1, 10, 1100]),
            )
        ]
        + [
            (
                f"{alg_name}_{task}_{'_'.join(map(str, random_state))}",
                task,
                alg_name,
                th,
                random_state,
            )
            for alg_name, task, th in (("MoleculeModel", TargetTasks.REGRESSION, None),)
            for random_state in ([None], [1, 42], [42, 42])
        ]
    )
    def testSingleTaskModel(
        self,
        _,
        task: TargetTasks,
        alg_name: str,
        th: float,
        random_state: list[int | None],
    ):
        """Test the DNNModel model in one configuration.

        Args:
            task: Task to test.
            alg_name: Name of the algorithm.
            th: Threshold to use for classification models.
            random_state: Seed to be used for random operations.
        """
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            name=f"{alg_name}_{task}",
            target_props=[{"name": "CL", "task": task, "th": th}],
            preparation_settings=None,
        )
        dataset.prepareDataset(
            feature_calculators=[SmilesDesc()],
            split=RandomSplit(test_fraction=0.1, dataset=dataset),
        )
        # initialize model for training from class
        alg_name = f"{alg_name}_{task}_th={th}"
        model = self.getModel(
            name=f"{alg_name}",
            random_state=random_state[0],
        )
        self.fitTest(model, dataset)
        predictor = ChempropModel(name=alg_name, base_dir=model.baseDir)
        self.predictorTest(predictor, dataset=dataset)

        # make predictions with the trained model and check if the results are (not)
        # equal if the random state is the (not) same and check if the output is the
        # same before and after saving and loading
        if random_state[0] is not None:
            comparison_model = self.getModel(
                name=alg_name,
                random_state=random_state[1],
            )
            self.fitTest(comparison_model, dataset)
            self.predictorTest(
                predictor, # model loaded from file
                dataset=dataset,
                comparison_model=comparison_model,  # model not loaded from file
                expect_equal_result=random_state[0] == random_state[1],
            )
        else:
            self.predictorTest(predictor, dataset=dataset)

    @parameterized.expand(
        [
            (f"{alg_name}_{task}", task, alg_name, [None])
            for alg_name, task in (("MoleculeModel", ModelTasks.MULTITASK_REGRESSION),)
        ]
        + [
            (
                f"{alg_name}_{task}_{'_'.join(map(str, random_state))}",
                task,
                alg_name,
                random_state,
            )
            for alg_name, task in (("MoleculeModel", ModelTasks.MULTITASK_SINGLECLASS),)
            for random_state in ([None], [1, 42], [42, 42])
        ]
    )
    def testMultiTaskmodel(
        self, _, task: TargetTasks, alg_name: str, random_state: list[int | None]
    ):
        """Test the DNNModel model in one configuration.

        Args:
            task: Task to test.
            alg_name: Name of the algorithm.
        """
        if task == ModelTasks.MULTITASK_REGRESSION:
            target_props = [
                {"name": "fu", "task": TargetTasks.SINGLECLASS, "th": [0.3]},
                {"name": "CL", "task": TargetTasks.SINGLECLASS, "th": [6.5]},
            ]
        else:
            target_props = [
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
            ]
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            name=f"{alg_name}_{task}",
            target_props=target_props,
            preparation_settings=None,
        )
        dataset.prepareDataset(
            feature_calculators=[SmilesDesc()],
            split=RandomSplit(test_fraction=0.1, dataset=dataset),
        )
        # initialize model for training from class
        alg_name = f"{alg_name}_{task}"
        model = self.getModel(
            name=alg_name,
            random_state=random_state[0],
        )
        self.fitTest(model, dataset)
        predictor = ChempropModel(name=alg_name, base_dir=model.baseDir)
        self.predictorTest(predictor, dataset=dataset)

        # make predictions with the trained model and check if the results are (not)
        # equal if the random state is the (not) same and check if the output is the
        # same before and after saving and loading
        if random_state[0] is not None:
            comparison_model = self.getModel(
                name=alg_name,
                random_state=random_state[1],
            )
            self.fitTest(comparison_model, dataset)
            self.predictorTest(
                predictor, # model loaded from file
                dataset=dataset,
                comparison_model=comparison_model, # model not loaded from file
                expect_equal_result=random_state[0] == random_state[1],
            )
        else:
            self.predictorTest(predictor, dataset=dataset)

    def testConsistency(self):
        """Test if QSPRpred Chemprop and Chemprop models are consistent."""
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            name="consistency_data",
            target_props=[{"name": "CL", "task": TargetTasks.REGRESSION}],
            preparation_settings=None,
        )
        dataset.prepareDataset(
            feature_calculators=[SmilesDesc()],
            split=RandomSplit(test_fraction=0.1, seed=dataset.randomState),
        )
        # initialize model for training from class
        model = self.getModel(name="consistency_data")

        # chemprop by default uses sklearn rmse (squared=False) as metric
        rmse = metrics.make_scorer(
            metrics.root_mean_squared_error, greater_is_better=False
        )

        # Run 1 fold of bootstrap cross validation (default cross validation in chemprop is bootstrap)
        assessor = CrossValAssessor(
            scoring=SklearnMetrics(rmse),
            split=ShuffleSplit(
                n_splits=1, test_size=0.1, random_state=dataset.randomState
            ),
        )
        qsprpred_score = assessor(
            model,
            dataset,
            split=RandomSplit(test_fraction=0.1, seed=dataset.randomState),
        )
        qsprpred_score = -qsprpred_score[0]  # qsprpred_score is negative rmse

        # save the cross-validation train, test and validation split to
        # compare with true Chemprop model from the base monitor
        df_train = pd.DataFrame(
            assessor.monitor.fits[0]["fitData"]["X_train"], columns=["SMILES"]
        )
        df_train["pchembl_value_Mean"] = assessor.monitor.fits[0]["fitData"]["y_train"]
        df_train.to_csv(
            f"{self.generatedModelsPath}/consistency_data_train.csv", index=False
        )

        df_val = pd.DataFrame(
            assessor.monitor.fits[0]["fitData"]["X_val"], columns=["SMILES"]
        )
        df_val["pchembl_value_Mean"] = assessor.monitor.fits[0]["fitData"]["y_val"]
        df_val.to_csv(
            f"{self.generatedModelsPath}/consistency_data_val.csv", index=False
        )

        df_test = pd.DataFrame(assessor.monitor.foldData[0]["X_test"])
        df_test.rename(columns={"Descriptor_SmilesDesc_SMILES": "SMILES"}, inplace=True)
        df_test["pchembl_value_Mean"] = assessor.monitor.foldData[0]["y_test"]
        df_test.to_csv(
            f"{self.generatedModelsPath}/consistency_data_test.csv", index=False
        )

        # run chemprop with the same data and parameters
        arguments = [
            "--data_path",
            f"{self.generatedModelsPath}/consistency_data_train.csv",
            "--separate_val_path",
            f"{self.generatedModelsPath}/consistency_data_val.csv",
            "--separate_test_path",
            f"{self.generatedModelsPath}/consistency_data_test.csv",
            "--dataset_type",
            "regression",
            "--save_dir",
            self.generatedModelsPath,
            "--epochs",
            "2",
            "--seed",
            str(model.randomState),
            "--pytorch_seed",
            str(model.randomState),
        ]

        if len(GPUS) > 0:
            arguments.extend(["--gpu", str(GPUS[0])])

        args = chemprop.args.TrainArgs().parse_args(arguments)
        chemprop_score, _ = chemprop.train.cross_validate(
            args=args, train_func=chemprop.train.run_training
        )

        # compare the scores
        print(f"QSPRpred score: {qsprpred_score}")
        print(f"Chemprop score: {chemprop_score}")
        self.assertAlmostEqual(qsprpred_score, chemprop_score, places=5)


@pytest.mark.skipif((spec := util.find_spec("cupy")) is None, reason="requires cupy")
class TestPyBoostModel(ModelDataSetsPathMixIn, ModelCheckMixIn, TestCase):
    """This class holds the tests for the PyBoostModel class."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def getModel(
        self,
        name: str,
        parameters: dict | None = None,
        random_state: int | None = None,
    ):
        """Create a PyBoostModel model.

        Args:
            name (str): the name of the model
            dataset (QSPRDataset, optional): the dataset to use. Defaults to None.
            parameters (dict, optional): the parameters to use. Defaults to None.
            random_state (int, optional): the random state to use. Defaults to None.

        Returns:
            PyBoostModel the model
        """
        if parameters is None:
            parameters = {}
        parameters["ntrees"] = 10

        return import_module("..pyboost", __name__).PyBoostModel(
            base_dir=self.generatedModelsPath,
            name=name,
            parameters=parameters,
            random_state=random_state,
        )

    @parameterized.expand(
        [
            (
                f"PyBoost_{'_'.join(map(str, random_state))}",
                TargetTasks.REGRESSION,
                "PyBoost",
                params,
                random_state,
            )
            for params in [
                {
                    "loss": "mse",
                    "metric": "r2_score",
                    "colsample": 0.5,  # introduce more randomness in the model
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
            for random_state in ([None], [1, 42], [42, 42])
        ]
    )
    def testRegressionPyBoostFit(self, _, task, model_name, parameters, random_state):
        """Test model training for regression models."""
        parameters["verbose"] = -1
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            target_props=[{"name": "CL", "task": task}],
            preparation_settings=self.getDefaultPrep(),
        )
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_{task}",
            parameters=parameters,
            random_state=random_state[0],
        )
        self.fitTest(model, dataset)
        predictor = import_module("..pyboost", __name__).PyBoostModel(
            name=f"{model_name}_{task}", base_dir=model.baseDir
        )
        self.predictorTest(predictor, dataset=dataset)

        # make predictions with the trained model and check if the results are (not)
        # equal if the random state is the (not) same and check if the output is the
        # same before and after saving and loading
        if random_state[0] is not None:
            comparison_model = self.getModel(
                name=f"{model_name}_{task}",
                dataset=dataset,
                parameters=parameters,
                random_state=random_state[1],
            )
            self.fitTest(comparison_model)
            self.predictorTest(
                predictor,
                dataset=dataset,
                comparison_model=comparison_model,
                expect_equal_result=random_state[0] == random_state[1],
            )
        else:
            self.predictorTest(predictor, dataset=dataset)

    @parameterized.expand(
        [
            (f"{'PyBoost'}_{task}", task, th, "PyBoost", params)
            for params, task, th in (
                ({"loss": "bce", "metric": "auc"}, TargetTasks.SINGLECLASS, [6.5]),
                ({"loss": "crossentropy"}, TargetTasks.MULTICLASS, [0, 1, 10, 1100]),
            )
        ]
    )
    def testClassificationPyBoostFit(self, _, task, th, model_name, parameters):
        """Test model training for classification models."""
        parameters["verbose"] = -1
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            target_props=[{"name": "CL", "task": task, "th": th}],
            preparation_settings=self.getDefaultPrep(),
        )
        # test classifier
        # initialize model for training from class
        model = self.getModel(
            base_dir=self.generatedModelsPath,
            name=f"{model_name}_{task}",
            parameters=parameters,
        )
        self.fitTest(model, dataset)
        predictor = import_module("..pyboost", __name__).PyBoostModel(
            name=f"{model_name}_{task}", base_dir=model.baseDir
        )
        self.predictorTest(predictor, dataset=dataset)

    @parameterized.expand(
        [
            ("PyBoost", "PyBoost", params)
            for params in [
                {"loss": "mse", "metric": "r2_score"},
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
            parameters=parameters,
        )
        self.fitTest(model, dataset)
        predictor = import_module("..pyboost", __name__).PyBoostModel(
            name=f"{model_name}_multitask_regression",
        )
        self.predictorTest(predictor, dataset=dataset)

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
    #         preparation_settings=self.getDefaultPrep(),
    #     )
    #     # test classifier
    #     # initialize model for training from class
    #     model = self.getModel(
    #         base_dir=self.generatedModelsPath,
    #         name=f"{model_name}_multitask_classification",
    #         parameters=parameters,
    #     )
    #     self.fitTest(model, dataset)
    #     predictor = import_module("..pyboost", __name__).PyBoostModel(
    #         name=f"{model_name}_multitask_classification", base_dir=model.baseDir
    #     )
    #     self.predictorTest(predictor)


class TestNNMonitoring(MonitorsCheckMixIn, TestCase):
    """This class holds the tests for the monitoring classes."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    @property
    def gridFile(self):
        """Return the path to the grid file with test
        search spaces for hyperparameter optimization.
        """
        return f"{os.path.dirname(__file__)}/test_files/search_space_test.json"

    def testBaseMonitor(self):
        model = DNNModel(
            base_dir=self.generatedModelsPath,
            name="STFullyConnected",
            gpus=GPUS,
            patience=3,
            tol=0.02,
            random_state=42,
        )
        self.runMonitorTest(
            model,
            self.createLargeTestDataSet(preparation_settings=self.getDefaultPrep()),
            BaseMonitor,
            self.baseMonitorTest,
            True,
        )

    def testFileMonitor(self):
        model = DNNModel(
            base_dir=self.generatedModelsPath,
            name="STFullyConnected",
            gpus=GPUS,
            patience=3,
            tol=0.02,
            random_state=42,
        )
        self.runMonitorTest(
            model,
            self.createLargeTestDataSet(preparation_settings=self.getDefaultPrep()),
            BaseMonitor,
            self.baseMonitorTest,
            True,
        )

    def testListMonitor(self):
        """Test the list monitor"""
        model = DNNModel(
            base_dir=self.generatedModelsPath,
            name="STFullyConnected",
            gpus=GPUS,
            patience=3,
            tol=0.02,
            random_state=42,
        )
        self.runMonitorTest(
            model,
            self.createLargeTestDataSet(preparation_settings=self.getDefaultPrep()),
            ListMonitor,
            self.listMonitorTest,
            True,
            [BaseMonitor(), FileMonitor()],
        )
