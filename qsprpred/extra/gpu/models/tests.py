import os
from typing import Type
from unittest import TestCase, skipIf

import pandas as pd
import torch
from parameterized import parameterized
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ShuffleSplit

import chemprop
from qsprpred.data.descriptors.sets import SmilesDesc
from qsprpred.data.sampling.splits import RandomSplit
from qsprpred.extra.gpu.utils.parallel import TorchJITGenerator
from qsprpred.tasks import ModelTasks, TargetTasks
from ....benchmarks import BenchmarkRunner
from ....benchmarks.tests import BenchMarkTestCase
from ....extra.gpu.models.chemprop import ChempropModel
from ....extra.gpu.models.dnn import DNNModel
from ....extra.gpu.models.neural_network import STFullyConnected
from ....models import CrossValAssessor, SklearnModel
from ....models.metrics import SklearnMetrics
from ....models.monitors import BaseMonitor, FileMonitor, ListMonitor
from ....utils.parallel import ThreadsJITGenerator
from ....utils.testing.check_mixins import ModelCheckMixIn, MonitorsCheckMixIn
from ....utils.testing.path_mixins import ModelDataSetsPathMixIn

GPUS = list(range(torch.cuda.device_count()))


@skipIf(
    len(GPUS) == 0,
    "No GPU is available. "
    "Skipping benchmarking tests that require a GPU to run swiftly. "
)
class BenchMarkTest(BenchMarkTestCase):
    """Test GPU models with benchmarks."""

    def testBasicTorchExecution(self):
        """Run single task tests for classification."""
        self.settings.models = [
            DNNModel(
                name="STFullyConnected",
                alg=STFullyConnected,
                base_dir=f"{self.generatedPath}/models",
                gpus=GPUS,
                patience=3,
                tol=0.02,
                random_state=42,
            ),
            SklearnModel(
                name="RandomForestClassifier",
                alg=RandomForestClassifier,
                base_dir=f"{self.generatedPath}/models",
                random_state=42,
            ),
        ]
        self.benchmark = BenchmarkRunner(
            self.settings,
            data_dir=f"{self.generatedPath}/benchmarks",
            results_file=f"{self.generatedPath}/benchmarks/results.tsv",
            parallel_generator_gpu=TorchJITGenerator(
                worker_type="gpu",
                jobs_per_gpu=os.cpu_count(),
                use_gpus=GPUS,
            ),
        )
        self.checkSettings()
        results = self.benchmark.run(raise_errors=True)
        self.checkRunResults(results)
        self.checkSettings()

    def testChemProp(self):
        """Run single task tests for classification."""
        self.settings.models = [
            ChempropModel(
                name="ChempropModel",
                base_dir=f"{self.generatedPath}/models",
                random_state=42,
            )
        ]
        self.settings.descriptors = [[SmilesDesc()]]
        self.settings.prep_settings[0].feature_standardizer = None
        self.checkSettings()
        self.benchmark = BenchmarkRunner(
            self.settings,
            data_dir=f"{self.generatedPath}/benchmarks",
            results_file=f"{self.generatedPath}/benchmarks/results.tsv",
            parallel_generator_gpu=ThreadsJITGenerator(
                worker_type="gpu",
                use_gpus=GPUS,
                jobs_per_gpu=os.cpu_count(),
            ),
        )
        results = self.benchmark.run(raise_errors=True)
        self.checkRunResults(results)
        self.checkSettings()


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
                    (
                            STFullyConnected, "STFullyConnected",
                            TargetTasks.REGRESSION, None),
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
                predictor,  # model loaded from file
                dataset=dataset,
                comparison_model=comparison_model,  # model not loaded from file
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
                predictor,  # model loaded from file
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
                predictor,  # model loaded from file
                dataset=dataset,
                comparison_model=comparison_model,  # model not loaded from file
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
        df_test.rename(columns={"SmilesDesc_SMILES": "SMILES"}, inplace=True)
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
