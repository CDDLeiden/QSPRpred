import os
from typing import Type
from unittest import TestCase

import torch
from parameterized import parameterized

from ...data.data import QSPRDataset
from ...deep.models.dnn import QSPRDNN
from ...deep.models.chemprop import Chemprop
from ...deep.models.neural_network import STFullyConnected
from ...models.tasks import TargetTasks, ModelTasks
from ...models.tests import ModelDataSetsMixIn, ModelTestMixIn
from ...data.utils.descriptorcalculator import MoleculeDescriptorsCalculator
from ...data.utils.descriptorsets import SmilesDesc
from ...data.utils.datasplitters import RandomSplit
from sklearn.impute import SimpleImputer

GPUS = list(range(torch.cuda.device_count()))


class NeuralNet(ModelDataSetsMixIn, ModelTestMixIn, TestCase):
    """This class holds the tests for the QSPRDNN class."""
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
    ):
        """Initialize model with data set.

        Args:
            base_dir: Base directory for model.
            name: Name of the model.
            alg: Algorithm to use.
            dataset: Data set to use.
            parameters: Parameters to use.
        """
        return QSPRDNN(
            base_dir=base_dir,
            alg=alg,
            data=dataset,
            name=name,
            parameters=parameters,
            gpus=GPUS,
            patience=3,
            tol=0.02,
        )

    @parameterized.expand(
        [
            (f"{alg_name}_{task}", task, alg_name, alg, th)
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
        ]
    )
    def testSingleTaskModel(
        self, _, task: TargetTasks, alg_name: str, alg: Type, th: float
    ):
        """Test the QSPRDNN model in one configuration.

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
            preparation_settings=self.getDefaultPrep(),
        )

        # initialize model for training from class
        alg_name = f"{alg_name}_{task}_th={th}"
        model = self.getModel(
            base_dir=self.generatedModelsPath,
            name=f"{alg_name}",
            alg=alg,
            dataset=dataset
        )
        self.fitTest(model)
        predictor = QSPRDNN(name=alg_name, base_dir=model.baseDir)
        self.predictorTest(predictor)


class ChemProp(ModelDataSetsMixIn, ModelTestMixIn, TestCase):
    """This class holds the tests for the QSPRDNN class."""
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
            parameters = {"gpu": GPUS[0] if len(GPUS) > 0 else None}
        else:
            parameters["gpu"] = GPUS[0] if len(GPUS) > 0 else None
        return Chemprop(
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
        """Test the QSPRDNN model in one configuration.

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
            preparation_settings=None
        )
        dataset.prepareDataset(
            feature_calculators=[
                MoleculeDescriptorsCalculator(desc_sets=[SmilesDesc()])
            ],
            split=RandomSplit(test_fraction=0.1)
        )
        # initialize model for training from class
        alg_name = f"{alg_name}_{task}_th={th}"
        model = self.getModel(
            base_dir=self.generatedModelsPath, name=f"{alg_name}", dataset=dataset
        )
        self.fitTest(model)
        predictor = Chemprop(name=alg_name, base_dir=model.baseDir)
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
        """Test the QSPRDNN model in one configuration.

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
            preparation_settings=None
        )
        dataset.prepareDataset(
            feature_calculators=[
                MoleculeDescriptorsCalculator(desc_sets=[SmilesDesc()])
            ],
            split=RandomSplit(test_fraction=0.1)
        )
        # initialize model for training from class
        alg_name = f"{alg_name}_{task}"
        model = self.getModel(
            base_dir=self.generatedModelsPath, name=f"{alg_name}", dataset=dataset
        )
        self.fitTest(model)
        predictor = Chemprop(name=alg_name, base_dir=model.baseDir)
        self.predictorTest(predictor)
