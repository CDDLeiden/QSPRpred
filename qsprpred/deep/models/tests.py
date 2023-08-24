"""
tests

Created by: Martin Sicho
On: 12.05.23, 18:31
"""
import os
from typing import Type
from unittest import TestCase

import torch
from parameterized import parameterized

from ...data.data import QSPRDataset
from ...deep.models.models import QSPRDNN
from ...deep.models.neural_network import STFullyConnected
from ...models.tasks import TargetTasks
from ...models.tests import ModelDataSetsMixIn, ModelTestMixIn

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
        random_state: int | None = None,
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
            random_state=random_state
        )

    @parameterized.expand(
        [
            (f"{alg_name}_{task}", task, alg_name, alg, th, random_state)
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
            for random_state in (
                None,
                # TODO: fix random_state
                # 42
            )
        ]
    )
    def test_qsprpred_model(
        self, _, task: TargetTasks, alg_name: str, alg: Type, th: float, random_state: int | None
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
            random_state=random_state
        )
        # initialize model for training from class
        alg_name = f"{alg_name}_{task}_th={th}"
        model = self.getModel(
            base_dir=self.generatedModelsPath, name=f"{alg_name}", alg=alg, dataset=dataset, random_state=random_state
        )
        self.fitTest(model, random_state)
        predictor = QSPRDNN(name=alg_name, base_dir=model.baseDir)
        pred_use_probas, pred_not_use_probas = self.predictorTest(predictor)
        if random_state is not None:
            model = self.getModel(
                base_dir=self.generatedModelsPath, name=f"{alg_name}", alg=alg, dataset=dataset, random_state=random_state
            )
            self.fitTest(model, random_state)
            predictor = QSPRDNN(name=alg_name, base_dir=model.baseDir)
            self.predictorTest(predictor,
                expected_pred_use_probas=pred_use_probas,
                expected_pred_not_use_probas=pred_not_use_probas)