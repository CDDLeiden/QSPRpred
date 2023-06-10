"""
tests

Created by: Martin Sicho
On: 12.05.23, 18:31
"""
import os
from unittest import TestCase

import torch
from parameterized import parameterized
from qsprpred.deep.models.models import QSPRDNN
from qsprpred.deep.models.neural_network import STFullyConnected
from qsprpred.models.tasks import TargetTasks
from qsprpred.models.tests import ModelDataSetsMixIn, ModelTestMixIn

GPUS = [idx for idx in range(torch.cuda.device_count())]


class NeuralNet(ModelDataSetsMixIn, ModelTestMixIn, TestCase):
    """This class holds the tests for the QSPRDNN class."""
    
    qsprModelsPath = f'{os.path.dirname(__file__)}/test_files/qspr/models'

    @property
    def gridFile(self):
        return f'{os.path.dirname(__file__)}/test_files/search_space_test.json'

    @staticmethod
    def get_model(name, alg=None, dataset=None, parameters=None):
        """Intialize dataset and model."""
        return QSPRDNN(
            base_dir=f'{os.path.dirname(__file__)}/test_files/',
            alg=alg,
            data=dataset,
            name=name,
            parameters=parameters,
            gpus=GPUS,
            patience=3,
            tol=0.02
        )

    # @parameterized.expand([
    #     (f"{alg_name}_{task}", task, alg_name, alg, th)
    #     for alg, alg_name, task, th in (
    #         (STFullyConnected, "STFullyConnected", TargetTasks.REGRESSION, None),
    #         (STFullyConnected, "STFullyConnected", TargetTasks.SINGLECLASS, [6.5]),
    #         (STFullyConnected, "STFullyConnected", TargetTasks.MULTICLASS, [0, 1, 10, 1200]),
    #     )
    # ])
    # def test_base_model(self, _, task, alg_name, alg, th):
    #     """Test the base DNN model."""
    #     # prepare test regression dataset
    #     is_reg = True if task == TargetTasks.REGRESSION else False
    #     no_features, trainloader, testloader = self.prep_testdata(
    #         name=f"{alg_name}_{task}", target_props=[{"name": "CL", "task": task, "th": th}])
    #
    #     # fit model with default settings
    #     model = alg(n_dim=no_features, is_reg=is_reg)
    #     model.fit(
    #         trainloader,
    #         testloader,
    #         out=f'{self.datapath}/{alg_name}_{task}',
    #         patience=3)
    #
    #     # fit model with non-default epochs and learning rate and tolerance
    #     model = alg(n_dim=no_features, n_epochs=50, lr=0.5, is_reg=is_reg)
    #     model.fit(
    #         trainloader,
    #         testloader,
    #         out=f'{self.datapath}/{alg_name}_{task}',
    #         patience=3,
    #         tol=0.01)
    #
    #     # fit model with non-default settings for model construction
    #     model = alg(
    #         n_dim=no_features,
    #         neurons_h1=2000,
    #         neurons_hx=500,
    #         extra_layer=True,
    #         is_reg=is_reg
    #     )
    #     model.fit(
    #         trainloader,
    #         testloader,
    #         out=f'{self.datapath}/{alg_name}_{task}',
    #         patience=3)

    @parameterized.expand([
        (f"{alg_name}_{task}", task, alg_name, alg, th)
        for alg, alg_name, task, th in (
            (STFullyConnected, "STFullyConnected", TargetTasks.REGRESSION, None),
            (STFullyConnected, "STFullyConnected", TargetTasks.SINGLECLASS, [6.5]),
            (STFullyConnected, "STFullyConnected", TargetTasks.MULTICLASS, [0, 1, 10, 1100]),
        )
    ])
    def test_qsprpred_model(self, _, task, alg_name, alg, th):
        """Test the QSPRDNN model."""
        # initialize dataset
        dataset = self.create_large_dataset(
            name=f"{alg_name}_{task}", target_props=[{"name": "CL", "task": task, "th": th}],
            preparation_settings=self.get_default_prep())

        # initialize model for training from class
        alg_name = f"{alg_name}_{task}_th={th}"
        model = self.get_model(
            name=f"{alg_name}",
            alg=alg,
            dataset=dataset
        )
        self.fitTest(model)
        predictor = QSPRDNN(name=alg_name, base_dir=model.baseDir)
        self.predictorTest(predictor)
