import os
from dataclasses import dataclass

from .data_prep import DataPrepSettings
from ...data.data import TargetProperty
from ...data.descriptors.sets import DescriptorSet
from ...data.sources.data_source import DataSource
from ...models.assessment_methods import ModelAssessor
from ...models.hyperparam_optimization import HyperparameterOptimization
from ...models.models import QSPRModel
from ...utils.serialization import JSONSerializable


@dataclass
class BenchmarkSettings(JSONSerializable):
    """Class that determines settings for a benchmarking run.

    Attributes:
        name (str): name of the benchmark
        n_replicas (int): number of repetitions per experiment
        random_seed (int): seed for random operations
        data_sources (list[DataSource]): data sources to use
        descriptors (list[list[DescriptorSet]]): descriptors to use
        target_props (list[list[TargetProperty]]): target properties to use
        prep_settings (list[qsprpred.benchmarks.settings.data_prep.DataPrepSettings]): data preparation settings to use
        models (list[QSPRModel]): models to use
        assessors (list[ModelAssessor]): model assessors to use
        optimizers (list[HyperparameterOptimization]): hyperparameter optimizers to use
    """

    _notJSON = ["models"]

    name: str
    n_replicas: int
    random_seed: int
    data_sources: list[DataSource]
    descriptors: list[list[DescriptorSet]]
    target_props: list[list[TargetProperty]]
    prep_settings: list[DataPrepSettings]
    models: list[QSPRModel]
    assessors: list[ModelAssessor]
    optimizers: list[HyperparameterOptimization]

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["models"] = []
        for model in self.models:
            o_dict["models"].append(model.save())
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        # logging.error(state["models"])
        # print([os.path.exists(model) for model in state["models"]])
        # print([json.loads(open(model).read()) for model in state["models"]])
        self.models = [QSPRModel.fromFile(model) for model in state["models"]]

    def checkConsistency(self):
        assert len(self.data_sources) > 0, "No data sources defined."
        assert len(self.descriptors) > 0, "No descriptors defined."
        assert len(self.target_props) > 0, "No target properties defined."
        assert len(self.prep_settings) > 0, "No data preparation settings defined."
        assert len(self.models) > 0, "No models defined."
        assert len(self.assessors) > 0, "No model assessors defined."
