from dataclasses import dataclass
from typing import ClassVar

from qsprpred.models.assessment.methods import ModelAssessor
from .data_prep import DataPrepSettings
from ...data.descriptors.sets import DescriptorSet
from ...data.sources.data_source import DataSource
from ...models.hyperparam_optimization import HyperparameterOptimization
from ...models.model import QSPRModel
from ...tasks import TargetProperty
from ...utils.serialization import JSONSerializable


@dataclass
class BenchmarkSettings(JSONSerializable):
    """Class that determines settings for a benchmarking run.

    Attributes:
        name (str):
            Name of the benchmarking run.
        n_replicas (int):
            Number of replicas to run.
        random_seed (int):
            Random seed to use.
        data_sources (list[DataSource]):
            Data sources to use.
        descriptors (list[list[DescriptorSet]]):
            Descriptor sets to use.
        target_props (list[list[TargetProperty]]):
            Target properties to use.
        prep_settings (list[DataPrepSettings]):
            Data preparation settings to use.
        models (list[QSPRModel]):
            Models to use.
        assessors (list[ModelAssessor]):
            Model assessors to use.
        optimizers (list[HyperparameterOptimization]):
            Hyperparameter optimizers to use.
    """

    _notJSON: ClassVar = ["models"]

    name: str
    n_replicas: int
    random_seed: int
    data_sources: list[DataSource]
    descriptors: list[list[DescriptorSet]]
    target_props: list[list[TargetProperty]]
    prep_settings: list[DataPrepSettings]
    models: list[QSPRModel]
    assessors: list[ModelAssessor]
    optimizers: list[HyperparameterOptimization] = tuple()

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["models"] = []
        for model in self.models:
            o_dict["models"].append(model.save())
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.models = [QSPRModel.fromFile(model) for model in state["models"]]

    def checkConsistency(self):
        """Checks if the settings are consistent.

        Raises:
            AssertionError:
                If the settings are inconsistent.
        """
        assert len(self.data_sources) > 0, "No data sources defined."
        assert len(self.descriptors) > 0, "No descriptors defined."
        assert len(self.target_props) > 0, "No target properties defined."
        assert len(self.prep_settings) > 0, "No data preparation settings defined."
        assert len(self.models) > 0, "No models defined."
        assert len(self.assessors) > 0, "No model assessors defined."
