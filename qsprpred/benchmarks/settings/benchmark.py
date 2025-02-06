from dataclasses import dataclass
from typing import ClassVar

from qsprpred.models.assessment.methods import ModelAssessor

from ...data.descriptors.sets import DescriptorSet
from ...data.sources.data_source import DataSource
from ...data.sampling.splits import DataSplit
from ...data.processing.pipeline import DatasetPipeline
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
        prep_settings (list[DatasetPipeline]):
            Data preparation settings to use.
        models (list[QSPRModel]):
            Models to use.
        assessors (list[ModelAssessor]):
            Model assessors to use.
        subsets (dict[str, tuple[DataSplit, str, int]]):
            Dictionary mapping assessor names to tuples of data split, set (Train/Test),
            and fold index. Used to apply assessors to subsets of the data.
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
    pipelines: list[DatasetPipeline]
    models: list[QSPRModel]
    assessors: list[ModelAssessor]
    subsets: dict[str, tuple[DataSplit, str, int]] = ()
    optimizers: list[HyperparameterOptimization] = ()

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
        assert len(self.pipelines) > 0, "No data preparation settings defined."
        assert len(self.models) > 0, "No models defined."
        assert len(self.assessors) > 0, "No model assessors defined."
        assessor_names = [assessor.name for assessor in self.assessors]
        if len(self.subsets) > 0:
            for assessor in self.subsets.keys():
                assert assessor in assessor_names, f"Assessor {assessor} in subsets not found in assessors."
                assert len(self.subsets[assessor]) == 3, "Subsets must be a tuple of DataSplit, set (Train/Test), and fold index."
                assert isinstance(self.subsets[assessor][0], DataSplit), "First element of subset must be a DataSplit."
                assert self.subsets[assessor][1] in ["Train", "Test"], "Second element of subset must be 'Train' or 'Test'."
                assert isinstance(self.subsets[assessor][2], int), "Third element of subset must be an integer, the fold index."
            
