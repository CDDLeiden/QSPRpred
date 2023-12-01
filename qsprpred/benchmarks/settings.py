import itertools
import json
import logging
import os.path
import random
from dataclasses import dataclass
from typing import Callable, Generator

import numpy as np

from .replica import Replica
from qsprpred.data.data import TargetProperty
from qsprpred.data.descriptors.sets import DescriptorSet

from qsprpred.data.processing.feature_standardizers import SKLearnStandardizer
from qsprpred.data.sampling.splits import DataSplit
from qsprpred.data.sources.data_source import DataSource
from qsprpred.models.assessment_methods import ModelAssessor
from qsprpred.models.hyperparam_optimization import HyperparameterOptimization
from qsprpred.models.models import QSPRModel
from qsprpred.utils.serialization import JSONSerializable


@dataclass
class DataPrepSettings:
    split: DataSplit = None
    smiles_standardizer: str | Callable = "chembl"
    feature_filters: list = None
    feature_standardizer: SKLearnStandardizer = None
    feature_fill_value: float = 0.0


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
        prep_settings (list[DataPrepSettings]): data preparation settings to use
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
            if not os.path.exists(model.metaFile):
                o_dict["models"].append(model.save())
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        # logging.error(state["models"])
        # print([os.path.exists(model) for model in state["models"]])
        # print([json.loads(open(model).read()) for model in state["models"]])
        self.models = [QSPRModel.fromFile(model) for model in state["models"]]

    @property
    def n_runs(self):
        """Returns the total number of benchmarking runs."""
        self.checkConsistency()
        ret = (self.n_replicas * len(self.data_sources)
                * len(self.descriptors) * len(self.target_props)
                * len(self.prep_settings) * len(self.models)
               )
        if len(self.optimizers) > 0:
            ret *= len(self.optimizers)
        return ret

    def get_seed_list(self, seed: int) -> list[int]:
        """
        Get a list of seeds for the replicas.

        Args:
            seed(int): master seed to generate the list of seeds from

        Returns:
            list[int]: list of seeds for the replicas

        """
        random.seed(seed)
        return random.sample(range(2**32 - 1), self.n_runs)

    def checkConsistency(self):
        assert len(self.data_sources) > 0, "No data sources defined."
        assert len(self.descriptors) > 0, "No descriptors defined."
        assert len(self.target_props) > 0, "No target properties defined."
        assert len(self.prep_settings) > 0, "No data preparation settings defined."
        assert len(self.models) > 0, "No models defined."
        assert len(self.assessors) > 0, "No model assessors defined."

    def iter_replicas(self) -> Generator[Replica, None, None]:
        np.random.seed(self.random_seed)
        # generate all combinations of settings with itertools
        self.checkConsistency()
        indices = [x+1 for x in range(self.n_replicas)]
        optimizers = self.optimizers if len(self.optimizers) > 0 else [None]
        product = itertools.product(
            indices,
            [self.name],
            self.data_sources,
            self.descriptors,
            self.target_props,
            self.prep_settings,
            self.models,
            optimizers,
        )
        seeds = self.get_seed_list(self.random_seed)
        for idx, settings in enumerate(product):
            yield Replica(
                *settings,
                random_seed=seeds[idx],
                assessors=self.assessors
            )
