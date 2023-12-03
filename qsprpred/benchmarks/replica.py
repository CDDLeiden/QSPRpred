import os

import json
import logging
from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

from .settings.benchmark import DataPrepSettings
from ..data.data import TargetProperty, QSPRDataset
from ..data.descriptors.calculators import MoleculeDescriptorsCalculator
from ..data.descriptors.sets import DescriptorSet
from ..data.sources.data_source import DataSource
from ..models.assessment_methods import ModelAssessor
from ..models.hyperparam_optimization import HyperparameterOptimization
from ..models.models import QSPRModel
from ..utils.serialization import JSONSerializable


@dataclass
class Replica(JSONSerializable):
    """Class that determines settings for a single replica of a benchmarking run.

    Attributes:
        idx (int): Index of the replica.
        name (str): Name of the replica.
        data_source (DataSource): Data source for the replica.
        descriptors (list[DescriptorSet]): Descriptor sets to use.
        target_props (list[TargetProperty]): Target properties to use.
        prep_settings (DataPrepSettings): Data preparation settings to use.
        model (QSPRModel): Model to use.
        optimizer (HyperparameterOptimization): Hyperparameter optimization to use.
        assessors (list[ModelAssessor]): Model assessors to use.
        random_seed (int): Random seed to use.
        ds (QSPRDataset): Data set used in the replica.
    """

    _notJSON = ["model"]

    idx: int
    name: str
    data_source: DataSource
    descriptors: list[DescriptorSet]
    target_props: list[TargetProperty]
    prep_settings: DataPrepSettings
    model: QSPRModel
    optimizer: HyperparameterOptimization
    assessors: list[ModelAssessor]
    random_seed: int
    ds: QSPRDataset | None = None
    results: pd.DataFrame | None = None

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["model"] = self.model.save()
        o_dict["ds"] = None
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.model = QSPRModel.fromFile(state["model"])
        self.ds = None

    def __str__(self):
        return self.id

    @property
    def id(self):
        """Returns the identifier of this replica."""
        return f"{self.name}_{self.random_seed}"

    def create_dataset(self, reload=False):
        # get the basics set from data source
        self.ds = self.data_source.getDataSet(
            deepcopy(self.target_props),
            overwrite=reload,
            random_state=self.random_seed,
        )
        # add descriptors
        self.add_descriptors(reload=reload)

    def add_descriptors(
            self,
            reload: bool = False
    ):
        # generate name for the data with descriptors
        desc_id = "_".join(sorted([str(d) for d in self.descriptors]))
        self.ds.name = f"{self.ds.name}_{desc_id}"
        # attempt to load the data set with descriptors
        if os.path.exists(self.ds.metaFile) and not reload:
            self.ds = QSPRDataset.fromFile(self.ds.metaFile)
            self.ds.setTargetProperties(deepcopy(self.target_props))
        else:
            logging.info(f"Data set {self.ds.name} not found. It will be created.")
            # calculate descriptors if necessary
            logging.info(f"Calculating descriptors for {self.ds.name}.")
            desc_calculator = MoleculeDescriptorsCalculator(
                desc_sets=deepcopy(self.descriptors)
            )
            self.ds.addDescriptors(desc_calculator, recalculate=True)
            self.ds.save()

    def prep_dataset(self):
        self.ds.prepareDataset(
            **deepcopy(self.prep_settings.__dict__),
        )

    def create_report(self):
        results = self.results.copy()
        results["ModelFile"] = self.model.metaFile
        results["Algorithm"] = self.model.alg.__name__
        results["AlgorithmParams"] = json.dumps(self.model.parameters)
        results["ReplicaID"] = self.id
        results["DataSet"] = self.ds.name
        out_file = f"{self.model.outPrefix}_replica.json"
        for assessor in self.assessors:
            # FIXME: some problems in monitor serialization now prevent this
            assessor.monitor = None
        self.model.data = None  # FIXME: model now does not support data serialization
        results["ReplicaFile"] = self.toFile(out_file)
        return results

    def init_model(self):
        self.model = deepcopy(self.model)
        self.model.name = self.id
        self.model.initFromData(self.ds)
        self.model.initRandomState(self.random_seed)
        if self.optimizer is not None:
            self.optimizer.optimize(self.model)

    def run_assessment(self):
        self.results = None
        for assessor in self.assessors:
            scores = assessor(self.model, save=True)
            scores = pd.DataFrame({
                "Assessor": assessor.__class__.__name__,
                "ScoreFunc": assessor.scoreFunc.name,
                "Score": scores,
                "TargetProperties": "~".join(
                    sorted([
                        tp.name for tp in self.target_props
                    ])),
                "TargetTasks": "~".join(
                    sorted([
                        tp.task for tp in self.target_props
                    ])),
            })
            if self.results is None:
                self.results = scores
            else:
                self.results = pd.concat([self.results, scores])
