import os

import json
import logging
from copy import deepcopy
from dataclasses import dataclass

import pandas as pd

from qsprpred.data.data import TargetProperty, QSPRDataset
from qsprpred.data.descriptors.calculators import MoleculeDescriptorsCalculator
from qsprpred.data.descriptors.sets import DescriptorSet
from qsprpred.data.sources.data_source import DataSource
from qsprpred.models.assessment_methods import ModelAssessor
from qsprpred.models.hyperparam_optimization import HyperparameterOptimization
from qsprpred.models.models import QSPRModel
from qsprpred.utils.serialization import JSONSerializable


@dataclass
class Replica(JSONSerializable):
    """Class that determines settings for a single replica of a benchmarking run.

    Attributes:
        idx (int): number of the replica
        name (str): name of the benchmark
        data_source (DataSource): data source to use
        target_props (TargetProperty | dict): target property to use
        prep_settings (DataPrepSettings): data preparation settings to use
        model (QSPRModel): model to use
        assessors (ModelAssessor): model assessor to use
        optimizer (HyperparameterOptimization): hyperparameter optimizer to use
        random_seed (int): seed for random operations
        ds (QSPRDataset): data set used for this replica
    """

    _notJSON = ["model"]

    idx: int
    name: str
    data_source: DataSource
    descriptors: list[DescriptorSet]
    target_props: list[TargetProperty]
    prep_settings: "DataPrepSettings"
    model: QSPRModel
    optimizer: HyperparameterOptimization
    assessors: list[ModelAssessor]
    random_seed: int
    ds: QSPRDataset | None = None

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["model"] = self.model.metaFile
        if not os.path.exists(o_dict["model"]):
            self.model.save()
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
            self.target_props,
            overwrite=reload,
            random_state=self.random_seed
        )
        # add descriptors
        self.add_descriptors(reload=reload)

    def add_descriptors(
            self,
            reload: bool = False
    ):
        # generate name for the data with descriptors
        desc_id = "_".join([str(d) for d in self.descriptors])
        # tp_id = "_".join([tp.name for tp in ds.targetProperties])
        ds_desc_name = f"{self.ds.name}_{desc_id}"
        # create or reload the data set
        try:
            ds_descs = QSPRDataset(
                name=ds_desc_name,
                store_dir=self.ds.baseDir,
                target_props=self.target_props,
                random_state=self.random_seed
            )
        except ValueError:
            logging.warning(f"Data set {ds_desc_name} not found. It will be created.")
            ds_descs = QSPRDataset(
                name=ds_desc_name,
                store_dir=self.ds.baseDir,
                target_props=self.target_props,
                random_state=self.random_seed,
                df=self.ds.getDF(),
            )
        # calculate descriptors if necessary
        if not ds_descs.hasDescriptors or reload:
            logging.info(f"Calculating descriptors for {ds_descs.name}.")
            desc_calculator = MoleculeDescriptorsCalculator(
                desc_sets=self.descriptors
            )
            ds_descs.addDescriptors(desc_calculator, recalculate=True)
            ds_descs.save()
        self.ds = ds_descs
        self.ds.save()

    def prep_dataset(self):
        self.ds.prepareDataset(
            **self.prep_settings.__dict__,
        )

    def create_report(self):
        self.initModel()
        results = self.run_assessment()
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

    def initModel(self):
        self.model = deepcopy(self.model)
        self.model.name = self.id
        self.model.initFromData(self.ds)
        self.model.initRandomState(self.random_seed)
        if self.optimizer is not None:
            self.optimizer.optimize(self.model)

    def run_assessment(self):
        results = None
        for assessor in self.assessors:
            scores = assessor(self.model, save=True)
            scores = pd.DataFrame({
                "Assessor": assessor.__class__.__name__,
                "ScoreFunc": assessor.scoreFunc.name,
                "Score": scores,
            })
            if results is None:
                results = scores
            else:
                results = pd.concat([results, scores])
        return results
