import json
import os
from copy import deepcopy

import pandas as pd

from .settings.benchmark import DataPrepSettings
from ..data import QSPRDataset
from ..data.descriptors.calculators import MoleculeDescriptorsCalculator
from ..data.descriptors.sets import DescriptorSet
from ..data.sources.data_source import DataSource
from ..logs import logger
from ..models.assessment_methods import ModelAssessor
from ..models.hyperparam_optimization import HyperparameterOptimization
from ..models.models import QSPRModel
from ..models.monitors import NullMonitor
from ..tasks import TargetProperty
from ..utils.serialization import JSONSerializable


class Replica(JSONSerializable):
    """Class that determines settings for a single replica of a benchmarking run.

    Attributes:
        idx (int):
            Index of the replica. This is not an identifier, but rather a number
            that indicates the order of the replica in the benchmarking run.
        name (str):
            Name of the replica.
        dataSource (DataSource):
            Data source to use.
        descriptors (list[DescriptorSet]):
            Descriptor sets to use.
        targetProps (list[TargetProperty]):
            Target properties to use.
        prepSettings (DataPrepSettings):
            Data preparation settings to use.
        model (QSPRModel):
            Current model. Use `initModel` to prepare it.
        optimizer (HyperparameterOptimization):
            Hyperparameter optimizer to use.
        assessors (list[ModelAssessor]):
            Model assessors to use.
        randomSeed (int):
            Random seed to use for all random operations withing the replica.
        ds (QSPRDataset):
            Initialized data set. Only available after `initData` has been called.
        results (pd.DataFrame):
            Results from the replica. Only available after
            `runAssessment` has been called.
    """

    def __init__(
        self,
        idx: int,
        name: str,
        data_source: DataSource,
        descriptors: list[DescriptorSet],
        target_props: list[TargetProperty],
        prep_settings: DataPrepSettings,
        model: QSPRModel,
        optimizer: HyperparameterOptimization,
        assessors: list[ModelAssessor],
        random_seed: int,
    ):
        self.idx = idx
        self.name = name
        self.dataSource = data_source
        self.descriptors = descriptors
        self.targetProps = target_props
        self.prepSettings = prep_settings
        self.optimizer = optimizer
        self.assessors = assessors
        self.randomSeed = random_seed
        self.ds = None
        self.results = None
        self.model = deepcopy(model)
        self.model.name = self.id
        self.model.save()

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["model"] = self.model.save()
        o_dict["ds"] = None
        o_dict["results"] = None
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.model = QSPRModel.fromFile(state["model"])
        self.ds = None
        self.results = None

    def __str__(self):
        return self.id

    @property
    def id(self) -> str:
        """A unique identifier for the replica.

        Returns:
            str:
                A unique identifier for the replica.
        """
        return f"{self.name}_{self.randomSeed}"

    def initData(self, reload=False):
        """Initializes the data set for this replica.

        Args:
            reload (bool, optional):
                Whether to overwrite all existing data and
                reinitialize from scratch. Defaults to `False`.
        """
        self.ds = self.dataSource.getDataSet(
            deepcopy(self.targetProps),
            overwrite=reload,
            random_state=self.randomSeed,
        )

    def addDescriptors(self, reload: bool = False):
        """Adds descriptors to the current data set. Make sure to call
        `initData` first to get it from the source.

        Args:
            reload (bool, optional):
                Whether to overwrite all existing data and
                reinitialize from scratch. Defaults to `False`.

        Raises:
            ValueError:
                If the data set has not been initialized.
        """
        if self.ds is None:
            raise ValueError("Data set not initialized. Call initData first.")
        desc_id = "_".join(sorted([str(d) for d in self.descriptors]))
        self.ds.name = f"{self.ds.name}_{desc_id}"
        # attempt to load the data set with descriptors
        if os.path.exists(self.ds.metaFile) and not reload:
            logger.info(f"Reloading existing {self.ds.name} from cache...")
            self.ds = QSPRDataset.fromFile(self.ds.metaFile)
            self.ds.setTargetProperties(deepcopy(self.targetProps))
        else:
            logger.info(f"Data set {self.ds.name} not yet found. It will be created.")
            # calculate descriptors if necessary
            logger.info(f"Calculating descriptors for {self.ds.name}.")
            desc_calculator = MoleculeDescriptorsCalculator(
                desc_sets=deepcopy(self.descriptors)
            )
            self.ds.addDescriptors(desc_calculator, recalculate=True)
            self.ds.save()

    def prepData(self):
        """Prepares the data set for this replica.

        Raises:
            ValueError:
                If the data set has not been initialized.
        """
        if self.ds is None:
            raise ValueError("Data set not initialized. Call initData first.")
        self.ds.prepareDataset(
            **deepcopy(self.prepSettings.__dict__),
        )

    def initModel(self):
        """Initializes the model for this replica. This includes
        initializing the model from the data set and optimizing
        the hyperparameters if an optimizer is specified.

        Raises:
            ValueError:
                If the data set has not been initialized.
        """
        if self.ds is None:
            raise ValueError("Data set not initialized. Call initData first.")
        self.model.initFromData(self.ds)
        self.model.initRandomState(self.randomSeed)
        if self.optimizer is not None:
            self.optimizer.optimize(self.model)
        self.model.save()

    def runAssessment(self):
        """Runs the model assessment for this replica. This includes
        running all model assessors and saving the results.

        The results are saved in the `results` attribute. They can be
        accessed by calling `createReport`, which combines the relevant information
        from the replica and the results into one `pd.DataFrame`.

        Raises:
            ValueError:
                If the model has not been initialized.
        """
        if self.model.data is None:
            raise ValueError("Model not initialized. Call initModel first.")
        self.results = None
        for assessor in self.assessors:
            scores = assessor(self.model, save=True)
            scores = pd.DataFrame(
                {
                    "Assessor": assessor.__class__.__name__,
                    "ScoreFunc": assessor.scoreFunc.name,
                    "Score": scores,
                    "TargetProperties": "~".join(
                        sorted([tp.name for tp in self.targetProps])
                    ),
                    "TargetTasks": "~".join(
                        sorted([str(tp.task) for tp in self.targetProps])
                    ),
                }
            )
            if self.results is None:
                self.results = scores
            else:
                self.results = pd.concat([self.results, scores])

    def createReport(self):
        """Creates a report from the results of this replica.

        Returns:
            pd.DataFrame:
                A `pd.DataFrame` with the results of this replica.

        Raises:
            ValueError:
                If the results have not been calculated.
        """
        if self.results is None:
            raise ValueError("Results not available. Call runAssessment first.")
        results = self.results.copy()
        results["ModelFile"] = self.model.metaFile
        results["Algorithm"] = self.model.alg.__name__
        results["AlgorithmParams"] = json.dumps(self.model.parameters)
        results["ReplicaID"] = self.id
        results["DataSet"] = self.ds.name
        out_file = f"{self.model.outPrefix}_replica.json"
        for assessor in self.assessors:
            # FIXME: some problems in monitor serialization now prevent this
            assessor.monitor = NullMonitor()
        self.model.data = None  # FIXME: model now does not support data serialization
        results["ReplicaFile"] = self.toFile(out_file)
        return results
