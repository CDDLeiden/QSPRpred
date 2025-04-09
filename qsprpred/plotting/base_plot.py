"""This module contains the base class for all model plots."""

import os
from abc import ABC, abstractmethod
from typing import Any

from ..models.model import QSPRModel
from ..data.tables.qspr import QSPRTable


class ModelPlot(ABC):
    """Base class for all model plots.

    Attributes:
        models (list[QSPRModel]):
            list of models to plot
        modelOuts (dict[QSPRModel, str]):
            dictionary of model output paths
        modelNames (dict[QSPRModel, str]):
            dictionary of model names
        assesmentPaths (dict[QSPRModel, dict[str, str]):
            dictionary of assessment names mapped to their
            paths for each model
        datasets (dict[str, QSPRTable]):
            dictionary of model names mapped to their datasets used for training,
            if datasets are provided, the plotter will use the dataset labels instead
            of the assessment labels
    """
    def __init__(self, models: list[QSPRModel], assessments: list[str], datasets: list[QSPRTable] = None):
        """Initialize the base class for all model plots.

        Args:
            models (list[QSPRModel]):
                list of models to plot
            assessments (list[str]):
                list of assessment names
            datasets (list[QSPRTable], optional):
                list of datasets used for training the models, if provided,
                the plotter will use the dataset labels instead of the assessment labels.
                Must be the same length as `models`, use None to skip a model.
        """
        self.models = models
        self.modelOuts = {model: model.outPrefix for model in self.models}
        self.modelNames = {model: model.name for model in self.models}
        self.assesmentPaths = {}
        for model in self.models:
            assesment_paths = self.checkModel(model, assessments)
            self.assesmentPaths[model] = assesment_paths
        if datasets is not None:
            if len(datasets) != len(models):
                raise ValueError(
                    "Length of datasets must be the same as the length of models."
                )
            self.datasets = {model.name: dataset for model, dataset in zip(models, datasets)}
        else:
            self.datasets = None

    def checkModel(self, model: QSPRModel, assessments: list[str]) -> tuple[str, str]:
        """Check if the model has been evaluated and saved. If not, raise an exception.

        Args:
            model (QSPRModel): model to check
            assessments (list[str]): list of assessment names
        Returns:
            assesment_paths (dict[str, str]): 
                dictionary of assessment names mapped to their paths

        Raises:
            ValueError: if the model type is not supported
        """
        if not os.path.exists(model.metaFile):
            raise ValueError(
                "Model output file does not exist: %s. "
                "Have you evaluated and saved the model, yet?" % model.metaFile
            )
        assesment_paths = {}
        for assessment in assessments:
            assesment_paths[assessment] = f"{self.modelOuts[model]}_{assessment}.tsv"
            if not os.path.exists(assesment_paths[assessment]):
                raise ValueError(
                    "Model output file does not exist: %s. "
                    "Have you evaluated the model, yet?" % assesment_paths[assessment]
                )
        if model.task not in self.getSupportedTasks():
            raise ValueError("Unsupported model type: %s" % model.task)
        return assesment_paths

    @abstractmethod
    def getSupportedTasks(self) -> list[str]:
        """Get the types of models this plotter supports.

        Returns:
            `list` of `TargetTasks`: list of supported `TargetTasks`
        """

    @abstractmethod
    def make(self, save: bool = True, show: bool = False) -> Any:
        """Make the plot.

        Opens a window to show the plot or returns a plot
        representation that can be directly shown in a notebook or saved to a file.

        Args:
            save (bool): whether to save the plot to a file
            show (bool): whether to show the plot in a window

        Returns:
            plot (Any): plot representation
        """
