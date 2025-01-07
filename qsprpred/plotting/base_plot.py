"""This module contains the base class for all model plots."""

import os
from abc import ABC, abstractmethod
from typing import Any

from ..models.model import QSPRModel


class ModelPlot(ABC):
    """Base class for all model plots.

    Attributes:
        models (list[QSPRModel]):
            list of models to plot
        modelOuts (dict[QSPRModel, str]):
            dictionary of model output paths
        modelNames (dict[QSPRModel, str]):
            dictionary of model names
        cvPaths (dict[QSPRModel, str]):
            dictionary of models mapped to their cross-validation set results paths
        indPaths (dict[QSPRModel, str]):
            dictionary of models mapped to their independent test set results paths
    """
    def __init__(self, models: list[QSPRModel], assessments: list[str]):
        """Initialize the base class for all model plots.

        Args:
            models (list[QSPRModel]):
                list of models to plot
        """
        self.models = models
        self.modelOuts = {model: model.outPrefix for model in self.models}
        self.modelNames = {model: model.name for model in self.models}
        self.assesmentPaths = {}
        for model in self.models:
            assesment_paths = self.checkModel(model, assessments)
            self.assesmentPaths[model] = assesment_paths

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
