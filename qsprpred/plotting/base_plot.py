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

    def __init__(self, models: list[QSPRModel]):
        """Initialize the base class for all model plots.

        Args:
            models (list[QSPRModel]):
                list of models to plot
        """
        self.models = models
        self.modelOuts = {model: model.outPrefix for model in self.models}
        self.modelNames = {model: model.name for model in self.models}
        self.cvPaths = {}
        self.indPaths = {}
        for model in self.models:
            cv_path, ind_path = self.checkModel(model)
            self.cvPaths[model] = cv_path
            self.indPaths[model] = ind_path

    def checkModel(self, model: QSPRModel) -> tuple[str, str]:
        """Check if the model has been evaluated and saved. If not, raise an exception.

        Args:
            model (QSPRModel): model to check

        Returns:
            cvPath (str): path to the cross-validation set results file
            indPath (str): path to the independent test set results file

        Raises:
            ValueError: if the model type is not supported
        """
        cv_path = f"{self.modelOuts[model]}.cv.tsv"
        ind_path = f"{self.modelOuts[model]}.ind.tsv"
        if model.task not in self.getSupportedTasks():
            raise ValueError("Unsupported model type: %s" % model.task)
        if not os.path.exists(model.metaFile):
            raise ValueError(
                "Model output file does not exist: %s. "
                "Have you evaluated and saved the model, yet?" % model.metaFile
            )
        if not os.path.exists(cv_path):
            raise ValueError(
                "Model output file does not exist: %s. "
                "Have you evaluated and saved the model, yet?" % cv_path
            )
        if not os.path.exists(ind_path):
            raise ValueError(
                "Model output file does not exist: %s. "
                "Have you evaluated and saved the model, yet?" % ind_path
            )
        return cv_path, ind_path

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
