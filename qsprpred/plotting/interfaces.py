"""This module contains the base class for all model plots."""
import os
from abc import ABC, abstractmethod
from typing import List

from qsprpred.models.interfaces import QSPRModel


class ModelPlot(ABC):
    """Base class for all model plots.

    Attributes:
        models (`list` of `QSPRModel`): list of models to plot
        modelOuts (`dict` of `QSPRModel` and `str`): dictionary mapping models to their output file paths
        modelNames (`dict` of `QSPRModel` and `str`): dictionary mapping models to their names
        cvPaths (`dict` of `QSPRModel` and `str`): dictionary mapping models to their cross-validation set results file paths
        indPaths (`dict` of `QSPRModel` and `str`): dictionary mapping models to their independent test set results file paths
    """

    def __init__(self, models: List[QSPRModel]):
        """Initialize the base class for all model plots.

        Args:
            models (`list` of `QSPRModel`): list of models to plot
        """
        self.models = models
        self.modelOuts = {model: model.outPrefix for model in self.models}
        self.modelNames = {model: model.name for model in self.models}
        self.cvPaths = dict()
        self.indPaths = dict()
        for model in self.models:
            cvPath, indPath = self.checkModel(model)
            self.cvPaths[model] = cvPath
            self.indPaths[model] = indPath

    def checkModel(self, model):
        """Check if the model has been evaluated and saved. If not, raise an exception.

        Args:
            model (`QSPRModel`): the model to check

        Returns:
            cvPath (`str`): path to the cross-validation set results file
            indPath (`str`): path to the independent test set results file
        """
        cvPath = f"{self.modelOuts[model]}.cv.tsv"
        indPath = f"{self.modelOuts[model]}.ind.tsv"
        if model.task not in self.getSupportedTasks():
            raise ValueError("Unsupported model type: %s" % model.task)
        if 'model_path' in model.metaInfo['model_path'] and not os.path.exists(model.metaInfo['model_path']):
            raise ValueError(
                "Model output file does not exist: %s. Have you evaluated and saved the model, yet?" %
                model.metaInfo['model_path'])
        if not os.path.exists(cvPath):
            raise ValueError(
                "Model output file does not exist: %s. Have you evaluated and saved the model, yet?" %
                cvPath)
        if not os.path.exists(indPath):
            raise ValueError(
                "Model output file does not exist: %s. Have you evaluated and saved the model, yet?" %
                indPath)

        return cvPath, indPath

    @abstractmethod
    def getSupportedTasks(self):
        """Get the types of models this plotter supports.

        Returns:
            `list` of `TargetTasks`: list of supported `TargetTasks`
        """
        pass

    @abstractmethod
    def make(self, save: bool = True, show: bool = False):
        """Make the plot. Opens a window to show the plot or returns a plot representation that can be directly shown in a notebook or saved to a file.

        Args:
            save (`bool`): if `True` the plot will be saved to a file. If `False` the plot will be shown in a window.
            show (`bool`): if `True` the plot will be shown in a window. If `False` the plot will be saved to a file.

        Returns:
            plot_instance `object` : representation of the plot
        """
        pass
