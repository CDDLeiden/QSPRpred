"""
interfaces

Created by: Martin Sicho
On: 16.11.22, 12:02
"""
import os
from abc import ABC, abstractmethod
from typing import List

from qsprpred.models.interfaces import QSPRModel


class ModelPlot(ABC):

    def __init__(self, models : List[QSPRModel]):
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
        cvPath = f"{self.modelOuts[model]}.cv.tsv"
        indPath = f"{self.modelOuts[model]}.ind.tsv"
        if model.task not in self.getSupportedTasks():
            raise ValueError("Unsupported model type: %s" % model.task)
        if 'model_path' in model.metaInfo['model_path'] and not os.path.exists(model.metaInfo['model_path']):
            raise ValueError("Model output file does not exist: %s. Have you evaluated and saved the model, yet?" % model.metaInfo['model_path'])
        if not os.path.exists(cvPath):
            raise ValueError("Model output file does not exist: %s. Have you evaluated and saved the model, yet?" % cvPath)
        if not os.path.exists(indPath):
            raise ValueError("Model output file does not exist: %s. Have you evaluated and saved the model, yet?" % indPath)

        return cvPath, indPath


    @abstractmethod
    def getSupportedTasks(self):
        """
        Get the types of models this plotter supports.

        Returns:
            `list` of `ModelTasks`: list of supporteclad `ModelTasks`
        """
        pass

    @abstractmethod
    def make(self, save : bool = True, show : bool = False):
        """
        Make the plot. Opens a window to show the plot or returns a plot representation that can be directly shown in a notebook or saved to a file.

        Args:
            save (`bool`): if `True` the plot will be saved to a file. If `False` the plot will be shown in a window.
            show (`bool`): if `True` the plot will be shown in a window. If `False` the plot will be saved to a file.

        Returns:
            plot_instance `object` : representation of the plot
        """
        pass
