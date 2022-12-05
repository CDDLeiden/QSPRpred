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
        self.nFolds = {model : model.data.n_folds for model in self.models}
        self.modelOuts = {model: model.out for model in self.models}
        self.modelNames = {model: model.alg_name for model in self.models}
        self.cvPaths = dict()
        self.indPaths = dict()
        for model in self.models:
            cvPath, indPath = self.checkModel(model)
            self.cvPaths[model] = cvPath
            self.indPaths[model] = indPath

    def checkModel(self, model):
        cvPath = f"{self.modelOuts[model]}.cv.tsv"
        indPath = f"{self.modelOuts[model]}.ind.tsv"
        if model.type not in self.getSupportedTypes():
            raise ValueError("Unsupported model type: %s" % model.type)
        if not os.path.exists(f"{model.out}.json"):
            raise ValueError("Model output file does not exist: %s. Have you fitted the model, yet?" % model.out)
        if not os.path.exists(cvPath):
            raise ValueError("Model output file does not exist: %s. Have you fitted the model, yet?" % cvPath)
        if not os.path.exists(indPath):
            raise ValueError("Model output file does not exist: %s. Have you fitted the model, yet?" % indPath)

        return cvPath, indPath


    @abstractmethod
    def getSupportedTypes(self):
        """
        Get the types of models this plotter supports.

        Returns:
            `list` of `str`: list of supporteclad model types
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
