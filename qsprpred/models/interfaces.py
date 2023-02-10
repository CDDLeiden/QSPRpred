"""This module holds the base class for QSPRmodels, model types should be a subclass."""
import json
import os
import shutil
import sys
from abc import ABC, abstractmethod
from inspect import isclass
from typing import Union, Type

import numpy as np
import pandas as pd

from qsprpred.data.data import QSPRDataset, MoleculeTable
from qsprpred.data.utils.descriptorcalculator import DescriptorsCalculator
from qsprpred.data.utils.feature_standardization import SKLearnStandardizer
from qsprpred.logs import logger
from qsprpred.models import SSPACE
from qsprpred.models.tasks import ModelTasks


class QSPRModel(ABC):
    """Model initialization, fit, cross validation and hyperparameter optimization for classifion/regression models.

    Attributes:
        data: instance QSPRDataset
        alg:  instance of estimator
        parameters (dict): dictionary of algorithm specific parameters
        njobs (int): the number of parallel jobs to run

    Methods:
        init_model: initialize model from saved hyperparameters
        fit: build estimator model from entire data set
        objective: objective used by bayesian optimization
        bayesOptimization: bayesian optimization of hyperparameters using optuna
        gridSearch: optimization of hyperparameters using gridSearch
    """

    def __init__(self, base_dir : str, alg = None, data : QSPRDataset = None, name : str =None, parameters : dict = None, autoload=True):
        """Initialize model from saved or default hyperparameters."""
        self.data = data
        self.name = name or alg.__class__.__name__
        self.baseDir = base_dir.rstrip('/')

        # load metadata and update parameters accordingly
        self.metaFile = f"{self.outPrefix}_meta.json"
        try:
            self.metaInfo = self.readMetadata(self.metaFile)
        except FileNotFoundError:
            if not self.data:
                raise FileNotFoundError(f"Metadata file '{self.metaFile}' not found")
            else:
                self.metaInfo = {}

        # get parameters from metadata if available
        self.parameters = parameters
        if self.metaInfo and not self.parameters:
            self.parameters = self.readParams(os.path.join(self.baseDir, self.metaInfo['parameters_path']))

        # initialize a feature calculator instance
        self.featureCalculator = self.data.descriptorCalculator if self.data else self.readDescriptorCalculator(os.path.join(self.baseDir, self.metaInfo['feature_calculator_path']))

        # initialize a standardizer instance
        self.featureStandardizer = self.data.feature_standardizer if self.data else self.readStandardizer(os.path.join(self.baseDir, self.metaInfo['feature_standardizer_path']))
        if not isinstance(self.featureStandardizer, SKLearnStandardizer):
            self.featureStandardizer = SKLearnStandardizer(self.featureStandardizer)

        # initialize a model instance with the given parameters
        self.alg = alg
        if autoload:
            self.model = self.loadModel(alg=self.alg, params=self.parameters)

    def __str__(self):
        return self.name

    @property
    def task(self):
        return self.data.task if self.data else self.metaInfo['task']

    @property
    def targetProperty(self):
        return self.data.targetProperty if self.data else self.metaInfo['target_property']
    @property
    def outDir(self):
        os.makedirs(f'{self.baseDir}/qspr/models/{self.name}', exist_ok=True)
        return f'{self.baseDir}/qspr/models/{self.name}'

    @property
    def outPrefix(self):
        return f'{self.outDir}/{self.name}'

    @staticmethod
    def readParams(path):
        with open(path, "r", encoding="utf-8") as j:
            logger.info(
                'loading model parameters from file: %s' % path)
            return json.loads(j.read())

    def saveParams(self, params):
        path = f'{self.outDir}/{self.name}_params.json'
        with open(path, "w", encoding="utf-8") as j:
            logger.info(
                'saving model parameters to file: %s' % path)
            j.write(json.dumps(params, indent=4))

        return path

    @staticmethod
    def readDescriptorCalculator(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Descriptor calculator file "{path}" does not exist.')
        return DescriptorsCalculator.fromFile(path)

    def saveDescriptorCalculator(self):
        path = f'{self.outDir}/{self.name}_descriptor_calculator.json'
        self.featureCalculator.toFile(path)
        return path

    @staticmethod
    def readStandardizer(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'Standardizer file "{path}" does not exist.')
        return SKLearnStandardizer.fromFile(path)

    def saveStandardizer(self):
        path = f'{self.outDir}/{self.name}_feature_standardizer.json'
        self.featureStandardizer.toFile(path)
        return path

    @classmethod
    def readMetadata(cls, path):
        if os.path.exists(path):
            with open(path) as j:
                metaInfo = json.loads(j.read())
                metaInfo['task'] = ModelTasks(metaInfo['task'])
        else:
            raise FileNotFoundError(f'Metadata file "{path}" does not exist.')

        return metaInfo

    def saveMetadata(self):
        with open(self.metaFile, "w", encoding="utf-8") as j:
            logger.info(
                'saving model metadata to file: %s' % self.metaFile)
            j.write(json.dumps(self.metaInfo, indent=4))

        return self.metaFile

    def save(self):
        self.metaInfo['name'] = self.name
        self.metaInfo['task'] = str(self.task)
        self.metaInfo['th'] = self.data.th
        self.metaInfo['target_property'] = self.targetProperty
        self.metaInfo['parameters_path'] = self.saveParams(self.parameters).replace(f"{self.baseDir}/", '')
        self.metaInfo['feature_calculator_path'] = self.saveDescriptorCalculator().replace(f"{self.baseDir}/", '')
        self.metaInfo['feature_standardizer_path'] = self.saveStandardizer().replace(f"{self.baseDir}/", '')
        self.metaInfo['model_path'] = self.saveModel().replace(f"{self.baseDir}/", '')
        return self.saveMetadata()

    def checkForData(self, exception=True):
        hasData = self.data is not None
        if exception and not hasData:
            raise ValueError('No data set specified. Make sure you initialized this model with a "QSPRDataset" instance to train on.')

        return hasData

    @abstractmethod
    def fit(self):
        """Build estimator model from entire data set."""
        pass

    @abstractmethod
    def evaluate(self, save=True):
        """Make predictions for crossvalidation and independent test set.

        Arguments:
            save (bool): don't save predictions when used in bayesian optimization
        """
        pass

    @abstractmethod
    def gridSearch(self):
        """Optimization of hyperparameters using gridSearch.

        Arguments:
            search_space_gs (dict): search space for the grid search
            save_m (bool): if true, after gs the model is refit on the entire data set
        """
        pass

    @abstractmethod
    def bayesOptimization(self):
        """Bayesian optimization of hyperparameters using optuna.

        Arguments:
            search_space_gs (dict): search space for the grid search
            n_trials (int): number of trials for bayes optimization
            save_m (bool): if true, after bayes optimization the model is refit on the entire data set
        """
        pass

    @staticmethod
    def loadParamsGrid(fname, optim_type, model_types):
        """Load parameter grids for bayes or grid search parameter optimization from json file.

        Arguments:
            fname (str): file name of json file containing array with three columns containing modeltype,
                            optimization type (grid or bayes) and model type
            optim_type (str): optimization type ('grid' or 'bayes')
            model_types (list of str): model type for hyperparameter optimization (e.g. RF)
        """
        if fname:
            try:
                with open(fname) as json_file:
                    optim_params = np.array(json.load(json_file), dtype=object)
            except FileNotFoundError:
                logger.error("Search space file (%s) not found" % fname)
                sys.exit()
        else:
            with open(SSPACE) as json_file:
                optim_params = np.array(json.load(json_file), dtype=object)

        # select either grid or bayes optimization parameters from param array
        optim_params = optim_params[optim_params[:, 2] == optim_type, :]

        # check all modeltypes to be used have parameter grid
        model_types = [model_types] if isinstance(
            model_types, str) else model_types

        if not set(list(model_types)).issubset(list(optim_params[:, 0])):
            logger.error("model types %s missing from models in search space dict (%s)" % (
                model_types, optim_params[:, 0]))
            sys.exit()
        logger.info("search space loaded from file")

        return optim_params

    @abstractmethod
    def loadModel(self, alg : Union[Type, object] = None, params : dict = None):
        """

        Initialize model instance with the given parameters. If no algorithm is given, the model is loaded from file based on available metadata. If no parameters are given, they are also loaded from the available file.

        Arguments:
            alg (object): algorithm class to instantiate
            params (dict): algorithm parameters
        """
        pass

    @abstractmethod
    def saveModel(self) -> str:
        """
        Save the underlying model to file.

        Returns:
            str: path to the saved model
        """
        pass

    @abstractmethod
    def predict(self, X : Union[pd.DataFrame, np.ndarray, QSPRDataset]):
        pass

    @abstractmethod
    def predictProba(self, X : Union[pd.DataFrame, np.ndarray, QSPRDataset]):
        pass

    def predictMols(self, mols, use_probas=False):
        dataset = MoleculeTable.fromSMILES(f"{self.__class__.__name__}_{hash(self)}", mols, drop_invalids=False)
        dataset.addProperty(self.targetProperty, np.nan)
        dataset = QSPRDataset.fromMolTable(dataset, self.targetProperty, drop_empty=False, drop_invalids=False)
        failed_mask = dataset.dropInvalids().to_list()
        failed_indices = [idx for idx,x in enumerate(failed_mask) if not x]
        if not self.featureCalculator:
            raise ValueError("No feature calculator set on this instance.")
        dataset.prepareDataset(
            standardize=True,
            sanitize=True,
            feature_calculator=self.featureCalculator,
            feature_standardizer=self.featureStandardizer
        )
        if self.task == ModelTasks.REGRESSION or not use_probas:
            predictions = self.predict(dataset)
            if (isclass(self.alg) and self.alg.__name__ == 'PLSRegression') or (type(self.alg).__name__ == 'PLSRegression'):
                predictions = predictions[:, 0]
        else:
            predictions = self.predictProba(dataset)

        if failed_indices:
            predictions = list(predictions)
            ret = []
            for idx, pred in enumerate(mols):
                if idx in failed_indices:
                    ret.append(None)
                else:
                    ret.append(predictions.pop(0))
            return np.array(ret)
        else:
            return predictions

    @classmethod
    def fromFile(cls, path):
        name = cls.readMetadata(path)['name']
        dir_name = os.path.dirname(path).replace(f"qspr/models/{name}", "")
        return cls(name=name, base_dir=dir_name)

    def cleanFiles(self):
        if os.path.exists(self.outDir):
            shutil.rmtree(self.outDir)
