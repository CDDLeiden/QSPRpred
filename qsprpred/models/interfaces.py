"""This module holds the base class for QSPRmodels, model types should be a subclass."""
import json
import os
import sys
from abc import ABC, abstractmethod

import numpy as np
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

    def __init__(self, base_dir, data, alg, alg_name, parameters={}):
        """Initialize model from saved or default hyperparameters."""
        self.data = data
        self.alg = alg
        self.parameters = parameters
        self.alg_name = alg_name

        d = '%s/qspr/models' % base_dir
        self.type = 'REG' if data.targetProperties[0].task == ModelTasks.REGRESSION else 'CLS'
        self.out = '%s/%s_%s_%s' % (d, alg_name,
                                    self.type, data.targetProperties[0].name)

        if os.path.isfile('%s_params.json' % self.out):
            with open('%s_params.json' % self.out) as j:
                if self.parameters:
                    self.parameters = json.loads(
                        j.read()).update(
                        self.parameters)
                else:
                    self.parameters = json.loads(j.read())
            logger.info(
                'loaded model parameters from file: %s_params.json' %
                self.out)

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
