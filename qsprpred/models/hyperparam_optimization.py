"""Module for hyper parameter optimization of QSPRModels."""
from datetime import datetime
from typing import Callable, Union

import numpy as np
from qsprpred.logs import logger
from qsprpred.models.interfaces import HyperParameterOptimization, QSPRModel
from sklearn.model_selection import ParameterGrid


class OptunaOptimization(HyperParameterOptimization):
    """Class for hyper parameter optimization of QSPRModels using Optuna.
    
    Args:
        score_func (Callable): qsprpred.metrics.score_func
        param_grid (dict): dictionary with parameter names as keys and lists of parameter settings to try as values
        n_trials (int): number of trials for bayes optimization
        best_score (float): best score of the optimization
        best_params (dict): best parameters of the optimization
        
    Example of OptunaOptimization for scikit-learn's MLPClassifier:
        >>> model = QSPRsklearn(base_dir='.', data=dataset,
        >>>                     alg = MLPClassifier(), alg_name="MLP")
        >>> search_space = {
        >>>    'learning_rate_init': ['float', 1e-5, 1e-3,],
        >>>    'power_t' : ['discrete_uniform', 0.2, 0.8, 0.1],
        >>>    'momentum': ['float', 0.0, 1.0],
        >>> }
        >>> optimizer = OptunaOptimization(scoring='average_precision', param_grid=search_space, n_trials=10)
        >>> best_params = optimizer.optimize(model)
        
    Available suggestion types:
        ['categorical', 'discrete_uniform', 'float', 'int', 'loguniform', 'uniform']
    """

    def __init__(self, scoring: Union[str, Callable], param_grid: dict, n_trials: int = 100, n_jobs: int = 1):
        """Initialize the class for hyper parameter optimization of QSPRModels using Optuna.

        Args:
            model (`QSPRModel`): the model to optimize
            scoring (`Optional[str, Callable]`): scoring function for the optimization.
            param_grid (`dict`): search space for bayesian optimization, keys are the parameter names, 
                                 values are lists with first element the type of the parameter and the following
                                 elements the parameter bounds or values.
            n_trials (`int`): number of trials for bayes optimization
            n_jobs (`int`): number of jobs to run in parallel. At the moment only n_jobs=1 is supported.
        """
        super().__init__(scoring, param_grid)
        self.n_trials = n_trials
        if n_jobs > 1:
            logger.warning("At the moment n_jobs>1 not available for bayes optimization. n_jobs set to 1")
        self.n_jobs = 1
        self.best_score = -np.inf
        self.best_params = None

    def optimize(self, model: QSPRModel, save_params=True):
        """Bayesian optimization of hyperparameters using optuna.

        Args:
            model (`QSPRModel`): the model to optimize
            save_params (`bool`): whether to set and save the best parameters to the model after optimization
        """
        import optuna
        logger.info('Bayesian optimization can take a while for some hyperparameter combinations')

        study = optuna.create_study(direction='maximize')
        logger.info('Bayesian optimization started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        study.optimize(lambda trial: self.objective(trial, model), self.n_trials, n_jobs=self.n_jobs)
        logger.info('Bayesian optimization ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        trial = study.best_trial

        logger.info('Bayesian optimization best params: %s' % trial.params)

        self.best_score = trial.value
        self.best_params = trial.params

        if save_params:
            model.saveParams(self.best_params)
        
        return self.best_params
    
    def objective(self, trial, model: QSPRModel):
        """Objective for bayesian optimization.

        Arguments:
            trial (`optuna.trial.Trial`): trial object for the optimization
            model (`QSPRModel`): the model to optimize
        """
        bayesian_params = {}

        for key, value in self.param_grid.items():
            if value[0] == 'categorical':
                bayesian_params[key] = trial.suggest_categorical(key, value[1])
            elif value[0] == 'discrete_uniform':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2], step=value[3])
            elif value[0] == 'float':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2])
            elif value[0] == 'int':
                bayesian_params[key] = trial.suggest_int(key, value[1], value[2])
            elif value[0] == 'loguniform':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2], log=True)
            elif value[0] == 'uniform':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2])

        y, y_ind = model.data.getTargetPropertiesValues()
        score = self.score_func(y, model.evaluate(save=False, parameters=bayesian_params, score_func=self.score_func))
        return score


class GridSearchOptimization(HyperParameterOptimization):
    """Class for hyperparameter optimization of QSPRModels using GridSearch.

    Args:
        score_func (Callable): qsprpred.metrics.score_func
        param_grid (dict): dictionary with parameter names as keys and lists of parameter settings to try as values
    """

    def __init__(self, scoring: Union[str, Callable], param_grid: dict):
        """Initialize the class.

        Args:
        scoring (Union[str, Callable]): metric name from sklearn.metrics or user-defined scoring function.
        param_grid (dict): dictionary with parameter names as keys and lists of parameter settings to try as values
        """
        super().__init__(scoring, param_grid)

    def optimize(self, model: QSPRModel, save_params=True):
        """Optimize the hyperparameters of the model.

        Args:
            model (`QSPRModel`): the model to optimize
            save_params (bool): whether to set and save the best parameters to the model after optimization
        """
        logger.info('Grid search started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        for params in ParameterGrid(self.param_grid):
            logger.info(params)

            # do 5 fold cross validation and take mean prediction on validation set as score of parameter settings
            y, y_ind = model.data.getTargetPropertiesValues()
            score = self.score_func(y, model.evaluate(save=False, parameters=params, score_func=self.score_func))
            logger.info('Score: %s' % score)
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
        
        logger.info('Grid search ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logger.info('Grid search best params: %s with score: %s' % (self.best_params, self.best_score))
        
        if save_params:
            model.saveParams(self.best_params)

        return self.best_params
