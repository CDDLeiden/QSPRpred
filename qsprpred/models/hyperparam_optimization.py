"""Module for hyperparameter optimization of QSPRModels."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Iterable

import numpy as np
import optuna.trial
from sklearn.model_selection import ParameterGrid

from qsprpred.models.assessment.methods import ModelAssessor
from ..data import QSPRDataset
from ..logs import logger
from ..models.model import QSPRModel
from ..models.monitors import BaseMonitor, HyperparameterOptimizationMonitor


class HyperparameterOptimization(ABC):
    """Base class for hyperparameter optimization.

    Attributes:
        runAssessment (ModelAssessor): evaluation method to use
        scoreAggregation (Callable[[Iterable], float]): function to aggregate scores
        paramGrid (dict): dictionary of parameters to optimize
        monitor (HyperparameterOptimizationMonitor): monitor to track the optimization
        bestScore (float): best score found during optimization
        bestParams (dict): best parameters found during optimization
    """

    def __init__(
        self,
        param_grid: dict,
        model_assessor: ModelAssessor,
        score_aggregation: Callable[[Iterable], float],
        monitor: HyperparameterOptimizationMonitor | None = None,
    ):
        """Initialize the hyperparameter optimization class.

        param_grid (dict):
            dictionary of parameters to optimize
        model_assessor (ModelAssessor):
            assessment method to use for determining the best parameters
        score_aggregation (Callable[[Iterable], float]): function to aggregate scores
        monitor (HyperparameterOptimizationMonitor): monitor to track the optimization,
            if None, a BaseMonitor is used
        """
        self.runAssessment = model_assessor
        self.scoreAggregation = score_aggregation
        self.paramGrid = param_grid
        self.bestScore = -np.inf
        self.bestParams = None
        self.monitor = monitor
        self.config = {
            "param_grid": param_grid,
            "model_assessor": model_assessor,
            "score_aggregation": score_aggregation,
        }

    @abstractmethod
    def optimize(
        self, model: QSPRModel, ds: QSPRDataset, refit_optimal: bool = False
    ) -> dict:
        """Optimize the model hyperparameters.

        Args:
            model (QSPRModel):
                model to optimize
            ds (QSPRDataset):
                dataset to use for the optimization
            refit_optimal (bool):
                whether to refit the model with the optimal parameters
                on the entire training set after optimization
        Returns:
            dict: dictionary of best parameters
        """

    def saveResults(
        self, model: QSPRModel, ds: QSPRDataset, save_params: bool, refit_optimal: bool
    ):
        """Handles saving of optimization results.

        Args:
            model (QSPRModel):
                model that was optimized
            ds (QSPRDataset):
                dataset used in the optimization
            save_params (bool):
                whether to re-initialize the model with the best parameters
            refit_optimal (bool):
                same as 'save_params', but also refits
                the model on the entire training set
        """
        if save_params:
            model.setParams(self.bestParams, reset_estimator=True)
            model.save()
        if refit_optimal:
            model.setParams(self.bestParams)
            model.fit(ds.getFeatures()[0], ds.getTargetPropertiesValues()[0])
            model.save()


class OptunaOptimization(HyperparameterOptimization):
    """Class for hyperparameter optimization of QSPRModels using Optuna.

    Attributes:
        nTrials (int):
            number of trials for bayes optimization
        nJobs (int):
            number of jobs to run in parallel. At the moment only n_jobs=1 is supported.
        bestScore (float):
            best score found during optimization
        bestParams (dict):
            best parameters found during optimization

    Example of OptunaOptimization for scikit-learn's MLPClassifier:
        >>> model = SklearnModel(base_dir=".",
        >>>                     alg = MLPClassifier(), alg_name="MLP")
        >>> search_space = {
        >>>    "learning_rate_init": ["float", 1e-5, 1e-3,],
        >>>    "power_t" : ["discrete_uniform", 0.2, 0.8, 0.1],
        >>>    "momentum": ["float", 0.0, 1.0],
        >>> }
        >>> optimizer = OptunaOptimization(
        >>>     scoring="average_precision",
        >>>     param_grid=search_space,
        >>>     n_trials=10
        >>> )
        >>> best_params = optimizer.optimize(model, dataset) # dataset is a QSPRDataset

    Available suggestion types:
        ["categorical", "discrete_uniform", "float", "int", "loguniform", "uniform"]
    """

    def __init__(
        self,
        param_grid: dict,
        model_assessor: ModelAssessor,
        score_aggregation: Callable[[Iterable], float] = np.mean,
        monitor: HyperparameterOptimizationMonitor | None = None,
        n_trials: int = 100,
        n_jobs: int = 1,
    ):
        """Initialize the class for hyperparameter optimization
        of QSPRModels using Optuna.

        Args:
            param_grid (dict):
                search space for bayesian optimization, keys are the parameter names,
                values are lists with first element the type of the parameter and the
                following elements the parameter bounds or values.
            model_assessor (ModelAssessor):
                assessment method to use for the optimization
                (default: CrossValAssessor)
            score_aggregation (Callable):
                function to aggregate the scores of different folds if the assessment
                method returns multiple predictions
            monitor (HyperparameterOptimizationMonitor):
                monitor for the optimization, if None, a BaseMonitor is used
            n_trials (int):
                number of trials for bayes optimization
            n_jobs (int):
                number of jobs to run in parallel.
                At the moment only n_jobs=1 is supported.
        """
        super().__init__(param_grid, model_assessor, score_aggregation, monitor)
        if monitor is None:
            self.monitor = BaseMonitor()
        search_space_types = [
            "categorical",
            "discrete_uniform",
            "float",
            "int",
            "loguniform",
            "uniform",
        ]
        if not all(v[0] in search_space_types for v in param_grid.values()):
            logger.error(
                f"Search space {param_grid} is missing or has invalid search type(s), "
                "see OptunaOptimization docstring for example."
            )
            raise ValueError(
                "Search space for optuna optimization is missing or "
                "has invalid search type(s)."
            )

        self.nTrials = n_trials
        self.nJobs = n_jobs
        if self.nJobs > 1:
            logger.warning(
                "At the moment n_jobs>1 not available for bayes optimization, "
                "n_jobs set to 1."
            )
            self.nJobs = 1
        self.bestScore = -np.inf
        self.bestParams = None
        self.config.update(
            {
                "n_trials": n_trials,
                "n_jobs": n_jobs,
            }
        )

    def optimize(
        self,
        model: QSPRModel,
        ds: QSPRDataset,
        save_params: bool = True,
        refit_optimal: bool = False,
        **kwargs,
    ) -> dict:
        """Bayesian optimization of hyperparameters using optuna.

        Args:
            model (QSPRModel): the model to optimize
            ds (QSPRDataset): dataset to use for the optimization
            save_params (bool):
                whether to set and save the best parameters to the model
                after optimization
            refit_optimal (bool):
                Whether to refit the model with the optimal parameters on the
                entire training set after optimization. This implies 'save_params=True'.
            **kwargs: additional arguments for the assessment method

        Returns:
            dict: best parameters found during optimization
        """
        import optuna

        self.monitor.onOptimizationStart(
            model, ds, self.config, self.__class__.__name__
        )

        logger.info(
            "Bayesian optimization can take a while "
            "for some hyperparameter combinations"
        )
        # create optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=model.randomState),
        )
        logger.info(
            "Bayesian optimization started: %s"
            % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        study.optimize(
            lambda t: self.objective(t, model, ds), self.nTrials, n_jobs=self.nJobs
        )
        logger.info(
            "Bayesian optimization ended: %s"
            % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        # save the best study
        trial = study.best_trial
        # log the best study
        logger.info("Bayesian optimization best params: %s" % trial.params)
        # save the best score and parameters, return the best parameters
        self.bestScore = trial.value
        self.bestParams = trial.params

        self.monitor.onOptimizationEnd(self.bestScore, self.bestParams)
        # save the best parameters to the model if requested
        self.saveResults(model, ds, save_params, refit_optimal)
        return self.bestParams

    def objective(
        self, trial: optuna.trial.Trial, model: QSPRModel, ds: QSPRDataset, **kwargs
    ) -> float:
        """Objective for bayesian optimization.

        Arguments:
            trial (optuna.trial.Trial): trial object for the optimization
            model (QSPRModel): the model to optimize
            ds (QSPRDataset): dataset to use for the optimization
            **kwargs: additional arguments for the assessment method

        Returns:
            float: score of the model with the current parameters
        """
        bayesian_params = {}
        # get the suggested parameters for the current trial
        for key, value in self.paramGrid.items():
            if value[0] == "categorical":
                bayesian_params[key] = trial.suggest_categorical(key, value[1])
            elif value[0] == "discrete_uniform":
                bayesian_params[key] = trial.suggest_float(
                    key, value[1], value[2], step=value[3]
                )
            elif value[0] == "float":
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2])
            elif value[0] == "int":
                bayesian_params[key] = trial.suggest_int(key, value[1], value[2])
            elif value[0] == "loguniform":
                bayesian_params[key] = trial.suggest_float(
                    key, value[1], value[2], log=True
                )
            elif value[0] == "uniform":
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2])
        self.monitor.onIterationStart(bayesian_params)
        # assess the model with the current parameters and return the score
        scores = self.runAssessment(
            model,
            ds=ds,
            save=False,
            parameters=bayesian_params,
            monitor=self.monitor,
            **kwargs,
        )
        score = self.scoreAggregation(scores)
        logger.info(bayesian_params)
        logger.info(f"Score: {score}, std: {np.std(scores)}")
        self.monitor.onIterationEnd(score, list(scores))
        return score


class GridSearchOptimization(HyperparameterOptimization):
    """Class for hyperparameter optimization of QSPRModels using GridSearch."""

    def __init__(
        self,
        param_grid: dict,
        model_assessor: ModelAssessor,
        score_aggregation: Callable = np.mean,
        monitor: HyperparameterOptimizationMonitor | None = None,
    ):
        """Initialize the class.

        Args:
            param_grid (dict):
                dictionary with parameter names as keys and lists of parameter settings
                to try as values
            model_assessor (ModelAssessor):
                assessment method to use for the optimization
            score_aggregation (Callable):
                function to aggregate the scores of different folds if the assessment
                method returns multiple predictions (default: np.mean)
            monitor (HyperparameterOptimizationMonitor):
                monitor for the optimization, if None, a BaseMonitor is used
        """
        super().__init__(param_grid, model_assessor, score_aggregation, monitor)
        if monitor is None:
            self.monitor = BaseMonitor()

    def optimize(
        self,
        model: QSPRModel,
        ds: QSPRDataset,
        save_params: bool = True,
        refit_optimal: bool = False,
        **kwargs,
    ) -> dict:
        """Optimize the hyperparameters of the model.

        Args:
            model (QSPRModel):
                the model to optimize
            ds (QSPRDataset):
                dataset to use for the optimization
            save_params (bool):
                whether to set and save the best parameters to the model
                after optimization
            refit_optimal (bool):
                whether to refit the model with the optimal parameters on the
                entire training set after optimization. This implies 'save_params=True'.
            **kwargs: additional arguments for the assessment method

        Returns:
            dict: best parameters found during optimization
        """
        self.monitor.onOptimizationStart(
            model, ds, self.config, self.__class__.__name__
        )
        logger.info(
            "Grid search started: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        for params in ParameterGrid(self.paramGrid):
            self.monitor.onIterationStart(params)
            logger.info(params)
            scores = self.runAssessment(
                model, ds, save=False, parameters=params, monitor=self.monitor, **kwargs
            )
            score = self.scoreAggregation(scores)
            logger.info(f"Score: {score}, std: {np.std(scores)}")
            if score > self.bestScore:
                self.bestScore = score
                self.bestParams = params
            self.monitor.onIterationEnd(score, scores)
        # log some info and return the best parameters
        logger.info(
            "Grid search ended: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        logger.info(
            "Grid search best params: %s with score: %s"
            % (self.bestParams, self.bestScore)
        )
        # save the best parameters to the model if requested
        self.saveResults(model, ds, save_params, refit_optimal)
        self.monitor.onOptimizationEnd(self.bestScore, self.bestParams)
        return self.bestParams
