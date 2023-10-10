"""Module for hyperparameter optimization of QSPRModels."""

from datetime import datetime
from typing import Callable, Iterable

import numpy as np
import optuna.trial
from sklearn.model_selection import ParameterGrid

from ..logs import logger
from ..models.assessment_methods import CrossValAssessor
from ..models.interfaces import (
    HyperParameterOptimization,
    HyperParameterOptimizationMonitor,
    ModelAssessor,
    QSPRModel,
)
from ..models.monitors import NullMonitor


class OptunaOptimization(HyperParameterOptimization):
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
        >>> model = SklearnModel(base_dir=".", data=dataset,
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
        >>> best_params = optimizer.optimize(model)

    Available suggestion types:
        ["categorical", "discrete_uniform", "float", "int", "loguniform", "uniform"]
    """
    def __init__(
        self,
        scoring: str | Callable[[Iterable, Iterable], float],
        param_grid: dict,
        model_assessor: ModelAssessor = CrossValAssessor(),
        score_aggregation: Callable[[Iterable], float] = np.mean,
        monitor: HyperParameterOptimizationMonitor = NullMonitor(),
        n_trials: int = 100,
        n_jobs: int = 1,
    ):
        """Initialize the class for hyperparameter optimization
        of QSPRModels using Optuna.

        Args:
            scoring (str | Callable[[Iterable, Iterable], float]]):
                scoring function for the optimization.
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
            n_trials (int):
                number of trials for bayes optimization
            n_jobs (int):
                number of jobs to run in parallel.
                At the moment only n_jobs=1 is supported.
        """
        super().__init__(
            scoring, param_grid, model_assessor, score_aggregation, monitor
        )
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
        if n_jobs > 1:
            logger.warning(
                "At the moment n_jobs>1 not available for bayes optimization, "
                "n_jobs set to 1."
            )
        self.nJobs = 1
        self.bestScore = -np.inf
        self.bestParams = None
        self.config = {
            "scoring": scoring,
            "param_grid": param_grid,
            "model_assessor": model_assessor,
            "score_aggregation": score_aggregation,
            "n_trials": n_trials,
            "n_jobs": n_jobs,
        }

    def optimize(self, model: QSPRModel, **kwargs) -> dict:
        """Bayesian optimization of hyperparameters using optuna.

        Args:
            model (QSPRModel): the model to optimize
            **kwargs: additional arguments for the assessment method

        Returns:
            dict: best parameters found during optimization
        """
        import optuna

        self.monitor.on_optimization_start(model, self.config, self.__class__.__name__)
        self.scoreFunc.checkMetricCompatibility(model.task, self.runAssessment.useProba)

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
            "Bayesian optimization started: %s" %
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        study.optimize(
            lambda t: self.objective(t, model), self.nTrials, n_jobs=self.nJobs
        )
        logger.info(
            "Bayesian optimization ended: %s" %
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        # save the best study
        trial = study.best_trial
        # log the best study
        logger.info("Bayesian optimization best params: %s" % trial.params)
        # save the best score and parameters, return the best parameters
        self.bestScore = trial.value
        self.bestParams = trial.params

        self.monitor.on_optimization_end(self.bestScore, self.bestParams)
        return self.bestParams

    def objective(self, trial: optuna.trial.Trial, model: QSPRModel, **kwargs) -> float:
        """Objective for bayesian optimization.

        Arguments:
            trial (optuna.trial.Trial): trial object for the optimization
            model (QSPRModel): the model to optimize
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
        self.monitor.on_iteration_start(bayesian_params)
        # assess the model with the current parameters and return the score
        predictions = self.runAssessment(
            model,
            save=False,
            parameters=bayesian_params,
            monitor=self.monitor,
            **kwargs,
        )
        scores = []
        # TODO: this should be removed once random seeds are fixed
        for pred in predictions:
            if model.task.isClassification():
                # check if more than 1 class in y_true
                if not len(np.unique(pred[0])) > 1:
                    logger.warning("Only 1 class in y_true, skipping fold.")
            scores.append(self.scoreFunc(*pred))
        assert len(scores) > 0, "No scores calculated, all folds skipped."

        score = self.scoreAggregation(scores)
        logger.info(bayesian_params)
        logger.info(f"Score: {score}, std: {np.std(scores)}")
        self.monitor.on_iteration_end(score, scores, predictions)
        return score


class GridSearchOptimization(HyperParameterOptimization):
    """Class for hyperparameter optimization of QSPRModels using GridSearch."""
    def __init__(
        self,
        scoring: str | Callable[[Iterable, Iterable], float],
        param_grid: dict,
        model_assessor: ModelAssessor = CrossValAssessor(),
        monitor: HyperParameterOptimizationMonitor = NullMonitor(),
        score_aggregation: Callable = np.mean,
    ):
        """Initialize the class.

        Args:
            scoring (Union[str, Callable[[Iterable, Iterable], Iterable]]):
                metric name from sklearn.metrics or user-defined scoring function.
            param_grid (dict):
                dictionary with parameter names as keys and lists of parameter settings
                to try as values
            model_assessor (ModelAssessor):
                assessment method to use for the optimization
            monitor (HyperParameterOptimizationMonitor):
                monitor for the optimization
            score_aggregation (Callable):
                function to aggregate the scores of different folds if the assessment
                method returns multiple predictions (default: np.mean)
        """
        super().__init__(
            scoring, param_grid, model_assessor, score_aggregation, monitor
        )

    def optimize(self, model: QSPRModel, save_params: bool = True, **kwargs) -> dict:
        """Optimize the hyperparameters of the model.

        Args:
            model (QSPRModel):
                the model to optimize
            save_params (bool):
                whether to set and save the best parameters to the model
                after optimization
            **kwargs: additional arguments for the assessment method

        Returns:
            dict: best parameters found during optimization
        """
        self.scoreFunc.checkMetricCompatibility(model.task, self.runAssessment.useProba)
        logger.info(
            "Grid search started: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        for params in ParameterGrid(self.paramGrid):
            logger.info(params)
            predictions = self.runAssessment(
                model, save=False, parameters=params, monitor=self.monitor, **kwargs
            )
            scores = []
            # TODO: this should be removed once random seeds are fixed
            for pred in predictions:
                if model.task.isClassification():
                    # check if more than 1 class in y_true
                    if not len(np.unique(pred[0])) > 1:
                        logger.warning("Only 1 class in y_true, skipping fold.")
                scores.append(self.scoreFunc(*pred))
            assert len(scores) > 0, "No scores calculated, all folds skipped."
            # scores = [self.scoreFunc(*pred) for pred in predictions]
            score = self.scoreAggregation(scores)
            logger.info(f"Score: {score}, std: {np.std(scores)}")
            if score > self.bestScore:
                self.bestScore = score
                self.bestParams = params
        # log some info and return the best parameters
        logger.info(
            "Grid search ended: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        logger.info(
            "Grid search best params: %s with score: %s" %
            (self.bestParams, self.bestScore)
        )
        # save the best parameters to the model if requested
        if save_params:
            model.saveParams(self.bestParams)
        return self.bestParams
