from typing import Any

import numpy as np

from .interfaces import (
    AssessorMonitor,
    FitMonitor,
    HyperParameterOptimizationMonitor,
    QSPRModel,
)


class NullFitMonitor(FitMonitor):
    """Null monitor that does nothing."""
    def on_fit_start(self, estimator: Any):
        """Called before the training has started.

        Args:
            estimator (Any): estimator to train
        """

    def on_fit_end(self, estimator: Any, best_epoch: int | None = None):
        """Called after the training has finished.

        Args:
            estimator (Any): estimator that was fitted
            best_epoch (int | None): index of the best epoch
        """

    def on_epoch_start(self, epoch: int):
        """Called before each epoch of the training.

        Args:
            epoch (int): index of the current epoch
        """

    def on_epoch_end(
        self, epoch: int, train_loss: float, val_loss: float | None = None
    ):
        """Called after each epoch of the training.

        Args:
            epoch (int): index of the current epoch
            train_loss (float): loss of the current epoch
            val_loss (float | None): validation loss of the current epoch
        """

    def on_batch_start(self, batch: int):
        """Called before each batch of the training.

        Args:
            batch (int): index of the current batch
        """

    def on_batch_end(self, batch: int, loss: float, predictions: np.ndarray):
        """Called after each batch of the training.

        Args:
            batch (int): index of the current batch
            loss (float): loss of the current batch
            predictions (np.ndarray): predictions of the current batch
        """


class NullAssessorMonitor(AssessorMonitor, NullFitMonitor):
    """Null monitor that does nothing."""
    def on_assessment_start(self, model: QSPRModel, assesment_type: str):
        """Called before the assessment has started.

        Args:
            model (QSPRModel): model to assess
            assesment_type (str): type of assessment
        """

    def on_assessment_end(self):
        """Called after the assessment has finished.

        Args:
            model (QSPRModel): model to assess
        """

    def on_fold_start(
        self,
        fold: int,
        X_train: np.array,
        y_train: np.array,
        X_test: np.array,
        y_test: np.array,
    ):
        """Called before each fold of the assessment.

        Args:
            fold (int): index of the current fold
            X_train (np.array): training data of the current fold
            y_train (np.array): training targets of the current fold
            X_test (np.array): test data of the current fold
            y_test (np.array): test targets of the current fold
        """

    def on_fold_end(
        self, fitted_estimator: Any | tuple[Any, int], predictions: np.ndarray
    ):
        """Called after each fold of the assessment.

        Args:
            fitted_estimator (Any |tuple[Any, int]):
                fitted estimator of the current fold
            predictions (np.ndarray):
                predictions of the current fold
        """


class NullMonitor(HyperParameterOptimizationMonitor, NullAssessorMonitor):
    """Null monitor that does nothing."""
    def on_optimization_start(
        self, model: QSPRModel, config: dict, optimization_type: str
    ):
        """Called before the hyperparameter optimization has started.

        Args:
            config (dict): configuration of the hyperparameter optimization
            optimization_type (str): type of optimization
        """

    def on_optimization_end(self, best_score: float, best_parameters: dict):
        """Called after the hyperparameter optimization has finished.

        Args:
            best_score (float): best score found during optimization
            best_parameters (dict): best parameters found during optimization
        """

    def on_iteration_start(self, params: dict):
        """Called before each iteration of the hyperparameter optimization.

        Args:
            params (dict): parameters used for the current iteration
        """

    def on_iteration_end(self, score: float, scores: list[float]):
        """Called after each iteration of the hyperparameter optimization.

        Args:
            score (float): (aggregated) score of the current iteration
            scores (list[float]): scores of the current iteration
                                  (e.g for cross-validation)
        """

class WandBAssesmentMonitor(AssessorMonitor):
    """Monitor assessment and fit to weights and biases."""
    def __init__(self, project_name: str, **kwargs):
        """Monitor assessment to weights and biases.

        Args:
            project_name (str): name of the project to log to
            kwargs: additional keyword arguments for wandb.init
        """
        super().__init__()
        try:
            import wandb
        except ImportError:
            raise ImportError("WandBMonitor requires wandb to be installed.")
        self.wandb = wandb

        wandb.login()

        self.projectName = project_name
        self.kwargs = kwargs

    def on_assessment_start(self, model: QSPRModel, assesment_type: str):
        """Called before the assessment has started.

        Args:
            model (QSPRModel): model to assess
            assesment_type (str): type of assessment
        """
        self.model = model
        self.data = model.data
        self.assessmentType = assesment_type

    def on_assessment_end(self):
        """Called after the assessment has finished.

        Args:
            model (QSPRModel): model to assess
        """

    def on_fold_start(
        self,
        fold: int,
        X_train: np.array,
        y_train: np.array,
        X_test: np.array,
        y_test: np.array,
    ):
        """Called before each fold of the assessment.

        Args:
            fold (int): index of the current fold
            X_train (np.array): training data of the current fold
            y_train (np.array): training targets of the current fold
            X_test (np.array): test data of the current fold
            y_test (np.array): test targets of the current fold
        """
        config = {
            "fold": fold,
            "model": self.model.name,
            "assessmentType": self.assessmentType,
        }
        # add hyperparameter optimization parameters if available
        if hasattr(self, "optimizationType"):
            config["optimizationType"] = self.optimizationType
            config.update(self.params)
            config["hyperParamOpt_iteration"] = self.numIterations
        else:
            config["optimizationType"] = None

        group = (
            f"{self.model.name}_{self.optimizationType}_{self.numIterations}"
            if hasattr(self, "optimizationType") else f"{self.model.name}"
        )
        name = f"{group}_{self.assessmentType}_{fold}"

        self.wandb.init(
            project=self.projectName,
            config=config,
            name=name,
            group=group,
            dir=f"{self.model.outDir}",
            **self.kwargs,
        )
        self.fold = fold

    def on_fold_end(
        self, fitted_estimator: Any | tuple[Any, int], predictions: np.ndarray
    ):
        """Called after each fold of the assessment.

        Args:
            fitted_estimator (Any |tuple[Any, int]):
                fitted estimator of the current fold
            predictions (np.ndarray):
                predictions of the current fold
        """
        self.wandb.log({"Test Results" : self.wandb.Table(data=predictions)})
        self.wandb.finish()

    def on_fit_start(self, estimator: Any):
        """Called before the training has started.

        Args:
            estimator (Any): estimator to train
        """

    def on_fit_end(self, estimator: Any, best_epoch: int | None = None):
        """Called after the training has finished.

        Args:
            estimator (Any): estimator that was fitted
            best_epoch (int | None): index of the best epoch
        """
        self.wandb.log({"best_epoch": best_epoch})

    def on_epoch_start(self, epoch: int):
        """Called before each epoch of the training.

        Args:
            epoch (int): index of the current epoch
        """
        self.epochnr = epoch

    def on_epoch_end(
        self, epoch: int, train_loss: float, val_loss: float | None = None
    ):
        """Called after each epoch of the training.

        Args:
            epoch (int): index of the current epoch
            train_loss (float): loss of the current epoch
            val_loss (float | None): validation loss of the current epoch
        """
        self.wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    def on_batch_start(self, batch: int):
        """Called before each batch of the training.

        Args:
            batch (int): index of the current batch
        """

    def on_batch_end(self, batch: int, loss: float, predictions: np.ndarray):
        """Called after each batch of the training.

        Args:
            batch (int): index of the current batch
            loss (float): loss of the current batch
            predictions (np.ndarray): predictions of the current batch
        """
        self.wandb.log({"batch": batch, "loss": loss})


class WandBMonitor(WandBAssesmentMonitor, HyperParameterOptimizationMonitor):
    def __init__(self, project_name: str, **kwargs):
        """Monitor hyperparameter optimization to weights and biases.

        Args:
            project_name (str): name of the project to log to
            kwargs: additional keyword arguments for wandb.init
        """
        super().__init__(project_name, **kwargs)
        self.numIterations = 0

    def on_optimization_start(
        self, model: QSPRModel, config: dict, optimization_type: str
    ):
        """Called before the hyperparameter optimization has started.

        Args:
            config (dict): configuration of the hyperparameter optimization
            optimization_type (str): type of optimization
        """
        self.optimizationType = optimization_type
        self.model = model
        self.data = model.data
        self.config = config

    def on_optimization_end(self, best_score: float, best_parameters: dict):
        """Called after the hyperparameter optimization has finished.

        Args:
            best_score (float): best score found during optimization
            best_parameters (dict): best parameters found during optimization
        """
        self.bestScore = best_score
        self.bestParameters = best_parameters

    def on_iteration_start(self, params: dict):
        """Called before each iteration of the hyperparameter optimization.

        Args:
            params (dict): parameters used for the current iteration
        """
        self.params = params

    def on_iteration_end(self, score: float, scores: list[float]):
        """Called after each iteration of the hyperparameter optimization.

        Args:
            score (float): (aggregated) score of the current iteration
            scores (list[float]): scores of the current iteration
                                (e.g for cross-validation)
        """
        self.scores.loc[self.numIterations] = [score, scores]
        self.numIterations += 1
