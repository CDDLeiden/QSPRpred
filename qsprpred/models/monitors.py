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
    def on_fit_start(self, model: QSPRModel):
        """Called before the training has started.

        Args:
            model (QSPRModel): model to train
        """

    def on_fit_end(self, model: QSPRModel):
        """Called after the training has finished.

        Args:
            model (QSPRModel): model to train
        """

    def on_epoch_start(self, epoch: int):
        """Called before each epoch of the training.

        Args:
            epoch (int): index of the current epoch
        """

    def on_epoch_end(self, epoch: int, score: float, predictions: np.ndarray):
        """Called after each epoch of the training.

        Args:
            epoch (int): index of the current epoch
            score (float): score of the current epoch
            predictions (np.ndarray): predictions of the current epoch
        """

    def on_batch_start(self, batch: int):
        """Called before each batch of the training.

        Args:
            batch (int): index of the current batch
        """

    def on_batch_end(self, batch: int, score: float, predictions: np.ndarray):
        """Called after each batch of the training.

        Args:
            batch (int): index of the current batch
            score (float): score of the current batch
            predictions (np.ndarray): predictions of the current batch
        """


class NullAssessorMonitor(AssessorMonitor, NullFitMonitor):
    """Null monitor that does nothing."""
    def on_assessment_start(self, model: QSPRModel):
        """Called before the assessment has started.

        Args:
            model (QSPRModel): model to assess
        """

    def on_assessment_end(self, model: QSPRModel):
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
    def on_optimization_start(self, model: QSPRModel, config: dict):
        """Called before the hyperparameter optimization has started.

        Args:
            model (QSPRModel): model to optimize
            config (dict): configuration of the hyperparameter optimization
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

    def on_iteration_end(
        self, score: float, scores: list[float], predictions: list[np.ndarray]
    ):
        """Called after each iteration of the hyperparameter optimization.

        Args:
            score (float): (aggregated) score of the current iteration
            scores (list[float]): scores of the current iteration
                                  (e.g for cross-validation)
            predictions (list[np.ndarray]): predictions of the current iteration
        """


class PrintMonitor(HyperParameterOptimizationMonitor):
    """Monitor that prints the progress of the hyperparameter optimization."""
    def on_optimization_start(self, model: QSPRModel, config: dict):
        """Called before the hyperparameter optimization has started.

        Args:
            model (QSPRModel): model to optimize
            config (dict): configuration of the hyperparameter optimization
        """
        print(f"Hyperparameter optimization started for {model.name}.")

    def on_optimization_end(self, best_score: float, best_parameters: dict):
        """Called after the hyperparameter optimization has finished.

        Args:
            best_score (float): best score found during optimization
            best_parameters (dict): best parameters found during optimization
        """
        print("Hyperparameter optimization finished.")
        print("Best score: %s" % best_score)
        print("Best parameters: %s" % best_parameters)

    def on_iteration_start(self, params: dict):
        """Called before each iteration of the hyperparameter optimization.

        Args:
            params (dict): parameters used for the current iteration
        """
        print("Iteration started.")
        print("Parameters: %s" % params)

    def on_iteration_end(
        self, score: float, scores: list[float], predictions: list[np.ndarray]
    ):
        """Called after each iteration of the hyperparameter optimization.

        Args:
            score (float): (aggregated) score of the current iteration
            scores (list[float]): scores of the current iteration
                                  (e.g for cross-validation)
            predictions (list[np.ndarray]): predictions of the current iteration
        """
        print("Iteration finished.")
        print("Score: %s" % score)
        print("Scores: %s" % scores)
        print("Predictions: %s" % predictions)


class WandBAssesmentMonitor(AssessorMonitor, NullFitMonitor):
    """Monitor assessment to weights and biases."""
    def __init__(self, project_name: str):
        try:
            import wandb
        except ImportError:
            raise ImportError("WandBMonitor requires wandb to be installed.")
        self.wandb = wandb

        wandb.login()

        self.project_name = project_name
        self.num_iterations = 0

    def on_assessment_start(self, model: QSPRModel):
        """Called before the assessment has started.

        Args:
            model (QSPRModel): model to assess
        """
        self.model = model

    def on_assessment_end(self, model: QSPRModel):
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
        config = {"fold": fold, "model": self.model.name}
        if hasattr(self, "params"):
            config.update(self.params)
        if hasattr(self, "num_iterations"):
            config["hyperParamOpt_iteration"] = self.num_iterations
        if hasattr(self, "wandb_runids"):
            new_runid = self.wandb.util.generate_id()
            self.wandb_runids.append(new_runid)

        group = (
            f"{self.model.name}_HyperParamOpt_{self.num_iterations}"
            if hasattr(self, "num_iterations") else f"{self.model.name}"
        )
        name = f"{group}_Assessment_{fold}"

        self.wandb.init(
            project=self.project_name,
            config=config,
            name=name,
            group=group,
            dir=f"{self.model.outDir}",
            id=new_runid if hasattr(self, "wandb_runids") else None,
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
        self.wandb.log(
            {
                "Test Results":
                    self.wandb.Table(
                        data=predictions.values, columns=list(predictions.columns)
                    )
            }
        )
        self.wandb.finish()


class WandBMonitor(HyperParameterOptimizationMonitor, WandBAssesmentMonitor):
    def __init__(self, project_name: str):
        try:
            import wandb
        except ImportError:
            raise ImportError("WandBMonitor requires wandb to be installed.")
        self.wandb = wandb

        wandb.login()

        self.project_name = project_name
        self.num_iterations = 0

    def on_optimization_start(self, model: QSPRModel, config: dict):
        """Called before the hyperparameter optimization has started.

        Args:
            config (dict): configuration of the hyperparameter optimization
        """
        self.model = model

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
        self.wandb_runids = []
        self.params = params

    def on_iteration_end(
        self, score: float, scores: list[float], predictions: list[np.ndarray]
    ):
        """Called after each iteration of the hyperparameter optimization.

        Args:
            score (float): (aggregated) score of the current iteration
            scores (list[float]): scores of the current iteration
                                (e.g for cross-validation)
            predictions (list[np.ndarray]): predictions of the current iteration
        """
        for i, runid in enumerate(self.wandb_runids):
            self.wandb.init(id=runid, resume="must", project=self.project_name)
            self.wandb.run.summary["Run scores"] = {
                "fold_score": scores[i],
                "aggregated_score": score,
            }
            self.wandb.finish()
        self.num_iterations += 1
        self.wandb_runids = []
