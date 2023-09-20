from .interfaces import HyperParameterOptimizationMonitor
import numpy as np

class NullMonitor(HyperParameterOptimizationMonitor):
    """Null monitor that does nothing."""

    def on_optimization_start(self, config: dict):
        """Called before the hyperparameter optimization has started.

        Args:
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

    def on_iteration_end(self, score: float, scores: list[float],
                         predictions: list[np.ndarray]):
        """Called after each iteration of the hyperparameter optimization.

        Args:
            score (float): (aggregated) score of the current iteration
            scores (list[float]): scores of the current iteration
                                  (e.g for cross-validation)
            predictions (list[np.ndarray]): predictions of the current iteration
        """

class PrintMonitor(HyperParameterOptimizationMonitor):
    """Monitor that prints the progress of the hyperparameter optimization."""

    def on_optimization_start(self, config: dict):
        """Called before the hyperparameter optimization has started.

        Args:
            config (dict): configuration of the hyperparameter optimization
        """
        print("Hyperparameter optimization started.")

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

    def on_iteration_end(self, score: float, scores: list[float],
                         predictions: list[np.ndarray]):
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
