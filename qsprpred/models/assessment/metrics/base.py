from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """Abstract class for scoring functions.

    Attributes:
        name (str): Name of the scoring function.
    """

    @abstractmethod
    def __call__(
            self, y_true: np.ndarray, y_pred: np.ndarray | list[np.ndarray]
    ) -> float:
        """Calculate the score.

        Args:
            y_true (np.ndarray): True values. Must be of shape (n_samples, n_targets)
            y_pred (np.ndarray | list[np.ndarray]): Predicted values. Shape
                (n_samples, n_tasks) for regression or discrete class predictions.
                List of arrays of shape (n_samples, n_classes) of length n_tasks for
                class probability predictions.

        Returns:
            float: Score of the predictions.
        """

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return self.name
