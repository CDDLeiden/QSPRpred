"""Wrapper for sklearn scoring functions."""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _BaseScorer


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


class SklearnMetrics(Metric):
    """Wrapper for sklearn scoring functions.

    Attributes:
        name (str): Name of the scoring function.
        scorer: Sklearn scorer object.
    """

    def __init__(self, scorer: str | _BaseScorer):
        """Initialize the scoring function.

        Args:
            scorer (str | _BaseScorer): Name of the scoring function or sklearn scorer
        """
        if isinstance(scorer, str):
            scorer = get_scorer(scorer)
        self.scorer = scorer  # Sklearn scorer object
        self.name = scorer._score_func.__name__  # Name of the scorer

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray | list[np.ndarray]
    ):
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
        # Convert predictions to correct shape for sklearn scorer
        if isinstance(y_pred, list):
            if self.scorer.__class__.__name__ == "_PredictScorer":
                # convert to discrete values
                y_pred = np.transpose([np.argmax(y_pred, axis=1) for y_pred in y_pred])
            else:
                # for each task if single class take second column
                y_pred = [
                    y_pred[:, 1].reshape(-1, 1) if y_pred.shape[1] == 2 else y_pred
                    for y_pred in y_pred
                ]

            if len(y_pred) > 1:
                # if multi-task concatenate arrays
                y_pred = np.concatenate(y_pred, axis=1)
            else:
                # if single class classification, convert to 1d array
                y_pred = y_pred[0]

        return self._scorerFunc(y_true, y_pred)

    def _scorerFunc(self, y_true, y_pred):
        """Return the scoring function of a sklearn scorer."""
        # From https://stackoverflow.com/questions/63943410/getting-a-scoring-function-by-name-in-scikit-learn
        return self.scorer._sign * self.scorer._score_func(y_true, y_pred, **self.scorer._kwargs)
