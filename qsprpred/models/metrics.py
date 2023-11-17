"""Wrapper for sklearn scoring functions."""

from abc import ABC, abstractmethod
from typing import Callable, Iterable

import numpy as np

from .tasks import ModelTasks
from sklearn.metrics import get_scorer


class Metric(ABC):
    """Abstract class for scoring functions.

    Attributes:
        name (str): Name of the scoring function.
    """

    @abstractmethod
    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray | list[np.ndarray], *args, **kwargs
    ) -> float:
        """Calculate the score.

        Args:
            y_true (np.ndarray): True values. Must be of shape (n_samples, n_targets)
            y_pred (np.ndarray | list[np.ndarray]): Predicted values. Must be of shape
                               (n_samples, n_tasks) or list of arrays of shape
                               (n_samples, n_classes) of length n_tasks.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            float: Score of the predictions.
        """

    def checkMetricCompatibility(self, task: ModelTasks, probas: bool):
        """Check if the metric supports the given task and prediction type.

        Args:
            task (ModelTasks): Task of the model.
            probas (bool): True if the predictions are probabilities.

        Raises:
            ValueError: If the metric does not support the given task or prediction
                        type.
        """
        if not self.supportsTask(task):
            raise ValueError("Scorer %s does not support task %s" % (self.name, task))
        if self.needsProbasToScore and not probas:
            raise ValueError("Scorer %s needs probabilities to score" % self.name)
        if self.needsDiscreteToScore and probas and not task.isRegression():
            raise ValueError("Scorer %s needs discrete values to score" % self.name)

    @abstractmethod
    def supportsTask(self, task: ModelTasks) -> bool:
        """Return true if the scorer supports the given task.

        Args:
            task (ModelTasks): Task of the model.

        Returns:
            bool: True if the scorer supports the given task.
        """

    @property
    @abstractmethod
    def needsProbasToScore(self) -> bool:
        """Return True if the scorer needs probabilities to score.

        Returns:
            bool: True if the scorer needs probabilities to score.
        """

    @property
    @abstractmethod
    def needsDiscreteToScore(self) -> bool:
        """Return True if the scorer needs discrete values to score.

        Returns:
            bool: True if the scorer needs discrete values to score.
        """

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return self.name


class SklearnMetrics(Metric):
    """Wrapper for sklearn scoring functions.

    Attributes:
        scorer: Sklearn scorer object.
    """

    def __init__(self, scorer: str | Callable[[Iterable, Iterable], float]):
        """Initialize the scoring function.

        Args:
            scorer (str | Callable): Name of the scoring function or sklearn scorer
        """
        if isinstance(scorer, str):
            scorer = get_scorer(scorer)
        self.scorer = scorer  # Sklearn scorer object
        self.name = scorer._score_func.__name__  # Name of the scorer

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray | list[np.ndarray], *args, **kwargs
    ):
        """Call the scoring function.

        Args:
            y_true (np.array): True values. Must be of shape (n_samples, n_targets)
            y_pred (Iterable or np.array): Predicted values. Must be of shape
                    (n_samples, n_targets) or list of arrays of shape
                    (n_samples, n_task) of length n_targets.
            *args: Additional arguments. Unused.
            **kwargs: Additional keyword arguments. Unused.
        """
        # Convert predictions to correct shape for sklearn scorer
        if isinstance(y_pred, list):
            if self.needsDiscreteToScore:
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

    @property
    def needsProbasToScore(self):
        """Return True if the scorer needs probabilities to score."""
        return self.scorer.__class__.__name__ in ["_ProbaScorer", "_ThresholdScorer"]

    @property
    def needsDiscreteToScore(self):
        """Return True if the scorer needs discrete values to score."""
        return self.scorer.__class__.__name__ == "_PredictScorer"

    def supportsTask(self, task: ModelTasks):
        """Return true if the scorer supports the given task."""
        if self.needsProbasToScore:
            return task in [
                ModelTasks.SINGLECLASS,
                ModelTasks.MULTITASK_SINGLECLASS,
                ModelTasks.MULTICLASS
            ]

        # No sklearn metric support multitask-multiclass
        if task in [ModelTasks.MULTITASK_MIXED, ModelTasks.MULTITASK_MULTICLASS]:
            return False

        return True

    def _scorerFunc(self, y_true, y_pred):
        """Return the scoring function of a sklearn scorer."""
        # From https://stackoverflow.com/questions/63943410/getting-a-scoring-function-by-name-in-scikit-learn
        return self.scorer._sign * self.scorer._score_func(y_true, y_pred, **self.scorer._kwargs)
