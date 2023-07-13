"""Wrapper for sklearn scoring functions."""

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, ClassVar, Iterable

import numpy as np
from sklearn import metrics

from .tasks import ModelTasks


class Metric(ABC):
    """Abstract class for scoring functions.

    Attributes:
        name (str): Name of the scoring function.
        func (Callable[[Iterable, Iterable], float]): Scoring function.
    """
    def __init__(self, name: str, func: Callable[[Iterable, Iterable, ...], float]):
        """Initialize the scoring function.

        Args:
            name (str): Name of the scoring function.
            func (Callable[[Iterable, Iterable], float]): Scoring function.
        """
        self.name = name
        self.func = func

    @abstractmethod
    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray | list[np.ndarray], *args, **kwargs
    ) -> float:
        """Calculate the score.

        Args:
            y_true (np.ndarray): True values. Must be of shape (n_samples, n_targets)
            y_pred (np.ndarray | list[np.ndarray]): Predicted values. Must be of shape
                               (n_samples, n_targets) or list of arrays of shape
                               (n_samples, n_task) of length n_targets.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

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

    @property
    def isClassificationMetric(self) -> bool:
        """Return true if the scorer supports any type of classification tasks."""
        return (
            self.supports_SINGLECLASS or self.supports_MULTICLASS or
            self.supports_MULTITASK_SINGLECLASS or self.supports_MULTITASK_MULTICLASS or
            self.supports_MULTITASK_MIXED
        )

    @property
    def isRegressionMetric(self) -> bool:
        """Return true if the scorer supports any type of regression tasks."""
        return (
            self.supports_REGRESSION or self.supports_MULTITASK_REGRESSION or
            self.supports_MULTITASK_MIXED
        )

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return self.name


class SklearnMetric(Metric):
    """Wrapper for sklearn scoring functions.

    Attributes:
        scorer: Sklearn scorer object.
        regressionMetrics: List of regression metrics.
        singleClassMetrics: List of single class classification metrics.
        multiClassMetrics: List of multi class classification metrics.
        multiTaskRegressionMetrics: List of multi task regression metrics.
        multiTaskSingleClassMetrics:
            List of multi task single class classification metrics.
        multiTaskMultiClassMetrics:
             List of multi task multi class classification metrics.
        multiTaskMixedMetrics: List of multi task mixed metrics.
    """

    regressionMetrics: ClassVar[list[str]] = [
        "explained_variance",
        "max_error",
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_root_mean_squared_error",
        "neg_mean_squared_log_error",
        "neg_median_absolute_error",
        "r2",
        "neg_mean_poisson_deviance",
        "neg_mean_gamma_deviance",
        "neg_mean_absolute_percentage_error",
    ]  # 'd2_absolute_error_score','d2_pinball_score', 'd2_tweedie_scor'
    singleClassMetrics: ClassVar[list[str]] = [
        "average_precision",
        "neg_brier_score",
        "neg_log_loss",
        "roc_auc",
        "roc_auc_ovo",
        "roc_auc_ovo_weighted",
        "roc_auc_ovr",
        "roc_auc_ovr_weighted",
        "accuracy",
        "balanced_accuracy",
        "top_k_accuracy",
        "f1",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "precision",
        "precision_micro",
        "precision_macro",
        "precision_weighted",
        "recall",
        "recall_micro",
        "recall_macro",
        "recall_weighted",
        "jaccard",
        "jaccard_micro",
        "jaccard_macro",
        "jaccard_weighted",
        "matthews_corrcoef",
    ]
    multiClassMetrics: ClassVar[list[str]] = [
        "neg_log_loss",
        "roc_auc_ovo",
        "roc_auc_ovo_weighted",
        "roc_auc_ovr",
        "roc_auc_ovr_weighted",
        "accuracy",
        "balanced_accuracy",
        "top_k_accuracy",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "precision_micro",
        "precision_macro",
        "precision_weighted",
        "recall_micro",
        "recall_macro",
        "recall_weighted",
        "jaccard_micro",
        "jaccard_macro",
        "jaccard_weighted",
    ]
    multiTaskRegressionMetrics: ClassVar[list[str]] = [
        "explained_variance",
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_root_mean_squared_error",
        "neg_mean_squared_log_error",
        "neg_median_absolute_error",
        "r2",
        "neg_mean_absolute_percentage_error",
    ]
    multiTaskSingleClassMetrics: ClassVar[list[str]] = [
        "roc_auc_ovo",
        "roc_auc_ovo_weighted",
        "roc_auc_ovr",
        "roc_auc_ovr_weighted",
        "accuracy",
        "average_precision",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "f1_samples",
        "precision_micro",
        "precision_macro",
        "precision_weighted",
        "precision_samples",
        "recall_micro",
        "recall_macro",
        "recall_weighted",
        "recall_samples",
        "jaccard_micro",
        "jaccard_macro",
        "jaccard_weighted",
        "jaccard_samples",
    ]
    multiTaskMultiClassMetrics: ClassVar[list[str]] = []
    multiTaskMixedMetrics: ClassVar[list[str]] = []

    def __init__(self, name, func, scorer):
        """Initialize the scoring function."""
        super().__init__(name, func)
        self.scorer = scorer  # Sklearn scorer object

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

        return self.func(y_true, y_pred, *args, **kwargs)

    @property
    def needsProbasToScore(self):
        """Return True if the scorer needs probabilities to score."""
        return self.name in [
            "average_precision",
            "neg_brier_score",
            "neg_log_loss",
            "roc_auc",
            "roc_auc_ovo",
            "roc_auc_ovo_weighted",
            "roc_auc_ovr",
            "roc_auc_ovr_weighted",
            "top_k_accuracy",
        ]

    @property
    def needsDiscreteToScore(self):
        """Return True if the scorer needs discrete values to score."""
        return self.name in [
            "accuracy",
            "balanced_accuracy",
            "f1",
            "f1_micro",
            "f1_macro",
            "f1_weighted",
            "f1_samples",
            "precision",
            "precision_micro",
            "precision_macro",
            "precision_weighted",
            "precision_samples",
            "recall",
            "recall_micro",
            "recall_macro",
            "recall_weighted",
            "recall_samples",
            "matthews_corrcoef",
        ]

    def supportsTask(self, task: ModelTasks):
        """Return true if the scorer supports the given task."""
        task_dict = {
            ModelTasks.REGRESSION: self.RegressionMetrics,
            ModelTasks.SINGLECLASS: self.SingleClassMetrics,
            ModelTasks.MULTICLASS: self.MultiClassMetrics,
            ModelTasks.MULTITASK_REGRESSION: self.multiTaskRegressionMetrics,
            ModelTasks.MULTITASK_SINGLECLASS: self.multiTaskSingleClassMetrics,
            ModelTasks.MULTITASK_MULTICLASS: self.multiTaskMultiClassMetrics,
            ModelTasks.MULTITASK_MIXED: self.multiTaskMixedMetrics,
        }
        return self.name in task_dict[task]

    @classmethod
    @property
    def supportedMetrics(cls):
        """Return a list of all supported metrics."""
        return list(
            set(
                cls.regressionMetrics + cls.singleClassMetrics + cls.multiClassMetrics +
                cls.multiTaskRegressionMetrics + cls.multiTaskSingleClassMetrics +
                cls.multiTaskMultiClassMetrics + cls.multiTaskMixedMetrics
            )
        )

    @staticmethod
    def scorerFunc(scorer, y_true, y_pred):
        """Return the scoring function of a sklearn scorer."""
        # From https://stackoverflow.com/questions/63943410/getting-a-scoring-function-by-name-in-scikit-learn
        return scorer._sign * scorer._score_func(y_true, y_pred, **scorer._kwargs)

    @classmethod
    def getMetric(cls, name: str):
        """Return a scorer function object from a sklearn scorer name.

        Args:
            name (str): Name of the scorer function.

        Returns:
            scorer (SklearnMetric): scorer function from sklearn.metrics wrapped in
                                    `SklearnMetric` class
        """
        scorer = metrics.get_scorer(name)

        return SklearnMetric(
            name=name, func=partial(SklearnMetric.scorerFunc, scorer), scorer=scorer
        )

    @classmethod
    def getDefaultMetric(
        cls,
        task: ModelTasks,
    ):
        """Get default scoring function for a QSPRModel from sklearn.metrics.

        Default scoring functions are:
            - explained_variance for regression
            - roc_auc_ovr_weighted for single class classification
            - roc_auc for multi class classification

        Args:
            task (ModelTasks): Task of the model.

        Raises:
            ValueError: If the model type is currently not supported by sklearn.metrics.

        Returns:
            scorer (SklearnMetric):
                scorer function from sklearn.metrics wrapped in `SklearnMetric` class
        """
        if task.isRegression():
            scorer = SklearnMetric.getMetric("explained_variance")
        elif task in [ModelTasks.MULTICLASS, ModelTasks.MULTITASK_SINGLECLASS]:
            scorer = SklearnMetric.getMetric("roc_auc_ovr_weighted")
        elif task in [ModelTasks.SINGLECLASS]:
            scorer = SklearnMetric.getMetric("roc_auc")
        else:
            raise ValueError("No supported scoring function for task %s" % task)
        return scorer
