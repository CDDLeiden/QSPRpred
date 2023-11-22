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
        if self.needsDiscreteToScore and probas:
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
            ModelTasks.REGRESSION: self.regressionMetrics,
            ModelTasks.SINGLECLASS: self.singleClassMetrics,
            ModelTasks.MULTICLASS: self.multiClassMetrics,
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


def calibration_error(
    y_true: np.array, y_prob: np.array, n_bins: int = 10, norm: str = "L1"
) -> float:
    """Compute the calibration error of a classifier.

    ECE is defined as the expected difference between the predicted probability
    and the observed frequency in each bin. The lower the ECE, the more
    calibrated the classifier is.

    If `y_prob` is 1d, it assumes that probablities correspond to the positive class of
    a binary classification. If `y_prob` is 2d, it assumes each column corresponds to a
    class and that the classes are in the columns in ascending order.

    If `norm` is 'L1', the expected calibration error is returned (ECE).
    If `norm` is 'L2', the root-mean-square calibration error is returned (RMSCE).
    If `norm` is 'infinity', the maximum calibration error is returned (MCE).

    Referece: Guo et al. (2017) On Calibration of Modern Neural Networks. https://arxiv.org/abs/1706.04599

    Args:
        y_true (np.array): True class labels. 1d array.
        y_prob (np.array): Raw probability/score of the positive class
            or multiclass probability. 1d or 2d array.
        n_bins (int): Number of bins to use for calibration.
            A bigger bin number requires more data. Defaults to 10.
        norm (str): The norm to use for the calibration error.
            Can be 'L1' or 'L2' or 'infinity'. Defaults to 'L1'.

    Returns:
        float: The calibration error.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    # If `y_prob` is 1d, convert it to 2d by adding
    # a new axis corresponding to the negative class
    if y_prob.ndim == 1:
        y_prob_0 = 1 - y_prob
        y_prob = np.stack([y_prob_0, y_prob], axis=1)
    # Get the highest probability and the predicted class
    y_prob_max = np.max(y_prob, axis=1)
    y_pred_class = np.argmax(y_prob, axis=1)
    # Sort data based on the highest probability
    sorted_indices = np.argsort(y_prob_max)
    sorted_y_true = y_true[sorted_indices]
    sorted_y_prob_max = y_prob_max[sorted_indices]
    sorted_y_pred_class = y_pred_class[sorted_indices]
    # Bin sorted data
    binned_y_true = np.array_split(sorted_y_true, n_bins)
    binned_y_prob_max = np.array_split(sorted_y_prob_max, n_bins)
    binned_y_pred_class = np.array_split(sorted_y_pred_class, n_bins)
    # Compute the calibration error by iterating over the bins
    calibration_error = 0.0
    for bin_y_true, bin_y_prob_max, bin_y_pred_class in zip(
        binned_y_true, binned_y_prob_max, binned_y_pred_class
    ):
        # Compute the accuracy and the mean probability for the bin
        mean_prob = np.mean(bin_y_prob_max)
        accuracy = np.mean(bin_y_true == bin_y_pred_class)
        # Compute the calibration error for the bin based on the norm
        if norm == "L1":
            calibration_error += (
                np.abs(mean_prob - accuracy) * len(bin_y_true) / len(y_true)
            )
        elif norm == "L2":
            calibration_error += (
                np.square(mean_prob - accuracy)**2 * len(bin_y_true) / len(y_true)
            )
        elif norm == "infinity":
            calibration_error = max(calibration_error, np.abs(mean_prob - accuracy))
        else:
            raise ValueError(f"Unknown norm {norm}")
    if norm == "L2":
        calibration_error = np.sqrt(calibration_error)

    return calibration_error
