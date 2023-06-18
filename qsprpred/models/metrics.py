"""Wrapper for sklearn scoring functions."""

from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Iterable

from sklearn import metrics

from .tasks import ModelTasks


class Metric(ABC):
    """Abstract class for scoring functions."""
    def __init__(self, name: str, func: Callable[[Iterable, Iterable, ...], float]):
        """Initialize the scoring function.

        Args:
            name (str): Name of the scoring function.
            func (Callable[[Iterable, Iterable], float]): Scoring function.
        """
        self.name = name
        self.func = func

    @abstractmethod
    def __call__(self, y_true: Iterable, y_pred: Iterable, *args, **kwargs) -> float:
        """Calculate the score.

        Args:
            y_true (Iterable): True values.
            y_pred (Iterable): Predicted values.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

    @property
    @abstractmethod
    def supports_REGRESSION(self) -> bool:
        """Return true if the scorer supports regression tasks."""

    @property
    @abstractmethod
    def supports_SINGLECLASS(self) -> bool:
        """Return true if the scorer supports single class classification tasks."""

    @property
    @abstractmethod
    def supports_MULTICLASS(self) -> bool:
        """Return true if the scorer supports multi class classification tasks."""

    @property
    @abstractmethod
    def supports_MULTITASK_REGRESSION(self) -> bool:
        """Return true if the scorer supports multi task regression tasks."""

    @property
    @abstractmethod
    def supports_MULTITASK_SINGLECLASS(self) -> bool:
        """Return true if the scorer supports
        multi task single class classification tasks."""

    @property
    @abstractmethod
    def supports_MULTITASK_MULTICLASS(self) -> bool:
        """Return true if the scorer supports
        multi task multi class classification tasks."""

    @property
    @abstractmethod
    def supports_MULTITASK_MIXED(self) -> bool:
        """Return true if the scorer supports multi task mixed tasks."""

    @property
    @abstractmethod
    def needsProbasToScore(self):
        """Return True if the scorer needs probabilities to score."""

    @property
    @abstractmethod
    def needsDiscreteToScore(self):
        """Return True if the scorer needs discrete values to score."""

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

    regressionMetrics = [
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
    singleClassMetrics = [
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
        'matthews_corrcoef',
    ]
    multiClassMetrics = [
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
    multiTaskRegressionMetrics = [
        "explained_variance",
        "neg_mean_absolute_error",
        "neg_mean_squared_error",
        "neg_root_mean_squared_error",
        "neg_mean_squared_log_error",
        "neg_median_absolute_error",
        "r2",
        "neg_mean_absolute_percentage_error",
    ]
    multiTaskSingleClassMetrics = [
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
    multiTaskMultiClassMetrics = []
    multiTaskMixedMetrics = []

    def __init__(self, name, func, scorer):
        """Initialize the scoring function."""
        super().__init__(name, func)
        self.scorer = scorer  # Sklearn scorer object

    def __call__(self, y_true, y_pred, *args, **kwargs):
        """Call the scoring function."""
        return self.func(y_true, y_pred, *args, **kwargs)

    @property
    def supports_REGRESSION(self):
        """Return true if the scorer supports regression tasks."""
        return self.name in self.regressionMetrics

    @property
    def supports_SINGLECLASS(self):
        """Return true if the scorer supports single class classification tasks."""
        return self.name in self.singleClassMetrics

    @property
    def supports_MULTICLASS(self):
        """Return true if the scorer supports multi class classification tasks."""
        return self.name in self.multiClassMetrics

    @property
    def supports_MULTITASK_REGRESSION(self):
        """Return true if the scorer supports multi task regression tasks."""
        return self.name in self.multiTaskRegressionMetrics

    @property
    def supports_MULTITASK_SINGLECLASS(self):
        """Return true if the scorer supports
        multi task single class classification tasks.
        """
        return self.name in self.multiTaskSingleClassMetrics

    @property
    def supports_MULTITASK_MULTICLASS(self):
        """Return true if the scorer supports multitask multiclass
        (or a mix of multiclass/single class) classification tasks.
        """
        return self.name in self.multiTaskMultiClassMetrics

    @property
    def supports_MULTITASK_MIXED(self):
        """Return true if the scorer supports
        multi task mixed regression/classification tasks."""
        return self.name in self.multiTaskMixedMetrics

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
        return getattr(self, f"supports_{task}")

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

    @staticmethod
    def getMetric(name):
        """Return a scorer function object from a sklearn scorer name."""
        scorer = metrics.get_scorer(name)

        return SklearnMetric(
            name=name, func=partial(SklearnMetric.scorerFunc, scorer), scorer=scorer
        )
