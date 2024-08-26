import numpy as np

from qsprpred.models.assessment.metrics.base import Metric


class MaskedMetric(Metric):
    """Wrapper for Metrics to handle missing target values."""

    def __init__(self, metric: Metric):
        """Initialize the masked metric.

        Args:
            metric (Metric): The metric to be masked.
        """
        self.metric = metric

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
        # Mask missing values
        mask = ~np.isnan(y_true)

        # if any target is missing in a row, make mask for that row False
        if y_true.ndim == 2:
            mask = np.all(mask, axis=1)

        y_true_masked = y_true[mask]
        if isinstance(y_pred, list):
            y_pred_masked = [yp[mask] for yp in y_pred]
        else:
            y_pred_masked = y_pred[mask]

        return self.metric(y_true_masked, y_pred_masked)

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return f"masked_{self.metric}"
