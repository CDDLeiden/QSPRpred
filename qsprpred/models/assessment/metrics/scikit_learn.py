import numpy as np
import sklearn
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _BaseScorer

from qsprpred.models.assessment.metrics.base import Metric


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

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray | list[np.ndarray]):
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
            convert_to_discrete = False
            if sklearn.__version__ < "1.4.0":
                if self.scorer.__class__.__name__ == "_PredictScorer":
                    convert_to_discrete = True
            elif self.scorer._response_method == "predict":
                convert_to_discrete = True
            elif "predict_proba" not in self.scorer._response_method:
                convert_to_discrete = True

            if convert_to_discrete:
                # convert to discrete values
                y_pred = [np.argmax(yp, axis=1) for yp in y_pred]
            else:
                # for each task if single class take second column
                y_pred = [
                    yp[:, 1].reshape(-1, 1) if yp.shape[1] == 2 else yp for yp in y_pred
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
        return self.scorer._sign * self.scorer._score_func(
            y_true, y_pred, **self.scorer._kwargs
        )
