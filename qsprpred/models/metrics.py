"""Wrapper for sklearn scoring functions."""

from abc import ABC, abstractmethod

import numpy as np
import sklearn
import scipy.stats
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


class CalibrationError(Metric):
    """Compute the calibration error of a classifier.

    ECE is defined as the expected difference between the predicted probability
    and the observed frequency in each bin. The lower the ECE, the more
    calibrated the classifier is.

    Referece: Guo et al. (2017) On Calibration of Modern Neural Networks.
    https://arxiv.org/abs/1706.04599

    Attributes:
        name (str): Name of the scoring function (calibration_error).
    """
    def __init__(self, n_bins: int = 10, norm: str = "L1"):
        """Initialize the calibration error scorer.
        
        If `norm` is 'L1', the expected calibration error is returned (ECE).
        If `norm` is 'L2', the root-mean-square calibration error is returned (RMSCE).
        If `norm` is 'infinity', the maximum calibration error is returned (MCE).

        Args:
            n_bins (int): Number of bins to use for calibration.
                A bigger bin number requires more data. Defaults to 10.
            norm (str): The norm to use for the calibration error.
                Can be 'L1' or 'L2' or 'infinity'. Defaults to 'L1'.
        """
        self.n_bins = n_bins
        self.norm = norm
    
    def __call__(
        self,
        y_true: np.array,
        y_pred: list[np.ndarray],
    ) -> float:
        """Compute the calibration error of a classifier.

        Referece: Guo et al. (2017) On Calibration of Modern Neural Networks.
        https://arxiv.org/abs/1706.04599

        Args:
            y_true (np.array): True class labels. 1d array.
            y_pred (list[np.array]): Predicted class probabilities.
                List of arrays of shape (n_samples, n_classes) of length n_tasks.
                Note. Multi-task predictions are not supported.

        Returns:
            float: The calibration error.
        """
        # Check if y_pred is a list of arrays of length 1
        if not isinstance(y_pred, list):
            raise ValueError("y_pred must be a list of 2D arrays.")
        if len(y_pred) > 1:
            raise ValueError("Multi-task predictions are not supported.")

        # TODO: support multi-task predictions
        # Convert y_pred from list to a 2D array
        y_pred = y_pred[0]
        
        assert len(y_true) >= self.n_bins, "Number of samples must be at least n_bins."

        # Get the highest probability and the predicted class
        y_pred_max = np.max(y_pred, axis=1)
        y_pred_class = np.argmax(y_pred, axis=1)
        # Sort data based on the highest probability
        sorted_indices = np.argsort(y_pred_max)
        sorted_y_true = y_true[sorted_indices]
        sorted_y_pred_max = y_pred_max[sorted_indices]
        sorted_y_pred_class = y_pred_class[sorted_indices]
        # Bin sorted data
        binned_y_true = np.array_split(sorted_y_true, self.n_bins)
        binned_y_pred_max = np.array_split(sorted_y_pred_max, self.n_bins)
        binned_y_pred_class = np.array_split(sorted_y_pred_class, self.n_bins)
        # Compute the calibration error by iterating over the bins
        calibration_error = 0.0
        for bin_y_true, bin_y_pred_max, bin_y_pred_class in zip(
            binned_y_true, binned_y_pred_max, binned_y_pred_class
        ):
            # Compute the accuracy and the mean probability for the bin
            mean_prob = np.mean(bin_y_pred_max)
            accuracy = np.mean(bin_y_true == bin_y_pred_class)
            # Compute the calibration error for the bin based on the norm
            if self.norm == "L1":
                calibration_error += (
                    np.abs(mean_prob - accuracy) * len(bin_y_true) / len(y_true)
                )
            elif self.norm == "L2":
                calibration_error += (
                    np.square(mean_prob - accuracy)**2 * len(bin_y_true) / len(y_true)
                )
            elif self.norm == "infinity":
                calibration_error = max(calibration_error, np.abs(mean_prob - accuracy))
            else:
                raise ValueError(f"Unknown norm {self.norm}")
        if self.norm == "L2":
            calibration_error = np.sqrt(calibration_error)

        return calibration_error

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "calibration_error"

class Specificity(Metric):
    """Calculate specificity (true postive rate).

    Attributes:
        name (str): Name of the scoring function (specificity).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the specificity (selectivity).

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (np.array): Predicted labels. 2D array (n_samples, 1)

        Returns:
            float: The specificity.

        """
        tn, fp, _, _ = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "specificity"


class NegativePredictedValue(Metric):
    """Calculate the negative predicted value.

    Attributes:
        name (str): Name of the scoring function (negative_predicted_value).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the negative predicted value.

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (np.array): Predicted labels. 2D array (n_samples, 1)

        Returns:
            float: The negative predicted value.

        """
        tn, _, fn, _ = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fn)

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "negative_predicted_value"


class BalancedMatthewsCorrcoeff(Metric):
    """Calculate the balanced Matthews correlation coefficient.

    Attributes:
        name (str): Name of the scoring function (balanced_matthews_corrcoeff).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the balanced Matthews correlation coefficient.

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (np.array): Predicted labels. 2D array (n_samples, 1)

        Returns:
            float: The correlation coefficient.

        """
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        prevalence = sum(y_true) / len(y_true)
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        return (sen + spe - 1) / (
            (sen + (1 - spe) * (1 - prevalence) / prevalence) *
            (spe + (1 - sen) * prevalence / (1 - prevalence))
        )**0.5

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "balanced_matthews_corrcoeff"


class Prevalence(Metric):
    """Calculate the prevalence.

    Attributes:
        name (str): Name of the scoring function (prevalence).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the prevalence.

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (np.array): Predicted labels. 2D array (n_samples, 1)

        Returns:
            float: The prevalence.

        """
        return sum(y_true) / len(y_true)

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "prevalence"


class BalancedPositivePredictedValue(Metric):
    """Calculate the balanced positive predicted value.

    Attributes:
        name (str): Name of the scoring function (balanced_positive_predicted_value).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the balanced positive predicted value.

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (np.array): Predicted labels. 2D array (n_samples, 1)

        Returns:
            float: The balanced positive predicted value.

        """
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        prevalence = sum(y_true) / len(y_true)
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        return (sen * prevalence) / (sen * prevalence + (1 - spe) * (1 - prevalence))

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "balanced_positive_predicted_value"


class BalancedNegativePredictedValue(Metric):
    """Calculate the balanced negative predicted value.

    Attributes:
        name (str): Name of the scoring function (balanced_negative_predicted_value).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the balanced negative predicted value.

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (np.array): Predicted labels. 2D array (n_samples, 1)

        Returns:
            float: The balanced negative predicted value.

        """
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        prevalence = sum(y_true) / len(y_true)
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        return (spe * (1 - prevalence)) / (
            spe * (1 - prevalence) + (1 - sen) * prevalence
        )

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "balanced_negative_predicted_value"


class BalancedCohenKappa(Metric):
    """Calculate the balanced Cohen kappa coefficient.

    Attributes:
        name (str): Name of the scoring function (balanced_cohen_kappa).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the balanced Cohen kappa coefficient.

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (np.array): Predicted labels. 2D array (n_samples, 1)

        Returns:
            float: The balanced Cohen kappa coefficient.

        """
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        prevalence = sum(y_true) / len(y_true)
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        return (2 * (sen + spe - 1)) / (
            (sen + (1 - spe) * (1 - prevalence) / prevalence) +
            (spe + (1 - sen) * prevalence / (1 - prevalence))
        )

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "balanced_cohen_kappa"


class KSlope(Metric):
    """Calculate the slope of the regression line through the origin
    between the predicted and observed values.
    
    Reference: Tropsha, A., & Golbraikh, A. (2010). In J.-L. Faulon & A. Bender (Eds.),
        Handbook of Chemoinformatics Algorithms.
    https://www.taylorfrancis.com/books/9781420082999

    Attributes:
        name (str): Name of the scoring function (k_slope).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the slope of the regression line through the origin
        between the predicted and observed values.

        Args:
            y_true (np.array): Ground truth (correct) target values. 1d array.
            y_pred (np.array): 2D array (n_samples, n_tasks)

        Returns:
            float: The coefficient of determination.

        """
        num, denom = 0, 0
        for i in range(len(y_true)):
            num += y_true[i] * y_pred[i]
            denom += y_true[i] ** 2
        return num / denom if len(y_pred) >= 2 else 0

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "k_slope"


class KPrimeSlope(Metric):
    """Calculate the slope of the regression line through the origin
    between the observed and predicted values.

    Reference: Tropsha, A., & Golbraikh, A. (2010). In J.-L. Faulon & A. Bender (Eds.),
    Handbook of Chemoinformatics Algorithms.
    https://www.taylorfrancis.com/books/9781420082999


    Attributes:
        name (str): Name of the scoring function (k_prime_slope).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the slope of the regression line through the origin
        between the observed and predicted values.

        Args:
            y_true (np.array): Ground truth (correct) target values. 1d array.
            y_pred (np.array): 2D array (n_samples, n_tasks)

        Returns:
            float: The coefficient of determination.

        """
        num, denom = 0, 0
        for i in range(len(y_true)):
            num += y_true[i] * y_pred[i]
            denom += y_pred[i] ** 2
        return num / denom if len(y_pred) >= 2 else 0

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "k_prime_slope"
    
class Pearson(Metric):
    """Calculate the Pearson correlation coefficient.

    Attributes:
        name (str): Name of the scoring function (pearson).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the Pearson correlation coefficient.

        Args:
            y_true (np.array): Ground truth (correct) target values. 1d array.
            y_pred (np.array): 2D array (n_samples, 1)

        Returns:
            float: The Pearson correlation coefficient.

        """
        y_pred = y_pred.flatten()
        return scipy.stats.pearsonr(y_true, y_pred)[0] if len(y_pred) >= 2 else 0

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "pearson"


class Spearman(Metric):
    """Calculate the Spearman correlation
    
    Attributes:
        name (str): Name of the scoring function (spearman).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the Spearman correlation
        
        Args:
            y_true (np.array): Ground truth (correct) target values. 1d array.
            y_pred (np.array): 2D array (n_samples, n_tasks)

        Returns:
            float: The Pearson Spearman coefficient.

        """
        return scipy.stats.spearmanr(y_true, y_pred)[0] if len(y_pred) >= 2 else 0

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "spearman"

class Kendall(Metric):
    """Calculate the Kendall rank correlation coefficient.

    Attributes:
        name (str): Name of the scoring function (kendall).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the Kendall rank correlation coefficient.

        Args:
            y_true (np.array): Ground truth (correct) target values. 1d array.
            y_pred (np.array): 2D array (n_samples, n_tasks)

        Returns:
            float: The Kendall rank correlation coefficient.

        """
        return scipy.stats.kendalltau(y_true, y_pred)[0] if len(y_pred) >= 2 else 0

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "kendall"

class R20(KSlope):
    """Calculate the coefficient of determination for regression line
    through the origin between the observed and predicted values.

    Reference: Tropsha, A., & Golbraikh, A. (2010). In J.-L. Faulon & A. Bender (Eds.),
    Handbook of Chemoinformatics Algorithms.
    https://www.taylorfrancis.com/books/9781420082999

    Attributes:
        name (str): Name of the scoring function (r_2_0).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the coefficient of determination for regression line
        through the origin between the observed and predicted values.

        Args:
            y_true (np.array): Ground truth (correct) target values. 1d array.
            y_pred (np.array): 2D array (n_samples, n_tasks)

        Returns:
            float: The coefficient of determination.

        """
        # get the slope of the regression line through the origin
        k_prime = super().__call__(y_true, y_pred)
        y_true_mean = y_true.mean()
        num, denom = 0, 0
        for i in range(len(y_true)):
            num += y_true[i] - k_prime * y_pred[i]
            denom += (y_true[i] - y_true_mean) ** 2
        return 1 - num / denom if len(y_pred) >= 2 else 0

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "r_2_0"

class RPrime20(KPrimeSlope):
    """Calculate the coefficient of determination for regression line
    through the origin between the predicted and observed values.

    Reference: Tropsha, A., & Golbraikh, A. (2010). In J.-L. Faulon & A. Bender (Eds.),
    Handbook of Chemoinformatics Algorithms.
    https://www.taylorfrancis.com/books/9781420082999

    Attributes:
        name (str): Name of the scoring function (r_prime_2_0).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the coefficient of determination for regression line
        through the origin between the predicted and observed values.

        Args:
            y_true (np.array): Ground truth (correct) target values. 1d array.
            y_pred (np.array): 2D array (n_samples, n_tasks)

        Returns:
            float: The coefficient of determination.

        """
        # get the slope of the regression line through the origin
        k = super().__call__(y_true, y_pred)
        y_pred_mean = y_pred.mean()
        num, denom = 0, 0
        for i in range(len(y_true)):
            num += y_pred[i] - k * y_true[i]
            denom += (y_pred[i] - y_pred_mean) ** 2
        return 1 - num / denom if len(y_pred) >= 2 else 0

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "r_prime_2_0"