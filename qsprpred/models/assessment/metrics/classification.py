import numpy as np
import sklearn
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment, CalcRIE

from qsprpred.models.assessment.metrics.base import Metric

# ------------------------------------------
#   Classification Metrics (Probabilistic)
# ------------------------------------------


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


class BEDROC(Metric):
    """Calculate the Boltzmann-enhanced discrimination of ROC (BEDROC).

    Reference: Truchon and Bayly, J. Chem. Inf. Model. 2007 47 (2), 488-508. DOI: 10.1021/ci600426e

    Attributes:
        name (str): Name of the scoring function (bedroc).
    """
    def __init__(self, alpha: float = 20):
        """Initialize the BEDROC scorer.

        Args:
            alpha (float): Weighting parameter (default: 20)
        """
        self.alpha = alpha

    def __call__(self, y_true: np.array, y_pred: list[np.array]) -> float:
        """Calculate the BEDROC score.

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (list[np.array]): Target probability scores.
                List of arrays of shape (n_samples, n_classes) of length n_tasks.
                Note. Multi-task predictions are not supported.

        Returns:
            float: The BEDROC score.
        """
        if isinstance(y_pred, list):
            y_pred = y_pred[0]
        return CalcBEDROC(
            [[y] for _, y in sorted(zip(y_pred[1], y_true), reverse=True)],
            col=0,
            alpha=self.alpha,
        )

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "bedroc"


class EnrichmentFactor(Metric):
    """Calculate the enrichment factor.

    Attributes:
        name (str): Name of the scoring function (enrichment_factor).
    """
    def __init__(self, chi: float = 0.05):
        """Initialize the enrichment factor scorer.

        Args:
            chi (float): Weighting parameter (default: 5%)
        """
        self.chi = chi

    def __call__(self, y_true: np.array, y_pred: list[np.array]) -> float:
        """Calculate the enrichment factor.

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (list[np.array]): Target probability scores.
                List of arrays of shape (n_samples, n_classes) of length n_tasks.
                Note. Multi-task predictions are not supported.

        Returns:
            float: The enrichment factor.
        """
        if isinstance(y_pred, list):
            y_pred = y_pred[0]
        return CalcEnrichment(
            [[y] for _, y in sorted(zip(y_pred[1], y_true), reverse=True)],
            col=0,
            fractions=[self.chi],
        )[0]

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "enrichment_factor"


class RobustInitialEnhancement(Metric):
    """Calculate the robust initial enhancement.

    Reference: Sheridan et al., J. Chem. Inf. Model. 2001 41 (5), 1395-1406. DOI: 10.1021/ci0100144

    Attributes:
        name (str): Name of the scoring function (robust_initial_enhancement).
    """
    def __init__(self, alpha: float = 100):
        """Initialize the robust initial enhancement scorer.

        Args:
            alpha (float): Weighting parameter (default: 100)
        """
        self.alpha = alpha

    def __call__(self, y_true: np.array, y_pred: list[np.array]) -> float:
        """Calculate the robust initial enhancement.

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (list[np.array]): Target probability scores.
                List of arrays of shape (n_samples, n_classes) of length n_tasks.
                Note. Multi-task predictions are not supported.

        Returns:
            float: The robust initial enhancement.
        """
        if isinstance(y_pred, list):
            y_pred = y_pred[0]
        return CalcRIE(
            [[y] for _, y in sorted(zip(y_pred[1], y_true), reverse=True)],
            col=0,
            alpha=self.alpha,
        )

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "robust_initial_enhancement"


# ------------------------------------------
#   Classification Metrics (Discrete)
# ------------------------------------------


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


class Sensitivity(Metric):
    """Calculate sensitivity (true positive rate).

    Attributes:
        name (str): Name of the scoring function (sensitivity).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the sensitivity (recall).

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (np.array): Predicted labels. 2D array (n_samples, 1)

        Returns:
            float: The sensitivity.

        """
        _, _, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn)

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "sensitivity"


class Specificity(Metric):
    """Calculate specificity (true negative rate).

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


class PositivePredictivity(Metric):
    """Calculate the Positive predictivity.

    Attributes:
        name (str): Name of the scoring function (Positive_predictivity).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the Positive predictivity.

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (np.array): Predicted labels. 2D array (n_samples, 1)

        Returns:
            float: The Positive predictivity.

        """
        _, fp, _, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fp)

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "Positive_predictivity"


class NegativePredictivity(Metric):
    """Calculate the negative predictivity.

    Attributes:
        name (str): Name of the scoring function (negative_predictivity).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the negative predictivity.

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (np.array): Predicted labels. 2D array (n_samples, 1)

        Returns:
            float: The negative predictivity.

        """
        tn, _, fn, _ = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fn)

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "negative_predictivity"


class CohenKappa(Metric):
    """Calculate the Cohen's kappa coefficient.

    Attributes:
        name (str): Name of the scoring function (cohen_kappa).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the Cohen kappa coefficient.

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (np.array): Predicted labels. 2D array (n_samples, 1)

        Returns:
            float: The Cohen kappa coefficient.

        """
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
        return (2 *
                (tp * tn - fp * fn)) / ((tp + fp) * (tn + fp) + (tp + fn) * (tn + fn))

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "cohen_kappa"


class BalancedPositivePredictivity(Metric):
    """Calculate the balanced positive predictivity.

    Guesné, S.J.J., Hanser, T., Werner, S. et al. Mind your prevalence!.
    J Cheminform 16, 43 (2024). https://doi.org/10.1186/s13321-024-00837-w

    Attributes:
        name (str): Name of the scoring function (balanced_positive_predictivity).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the balanced positive predictivity.

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (np.array): Predicted labels. 2D array (n_samples, 1)

        Returns:
            float: The balanced positive predictivity.

        """
        _, sen, spe = derived_confusion_matrix(y_true, y_pred)
        return sen / (1 + sen - spe)

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "balanced_positive_predictivity"


class BalancedNegativePredictivity(Metric):
    """Calculate the balanced negative predictivity.

    Guesné, S.J.J., Hanser, T., Werner, S. et al. Mind your prevalence!.
    J Cheminform 16, 43 (2024). https://doi.org/10.1186/s13321-024-00837-w

    Attributes:
        name (str): Name of the scoring function (balanced_negative_predictivity).
    """
    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the balanced negative predictivity.

        Args:
            y_true (np.array): Ground truth (correct) labels. 1d array.
            y_pred (np.array): Predicted labels. 2D array (n_samples, 1)

        Returns:
            float: The balanced negative predictivity.

        """
        _, sen, spe = derived_confusion_matrix(y_true, y_pred)
        return spe / (1 + sen + spe)

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "balanced_negative_predictivity"


class BalancedMatthewsCorrcoeff(Metric):
    """Calculate the balanced Matthews correlation coefficient.

    Guesné, S.J.J., Hanser, T., Werner, S. et al. Mind your prevalence!.
    J Cheminform 16, 43 (2024). https://doi.org/10.1186/s13321-024-00837-w

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
        _, sen, spe = derived_confusion_matrix(y_true, y_pred)
        return (sen + spe - 1) / np.sqrt(1 - (sen - spe)**2)

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "balanced_matthews_corrcoeff"


class BalancedCohenKappa(Metric):
    """Calculate the balanced Cohen kappa coefficient.

    Guesné, S.J.J., Hanser, T., Werner, S. et al. Mind your prevalence!.
    J Cheminform 16, 43 (2024). https://doi.org/10.1186/s13321-024-00837-w

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
        _, sen, spe = derived_confusion_matrix(y_true, y_pred)
        return sen + spe - 1

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "balanced_cohen_kappa"


def derived_confusion_matrix(y_true: np.array,
                             y_pred: np.array) -> tuple[int, int, int]:
    """Calculate the derived confusion matrix.

    Args:
        y_true (np.array): Ground truth (correct) labels. 1d array.
        y_pred (np.array): Predicted labels. 2D array (n_samples, 1)

    Returns:
        tuple[int, int, int]: The derived confusion matrix.
                              Prevalence, sensitivity and specificity.
    """
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    pre = sum(y_true) / len(y_true)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    return pre, sen, spe
