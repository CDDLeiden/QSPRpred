import numpy as np
import scipy.stats

from qsprpred.models.assessment.metrics.base import Metric


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


class AverageFoldError(Metric):
    """Calculate the average fold error (AFE).

    Attributes:
        name (str): Name of the scoring function (fold_error).
    """

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the fold error.

        Args:
            y_true (np.array): Ground truth (correct) target values. 1d array.
            y_pred (np.array): 2D array (n_samples, n_tasks)

        Returns:
            float: The fold error.

        """
        return 10 ** (np.mean(np.log10(y_pred / y_true)))

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "average_fold_error"


class AbsoluteAverageFoldError(Metric):
    """Calculate the absolute average fold error (AAFE).

    The AAFE is also known as the geometric mean fold error (GMFE).

    Attributes:
        name (str): Name of the scoring function (absolute_average_fold_error).
    """

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the absolute fold error.

        Args:
            y_true (np.array): Ground truth (correct) target values. 1d array.
            y_pred (np.array): 2D array (n_samples, n_tasks)

        Returns:
            float: The absolute average fold error.

        """
        return 10 ** (np.mean(np.abs(np.log10(y_pred / y_true))))

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return "absolute_average_fold_error"


class PercentageWithinFoldError(Metric):
    """Calculate the percentage of predictions within a certain fold error.

    Attributes:
        name (str): Name of the scoring function (percentage_within_{x}_fold_error).
    """

    def __init__(self, fold_error: float = 2):
        """Initialize the percentage within fold error scorer.

        Args:
            fold_error (float): The fold error threshold. Defaults to 2.
        """
        self.fold_error = fold_error

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """Calculate the percentage of predictions within a specified fold error.

        Args:
            y_true (np.array): Ground truth (correct) target values. 1d array.
            y_pred (np.array): 2D array (n_samples, n_tasks)

        Returns:
            float: The percentage of predictions within a fold error.

        """
        fold_errors = np.abs(np.log10(y_pred / y_true))
        return np.mean(fold_errors < np.log10(self.fold_error)) * 100

    def __str__(self) -> str:
        """Return the name of the scorer."""
        return f"percentage_within_{self.fold_error}_fold_error"
