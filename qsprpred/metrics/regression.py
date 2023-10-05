"""  Metrics for regression tasks.  """

import scipy.stats
import sklearn.metrics


def _k_slope(y_true, y_pred):
    """Calculate the slope of the regression line through the origin
    between the predicted and observed values.

    Reference: Tropsha, A., & Golbraikh, A. (2010). In J.-L. Faulon & A. Bender (Eds.), Handbook of Chemoinformatics Algorithms.
    https://www.taylorfrancis.com/books/9781420082999

    Args:
        y_true (np.array): Ground truth (correct) target values. 1d array.
        y_pred (np.array): Estimate values. 1d array.

    Returns:
        float: The coefficient of determination.

    """
    if len(y_true) != len(y_pred):
        raise ValueError(f'y_true and y_pred must have the same dimensions: {len(y_true)} & {len(y_pred)}')
    num, denom = 0, 0
    for i in range(len(y_true)):
        num += y_true[i] * y_pred[i]
        denom += y_true[i] ** 2
    return num / denom if len(y_pred) >= 2 else 0


def _k_prime_slope(y_true, y_pred):
    """Calculate the slope of the regression line through the origin
    between the observed and predicted values.

    Reference: Tropsha, A., & Golbraikh, A. (2010). In J.-L. Faulon & A. Bender (Eds.), Handbook of Chemoinformatics Algorithms.
    https://www.taylorfrancis.com/books/9781420082999

    Args:
        y_true (np.array): Ground truth (correct) target values. 1d array.
        y_pred (np.array): Estimate values. 1d array.

    Returns:
        float: The coefficient of determination.

    """
    if len(y_true) != len(y_pred):
        raise ValueError(f'y_true and y_pred must have the same dimensions: {len(y_true)} & {len(y_pred)}')
    num, denom = 0, 0
    for i in range(len(y_true)):
        num += y_true[i] * y_pred[i]
        denom += y_pred[i] ** 2
    return num / denom if len(y_pred) >= 2 else 0

def _pearson(y_true, y_pred):
    """Calculate the Pearson correlation coefficient.

    Args:
        y_true (np.array): Ground truth (correct) target values. 1d array.
        y_pred (np.array): Estimate values. 1d array.

    Returns:
        float: The Pearson correlation coefficient.

    """
    return scipy.stats.pearsonr(y_true, y_pred)[0] if len(y_pred) >= 2 else 0


def _spearman(y_true, y_pred):
    """Calculate the Spearman correlation coefficient.

    Args:
        y_true (np.array): Ground truth (correct) target values. 1d array.
        y_pred (np.array): Estimate values. 1d array.

    Returns:
        float: The Pearson Spearman coefficient.

    """
    return scipy.stats.spearmanr(y_true, y_pred)[0] if len(y_pred) >= 2 else 0


def _kendall(y_true, y_pred):
    """Calculate the Kendall rank correlation coefficient.

    Args:
        y_true (np.array): Ground truth (correct) target values. 1d array.
        y_pred (np.array): Estimate values. 1d array.

    Returns:
        float: The Kendall rank correlation coefficient.

    """
    return scipy.stats.kendalltau(y_true, y_pred)[0] if len(y_pred) >= 2 else 0


def _r_2_0(y_true, y_pred):
    """Calculate the coefficient of determination for regression line
    through the origin between the observed and predicted values.

    Reference: Tropsha, A., & Golbraikh, A. (2010). In J.-L. Faulon & A. Bender (Eds.), Handbook of Chemoinformatics Algorithms.
    https://www.taylorfrancis.com/books/9781420082999

    Args:
        y_true (np.array): Ground truth (correct) target values. 1d array.
        y_pred (np.array): Estimate values. 1d array.

    Returns:
        float: The coefficient of determination.

    """
    if len(y_true) != len(y_pred):
        raise ValueError(f'y_true and y_pred must have the same dimensions: {len(y_true)} & {len(y_pred)}')
    k_prime = _k_prime_slope(y_true, y_pred)
    y_true_mean = y_true.mean()
    num, denom = 0, 0
    for i in range(len(y_true)):
        num += y_true[i] - k_prime * y_pred[i]
        denom += (y_true[i] - y_true_mean) ** 2
    return 1 - num / denom if len(y_pred) >= 2 else 0


def _r_prime_2_0(y_true, y_pred):
    """Calculate the coefficient of determination for regression line
    through the origin between the predicted and observed values.

    Reference: Tropsha, A., & Golbraikh, A. (2010). In J.-L. Faulon & A. Bender (Eds.), Handbook of Chemoinformatics Algorithms.
    https://www.taylorfrancis.com/books/9781420082999

    Args:
        y_true (np.array): Ground truth (correct) target values. 1d array.
        y_pred (np.array): Estimate values. 1d array.

    Returns:
        float: The coefficient of determination.

    """
    if len(y_true) != len(y_pred):
        raise ValueError(f'y_true and y_pred must have the same dimensions: {len(y_true)} & {len(y_pred)}')
    k = _k_slope(y_true, y_pred)
    y_pred_mean = y_pred.mean()
    num, denom = 0, 0
    for i in range(len(y_true)):
        num += y_pred[i] - k * y_true[i]
        denom += (y_pred[i] - y_pred_mean) ** 2
    return 1 - num / denom if len(y_pred) >= 2 else 0


pearson_corrcoeff = sklearn.metrics.make_scorer(_pearson)
spearman_corrcoeff = sklearn.metrics.make_scorer(_spearman)
kendall_coeff = sklearn.metrics.make_scorer(_kendall)
k_slope = sklearn.metrics.make_scorer(_k_slope)
k_prime_slope = sklearn.metrics.make_scorer(_k_prime_slope)
r_2_0_score = sklearn.metrics.make_scorer(_r_2_0)
r_prime_2_0_score = sklearn.metrics.make_scorer(_r_prime_2_0)
