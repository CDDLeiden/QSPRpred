"""  Metrics for classification tasks.  """

from importlib import import_module

import sklearn.metrics


# Include RDKit defined classification metrics if is installed
try:
    _ = import_module('rdkit')
    from ..extra.metrics.classification import *
except:
    pass

def _specificity_score(y_true, y_pred):
    """Calculate the specificity (selectivity).

    Args:
        y_true (np.array): Ground truth (correct) labels. 1d array.
        y_pred (np.array): Predicted labels. 1d array.

    Returns:
        float: The specificity.

    """
    tn, fp, _, _ = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def _negative_predicted_value_score(y_true, y_pred):
    """Calculate the negative predicted value.

    Args:
        y_true (np.array): Ground truth (correct) labels. 1d array.
        y_pred (np.array): Predicted labels. 1d array.

    Returns:
        float: The negative predicted value.

    """
    tn, _, fn, _ = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn)


def _balanced_matthews_corrcoeff(y_true, y_pred):
    """Calculate the balanced Matthews correlation coefficient.

    Reference:

    Args:
        y_true (np.array): Ground truth (correct) labels. 1d array.
        y_pred (np.array): Predicted labels. 1d array.

    Returns:
        float: The correlation coefficient.

    """
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    prevalence = sum(y_true) / len(y_true)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    return ((sen + spe - 1) / ((sen + (1 - spe) * (1 - prevalence) / prevalence) * (spe + (1-sen) * prevalence / (1 - prevalence))) ** 0.5)


def _prevalence_score(y_true, y_pred):
    """Calculate the prevalence.

    Args:
        y_true (np.array): Ground truth (correct) labels. 1d array.
        y_pred (np.array): Predicted labels. 1d array.

    Returns:
        float: The prevalence.

    """
    return sum(y_true) / len(y_true)


def _balanced_positive_predicted_value_score(y_true, y_pred):
    """Calculate the balanced positive predicted value.

    Reference:

    Args:
        y_true (np.array): Ground truth (correct) labels. 1d array.
        y_pred (np.array): Predicted labels. 1d array.

    Returns:
        float: The balanced positive predicted value.

    """
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    prevalence = sum(y_true) / len(y_true)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    return (sen * prevalence) / (sen * prevalence + (1 - spe) * (1 - prevalence))


def _balanced_negative_predicted_value_score(y_true, y_pred):
    """Calculate the balanced negative predicted value.

    Reference:

    Args:
        y_true (np.array): Ground truth (correct) labels. 1d array.
        y_pred (np.array): Predicted labels. 1d array.

    Returns:
        float: The balanced negative predicted value.

    """
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    prevalence = sum(y_true) / len(y_true)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    return (spe * (1 - prevalence)) / (spe * (1 - prevalence) + (1 - sen) * prevalence)


def _balanced_cohen_kappa_score(y_true, y_pred):
    """Calculate the balanced Cohen kappa coefficient.

    Reference:

    Args:
        y_true (np.array): Ground truth (correct) labels. 1d array.
        y_pred (np.array): Predicted labels. 1d array.

    Returns:
        float: The balanced Cohen kappa coefficient.

    """
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    prevalence = sum(y_true) / len(y_true)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    return (2 * (sen + spe - 1)) / ((sen + (1 - spe) * (1 - prevalence) / (prevalence)) + (spe + (1 - sen) * (prevalence) / (1 - prevalence)))


# Make scikit-learn scorers
sensitivity_score = sklearn.metrics.recall_score
positive_predicted_value_score = sklearn.metrics.precision_score
specificity_score = sklearn.metrics.make_scorer(_specificity_score)
negative_predicted_value_score = sklearn.metrics.make_scorer(_negative_predicted_value_score)
balanced_matthews_corrcoeff = sklearn.metrics.make_scorer(_balanced_matthews_corrcoeff)
prevalence_score = sklearn.metrics.make_scorer(_prevalence_score)
balanced_positive_predicted_value_score = sklearn.metrics.make_scorer(_balanced_positive_predicted_value_score)
balanced_negative_predicted_value_score = sklearn.metrics.make_scorer(_balanced_negative_predicted_value_score)
balanced_cohen_kappa_score = sklearn.metrics.make_scorer(_balanced_cohen_kappa_score)
