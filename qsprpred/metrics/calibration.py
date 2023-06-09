"""  Calibration error metrics.  """

import numpy as np

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
