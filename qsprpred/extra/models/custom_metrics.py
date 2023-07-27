"""This module holds custom metric wrappers for pyboost,
as adapted from the tutorials in https://github.com/sb-ai-lab/Py-Boost
"""
import numpy as np
from py_boost.gpu.losses.metrics import Metric


class NaNRMSEScore(Metric):
    "Masked RMSE score with weights based on number of non-zero values per task"
    def __call__(self, y_true, y_pred, sample_weight=None):
        mask = ~np.isnan(y_true)

        if sample_weight is not None:
            err = ((y_true - y_pred)[mask]**2 *
                   sample_weight[mask]).sum(axis=0) / sample_weight[mask].sum()
        else:
            err = np.nanmean((np.where(mask, (y_true - y_pred), np.nan)**2), axis=0)

        return np.average(err, weights=np.count_nonzero(mask, axis=0))

    def compare(self, v0, v1):

        return v0 > v1


class NaNR2Score(Metric):
    "Masked R2 score with weights based on number of non-zero values per task"
    def __call__(self, y_true, y_pred, sample_weight=None):
        mask = ~np.isnan(y_true)

        if sample_weight is not None:
            err = ((y_true - y_pred)[mask]**2 *
                   sample_weight[mask]).sum(axis=0) / sample_weight[mask].sum()
            std = ((y_true[mask] - y_true[mask].mean(axis=0))**2 *
                   sample_weight[mask]).sum(axis=0) / sample_weight[mask].sum()
        else:
            err = np.nanmean((np.where(mask, (y_true - y_pred), np.nan)**2), axis=0)
            std = np.nanvar(np.where(mask, y_true, np.nan), axis=0)

        return np.average(1 - err / std, weights=np.count_nonzero(mask, axis=0))

    def compare(self, v0, v1):

        return v0 > v1
