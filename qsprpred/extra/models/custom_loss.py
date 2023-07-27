"""This module holds two custom loss wrappers for pyboost,
as adapted from the tutorials in https://github.com/sb-ai-lab/Py-Boost
"""

import cupy as cp
from py_boost.gpu.losses import BCELoss, MSELoss


class MSEwithNaNLoss(MSELoss):
    "Masked MSE loss function"
    def base_score(self, y_true):
        # Replace .mean with nanmean function to calc base score
        return cp.nanmean(y_true, axis=0)

    def get_grad_hess(self, y_true, y_pred):
        # first, get nan mask for y_true
        mask = cp.isnan(y_true)
        # then, compute loss with any values at nan places just to prevent the exception
        grad, hess = super().get_grad_hess(cp.where(mask, 0, y_true), y_pred)
        # invert mask
        mask = (~mask).astype(cp.float32)
        # multiply grad and hess on inverted mask
        # now grad and hess eq. 0 on NaN points
        # that actually means that prediction on that place should not be updated
        grad = grad * mask
        hess = hess * mask

        return grad, hess


class BCEWithNaNLoss(BCELoss):
    "Masked BCE loss function"
    def base_score(self, y_true):
        # Replace .mean with nanmean function to calc base score
        means = cp.clip(
            cp.nanmean(y_true, axis=0), self.clip_value, 1 - self.clip_value
        )
        return cp.log(means / (1 - means))

    def get_grad_hess(self, y_true, y_pred):
        # first, get nan mask for y_true
        mask = cp.isnan(y_true)
        # then, compute loss with any values at nan places just to prevent the exception
        grad, hess = super().get_grad_hess(cp.where(mask, 0, y_true), y_pred)
        # invert mask
        mask = (~mask).astype(cp.float32)
        # multiply grad and hess on inverted mask
        # now grad and hess eq. 0 on NaN points
        # that actually means that prediction on that place should not be updated
        grad = grad * mask
        hess = hess * mask

        return grad, hess
