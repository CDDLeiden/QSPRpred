"""Here model classes with GPU dependencies can be found.

At the moment this contains a class for py-boost models.
To add more a model class implementing the `QSPRModel` interface can be added,
see tutorial adding_new_components.


Created by: Linde Schoenmaker
On: 03.08.2023, 15:26
"""
import os
from copy import deepcopy
from importlib import import_module
from typing import Any, Optional, Type

import cupy as cp
import joblib
import numpy as np
import pandas as pd
from py_boost.gpu.losses import BCELoss, MSELoss
from py_boost.gpu.losses.metrics import Metric, auc
from sklearn.model_selection import ShuffleSplit

from ....data.sampling.splits import DataSplit
from ....data.tables.qspr import QSPRDataset
from ....models.early_stopping import EarlyStoppingMode, early_stopping
from ....models.model import QSPRModel
from ....models.monitors import BaseMonitor, FitMonitor
from ....tasks import ModelTasks


class PyBoostModel(QSPRModel):
    """PyBoostModel class for pyboost that can handle missing data models.
    Pyboost does gradient boosting with option to do multioutput and
    customizable loss and evaluation.
    For more information and tutorials see: https://github.com/sb-ai-lab/Py-Boost

    Wrap your pyboost model class in this class
    to use it with the `QSPRModel` interface.

    Example
    --------
    >>> from qsprpred.extra.gpu.models.models import PyBoostModel
    >>> parameters = {'loss':  'mse', 'metric': 'r2_score', 'verbose': -1}
    >>> model = PyBoostModel(
    ...     base_dir='qspr/models/',
    ...     name="PyBoost",
    ...     parameters=parameters
    ... )
    """

    def __init__(
        self,
        base_dir: str,
        name: Optional[str] = None,
        parameters: Optional[dict] = None,
        autoload=True,
        random_state: Optional[int] = None,
    ):
        """Initialize a QSPR model instance.

        If the model is loaded from file, the data set is not required.
        Note that the data set is required for fitting and optimization.

        Args:
            base_dir (str):
                base directory of the model,
                the model files are stored in a subdirectory `{baseDir}/{outDir}/`
            name (str): name of the model
            parameters (dict): dictionary of algorithm specific parameters
            autoload (bool):
                if `True`, the estimator is loaded from the serialized file
                if it exists, otherwise a new instance of alg is created
            random_state (int): random state to use for the model
        """
        super().__init__(
            base_dir,
            import_module("py_boost").GradientBoosting,
            name,
            parameters,
            autoload,
            random_state,
        )

    @property
    def supportsEarlyStopping(self) -> bool:
        """Check if the model supports early stopping.

        Returns:
            (bool): whether the model supports early stopping or not
        """
        return True

    @early_stopping
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        estimator: Optional[Type[import_module("py_boost").GradientBoosting]] = None,
        mode: EarlyStoppingMode = EarlyStoppingMode.NOT_RECORDING,
        split: DataSplit | None = None,
        monitor: FitMonitor | None = None,
        **kwargs,
    ) -> import_module("py_boost").GradientBoosting:
        """Fit the model to the given data matrix or `QSPRDataset`.

        Args:
            X (pd.DataFrame, np.ndarray): data matrix to fit
            y (pd.DataFrame, np.ndarray): target matrix to fit
            estimator (Any): estimator instance to use for fitting
            mode (EarlyStoppingMode): mode to use for early stopping
            split (DataSplit): data split to use for early stopping,
                if None, a ShuffleSplit with 10% validation set size is used
            monitor (FitMonitor): monitor to use for fitting, if None, a BaseMonitor
            kwargs: additional keyword arguments for the fit function

        Returns:
            (Pyboost): fitted estimator instance
        """
        if self.task == ModelTasks.MULTITASK_MIXED:
            raise ValueError(
                "MultiTask with a mix of classification and regression tasks "
                "is not supported for pyboost that can handle missing data models."
            )
        if self.task == ModelTasks.MULTITASK_MULTICLASS:
            raise NotImplementedError(
                "Multi-task multi-class is not supported for "
                "pyboost that can handle missing data models."
            )
        if self.task == ModelTasks.MULTITASK_SINGLECLASS:
            # FIX ME:  PyBoost default auc loss does not handle multitask data
            # and the custom NaN AUC metric is not JSON serializable.
            raise NotImplementedError(
                "Multi-class is not supported for pyboost "
                "that can handle missing data models."
            )
        monitor = BaseMonitor() if monitor is None else monitor
        estimator = self.estimator if estimator is None else estimator
        split = split or ShuffleSplit(
            n_splits=1, test_size=0.1, random_state=self.randomState
        )
        X, y = self.convertToNumpy(X, y)

        if self.task == ModelTasks.MULTICLASS:
            y = np.squeeze(y)

        if self.earlyStopping:
            # split cross validation fold train set into train
            # and validation set for early stopping
            train_index, val_index = next(split.split(X, y))
            monitor.onFitStart(
                self,
                X[train_index, :],
                y[train_index],
                X[val_index, :],
                y[val_index],
            )
            estimator.fit(
                X[train_index, :],
                y[train_index],
                eval_sets=[{"X": X[val_index, :], "y": y[val_index]}],
            )
            monitor.onFitEnd(estimator, estimator.best_round)
            return estimator, estimator.best_round

        monitor.onFitStart(self, X, y)
        estimator.params.update({"ntrees": self.earlyStopping.getEpochs()})
        estimator.fit(X, y)

        monitor.onFitEnd(estimator)
        return estimator, self.earlyStopping.getEpochs()

    def predict(
        self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ) -> np.ndarray:
        """Make predictions for the given data matrix or `QSPRDataset`.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to predict
            estimator (Any): estimator instance to use for fitting

        Returns:
            np.ndarray:
                2D array containing the predictions, where each row corresponds
                to a sample in the data and each column to a target property
        """
        estimator = self.estimator if estimator is None else estimator
        X = self.convertToNumpy(X)

        preds = estimator.predict(X)

        if self.task.isClassification():
            if preds.shape[1] == 1:
                preds = np.concatenate(
                    (1 - preds, preds), axis=1
                )  # return 1 if predict proba > 0.5
                return np.argmax(preds, axis=1, keepdims=True)
            elif self.task.isMultiTask():  # multitask
                preds_mt = np.array([]).reshape(preds.shape[0], 0)
                for i in range(preds.shape[1]):
                    preds_task = preds[:, i].reshape(-1, 1)
                    preds_task = np.concatenate((1 - preds_task, preds_task), axis=1)
                    preds_task = np.argmax(preds_task, axis=1, keepdims=True)
                    preds_mt = np.hstack([preds_mt, preds_task])
                return preds_mt
            else:  # multiclass
                return np.argmax(preds, axis=1, keepdims=True)
        else:
            return preds

    def predictProba(
        self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ) -> np.ndarray:
        estimator = self.estimator if estimator is None else estimator
        X = self.convertToNumpy(X)

        preds = estimator.predict(X)
        if self.task.isClassification():
            if preds.shape[1] == 1:
                preds = np.concatenate((preds, 1 - preds), axis=1)
            elif self.task.isMultiTask():  # multitask
                preds_mt = []
                for i in range(preds.shape[1]):
                    preds_task = preds[:, i].reshape(-1, 1)
                    preds_mt.append(
                        np.concatenate((preds_task, 1 - preds_task), axis=1)
                    )
                return preds_mt

        # if preds is a numpy array, convert it to a list
        # to be consistent with the multiclass-multitask case
        if isinstance(preds, np.ndarray):
            preds = [preds]
        return preds

    def loadEstimator(self, params: Optional[dict] = None) -> object:
        """Initialize estimator instance with the given parameters.

        If `params` is `None`, the default parameters will be used.

        Arguments:
            params (dict): algorithm parameters

        Returns:
            object: initialized estimator instance
        """
        if params:
            if self.parameters is not None:
                temp_params = deepcopy(self.parameters)
                temp_params.update(params)
                return self.alg(**temp_params)
            else:
                return self.alg(**params)
        elif self.parameters is not None:
            return self.alg(**self.parameters)
        else:
            return self.alg()

    def loadEstimatorFromFile(
        self, params: Optional[dict] = None, fallback_load=True
    ) -> object:
        """Load estimator instance from file and apply the given parameters.

        Args:
            params (dict): algorithm parameters

        Returns:
            object: initialized estimator instance
        """
        path = f"{self.outPrefix}.pkl"
        if os.path.isfile(path):
            estimator = joblib.load(path)
            new_parameters = self.getParameters(params)
            if new_parameters is not None:
                estimator.params.update(new_parameters)
            return estimator
        elif fallback_load:
            return self.loadEstimator(params)
        else:
            raise FileNotFoundError(
                f"No estimator found at {path}, loading estimator from file failed."
            )

    def saveEstimator(self) -> str:
        """Save the underlying estimator to file.

        Returns:
            path (str): path to the saved estimator
        """
        estimator_path = f"{self.outPrefix}.pkl"

        joblib.dump(self.estimator, estimator_path)

        return estimator_path


class MSEwithNaNLoss(MSELoss):
    """
    Masked MSE loss function. Custom loss wrapper for pyboost that can handle
    missing data, as adapted from the tutorials in https://github.com/sb-ai-lab/Py-Boost
    """

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
    """
    Masked BCE loss function. Custom loss wrapper for pyboost that can handle missing
    data, as adapted from the tutorials in https://github.com/sb-ai-lab/Py-Boost
    """

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


class NaNRMSEScore(Metric):
    """
    Masked RMSE score with weights based on number of non-zero values per task.
    Custom metric wrapper for pyboost that can handle missing data,
    as adapted from  the tutorials in https://github.com/sb-ai-lab/Py-Boost
    """

    def __call__(self, y_true, y_pred, sample_weight=None):
        mask = ~np.isnan(y_true)

        if sample_weight is not None:
            err = ((y_true - y_pred)[mask] ** 2 * sample_weight[mask]).sum(
                axis=0
            ) / sample_weight[mask].sum()
        else:
            err = np.nanmean((np.where(mask, (y_true - y_pred), np.nan) ** 2), axis=0)

        return np.average(err, weights=np.count_nonzero(mask, axis=0))

    def compare(self, v0, v1):
        return v0 > v1


class NaNR2Score(Metric):
    """
    Masked R2 score with weights based on number of non-zero values per task.
    Custom metric wrapper for pyboost that can handle missing data,
    as adapted from  the tutorials in https://github.com/sb-ai-lab/Py-Boost
    """

    def __call__(self, y_true, y_pred, sample_weight=None):
        mask = ~np.isnan(y_true)

        if sample_weight is not None:
            err = ((y_true - y_pred)[mask] ** 2 * sample_weight[mask]).sum(
                axis=0
            ) / sample_weight[mask].sum()
            std = (
                (y_true[mask] - y_true[mask].mean(axis=0)) ** 2 * sample_weight[mask]
            ).sum(axis=0) / sample_weight[mask].sum()
        else:
            err = np.nanmean((np.where(mask, (y_true - y_pred), np.nan) ** 2), axis=0)
            std = np.nanvar(np.where(mask, y_true, np.nan), axis=0)

        return np.average(1 - err / std, weights=np.count_nonzero(mask, axis=0))

    def compare(self, v0, v1):
        return v0 > v1


class NaNAucMetric(Metric):
    """
    Masked AUC score with weights based on number of non-zero values per task.
    Custom metric wrapper for pyboost that can handle missing data,
    as adapted from  the tutorials in https://github.com/sb-ai-lab/Py-Boost
    """

    def __call__(self, y_true, y_pred, sample_weight=None):
        aucs = []
        mask = ~cp.isnan(y_true)

        for i in range(y_true.shape[1]):
            m = mask[:, i]
            w = None if sample_weight is None else sample_weight[:, 0][m]
            aucs.append(auc(y_true[:, i][m], y_pred[:, i][m], w))

        return np.mean(aucs)

    def compare(self, v0, v1):
        return v0 > v1
