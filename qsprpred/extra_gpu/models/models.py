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

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ...data.data import QSPRDataset
from ...models.early_stopping import EarlyStoppingMode, early_stopping
from ...models.interfaces import QSPRModel
from ...models.tasks import ModelTasks


class PyBoostModel(QSPRModel):
    """PyBoostModel class for pyboost models.
    Pyboost does gradient boosting with option to do multioutput and
    customizable loss and evaluation.
    For more information and tutorials see: https://github.com/sb-ai-lab/Py-Boost

    Wrap your pyboost model class in this class
    to use it with the `QSPRModel` interface.

    Example
    --------
    >>> from qsprpred.deep.models.models import PyBoostModel
    >>> parameters = {'loss':  'mse', 'metric': 'r2_score', 'verbose': -1}
    >>> model = PyBoostModel(
    ...     base_dir='qspr/models/',
    ...     data=dataset,
    ...     name="PyBoost",
    ...     parameters=parameters
    ... )
    """
    def __init__(
        self,
        base_dir: str,
        data: Optional[QSPRDataset] = None,
        name: Optional[str] = None,
        parameters: Optional[dict] = None,
        autoload=True
    ):
        """Initialize a QSPR model instance.

        If the model is loaded from file, the data set is not required.
        Note that the data set is required for fitting and optimization.

        Args:
            base_dir (str):
                base directory of the model,
                the model files are stored in a subdirectory `{baseDir}/{outDir}/`
            data (QSPRDataset): data set used to train the model
            name (str): name of the model
            parameters (dict): dictionary of algorithm specific parameters
            autoload (bool):
                if `True`, the estimator is loaded from the serialized file
                if it exists, otherwise a new instance of alg is created
        """
        super().__init__(
            base_dir,
            import_module("py_boost").GradientBoosting, data, name, parameters, autoload
        )
        if self.task == ModelTasks.MULTITASK_MIXED:
            raise ValueError(
                "MultiTask with a mix of classification and regression tasks "
                "is not supported for pyboost models."
            )
        if self.task == ModelTasks.MULTITASK_MULTICLASS:
            raise NotImplementedError(
                "Multi-task multi-class is not supported for pyboost models."
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
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        y: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Optional[Type[import_module("py_boost").GradientBoosting]] = None,
        mode: EarlyStoppingMode = EarlyStoppingMode.NOT_RECORDING,
        **kwargs
    ) -> import_module("py_boost").GradientBoosting:
        """Fit the model to the given data matrix or `QSPRDataset`.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to fit
            y (pd.DataFrame, np.ndarray, QSPRDataset): target matrix to fit
            estimator (Any): estimator instance to use for fitting
            early_stopping (bool): if True, early stopping is used
            kwargs: additional keyword arguments for the fit function

        Returns:
            (GzipKNNAlgorithm): fitted estimator instance
        """
        estimator = self.estimator if estimator is None else estimator
        X, y = self.convertToNumpy(X, y)

        if self.task == ModelTasks.MULTICLASS:
            y = np.squeeze(y)

        if self.earlyStopping:
            # split cross validation fold train set into train
            # and validation set for early stopping
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
            estimator.fit(X_train, y_train, eval_sets=[{"X": X_val, "y": y_val}])

            return estimator, estimator.best_round

        estimator.ntrees = self.earlyStopping.getEpochs()
        estimator.fit(X, y)

        return estimator, estimator.ntrees

    def predict(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any = None
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
            elif self.task.isMultiTask():  #multitask
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
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any = None
    ) -> np.ndarray:
        estimator = self.estimator if estimator is None else estimator
        X = self.convertToNumpy(X)

        preds = estimator.predict(X)
        if self.task.isClassification():
            if preds.shape[1] == 1:
                preds = np.concatenate((preds, 1 - preds), axis=1)
            elif self.task.isMultiTask():  #multitask
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
