"""Here model classes with GPU dependencies can be found.

At the moment this contains a class for fully-connected DNNs for py-boost models.
To add more a model class implementing the `QSPRModel` interface can be added,
see tutorial adding_new_components.


Created by: Martin Sicho
On: 12.05.23, 16:39
"""
import os
from copy import deepcopy
from importlib import import_module, util
from typing import Any, Optional, Type

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import train_test_split

from ...data.data import QSPRDataset
from ...deep import DEFAULT_DEVICE, DEFAULT_GPUS, SSPACE
from ...deep.models.neural_network import STFullyConnected
from ...models.interfaces import QSPRModel
from ...models.tasks import ModelTasks


class QSPRDNN(QSPRModel):
    """This class holds the methods for training and fitting a
    Deep Neural Net QSPR model initialization.

    Here the model instance is created and parameters can be defined.

    Attributes:
        name (str): name of the model
        data (QSPRDataset): data set used to train the model
        alg (estimator): estimator instance or class
        parameters (dict): dictionary of algorithm specific parameters
        estimator (object):
            the underlying estimator instance, if `fit` or optimization is perforemed,
            this model instance gets updated accordingly
        featureCalculators (MoleculeDescriptorsCalculator):
            feature calculator instance taken from the data set
            or deserialized from file if the model is loaded without data
        featureStandardizer (SKLearnStandardizer):
            feature standardizer instance taken from the data set
            or deserialized from file if the model is loaded without data
        metaInfo (dict):
            dictionary of metadata about the model, only available
            after the model is saved
        baseDir (str):
            base directory of the model, the model files
            are stored in a subdirectory `{baseDir}/{outDir}/`
        metaFile (str):
            absolute path to the metadata file of the model (`{outPrefix}_meta.json`)
        device (cuda device): cuda device
        gpus (int/ list of ints): gpu number(s) to use for model fitting
        patience (int):
            number of epochs to wait before early stop if no progress
            on validation set score
        tol (float):
            minimum absolute improvement of loss necessary to count as
            progress on best validation score
        nClass (int): number of classes
        nDim (int): number of features
        optimalEpochs (int): number of epochs to train the model for optimal performance
        device (torch.device): cuda device, cpu or gpu
        gpus (list[int]): gpu number(s) to use for model fitting
        patience (int):
            number of epochs to wait before early stop
            if no progress on validation set score
        optimalEpochs (int):
            number of epochs to train the model for optimal performance
    """
    def __init__(
        self,
        base_dir: str,
        alg: Type = STFullyConnected,
        data: QSPRDataset | None = None,
        name: str | None = None,
        parameters: dict | None = None,
        autoload: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        gpus: list[int] = DEFAULT_GPUS,
        patience: int = 50,
        tol: float = 0,
    ):
        """Initialize a QSPRDNN model.

        Args:
            base_dir (str):
                base directory of the model, the model files are stored in
                a subdirectory `{baseDir}/{outDir}/`
            alg (Type, optional):
                model class or instance. Defaults to STFullyConnected.
            data (QSPRDataset, optional):
                data set used to train the model. Defaults to None.
            name (str, optional):
                name of the model. Defaults to None.
            parameters (dict, optional):
                dictionary of algorithm specific parameters. Defaults to None.
            autoload (bool, optional):
                whether to load the model from file or not. Defaults to True.
            device (torch.device, optional):
                The cuda device. Defaults to `DEFAULT_DEVICE`.
            gpus (list[int], optional):
                gpu number(s) to use for model fitting. Defaults to `DEFAULT_GPUS`.
            patience (int, optional):
                number of epochs to wait before early stop if no progress
                on validation set score. Defaults to 50.
            tol (float, optional):
                minimum absolute improvement of loss necessary to count as progress
                on best validation score. Defaults to 0.
        """
        self.device = device
        self.gpus = gpus
        self.patience = patience
        self.tol = tol
        self.nClass = None
        self.nDim = None
        super().__init__(base_dir, alg, data, name, parameters, autoload=autoload)
        if self.task.isMultiTask():
            raise NotImplementedError(
                "Multitask modelling is not implemented for QSPRDNN models."
            )
        self.optimalEpochs = (
            self.parameters["n_epochs"]
            if self.parameters is not None and "n_epochs" in self.parameters else -1
        )

    @property
    def supportsEarlyStopping(self) -> bool:
        """Whether the model supports early stopping or not."""
        return True

    @classmethod
    def getDefaultParamsGrid(cls) -> list[list]:
        return SSPACE

    def loadEstimator(self, params: dict | None = None) -> object:
        """Load model from file or initialize new model.

        Args:
            params (dict, optional): model parameters. Defaults to None.

        Returns:
            model (object): model instance
        """
        if self.task.isRegression():
            self.nClass = 1
        else:
            self.nClass = (
                self.data.targetProperties[0].nClasses
                if self.data else self.metaInfo["n_class"]
            )
        self.nDim = self.data.X.shape[1] if self.data else self.metaInfo["n_dim"]
        # initialize model
        estimator = self.alg(
            n_dim=self.nDim,
            n_class=self.nClass,
            device=self.device,
            gpus=self.gpus,
            is_reg=self.task == ModelTasks.REGRESSION,
            patience=self.patience,
            tol=self.tol,
        )
        # set parameters if available and return
        new_parameters = self.getParameters(params)
        if new_parameters is not None:
            estimator.set_params(**new_parameters)
        return estimator

    def loadEstimatorFromFile(
        self, params: dict | None = None, fallback_load: bool = True
    ) -> object:
        """Load estimator from file.

        Args:
            params (dict): parameters
            fallback_load (bool):
                if `True`, init estimator from `alg` and `params` if no estimator
                found at path

        Returns:
            estimator (object): estimator instance
        """
        path = f"{self.outPrefix}_weights.pkg"
        estimator = self.loadEstimator(params)
        # load states if available
        if os.path.exists(path):
            estimator.load_state_dict(torch.load(path))
        elif not fallback_load:
            raise FileNotFoundError(
                f"No estimator found at {path}, "
                f"loading estimator weights from file failed."
            )
        return estimator

    def saveEstimator(self) -> str:
        """Save the QSPRDNN model.

        Returns:
            str: path to the saved model
        """
        path = f"{self.outPrefix}_weights.pkg"
        torch.save(self.estimator.state_dict(), path)
        return path

    def saveParams(self, params: dict) -> str:
        """Save model parameters to file.

        Args:
            params (dict): model parameters

        Returns:
            str: path to the saved parameters as json
        """
        return super().saveParams(
            {
                k: params[k]
                for k in params
                if not k.startswith("_") and k not in ["training", "device", "gpus"]
            }
        )

    def save(self) -> str:
        """Save the QSPRDNN model and meta information.

        Returns:
            str: path to the saved model
        """
        self.metaInfo["n_dim"] = self.nDim
        self.metaInfo["n_class"] = self.nClass
        return super().save()

    def fit(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        y: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any | None = None,
        early_stopping: bool | None = True,
        **kwargs
    ):
        """Fit the model to the given data matrix or `QSPRDataset`.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to fit
            y (pd.DataFrame, np.ndarray, QSPRDataset): target matrix to fit
            estimator (Any): estimator instance to use for fitting
            early_stopping (bool): if True, early stopping is used
            kwargs (dict): additional keyword arguments for the estimator's fit method

        Returns:
            Any: fitted estimator instance
            int, optional: in case of early stopping, the number of iterations
                after which the model stopped training
        """
        estimator = self.estimator if estimator is None else estimator
        X, y = self.convertToNumpy(X, y)

        if early_stopping:
            # split cross validation fold train set into train
            # and validation set for early stopping
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
            return estimator.fit(X_train, y_train, X_val, y_val, **kwargs)

        estimator, _ = estimator.fit(X, y, **kwargs)
        return estimator

    def predict(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any = None
    ) -> np.ndarray:
        """See `QSPRModel.predict`."""
        estimator = self.estimator if estimator is None else estimator
        scores = self.predictProba(X, estimator)
        # return class labels for classification
        if self.task.isClassification():
            return np.argmax(scores[0], axis=1, keepdims=True)
        else:
            return scores[0]

    def predictProba(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any = None
    ) -> np.ndarray:
        """See `QSPRModel.predictProba`."""
        estimator = self.estimator if estimator is None else estimator
        X = self.convertToNumpy(X)

        return [estimator.predict(X)]


@pytest.mark.skipif((spec := util.find_spec("cupy")) is None, reason="requires cupy")
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
        return False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        y: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Optional[Type[import_module("py_boost").GradientBoosting]] = None,
        early_stopping: bool = False,
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

        if early_stopping:
            # split cross validation fold train set into train
            # and validation set for early stopping
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
            estimator.fit(X_train, y_train, eval_sets=[{"X": X_val, "y": y_val}])

            return estimator, estimator.best_round

        estimator.fit(X, y)

        return estimator

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
