"""Here the DNN model originally from DrugEx can be found.

At the moment this contains a class for fully-connected DNNs.
"""
import os
from typing import Any, Type

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ShuffleSplit

from qsprpred.tasks import ModelTasks
from .. import DEFAULT_TORCH_DEVICE, DEFAULT_TORCH_GPUS
from ....data.sampling.splits import DataSplit
from ....data.tables.qspr import QSPRDataset
from ....extra.gpu.models.neural_network import STFullyConnected
from ....models.early_stopping import EarlyStoppingMode, early_stopping
from ....models.model import QSPRModel
from ....models.monitors import BaseMonitor, FitMonitor


class DNNModel(QSPRModel):
    """This class holds the methods for training and fitting a
    Deep Neural Net QSPR model initialization.

    Here the model instance is created and parameters can be defined.

    Attributes:
        name (str): name of the model
        alg (estimator): estimator instance or class
        parameters (dict): dictionary of algorithm specific parameters
        estimator (object):
            the underlying estimator instance, if `fit` or optimization is performed,
            this model instance gets updated accordingly
        featureCalculators (MoleculeDescriptorsCalculator):
            feature calculator instance taken from the data set
            or deserialized from file if the model is loaded without data
        featureStandardizer (SKLearnStandardizer):
            feature standardizer instance taken from the data set
            or deserialized from file if the model is loaded without data
        baseDir (str):
            base directory of the model, the model files
            are stored in a subdirectory `{baseDir}/{outDir}/`
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
        device (torch.device): cuda device, cpu or gpu
        gpus (list[int]): gpu number(s) to use for model fitting
        patience (int):
            number of epochs to wait before early stop
            if no progress on validation set score
    """

    def __init__(
        self,
        base_dir: str,
        alg: Type = STFullyConnected,
        name: str | None = None,
        parameters: dict | None = None,
        random_state: int | None = None,
        autoload: bool = True,
        device: torch.device = DEFAULT_TORCH_DEVICE,
        gpus: list[int] = DEFAULT_TORCH_GPUS,
        patience: int = 50,
        tol: float = 0,
    ):
        """Initialize a DNNModel model.

        Args:
            base_dir (str):
                base directory of the model, the model files are stored in
                a subdirectory `{baseDir}/{outDir}/`
            alg (Type, optional):
                model class or instance. Defaults to STFullyConnected.
            name (str, optional):
                name of the model. Defaults to None.
            parameters (dict, optional):
                dictionary of algorithm specific parameters. Defaults to None.
            autoload (bool, optional):
                whether to load the model from file or not. Defaults to True.
            device (torch.device, optional):
                The cuda device. Defaults to `DEFAULT_TORCH_DEVICE`.
            gpus (list[int], optional):
                gpu number(s) to use for model fitting. Defaults to `DEFAULT_TORCH_GPUS`.
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
        super().__init__(
            base_dir,
            alg,
            name,
            parameters,
            autoload=autoload,
            random_state=random_state,
        )

    def initRandomState(self, random_state):
        """Set random state if applicable.
        Defaults to random state of dataset if no random state is provided by the constructor.

        Args:
            random_state (int): Random state to use for shuffling and other random operations.
        """
        super().initRandomState(random_state)
        if random_state is not None:
            torch.manual_seed(random_state)

    @property
    def supportsEarlyStopping(self) -> bool:
        """Whether the model supports early stopping or not."""
        return True

    def initFromDataset(self, data: QSPRDataset | None):
        super().initFromDataset(data)
        if self.targetProperties[0].task.isRegression():
            self.nClass = 1
        elif data is not None:
            self.nClass = self.targetProperties[0].nClasses
        if data is not None:
            self.nDim = data.getFeatures()[0].shape[1]

    def loadEstimator(self, params: dict | None = None) -> object:
        """Load model from file or initialize new model.

        Args:
            params (dict, optional): model parameters. Defaults to None.

        Returns:
            model (object): model instance
        """
        if self.nClass is None or self.nDim is None:
            return "Uninitialized model."
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
        """Save the DNNModel model.

        Returns:
            str: path to the saved model
        """
        path = f"{self.outPrefix}_weights.pkg"
        if not isinstance(self.estimator, str):
            torch.save(self.estimator.state_dict(), path)
        else:
            # just save the estimator message
            with open(path, "w") as f:
                f.write(self.estimator)
        return path

    @early_stopping
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        estimator: Any | None = None,
        mode: EarlyStoppingMode = EarlyStoppingMode.NOT_RECORDING,
        split: DataSplit | None = None,
        monitor: FitMonitor | None = None,
        **kwargs,
    ):
        """Fit the model to the given data matrix or `QSPRDataset`.

        Args:
            X (pd.DataFrame, np.ndarray): data matrix to fit
            y (pd.DataFrame, np.ndarray): target matrix to fit
            estimator (Any): estimator instance to use for fitting
            mode (EarlyStoppingMode): early stopping mode
            split (DataSplit): data split to use for early stopping,
                if None, a ShuffleSplit with 10% validation set size is used
            monitor (FitMonitor): fit monitor instance, if None, a BaseMonitor is used
            kwargs (dict): additional keyword arguments for the estimator's fit method

        Returns:
            Any: fitted estimator instance
            int, optional: in case of early stopping, the number of iterations
                after which the model stopped training
        """
        if self.task.isMultiTask():
            raise NotImplementedError(
                "Multitask modelling is not implemented for this model."
            )
        monitor = BaseMonitor() if monitor is None else monitor
        estimator = self.estimator if estimator is None else estimator
        split = split or ShuffleSplit(
            n_splits=1, test_size=0.1, random_state=self.randomState
        )
        X, y = self.convertToNumpy(X, y)
        # fit with early stopping
        if self.earlyStopping:
            # split cross validation fold train set into train
            # and validation set for early stopping
            train_index, val_index = next(split.split(X, y))
            monitor.onFitStart(
                self, X[train_index, :], y[train_index], X[val_index, :], y[val_index]
            )
            estimator_fit = estimator.fit(
                X[train_index, :],
                y[train_index],
                X[val_index, :],
                y[val_index],
                monitor=monitor,
                **kwargs,
            )
            monitor.onFitEnd(estimator_fit[0], estimator_fit[1])
            return estimator_fit
        monitor.onFitStart(self, X, y)
        # set fixed number of epochs if early stopping is not used
        estimator.n_epochs = self.earlyStopping.getEpochs()
        estimator_fit = estimator.fit(X, y, monitor=monitor, **kwargs)
        monitor.onFitEnd(estimator_fit[0])
        return estimator_fit

    def predict(
        self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
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
        self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ) -> np.ndarray:
        """See `QSPRModel.predictProba`."""
        estimator = self.estimator if estimator is None else estimator
        X = self.convertToNumpy(X)

        return [estimator.predict(X)]
