"""Here the DNN model originally from DrugEx can be found.

At the moment this contains a class for fully-connected DNNs.
To add more a model class implementing the `QSPRModel` interface can be added,
see tutorial adding_new_components.
"""
import os
from typing import Any, Type

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ShuffleSplit

from ....logs import logger
from ....data.data import QSPRDataset
from ....data.interfaces import DataSplit
from ....extra.gpu import DEFAULT_DEVICE, DEFAULT_GPUS, SSPACE
from ....extra.gpu.models.neural_network import STFullyConnected
from ....models.early_stopping import EarlyStoppingMode, early_stopping
from ....models.interfaces import FitMonitor, QSPRModel
from ....models.monitors import BaseMonitor
from ....models.tasks import ModelTasks


class DNNModel(QSPRModel):
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
        random_state (int):
            seed for the random state
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
        random_state: int | None = None,
        autoload: bool = True,
        device: torch.device = DEFAULT_DEVICE,
        gpus: list[int] = DEFAULT_GPUS,
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
        super().__init__(
            base_dir,
            alg,
            data,
            name,
            parameters,
            autoload=autoload,
            random_state=random_state,
        )
        if self.task.isMultiTask():
            raise NotImplementedError(
                "Multitask modelling is not implemented for DNNModel models."
            )

    def initRandomState(self, random_state):
        """Set random state if applicable.
        Defaults to random state of dataset if no random state is provided by the constructor.

        Args:
            random_state (int): Random state to use for shuffling and other random operations.
        """
        new_random_state = random_state or (
            self.data.randomState if self.data is not None else
            int(np.random.randint(0, 2**32 - 1, dtype=np.int64))
        )
        self.randomState = new_random_state
        if new_random_state is None:
            logger.warning(
                "No random state supplied, "
                "and could not find random state on the dataset."
            )
        self.randomState = random_state
        if random_state is not None:
            torch.manual_seed(random_state)

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
        self.initRandomState(self.randomState)
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
        """Save the DNNModel model.

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
        if params is None:
            return super().saveParams(params)
        return super().saveParams(
            {
                k: params[k]
                for k in params
                if not k.startswith("_") and k not in ["training", "device", "gpus"]
            }
        )

    def save(self) -> str:
        """Save the DNNModel model and meta information.

        Returns:
            str: path to the saved model
        """
        self.metaInfo["n_dim"] = self.nDim
        self.metaInfo["n_class"] = self.nClass
        return super().save()

    @early_stopping
    def fit(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        y: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any | None = None,
        mode: EarlyStoppingMode = EarlyStoppingMode.NOT_RECORDING,
        split: DataSplit | None = None,
        monitor: FitMonitor | None = None,
        **kwargs,
    ):
        """Fit the model to the given data matrix or `QSPRDataset`.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to fit
            y (pd.DataFrame, np.ndarray, QSPRDataset): target matrix to fit
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
        monitor = BaseMonitor() if monitor is None else monitor
        estimator = self.estimator if estimator is None else estimator
        split = split or ShuffleSplit(
            n_splits=1, test_size=0.1, random_state=self.data.randomState
        )
        X, y = self.convertToNumpy(X, y)
        monitor.onFitStart(self)

        if self.earlyStopping:
            # split cross validation fold train set into train
            # and validation set for early stopping
            train_index, val_index = next(split.split(X, y))
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

        # set fixed number of epochs if early stopping is not used
        estimator.n_epochs = self.earlyStopping.getEpochs()
        estimator_fit = estimator.fit(X, y, monitor=monitor, **kwargs)
        monitor.onFitEnd(estimator_fit[0])
        return estimator_fit

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