"""Early stopping for training of models."""
from enum import Enum
import numpy as np
import json
from typing import Callable, Any
import pandas as pd
from ..data.data import QSPRDataset


class EarlyStoppingMode(Enum):
    """Enum representing the type of early stopping to use.

    Attributes:
        NOT_RECORDING (str): early stopping, not recording number of epochs
        RECORDING (str):early stopping, recording number of epochs
        FIXED (str): no early stopping, specified number of epochs
        OPTIMAL (str): no early stopping, optimal number of epochs determined by
            previous training runs with early stopping (e.g. average number of epochs
            trained in cross validation with early stopping)
    """

    NOT_RECORDING = "NOT_RECORDING"
    RECORDING = "RECORDING"
    FIXED = "FIXED"
    OPTIMAL = "OPTIMAL"

    def __str__(self) -> str:
        """Return the name of the task."""
        return self.name

    def __bool__(self) -> bool:
        """Return whether early stopping is used."""
        return self in [EarlyStoppingMode.NOT_RECORDING, EarlyStoppingMode.RECORDING]


class EarlyStopping():
    """Early stopping for training of models.

    Attributes:
        mode (EarlyStoppingMode): early stopping mode
        numEpochs (int): number of epochs to train in FIXED mode.
        aggregatefunc (function): numpy function to aggregate trained epochs in OPTIMAL
            mode. Defaults to np.mean.
        trainedEpochs (list[int]): list of number of epochs trained in a model training
            with early stopping on RECORDING mode.
    """
    def __init__(
        self,
        mode: EarlyStoppingMode = EarlyStoppingMode.NOT_RECORDING,
        num_epochs: int | None = None,
        aggregate_func: Callable[[list[int]], int] = np.mean
    ):
        """Initialize early stopping.

        Args:
            mode (EarlyStoppingMode): early stopping mode
            num_epochs (int, optional): number of epochs to train in FIXED mode.
            aggregatefunc (function, optional): numpy function to aggregate trained
                epochs in OPTIMAL mode. Note, non-numpy functions are not supported.
        """
        self.mode = mode
        self.numEpochs = num_epochs
        self._trainedEpochs = []
        self.aggregateFunc = aggregate_func

    @property
    def optimalEpochs(self) -> int:
        """Number of epochs to train in OPTIMAL mode."""
        if len(self._trainedEpochs) == 0:
            raise ValueError(
                "No number of epochs have been recorded yet, first run fit with early "
                "stopping mode set to RECORDING or set the optimal number of epochs "
                "manually."
            )
        return int(np.round(self.aggregateFunc(self._trainedEpochs)))

    @property
    def trainedEpochs(self) -> list[int]:
        """"List of number of epochs trained in a model training with early stopping."""
        return self._trainedEpochs.copy()

    @trainedEpochs.setter
    def trainedEpochs(self, epochs: list[int]):
        """Set number of epochs trained in a model training with early stopping.

        Args:
            epochs (list[int]): list of number of epochs
        """
        self._trainedEpochs = epochs

    def recordEpochs(self, epochs):
        """Record number of epochs.

        Args:
            epochs (int): number of epochs
        """
        self._trainedEpochs.append(epochs)

    def getEpochs(self) -> int:
        """Get the number of epochs to train in a non-early stopping mode."""
        if self.mode == EarlyStoppingMode.FIXED:
            return self.numEpochs
        else:
            return self.optimalEpochs

    def __str__(self) -> str:
        """Return the name of the task."""
        return self.mode.name

    def toFile(self, path: str):
        """Save early stopping object to file.

        Args:
            path (str): path to file to save early stopping object to
        """
        with open(path, "w") as f:
            json.dump(
                {
                    "mode": self.mode.name,
                    "num_epochs": self.numEpochs,
                    "trained_epochs": self.trainedEpochs,
                    "aggregate_func_name": self.aggregateFunc.__name__,
                }, f
            )

    @classmethod
    def fromFile(cls, path: str) -> "EarlyStopping":
        """Load early stopping object from file.

        Args:
            path (str): path to file containing early stopping object
        """
        with open(path, "r") as f:
            data = json.load(f)
        mode = EarlyStoppingMode[data["mode"]]
        aggregate_func = getattr(np, data["aggregate_func_name"])
        early_stopping = cls(mode, data["num_epochs"], aggregate_func)
        early_stopping.trainedEpochs = data["trained_epochs"]
        return early_stopping

    def __bool__(self) -> bool:
        """Return whether early stopping is used."""
        return self.mode.__bool__()


def early_stopping(func):
    """Early stopping decorator for fit method of models that support early stopping."""
    def wrapper_fit(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        y: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any | None = None,
        mode: EarlyStoppingMode = EarlyStoppingMode.NOT_RECORDING,
        **kwargs
    ):
        """Wrapper for fit method of models that support early stopping.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to fit
            y (pd.DataFrame, np.ndarray, QSPRDataset): target matrix to fit
            estimator (Any): estimator instance to use for fitting
            early_stopping (EarlyStopping): early stopping object
            kwargs (dict): additional keyword arguments for the estimator's fit method

        Returns:
            Any: fitted estimator instance
            int, optional: in case of early stopping, the number of iterations
                after which the model stopped training
        """
        assert self.supportsEarlyStopping, (
            "early_stopping decorator can only be used for models that support"
            "early stopping."
        )
        self.earlyStopping.mode = mode
        estimator, best_epoch = func(self, X, y, estimator, mode, **kwargs)
        if mode == EarlyStoppingMode.RECORDING:
            self.earlyStopping.recordEpochs(best_epoch)
        return estimator

    return wrapper_fit
