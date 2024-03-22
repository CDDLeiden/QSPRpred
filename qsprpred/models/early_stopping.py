"""Early stopping for training of models."""
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..data.tables.qspr import QSPRDataset
from ..logs import logger
from ..utils.serialization import JSONSerializable


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


class EarlyStopping(JSONSerializable):
    """Early stopping tracker for training of QSPRpred models.

    An instance of this class is used to track the number of epochs trained in a model
    when early stopping (mode RECORDING) is used. This information can then be used
    to determine the optimal number of epochs to train in a model training without
    early stopping (mode OPTIMAL). The optimal number of epochs is determined by
    aggregating the number of epochs trained in previous model trainings with early
    stopping. The aggregation function can be specified by the user.
    The number of epochs to train in a model training without early stopping can also
    be specified manually (mode FIXED). Models can also be trained with early stopping
    without recording the number of epochs trained (mode NOT_RECORDING), e.g. useful
    when hyperparameter tuning is performed with early stopping.

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
        aggregate_func: Callable[[list[int]], int] = np.mean,
    ):
        """Initialize early stopping.

        Args:
            mode (EarlyStoppingMode): early stopping mode
            num_epochs (int, optional): number of epochs to train in FIXED mode.
            aggregate_func (function, optional): numpy function to aggregate trained
                epochs in OPTIMAL mode. Note, non-numpy functions are not supported.
        """
        self.mode = mode
        self.numEpochs = num_epochs
        self._trainedEpochs = []
        self.aggregateFunc = aggregate_func

    def __getstate__(self):
        state = super().__getstate__()
        state["aggregateFunc"] = self.aggregateFunc.__name__
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self.aggregateFunc = getattr(np, self.aggregateFunc)

    @property
    def optimalEpochs(self) -> int:
        """Return number of epochs to train in OPTIMAL mode."""
        if len(self._trainedEpochs) == 0:
            raise ValueError(
                "No number of epochs have been recorded yet, first run fit with early "
                "stopping mode set to RECORDING or set the optimal number of epochs "
                "manually."
            )
        optimal_epochs = int(np.round(self.aggregateFunc(self._trainedEpochs)))
        logger.debug(f"Optimal number of epochs: {optimal_epochs}")
        return optimal_epochs

    @property
    def trainedEpochs(self) -> list[int]:
        """Return list of number of epochs trained in early stopping mode RECORDING."""
        return self._trainedEpochs.copy()

    @trainedEpochs.setter
    def trainedEpochs(self, epochs: list[int]):
        """Set list of number of epochs trained in early stopping mode RECORDING."

        Args:
            epochs (list[int]): list of number of epochs
        """
        self._trainedEpochs = epochs

    def recordEpochs(self, epochs: int):
        """Record number of epochs.

        Args:
            epochs (int): number of epochs
        """
        logger.debug(f"Recorded best epoch: {epochs}")
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

    def __bool__(self) -> bool:
        """Return whether early stopping is used."""
        return self.mode.__bool__()

    def clean(self):
        """Clean early stopping object."""
        self._trainedEpochs = []


def early_stopping(func: Callable) -> Callable:
    """Early stopping decorator for fit method of models that support early stopping.

    Returns:
        function: decorated fit method
    """

    def wrapper_fit(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        y: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any | None = None,
        mode: EarlyStoppingMode | None = None,
        split: "DataSplit" = None,
        monitor: "FitMonitor" = None,
        **kwargs,
    ) -> Any:
        """Wrapper for fit method of models that support early stopping.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to fit
            y (pd.DataFrame, np.ndarray, QSPRDataset): target matrix to fit
            estimator (Any): estimator instance to use for fitting
            mode (EarlyStoppingMode): early stopping mode
            split (DataSplit): data split to use for early stopping,
                if None, a ShuffleSplit with 10% validation set size is used
            monitor (FitMonitor): monitor to use for fitting, if None, a BaseMonitor
                is used
            kwargs (dict): additional keyword arguments for the estimator's fit method

        Returns:
            Any: fitted estimator instance
        """
        assert self.supportsEarlyStopping, (
            "early_stopping decorator can only be used for models that support"
            " early stopping."
        )
        self.earlyStopping.mode = mode if mode is not None else self.earlyStopping.mode
        estimator, best_epoch = func(
            self, X, y, estimator, mode, split, monitor, **kwargs
        )
        if self.earlyStopping.mode == EarlyStoppingMode.RECORDING:
            self.earlyStopping.recordEpochs(best_epoch + 1)  # +1 for 0-indexing
        return estimator

    return wrapper_fit
