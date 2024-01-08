import json
import numpy as np
import pandas as pd
import os
import types

from .models import QSPRModel
from ..data.tables.qspr import QSPRDataset
from typing import Any

class RandomModel(QSPRModel):
    def __init__(self,
        base_dir: str,
        alg=None,
        data: QSPRDataset = None,
        name: str | None = None,
        parameters: dict | None = None,
        autoload: bool = True,
        random_state: int | None = None,
    ):
        self.baseDir = os.path.abspath(base_dir.rstrip("/"))
        self.name = name
        self.parameters = parameters
        self.estimator = self.loadEstimator(self.parameters)

        # Make alg an anonymous object having all necessary properties
        self.alg = types.SimpleNamespace()
        self.alg.__name__ = "Random"
        super().initFromData(data)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        y: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any = None,
        mode: Any = None,
        monitor: None = None,
        **kwargs,
    ):
        # Values of X are irrelevant
        if (self.task.isClassification()):
            # Save ratios, these will be used as probability that a given bucket is
            # chosen per target

            # TODO: this is probably clumsy, I shouldn't have to go from dataframe
            # to dict to dataframe
            estimator = pd.DataFrame({
                "ratios": pd.DataFrame.from_dict({col: y[col].value_counts() / y.shape[0] for col in list(y)})
            })

        if (self.task.isRegression()):
            # Calculate the mean and standard deviation of each column
            estimator = pd.DataFrame({
                "mean": y.mean(),
                "std": y.std()
            })
        
        self.estimator = estimator

    def predict(
        self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ):
        # Values of X are irrelevant
        estimator = self.estimator if estimator is None else estimator
        if (self.task.isClassification()):
            y_list = [np.random.choice(estimator["ratios"].shape[0], len(X), p=estimator["ratios"][col]) for col in list(estimator["ratios"])]
        if (self.task.isRegression()):
            y_list = [np.random.normal(loc=estimator["mean"][col], scale=estimator["std"][col], size=len(X)) for col in range(len(estimator["mean"]))]
        y = np.column_stack(y_list)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def predictProba(
        self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ):
        # Values of X are irrelevant
        estimator = self.estimator if estimator is None else estimator
        if (self.task.isClassification()):
            y_list = [np.random.choice(estimator["ratios"].shape[0], len(X), p=estimator["ratios"][col]) for col in list(estimator["ratios"])]
        if (self.task.isRegression()):
            y_list = [np.random.normal(loc=estimator["mean"][col], scale=estimator["std"][col], size=len(X)) for col in range(len(estimator["mean"]))]
        y = np.column_stack(y_list)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    @property
    def supportsEarlyStopping(self) -> bool:
        """Whether the model supports early stopping or not."""
        return False

    def loadEstimator(self, params: dict | None = None) -> object:
        new_parameters = self.getParameters(params)
        if new_parameters is not None:
            return new_parameters
        else:
            return pd.DataFrame({})


    def loadEstimatorFromFile(self, params: dict | None = None) -> object:
        path = f"{self.outPrefix}.json"
        if os.path.isfile(path):
            estimator = pd.read_json(path)
            new_parameters = self.getParameters(params)
            if new_parameters is not None:
                return estimator.set_params(**new_parameters)
            else:
                return estimator
        else:
            raise FileNotFoundError(
                f"No estimator found at {path}, loading estimator from file failed."
            )

    
    def saveEstimator(self) -> str:
        """See `QSPRModel.saveEstimator`."""
        estimator_path = f"{self.outPrefix}.json"
        self.estimator.to_json(estimator_path)
        return estimator_path
    