import json
import numpy as np
import pandas as pd
import os

from .models import QSPRModel
from ..data.tables.qspr import QSPRDataset
from typing import Any

import jsonpickle

jsonpickle.set_encoder_options('json', indent=4)

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
        self.alg = None
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
        #TODO: y might need to be transposed everywhere, didn't test with actual values yet
        # Values of X are irrelevant
        if (self.task.isClassification()):
            # Save ratios, these will be used as probability that a given bucket is
            # chosen per target

            # TODO: this is probably clumsy, I shouldn't have to go from dataframe
            # to dict to dataframe
            estimator["ratios"] = pd.DataFrame.from_dict({col: y[col].value_counts() / y.shape[0] for col in list(y)})

        if (self.task.isRegression()):
            # Calculate the mean and standard deviation of each column
            estimator["mean"] = y.mean()
            estimator["std"] = y.std()
        
        self.estimator = estimator

    def predict(
        self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ):
        # Values of X are irrelevant
        # TODO: might need to be len(X.T)
        # TODO: clumsy

        if (self.task.isClassification()):
            y_list = [pd.DataFrame(np.random.choice(self.estimator["ratios"].shape[0], len(X), p=self.estimator["ratios"][col]), columns=[col]) for col in list(self.estimator["ratios"])]
            return pd.concat(y_list, axis=1)
        if (self.task.isRegression()):
            y_list = [pd.DataFrame(np.random.normal(loc=self.estimator["mean"][col], scale=self.estimator["std"][col], size=len(X)), columns=[col]) for col in range(len(self.estimator["mean"]))]
            y = pd.concat(y_list, axis=1).to_numpy()
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            return y

    def predictProba(
        self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ):
        # Values of X are irrelevant
        # TODO: might need to be len(X.T)
        # TODO: clumsy

        if (self.task.isClassification()):
            y_list = [pd.DataFrame(np.random.choice(estimator["ratios"].shape[0], len(X), p=estimator["ratios"][col]), columns=[col]) for col in list(estimator["ratios"])]
            return pd.concat(y_list, axis=1)
        if (self.task.isRegression()):
            y_list = [pd.DataFrame(np.random.normal(loc=estimator["mean"][col], scale=estimator["std"][col], size=len(X)), columns=[col]) for col in list(estimator["mean"])]
            return pd.concat(y_list, axis=1)

    @property
    def supportsEarlyStopping(self) -> bool:
        """Whether the model supports early stopping or not."""
        return False

    def loadEstimator(self, params: dict | None = None) -> object:
        new_parameters = self.getParameters(params)
        if new_parameters is not None:
            return new_parameters
        else:
            return {}


    def loadEstimatorFromFile(self, params: dict | None = None) -> object:
        path = f"{self.outPrefix}.json"
        if os.path.isfile(path):
            with open(path, 'r') as f:
                estimator = self.estimatorFromJson(f.read())
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
        with open(estimator_path, 'w') as f:
            f.write(self.estimatorToJson())
        return estimator_path

    def estimatorToJson(self) -> str:
        return jsonpickle.encode(self.estimator, unpicklable=True)
    
    def estimatorFromJson(self, json):
        return jsonpickle.decode(json)
    