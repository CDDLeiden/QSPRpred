import numpy as np
import pandas as pd

from qsprpred.tasks import ModelTasks
from .models import QSPRModel
from ..data.tables.qspr import QSPRDataset
from typing import Any

class RandomModel(QSPRModel):
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
            self.ratios = pd.DataFrame.from_dict({col: y[col].value_counts() / y.shape[0] for col in list(y)})

        if (self.task.isRegression()):
            # Calculate the mean and standard deviation of each column
            self.mean = y.mean()
            self.std = y.std()

    def predict(
        self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ):
        # Values of X are irrelevant
        # TODO: might need to be len(X.T)
        # TODO: clumsy

        if (self.task.isClassification()):
            y_list = [pd.DataFrame(np.random.choice(self.ratios.shape[0], len(X), p=self.ratios[col]), columns=[col]) for col in list(self.ratios)]
            return pd.concat(y_list, axis=1)
        if (self.task.isRegression()):
            y_list = [pd.DataFrame(np.random.normal(loc=self.mean[col], scale=self.std[col], size=len(X)), columns=[col]) for col in len(self.mean)]
            return pd.concat(y_list, axis=1)

    @property
    def supportsEarlyStopping(self) -> bool:
        """Whether the model supports early stopping or not."""
        return False
