import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

import numpy as np
import pandas as pd
import scipy as sp

from qsprpred.models.early_stopping import EarlyStoppingMode
from qsprpred.models.monitors import FitMonitor

from ...data.tables.qspr import QSPRDataset
from ...models.model import QSPRModel


class RandomDistributionAlgorithm(ABC):
    @abstractmethod
    def __call__(self, X_test: np.ndarray):
        pass

    @abstractmethod
    def fit(self, y_df: pd.DataFrame):
        pass

    @abstractmethod
    def get_probas(self, X_test: np.ndarray):
        pass

    @abstractmethod
    def from_dict(self, loaded_dict):
        pass

    @abstractmethod
    def to_dict(self):
        pass


class RatioDistributionAlgorithm(RandomDistributionAlgorithm):
    """Categorical distribution using ratio of categories as probabilities
    
    Values of X are irrelevant, only distribution of y is used
    
    Attributes:
        ratios (pd.DataFrame): ratio of each category in y
        random_state (int): random state for reproducibility
    """
    def __init__(self, random_state=None):
        self.ratios = None
        self.random_state = random_state

    def __call__(self, X_test: np.ndarray):
        """Generate random samples from the distribution of y"""
        rng = np.random.default_rng(seed=self.random_state)
        y_list = [
            rng.choice(
                len(self.ratios.values),
                len(X_test),
                p=[r[col] for r in self.ratios.values],
            ) for col in range(len(self.ratios.values[0]))
        ]
        y = np.column_stack(y_list)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        return y

    def get_probas(self, X_test: np.ndarray):
        """Get probabilities of each category for each sample in X_test"""
        y_list = [
            np.array(
                [[ratio[col] for ratio in self.ratios.values] for _ in range(len(X_test))]
            ) for col in range(len(self.ratios.values[0]))
        ]
        return y_list

    def fit(self, y_df: pd.DataFrame):
        """Calculate ratio of each category in y_df and store as probability distribution"""
        self.ratios = pd.DataFrame.from_dict(
            {col: y_df[col].value_counts() / y_df.shape[0]
             for col in list(y_df)}
        ).sort_index()

    def from_dict(self, loaded_dict):
        self.ratios = (
            pd.DataFrame(json.loads(loaded_dict["ratios"]))
            if loaded_dict["ratios"] is not None else None
        )

    def to_dict(self):
        param_dictionary = {
            "parameters": {},
            "ratios": self.ratios.to_json() if self.ratios is not None else None,
        }
        return param_dictionary


class MedianDistributionAlgorithm(RandomDistributionAlgorithm):
    def __init__(self):
        self.median = None

    def __call__(self, X_test: np.ndarray):
        y_list = [
            np.full(len(X_test), self.median.values[col])
            for col in range(len(self.median))
        ]
        y = np.column_stack(y_list)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        return y

    def get_probas(self, X_test: np.ndarray):
        raise Exception("No probas supported for this algorithm.")

    def fit(self, y_df: pd.DataFrame):
        self.median = y_df.median()

    def from_dict(self, loaded_dict):
        self.median = (
            pd.DataFrame({"median": json.loads(loaded_dict["median"])})
            if loaded_dict["median"] is not None else None
        )

    def to_dict(self):
        param_dictionary = {
            "parameters": {},
            "median": self.median.to_json() if self.median is not None else None,
        }
        return param_dictionary


class ScipyDistributionAlgorithm(RandomDistributionAlgorithm):
    def __init__(
        self,
        distribution: sp.stats.rv_continuous = sp.stats.norm,
        params={},
        random_state=None,
    ):
        self.fitted_parameters = None
        self.distribution = distribution
        self.params = params
        self.random_state = random_state

    def __call__(self, X_test: np.ndarray):
        y_list = [
            self.distribution.rvs(
                *(tuple(self.fitted_parameters.values.T[col])),
                size=len(X_test),
                random_state=self.random_state,
            ) for col in range(len(self.fitted_parameters.values.T))
        ]
        y = np.column_stack(y_list)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        return y

    def get_probas(self, X_test: np.ndarray):
        raise Exception("No probas supported for this algorithm.")

    def fit(self, y_df: pd.DataFrame):
        self.fitted_parameters = pd.DataFrame.from_dict(
            {col: self.distribution.fit(y_df[col], *self.params)
             for col in list(y_df)}
        )

    def from_dict(self, loaded_dict):
        self.fitted_parameters = (
            pd.DataFrame(json.loads(loaded_dict["fitted_parameters"]))
            if loaded_dict["fitted_parameters"] is not None else None
        )

    def to_dict(self):
        param_dictionary = {
            "parameters": {},
            "fitted_parameters":
                self.fitted_parameters.to_json()
                if self.fitted_parameters is not None else None,
        }
        return param_dictionary


class RandomModel(QSPRModel):
    def __init__(
        self,
        base_dir: str,
        alg: RandomDistributionAlgorithm,
        name: Optional[str] = None,
        parameters: Optional[dict] = None,
        autoload=True,
        random_state: int | None = None,
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
        """
        super().__init__(
            base_dir, alg, name, parameters, autoload, random_state=random_state
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
        estimator: Type[RandomDistributionAlgorithm] = None,
        mode: EarlyStoppingMode = None,
        monitor: FitMonitor | None = None,
        **kwargs,
    ) -> RandomDistributionAlgorithm:
        """Fit the model to the given data matrix or `QSPRDataset`.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to fit
            y (pd.DataFrame, np.ndarray, QSPRDataset): target matrix to fit
            estimator (Any): estimator instance to use for fitting
            mode (EarlyStoppingMode): early stopping mode, unused
            monitor (FitMonitor): monitor instance to track the fitting process, unused
            kwargs: additional keyword arguments for the fit function

        Returns:
            (RandomDistributionAlgorithm): fitted estimator instance
        """
        estimator = self.estimator if estimator is None else estimator
        if isinstance(y, pd.DataFrame):
            y_df = y
        elif isinstance(y, np.ndarray):
            y_df = pd.DataFrame(y)
        else:
            y_df = y.df

        estimator.fit(y_df)

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
        return estimator(X)

    def predictProba(
        self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ):
        estimator = self.estimator if estimator is None else estimator
        return estimator.get_probas(X)

    def loadEstimator(self, params: Optional[dict] = None) -> object:
        """Initialize estimator instance with the given parameters.

        If `params` is `None`, the default parameters will be used.

        Arguments:
            params (dict): algorithm parameters

        Returns:
            object: initialized estimator instance
        """
        new_parameters = self.getParameters(
            params
        )  # combine self.parameters and params
        if new_parameters is not None:
            return self.alg(**new_parameters)
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
        path = f"{self.outPrefix}.json"
        if os.path.isfile(path):
            with open(path, "r") as f:
                loaded_dict = json.load(f)
                loaded_params = loaded_dict["parameters"]
            if params is not None:
                loaded_params.update(params)

            estimator = self.loadEstimator(loaded_params)
            estimator.from_dict(loaded_dict)
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
        estimator_path = f"{self.outPrefix}.json"

        param_dictionary = self.estimator.to_dict()
        with open(estimator_path, "w") as outfile:
            json.dump(param_dictionary, outfile)

        return estimator_path
