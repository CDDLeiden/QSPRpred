import json
import numpy as np
import pandas as pd
import os

from qsprpred.models.early_stopping import EarlyStoppingMode
from qsprpred.models.monitors import FitMonitor

from ...models.models import QSPRModel
from ...data.tables.qspr import QSPRDataset
from typing import Any, Optional, Type
from abc import ABC, abstractmethod

class RandomDistributionAlgorithm(ABC):
    @abstractmethod
    def __call__(self, X_test: np.ndarray):
        pass

    @abstractmethod
    def fit(self, y_df: pd.DataFrame):
        pass

    @abstractmethod
    def from_dict(self, loaded_dict):
        pass

    @abstractmethod
    def to_dict(self):
        pass


class RatioDistributionAlgorithm(RandomDistributionAlgorithm):
    """
    Categorical distribution using ratio of categories as probabilities
    Values of X are irrelevant, only distribution of y is used
    """
    def __init__(self, random_state=None):
        self.ratios = None
        self.random_state=random_state
        self.rng = np.random.default_rng(seed=random_state)

    def __call__(self, X_test: np.ndarray):
        # TODO: fix
        y_list = [self.rng.choice(len(self.ratios.values), len(X_test), p=[r[col] for r in self.ratios.values]) for col in range(len(self.ratios.values[0]))]
        y = np.column_stack(y_list)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        print(y)
        return y

    def fit(self, y_df: pd.DataFrame):
        self.ratios = pd.DataFrame.from_dict({col: y_df[col].value_counts() / y_df.shape[0] for col in list(y_df)})

    def from_dict(self, loaded_dict):
        self.ratios = pd.DataFrame({"ratios": json.loads(loaded_dict["ratios"])}) if loaded_dict[
                                                "ratios"] is not None else None


    def to_dict(self):
        param_dictionary = {"parameters": {},
                "ratios": self.ratios.to_json() if self.ratios is not None else None,
                }
        return param_dictionary

class NormalDistributionAlgorithm(RandomDistributionAlgorithm):
    """
    Distributions used: normal distribution for regression, not to be used for classification
    Values of X are irrelevant, only distribution of y is used
    """

    def __init__(self, random_state=None):
        self.mean = None
        self.std = None
        self.random_state=random_state
        self.rng = np.random.default_rng(seed=random_state)

    def __call__(self, X_test: np.ndarray):
        y_list = [self.rng.normal(loc=self.mean.values[col], scale=self.std.values[col], size=len(X_test)) for col in range(len(self.mean))]
        y = np.column_stack(y_list)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        return y

    def fit(self, y_df: pd.DataFrame):
        self.mean = y_df.mean()
        self.std = y_df.std()

    def from_dict(self, loaded_dict):
        self.mean = pd.DataFrame({"mean": json.loads(loaded_dict["mean"])}) if loaded_dict[
                                                "mean"] is not None else None
        self.std = pd.DataFrame({"std": json.loads(loaded_dict["std"])}) if loaded_dict[
                                                "std"] is not None else None

    def to_dict(self):
        param_dictionary = {"parameters": {},
                "mean": self.mean.to_json() if self.mean is not None else None,
                "std": self.std.to_json() if self.std is not None else None,
                }
        return param_dictionary

class MedianDistributionAlgorithm(RandomDistributionAlgorithm):
    def __init__(self):
        self.median = None

    def __call__(self, X_test: np.ndarray):
        raise Exception("TODO: return constant array")
        #y_list = [self.rng.normal(loc=self.mean.values[col], scale=self.std.values[col], size=len(X_test)) for col in range(len(self.mean))]
        y = np.column_stack(y_list)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        return y


    def fit(self, y: pd.DataFrame):
        estimator.median = y_df.median()

    def from_dict(self, loaded_dict):
        pass

    def to_dict(self):
        pass


class RandomModel(QSPRModel):
    def __init__(
        self,
        base_dir: str,
        alg: RandomDistributionAlgorithm,
        data: Optional[QSPRDataset] = None,
        name: Optional[str] = None,
        parameters: Optional[dict] = None,
        autoload=True,
        random_state: int | None = None
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
        super().__init__(base_dir, alg, data, name, parameters, autoload, random_state=random_state)

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
            **kwargs
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
            self,
            X: pd.DataFrame | np.ndarray | QSPRDataset,
            estimator: Any = None
    ):
        #TODO
        return self.predict(X, estimator)
    
    def loadEstimator(self, params: Optional[dict] = None) -> object:
        """Initialize estimator instance with the given parameters.

        If `params` is `None`, the default parameters will be used.

        Arguments:
            params (dict): algorithm parameters

        Returns:
            object: initialized estimator instance
        """
        new_parameters = self.getParameters(
            params)  # combine self.parameters and params
        if new_parameters is not None:
            return self.alg(**new_parameters)
        else:
            return self.alg()
        
    def loadEstimatorFromFile(self, params: Optional[dict] = None,
                              fallback_load=True) -> object:
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