"""Here the QSPRmodel classes can be found.

At the moment there is a class for sklearn type models. However, one for a pytorch DNN
model can be found in `qsprpred.deep`. To add more types a model class implementing
the `QSPRModel` interface can be added.
"""

import os
from typing import Any

import ml2json
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .model import QSPRModel
from ..data.tables.qspr import QSPRDataset
from ..logs import logger
from ..tasks import ModelTasks


class SklearnModel(QSPRModel):
    """QSPRModel class for sklearn type models.

    Wrap your sklearn model class in this class
    to use it with the `QSPRModel` interface.
    """

    def __init__(
        self,
        base_dir: str,
        alg=None,
        name: str | None = None,
        parameters: dict | None = None,
        autoload: bool = True,
        random_state: int | None = None,
    ):
        """Initialize SklearnModel model.

        Args:
            base_dir (str): base directory for model
            alg (Type): sklearn model class
            name (str): customized model name
            parameters (dict): model parameters
            autoload (bool): load model from file
            random_state (int): seed for the random state
        """
        super().__init__(base_dir, alg, name, parameters, autoload, random_state)
        # initialize models with defined parameters
        try:
            # check if alg can be initialized with parameters
            if self.parameters is not None:
                self.alg(**self.parameters)
            else:
                self.alg()
        except:
            logger.error(
                f"Cannot initialize alg {self.alg} with parameters {self.parameters}."
            )
            raise
        # set parameters if defined
        if (
            (self.parameters not in [None, {}])
            and hasattr(self, "estimator")
            and self.estimator is not None
        ):
            try:
                check_is_fitted(self.estimator)
            except NotFittedError:
                self.estimator.set_params(**self.parameters)
        # log some things
        logger.info("parameters: %s" % self.parameters)
        logger.debug(f'Model "{self.name}" initialized in: "{self.baseDir}"')

    @property
    def supportsEarlyStopping(self) -> bool:
        """Whether the model supports early stopping or not."""
        return False

    def loadEstimator(self, params: dict | None = None) -> Any:
        """Load estimator from alg and params.

        Args:
            params (dict): parameters
        """
        new_parameters = self.getParameters(params)
        if new_parameters is not None:
            return self.alg(**new_parameters)
        else:
            return self.alg()

    def loadEstimatorFromFile(
        self, params: dict | None = None, fallback_load: bool = True
    ):
        """Load estimator from file.

        Args:
            params (dict): parameters
            fallback_load (bool):
                if `True`, init estimator from alg
                and params if no estimator found at path
        """
        path = f"{self.outPrefix}.json"
        if os.path.isfile(path):
            estimator = ml2json.from_json(path)
            self.alg = estimator.__class__
            if params is not None:
                new_parameters = self.getParameters(params)
                if new_parameters is not None:
                    estimator = estimator.set_params(**new_parameters)
            return estimator
        elif fallback_load:
            logger.warning(
                f"No estimator found at {path}, creating unfitted estimator instead. "
                f"Set fallback_load to False to prevent this."
            )
            return self.loadEstimator(params)
        else:
            raise FileNotFoundError(
                f"No estimator found at {path}, loading estimator from file failed."
            )

    def saveEstimator(self) -> str:
        """See `QSPRModel.saveEstimator`."""
        estimator_path = f"{self.outPrefix}.json"
        ml2json.to_json(self.estimator, estimator_path)
        return estimator_path

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
        estimator: Any = None,
        mode: Any = None,
        monitor: None = None,
        **kwargs,
    ):
        # check for incompatible tasks
        if self.task == ModelTasks.MULTITASK_MIXED:
            raise ValueError(
                "MultiTask with a mix of classification and regression tasks "
                "is not supported for sklearn models."
            )
        if self.task == ModelTasks.MULTITASK_MULTICLASS:
            raise NotImplementedError(
                "At the moment there are no supported metrics "
                "for multi-task multi-class/mix multi-and-single class classification."
            )
        estimator = self.estimator if estimator is None else estimator
        X, y = self.convertToNumpy(X, y)
        # sklearn models expect 1d arrays
        # for single target regression and classification
        if not self.task.isMultiTask():
            y = y.ravel()
        return estimator.fit(X, y)

    def predict(
        self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ):
        """See `QSPRModel.predict`."""
        estimator = self.estimator if estimator is None else estimator
        X = self.convertToNumpy(X)
        preds = estimator.predict(X)
        # Most sklearn regression models return 1d arrays for single target regression
        # and sklearn single task classification models return 1d arrays
        # However, QSPRpred expects 2d arrays in every case
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        return preds

    def predictProba(
        self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ):
        """See `QSPRModel.predictProba`."""
        estimator = self.estimator if estimator is None else estimator
        X = self.convertToNumpy(X)
        preds = estimator.predict_proba(X)
        # if preds is a numpy array, convert it to a list
        # to be consistent with the multiclass-multitask case
        if isinstance(preds, np.ndarray):
            preds = [preds]
        return preds
