"""Here the QSPRmodel classes can be found.

At the moment there is a class for sklearn type models. However, one for a pytorch DNN
model can be found in `qsprpred.deep`. To add more types a model class implementing
the `QSPRModel` interface can be added.
"""

import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import sklearn_json as skljson
from sklearn.svm import SVC, SVR

from ..data.data import QSPRDataset
from ..logs import logger
from ..models.interfaces import QSPRModel
from ..models.tasks import ModelTasks


class QSPRsklearn(QSPRModel):
    """QSPRModel class for sklearn type models.

    Wrap your sklearn model class in this class
    to use it with the `QSPRModel` interface.
    """
    def __init__(
        self,
        base_dir: str,
        alg=None,
        data: QSPRDataset = None,
        name: Optional[str] = None,
        parameters: Optional[dict] = None,
        autoload: bool = True,
        scoring=None,
    ):
        """Initialize QSPRsklearn model.

        Args:
            base_dir (str): base directory for model
            alg (Type): sklearn model class
            data (QSPRDataset): data set to use for model
            name (str): customized model name
            parameters (dict): model parameters
            autoload (bool): load model from file
            scoring (str): scoring function to use for model evaluation
        """
        super().__init__(base_dir, alg, data, name, parameters, autoload, scoring)
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
        # initialize models with defined parameters
        if type(self.estimator) in [SVC, SVR]:
            logger.warning(
                "parameter max_iter set to 10000 to avoid training getting stuck. \
                            Manually set this parameter if this is not desired."
            )
            if self.parameters:
                self.parameters.update({"max_iter": 10000})
            else:
                self.parameters = {"max_iter": 10000}
        # set parameters if defined
        if self.parameters not in [None, {}] and hasattr(self, "estimator"):
            self.estimator.set_params(**self.parameters)
        # log some things
        logger.info("parameters: %s" % self.parameters)
        logger.debug(f'Model "{self.name}" initialized in: "{self.baseDir}"')

    @property
    def supportsEarlyStopping(self) -> bool:
        """Whether the model supports early stopping or not."""
        return False

    def fitAllData(self) -> str:
        """Fit the underlying scikit-learn estimator.

        Returns:
            str: path to saved model
        """
        # check if data is available
        self.checkForData()
        # get data into fit set
        X_all = self.data.getFeatures(concat=True).values
        y_all = self.data.getTargetPropertiesValues(concat=True)
        # fit model
        logger.info(
            "Model fit started: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        self.fit(X_all, y_all)
        logger.info(
            "Model fit ended: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        # save model
        return self.save()

    def loadEstimator(self, params: Optional[dict] = None) -> Any:
        """Load estimator from alg and params.

        Args:
            params (dict): parameters
        """
        if params:
            if self.parameters is not None:
                temp_params = deepcopy(self.parameters)
                temp_params.update(params)
                return self.alg(**temp_params)
            else:
                return self.alg(**params)
        elif self.parameters is not None:
            return self.alg(**self.parameters)
        else:
            return self.alg()

    def loadEstimatorFromFile(
        self, params: Optional[dict] = None, fallback_load: bool = True
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
            estimator = skljson.from_json(path)
            self.alg = estimator.__class__
            if params:
                return estimator.set_params(**params)
            else:
                return estimator
        elif fallback_load:
            return self.loadEstimator(params)
        else:
            raise FileNotFoundError(
                f"No estimator found at {path}, loading estimator from file failed."
            )

    def saveEstimator(self) -> str:
        """See `QSARModel.saveEstimator`."""
        estimator_path = f"{self.outPrefix}.json"
        skljson.to_json(self.estimator, estimator_path)
        return estimator_path

    def fit(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        y: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any = None,
        early_stopping: Any = None
    ):
        """See `QSARModel.fit`."""
        estimator = self.estimator if estimator is None else estimator
        if isinstance(X, QSPRDataset):
            X = X.getFeatures(raw=True, concat=True)
            y = y.getTargetPropertiesValues(concat=True)
        if self.featureStandardizer:
            X = self.featureStandardizer(X)

        if not self.task.isMultiTask():
            if isinstance(y, pd.DataFrame):
                y = y.squeeze()
            else:
                y = y.ravel()
        return estimator.fit(X, y)

    def predict(
        self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
    ):
        """See `QSARModel.predict`."""
        if estimator is None:
            estimator = self.estimator
        if isinstance(X, QSPRDataset):
            X = X.getFeatures(raw=True, concat=True)
        if self.featureStandardizer:
            X = self.featureStandardizer(X)
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
        """See `QSARModel.predictProba`."""
        if estimator is None:
            estimator = self.estimator
        if isinstance(X, QSPRDataset):
            X = X.getFeatures(raw=True, concat=True)
        if self.featureStandardizer:
            X = self.featureStandardizer(X)
        preds = estimator.predict_proba(X)
        # if preds is a numpy array, convert it to a list
        # to be consistent with the multiclass-multitask case
        if isinstance(preds, np.ndarray):
            preds = [preds]
        return preds
