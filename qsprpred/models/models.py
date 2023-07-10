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

    def evaluate(
        self,
        save: bool = True,
        parameters: Optional[dict] = None,
        score_func=None,
        **kwargs
    ) -> float | np.ndarray:
        """Make predictions for crossvalidation and independent test set.

        Args:
            save (bool):
                save predictions to file
                (don't save predictions when used in bayesian optimization)
            parameters (dict):
                model parameters, if None, the parameters from the model are used
            score_func (Metric):
                metric to use for scoring, if None, the metric from the model is used
            **kwargs:
                additional keyword arguments for the estimator's predict method

        Returns:
            float | np.ndarray:
                predictions for evaluation
        """
        evalparams = self.parameters if parameters is None else parameters
        score_func = self.scoreFunc if score_func is None else score_func
        # check if data is available
        self.checkForData()
        folds = self.data.createFolds()
        X, X_ind = self.data.getFeatures()
        y, y_ind = self.data.getTargetPropertiesValues()
        # prepare arrays to store molecule ids
        cvs_ids = np.array([None] * len(X))
        inds_ids = X_ind.index.to_numpy()
        # cvs and inds are used to store the predictions for the cross validation
        # and the independent test set
        if self.task.isRegression():
            cvs = np.zeros((y.shape[0], self.nTargets))
        else:
            # cvs, inds need to be lists of arrays
            # for multiclass-multitask classification
            cvs = [
                np.zeros((y.shape[0], prop.nClasses)) for prop in self.targetProperties
            ]
            inds = [
                np.zeros((y_ind.shape[0], prop.nClasses))
                for prop in self.targetProperties
            ]

        fold_counter = np.zeros(y.shape[0])
        # cross validation
        for i, (X_train, X_test, y_train, y_test, idx_train,
                idx_test) in enumerate(folds):
            crossvalmodel = self.loadEstimator(evalparams)
            # log some things
            logger.info(
                "cross validation fold %s started: %s" %
                (i, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            # store molecule indices
            fold_counter[idx_test] = i
            # fit model
            self.fit(X_train, y_train, crossvalmodel)
            # predict and store predictions
            if self.task.isRegression():
                cvs[idx_test] = self.predict(X_test, crossvalmodel)
            else:
                preds = self.predictProba(X_test, crossvalmodel)
                for idx in range(self.nTargets):
                    cvs[idx][idx_test] = preds[idx]

            cvs_ids[idx_test] = X.iloc[idx_test].index.to_numpy()
            # log some things
            logger.info(
                "cross validation fold %s ended: %s" %
                (i, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
        # fitting on whole trainingset and predicting on test set
        indmodel = self.loadEstimator(evalparams)
        self.fit(X, y, indmodel)
        # if independent test set is available, predict on it
        if X_ind.shape[0] > 0:
            if self.task.isRegression():
                inds = self.predict(X_ind, indmodel)
            else:
                preds = self.predictProba(X_ind, indmodel)
                for idx in range(self.nTargets):
                    inds[idx] = preds[idx]
        else:
            logger.warning(
                "No independent test set available. "
                "Skipping prediction on independent test set."
            )
        # save crossvalidation results
        if save:
            index_name = self.data.getDF().index.name
            ind_index = pd.Index(inds_ids, name=index_name)
            cvs_index = pd.Index(cvs_ids, name=index_name)
            train, test = y.add_suffix("_Label"), y_ind.add_suffix("_Label")
            train, test = pd.DataFrame(
                train.values, columns=train.columns, index=cvs_index
            ), pd.DataFrame(test.values, columns=test.columns, index=ind_index)
            for idx, prop in enumerate(self.data.targetProperties):
                if prop.task.isClassification():
                    # convert one-hot encoded predictions to class labels
                    # and add to train and test
                    (
                        train[f"{prop.name}_Prediction"],
                        test[f"{prop.name}_Prediction"],
                    ) = np.argmax(cvs[idx], axis=1), np.argmax(inds[idx], axis=1)
                    # add probability columns to train and test set
                    train = pd.concat(
                        [
                            train,
                            pd.DataFrame(cvs[idx], index=cvs_index
                                        ).add_prefix(f"{prop.name}_ProbabilityClass_"),
                        ],
                        axis=1,
                    )
                    test = pd.concat(
                        [
                            test,
                            pd.DataFrame(inds[idx], index=ind_index
                                        ).add_prefix(f"{prop.name}_ProbabilityClass_"),
                        ],
                        axis=1,
                    )
                else:
                    (
                        train[f"{prop.name}_Prediction"],
                        test[f"{prop.name}_Prediction"],
                    ) = (cvs[:, idx], inds[:, idx])
            train["Fold"] = fold_counter
            train.to_csv(self.outPrefix + ".cv.tsv", sep="\t")
            test.to_csv(self.outPrefix + ".ind.tsv", sep="\t")
        # Return predictions in the right format for scorer
        if self.task.isRegression():
            return cvs
        elif self.scoreFunc.needsProbasToScore:
            if self.task in [
                ModelTasks.SINGLECLASS,
                ModelTasks.MULTITASK_SINGLECLASS,
            ]:
                return np.transpose([y_pred[:, 1] for y_pred in cvs])
            elif self.task.isMultiTask():
                return cvs
            else:
                return cvs[0]
        else:
            return np.transpose([np.argmax(y_pred, axis=1) for y_pred in cvs])

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
        estimator: Any = None
    ):
        """See `QSARModel.fit`."""
        estimator = self.estimator if estimator is None else estimator
        if isinstance(X, QSPRDataset):
            X = X.getFeatures(raw=True, concat=True)
            y = y.getTargetPropertiesValues(concat=True)
        if self.featureStandardizer:
            X = self.featureStandardizer(X)

        if not self.task.isMultiTask:
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
