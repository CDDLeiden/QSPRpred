"""This module holds evaluation methods for QSPRModels"""

import math
from datetime import datetime
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from ..logs import logger
from ..models.tasks import ModelTasks
from .interfaces import EvaluationMethod, QSPRModel
from .metrics import SklearnMetric


class CrossValidation(EvaluationMethod):
    def __init__(
        self, score_func: Callable[[Iterable, Iterable], float] | SklearnMetric = None
    ):
        """Initialize the cross validation evaluation method.

        Args:
            score_func (Callable[[Iterable, Iterable], float] | SklearnMetric):
                scoring function to use to determine the best model in the
                cross validation, if None defaults to the score function of the model
        """
        self.scoreFunc = score_func

    def __call__(
        self,
        model: QSPRModel,
        save: bool = True,
        parameters: Optional[dict] = None,
        **kwargs
    ) -> float | np.ndarray:
        """Perform cross validation on the model with the given parameters.

        If save is True, the predictions on the validation set are saved to a file in
        the output directory.

        Arguments:
            save (bool): don't save predictions when used in bayesian optimization
            parameters (dict): optional model parameters to use for evaluation
            score_func (str or callable): scoring function to use for evaluation
            **kwargs: additional keyword arguments for the evaluation function

        Returns:
            float | np.ndarray: predictions on the validation set"""
        evalparams = model.parameters if parameters is None else parameters
        # check if data is available
        model.checkForData()
        X, X_ind = model.data.getFeatures()
        y, y_ind = model.data.getTargetPropertiesValues()
        # prepare arrays to store molecule ids
        cvs_ids = np.array([None] * len(X))
        # cvs and inds are used to store the predictions for the cross validation
        # and the independent test set
        if model.task.isRegression():
            cvs = np.zeros((y.shape[0], model.nTargets))
        else:
            # cvs, inds need to be lists of arrays
            # for multiclass-multitask classification
            cvs = [
                np.zeros((y.shape[0], prop.nClasses)) for prop in model.targetProperties
            ]
        # cross validation
        folds = model.data.createFolds()
        fold_counter = np.zeros(y.shape[0])
        last_save_epochs = 0
        for i, (X_train, X_test, y_train, y_test, idx_train,
                idx_test) in enumerate(folds):
            crossvalmodel = model.loadEstimator(evalparams)
            logger.info(
                "cross validation fold %s started: %s" %
                (i, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            # store molecule indices
            fold_counter[idx_test] = i
            # fit model
            if model.supportsEarlyStopping:
                crossvalmodel, last_save_epoch = model.fit(
                    X_train, y_train, crossvalmodel
                )
                last_save_epochs += last_save_epoch
                logger.info(
                    f"cross validation fold {i}: last save epoch {last_save_epoch}"
                )
            model.fit(X_train, y_train, crossvalmodel)
            # predict and store predictions
            if model.task.isRegression():
                cvs[idx_test] = model.predict(X_test, crossvalmodel)
            else:
                preds = model.predictProba(X_test, crossvalmodel)
                for idx in range(model.nTargets):
                    cvs[idx][idx_test] = preds[idx]

            cvs_ids[idx_test] = X.iloc[idx_test].index.to_numpy()
            logger.info(
                "cross validation fold %s ended: %s" %
                (i, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            if model.supportsEarlyStopping:
                n_folds = max(fold_counter) + 1
                model.optimalEpochs = int(math.ceil(last_save_epochs / n_folds)) + 1
        # save results
        if save:
            index_name = model.data.getDF().index.name
            cvs_index = pd.Index(cvs_ids, name=index_name)
            train = y.add_suffix("_Label")
            train = pd.DataFrame(train.values, columns=train.columns, index=cvs_index)
            for idx, prop in enumerate(model.data.targetProperties):
                if prop.task.isClassification():
                    # convert one-hot encoded predictions to class labels
                    # and add to train and test
                    train[f"{prop.name}_Prediction"] = np.argmax(cvs[idx], axis=1)
                    # add probability columns to train and test set
                    train = pd.concat(
                        [
                            train,
                            pd.DataFrame(cvs[idx], index=cvs_index
                                        ).add_prefix(f"{prop.name}_ProbabilityClass_"),
                        ],
                        axis=1,
                    )
                else:
                    train[f"{prop.name}_Prediction"] = cvs[:, idx]
            train["Fold"] = fold_counter
            train.to_csv(model.outPrefix + ".cv.tsv", sep="\t")
        # Return predictions in the right format for scorer
        if model.task.isRegression():
            return cvs
        elif model.scoreFunc.needsProbasToScore:
            if model.task in [
                ModelTasks.SINGLECLASS,
                ModelTasks.MULTITASK_SINGLECLASS,
            ]:
                return np.transpose([y_pred[:, 1] for y_pred in cvs])
            elif model.task.isMultiTask():
                return cvs
            else:
                return cvs[0]
        else:
            return np.transpose([np.argmax(y_pred, axis=1) for y_pred in cvs])


class EvaluateTestSetPerformance(EvaluationMethod):
    def __call__(
        self,
        model: QSPRModel,
        save: bool = True,
        parameters: Optional[dict] = None,
        **kwargs
    ):
        """Make predictions for independent test set.

        If save is True, the predictions are saved to a file in the output directory.

        Arguments:
            save (bool): don't save predictions when used in bayesian optimization
            parameters (dict): optional model parameters to use for evaluation
            **kwargs: additional keyword arguments for the evaluation function

        Returns:
            float | np.ndarray: predictions for evaluation
        """
        evalparams = model.parameters if parameters is None else parameters
        # check if data is available
        model.checkForData()
        X, X_ind = model.data.getFeatures()
        y, y_ind = model.data.getTargetPropertiesValues()
        # prepare arrays to store molecule ids and predictions
        inds_ids = X_ind.index.to_numpy()
        if not model.task.isRegression():
            inds = [
                np.zeros((y_ind.shape[0], prop.nClasses))
                for prop in model.targetProperties
            ]

        # predict values for independent test set and save results
        if save:
            indmodel = model.loadEstimator(evalparams)
            # fitting on whole trainingset and predicting on test set
            if model.supportsEarlyStopping:
                indmodel = indmodel.set_params(n_epochs=model.optimalEpochs)

            indmodel = model.fit(X, y, indmodel, early_stopping=False)
            # if independent test set is available, predict on it
            if X_ind.shape[0] > 0:
                if model.task.isRegression():
                    inds = model.predict(X_ind, indmodel)
                else:
                    preds = model.predictProba(X_ind, indmodel)
                    for idx in range(model.nTargets):
                        inds[idx] = preds[idx]
            else:
                logger.warning(
                    "No independent test set available. "
                    "Skipping prediction on independent test set."
                )
            index_name = model.data.getDF().index.name
            ind_index = pd.Index(inds_ids, name=index_name)
            test = y_ind.add_suffix("_Label")
            test = pd.DataFrame(test.values, columns=test.columns, index=ind_index)
            for idx, prop in enumerate(model.data.targetProperties):
                if prop.task.isClassification():
                    # convert one-hot encoded predictions to class labels
                    # and add to train and test
                    test[f"{prop.name}_Prediction"] = np.argmax(inds[idx], axis=1)
                    # add probability columns to train and test set
                    test = pd.concat(
                        [
                            test,
                            pd.DataFrame(inds[idx], index=ind_index
                                        ).add_prefix(f"{prop.name}_ProbabilityClass_"),
                        ],
                        axis=1,
                    )
                else:
                    test[f"{prop.name}_Prediction"] = inds[:, idx]
            test.to_csv(model.outPrefix + ".ind.tsv", sep="\t")
