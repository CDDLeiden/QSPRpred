"""This module holds assessment methods for QSPRModels"""

import math
from datetime import datetime

import numpy as np
import pandas as pd

from ..logs import logger
from .interfaces import ModelAssessor, QSPRModel


class CrossValAssessor(ModelAssessor):
    """Perform cross validation on a model.

    Attributes:
        useProba (bool): use predictProba instead of predict for classification
    """
    def __call__(
        self,
        model: QSPRModel,
        save: bool = True,
        parameters: dict | None = None,
        **kwargs
    ) -> float | np.ndarray:
        """Perform cross validation on the model with the given parameters.

        Arguments:
            model (QSPRModel): model to assess
            save (bool): whether to save predictions to file
            parameters (dict): optional model parameters to use in assessment
            use_proba (bool): use predictProba instead of predict for classification
            **kwargs: additional keyword arguments for the fit function

        Returns:
            float | np.ndarray: predictions on the validation set
        """
        evalparams = model.parameters if parameters is None else parameters
        # check if data is available
        model.checkForData()
        X, _ = model.data.getFeatures()
        y, _ = model.data.getTargetPropertiesValues()
        # prepare arrays to store molecule ids
        cvs_ids = np.array([None] * len(X))
        # cvs and inds are used to store the predictions for the cross validation
        # and the independent test set
        if model.task.isRegression() or not self.useProba:
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
            logger.debug(
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
            else:
                model.fit(X_train, y_train, crossvalmodel)
            # make predictions
            if model.task.isRegression() or not self.useProba:
                cvs[idx_test] = model.predict(X_test, crossvalmodel)
            else:
                preds = model.predictProba(X_test, crossvalmodel)
                for idx in range(model.nTargets):
                    cvs[idx][idx_test] = preds[idx]

            cvs_ids[idx_test] = X.iloc[idx_test].index.to_numpy()
            logger.debug(
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
            self.savePredictionsToFile(
                model, y, cvs, cvs_index, "cv", extra_columns={"Fold": fold_counter}
            )
        # create output list with tuples of true values and predictions for each fold
        output = []
        for i in np.unique(fold_counter):
            if model.task.isRegression():
                predictions = cvs[fold_counter == i]
            else:
                predictions = [cvs_task[fold_counter == i] for cvs_task in cvs]
            output.append((y[fold_counter == i], predictions))

        return output


class TestSetAssessor(ModelAssessor):
    def __call__(
        self,
        model: QSPRModel,
        save: bool = True,
        parameters: dict | None = None,
        **kwargs
    ):
        """Make predictions for independent test set.

        Arguments:
            model (QSPRModel): model to assess
            save (bool): whether to save predictions to file
            parameters (dict): optional model parameters to use in assessment
            use_proba (bool): use predictProba instead of predict for classification
            **kwargs: additional keyword arguments for the fit function

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
        if not model.task.isRegression() and self.useProba:
            inds = [
                np.zeros((y_ind.shape[0], prop.nClasses))
                for prop in model.targetProperties
            ]

        indmodel = model.loadEstimator(evalparams)
        # fitting on whole trainingset and predicting on test set
        if model.supportsEarlyStopping:
            indmodel = indmodel.set_params(n_epochs=model.optimalEpochs)

        indmodel = model.fit(X, y, indmodel, early_stopping=False)
        # if independent test set is available, predict on it
        if X_ind.shape[0] > 0:
            if model.task.isRegression() or not self.useProba:
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

        # predict values for independent test set and save results
        if save:
            index_name = model.data.getDF().index.name
            ind_index = pd.Index(inds_ids, name=index_name)
            self.savePredictionsToFile(model, y_ind, inds, ind_index, "ind")
        return [(y_ind, inds)]
