"""This module holds assessment methods for QSPRModels"""

from datetime import datetime

import numpy as np
import pandas as pd

from ..logs import logger
from .early_stopping import EarlyStoppingMode
from .interfaces import AssessorMonitor, ModelAssessor, QSPRModel
from .monitors import NullAssessorMonitor


class CrossValAssessor(ModelAssessor):
    """Perform cross validation on a model.

    Attributes:
        useProba (bool): use predictProba instead of predict for classification
        monitor (AssessorMonitor): monitor to use for assessment
        mode (EarlyStoppingMode): mode to use for early stopping
    """
    def __init__(
        self,
        monitor: AssessorMonitor = NullAssessorMonitor(),
        use_proba: bool = True,
        mode: EarlyStoppingMode | None = None,
    ):
        super().__init__(monitor, use_proba, mode)

    def __call__(
        self,
        model: QSPRModel,
        save: bool = True,
        parameters: dict | None = None,
        monitor: AssessorMonitor | None = None,
        **kwargs,
    ) -> float | np.ndarray:
        """Perform cross validation on the model with the given parameters.

        Arguments:
            model (QSPRModel): model to assess
            save (bool): whether to save predictions to file
            parameters (dict): optional model parameters to use in assessment
            use_proba (bool): use predictProba instead of predict for classification
            monitor (AssessorMonitor): optional, overrides monitor set in constructor
            **kwargs: additional keyword arguments for the fit function

        Returns:
            float | np.ndarray: predictions on the validation set
        """
        monitor = monitor or self.monitor
        evalparams = model.parameters if parameters is None else parameters
        # check if data is available
        model.checkForData()
        X, _ = model.data.getFeatures()
        y, _ = model.data.getTargetPropertiesValues()
        monitor.on_assessment_start(model)
        # prepare arrays to store molecule ids and predictions
        cvs_ids = np.array([None] * len(X))
        if model.task.isRegression() or not self.useProba:
            cvs = np.zeros((y.shape[0], model.nTargets))
        else:
            cvs = [
                np.zeros((y.shape[0], prop.nClasses)) for prop in model.targetProperties
            ]
        # cross validation
        folds = model.data.createFolds()
        fold_counter = np.zeros(y.shape[0])
        for i, (X_train, X_test, y_train, y_test, idx_train,
                idx_test) in enumerate(folds):
            logger.debug(
                "cross validation fold %s started: %s" %
                (i, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            monitor.on_fold_start(
                fold=i, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
            )
            # fit model
            crossval_estimator = model.loadEstimator(evalparams)
            model_fit = model.fit(
                X_train, y_train, monitor, crossval_estimator, mode=self.mode, **kwargs
            )
            # make predictions
            if model.task.isRegression() or not self.useProba:
                cvs[idx_test] = preds = model.predict(X_test, crossval_estimator)
            else:
                preds = model.predictProba(X_test, crossval_estimator)
                for idx in range(model.nTargets):
                    cvs[idx][idx_test] = preds[idx]
            # save molecule ids and fold number
            fold_counter[idx_test] = i
            cvs_ids[idx_test] = X.iloc[idx_test].index.to_numpy()
            logger.debug(
                "cross validation fold %s ended: %s" %
                (i, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            monitor.on_fold_end(
                model_fit,
                self.predictionsToDataFrame(
                    model,
                    y.iloc[idx_test],
                    preds,
                    idx_test,
                    extra_columns={"Fold": fold_counter[idx_test]},
                ),
            )
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

        monitor.on_assessment_end(output)
        return output


class TestSetAssessor(ModelAssessor):
    """Assess a model on a test set.

    Attributes:
        useProba (bool): use predictProba instead of predict for classification
        monitor (AssessorMonitor): monitor to use for assessment
        mode (EarlyStoppingMode): mode to use for early stopping
    """
    def __init__(
        self,
        monitor: AssessorMonitor = NullAssessorMonitor(),
        use_proba: bool = True,
        mode: EarlyStoppingMode | None = None,
    ):
        super().__init__(monitor, use_proba, mode)

    def __call__(
        self,
        model: QSPRModel,
        save: bool = True,
        parameters: dict | None = None,
        monitor: AssessorMonitor | None = None,
        **kwargs,
    ):
        """Make predictions for independent test set.

        Arguments:
            model (QSPRModel): model to assess
            save (bool): whether to save predictions to file
            parameters (dict): optional model parameters to use in assessment
            use_proba (bool): use predictProba instead of predict for classification
            monitor (AssessorMonitor): optional, overrides monitor set in constructor
            **kwargs: additional keyword arguments for the fit function

        Returns:
            float | np.ndarray: predictions for evaluation
        """
        monitor = monitor or self.monitor
        evalparams = model.parameters if parameters is None else parameters
        # check if data is available
        model.checkForData()
        X, X_ind = model.data.getFeatures()
        y, y_ind = model.data.getTargetPropertiesValues()
        monitor.on_assessment_start(model)
        # prepare arrays to store molecule ids and predictions
        inds_ids = X_ind.index.to_numpy()
        if not model.task.isRegression() and self.useProba:
            inds = [
                np.zeros((y_ind.shape[0], prop.nClasses))
                for prop in model.targetProperties
            ]
        monitor.on_fold_start(
                fold=1, X_train=X, y_train=y, X_test=X_ind, y_test=y_ind
            )
        # fit model
        ind_estimator = model.loadEstimator(evalparams)
        ind_estimator = model.fit(X, y, monitor, ind_estimator, mode=self.mode, **kwargs)
        # if independent test set is available, predict on it
        if X_ind.shape[0] > 0:
            if model.task.isRegression() or not self.useProba:
                inds = preds = model.predict(X_ind, ind_estimator)
            else:
                preds = model.predictProba(X_ind, ind_estimator)
                for idx in range(model.nTargets):
                    inds[idx] = preds[idx]
        else:
            logger.warning(
                "No independent test set available. "
                "Skipping prediction on independent test set."
            )
        index_name = model.data.getDF().index.name
        ind_index = pd.Index(inds_ids, name=index_name)
        monitor.on_fold_end(
                ind_estimator,
                self.predictionsToDataFrame(
                    model,
                    y_ind,
                    preds,
                    ind_index
                ),
            )
        # predict values for independent test set and save results
        if save:
            self.savePredictionsToFile(model, y_ind, inds, ind_index, "ind")
        return [(y_ind, inds)]
