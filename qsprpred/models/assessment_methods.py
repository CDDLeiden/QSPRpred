"""This module holds assessment methods for QSPRModels"""

from datetime import datetime
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ..data.interfaces import DataSplit
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
        scoring: str | Callable[[Iterable, Iterable], float],
        split: DataSplit | None = None,
        monitor: AssessorMonitor = NullAssessorMonitor(),
        use_proba: bool = True,
        mode: EarlyStoppingMode | None = None,
    ):
        super().__init__(scoring, monitor, use_proba, mode)
        self.split = split

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
            scoring (str | Callable): scoring function to use
            save (bool): whether to save predictions to file
            parameters (dict): optional model parameters to use in assessment
            monitor (AssessorMonitor): optional, overrides monitor set in constructor
            **kwargs: additional keyword arguments for the fit function

        Returns:
            float | np.ndarray: predictions on the validation set
        """
        monitor = monitor or self.monitor
        model.checkForData()
        data = model.data
        split = self.split or KFold(
            n_splits=5, shuffle=True, random_state=data.randomState
        )
        evalparams = model.parameters if parameters is None else parameters
        self.scoreFunc.checkMetricCompatibility(model.task, self.useProba)
        # check if data is available
        X, _ = data.getFeatures()
        y, _ = data.getTargetPropertiesValues()
        monitor.on_assessment_start(model, self.__class__.__name__)
        # cross validation
        fold_counter = np.zeros(y.shape[0])
        predictions = []
        scores = []
        for i, (X_train, X_test, y_train, y_test, idx_train, idx_test) in enumerate(
                data.iterFolds(split=split)
        ):
            logger.debug(
                "cross validation fold %s started: %s"
                % (i, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
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
                fold_predictions = model.predict(X_test, crossval_estimator)
            else:
                fold_predictions = model.predictProba(X_test, crossval_estimator)

            # score
            score = self.scoreFunc(y.iloc[idx_test], fold_predictions)
            scores.append(score)
            # save molecule ids and fold number
            fold_counter[idx_test] = i
            logger.debug(
                "cross validation fold %s ended: %s"
                % (i, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            fold_predictions_df = self.predictionsToDataFrame(
                model,
                y.iloc[idx_test],
                fold_predictions,
                idx_test,
                extra_columns={"Fold": fold_counter[idx_test]},
            )
            monitor.on_fold_end(model_fit, fold_predictions_df)
            predictions.append(fold_predictions_df)
        # save results
        if save:
            pd.concat(predictions).to_csv(f"{model.outPrefix}.cv.tsv", sep="\t")
        monitor.on_assessment_end()
        return scores


class TestSetAssessor(ModelAssessor):
    """Assess a model on a test set.

    Attributes:+
        useProba (bool): use predictProba instead of predict for classification
        monitor (AssessorMonitor): monitor to use for assessment
        mode (EarlyStoppingMode): mode to use for early stopping
    """

    def __init__(
        self,
        scoring: str | Callable[[Iterable, Iterable], float],
        monitor: AssessorMonitor = NullAssessorMonitor(),
        use_proba: bool = True,
        mode: EarlyStoppingMode | None = None,
    ):
        super().__init__(scoring, monitor, use_proba, mode)

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
            scoring (str | Callable): scoring function to use
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
        self.scoreFunc.checkMetricCompatibility(model.task, self.useProba)
        # check if data is available
        model.checkForData()
        X, X_ind = model.data.getFeatures()
        y, y_ind = model.data.getTargetPropertiesValues()
        monitor.on_assessment_start(model, self.__class__.__name__)
        monitor.on_fold_start(fold=1, X_train=X, y_train=y, X_test=X_ind, y_test=y_ind)
        # fit model
        ind_estimator = model.loadEstimator(evalparams)
        ind_estimator = model.fit(
            X, y, monitor, ind_estimator, mode=self.mode, **kwargs
        )
        # predict values for independent test set
        if model.task.isRegression() or not self.useProba:
            predictions = model.predict(X_ind, ind_estimator)
        else:
            predictions = model.predictProba(X_ind, ind_estimator)
        # score
        score = self.scoreFunc(y_ind, predictions)
        predictions_df = self.predictionsToDataFrame(
            model, y_ind, predictions, y_ind.index
        )
        monitor.on_fold_end(ind_estimator, predictions_df)
        # predict values for independent test set and save results
        if save:
            predictions_df.to_csv(f"{model.outPrefix}.ind.tsv", sep="\t")
        return [score]
