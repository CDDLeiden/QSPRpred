"""This module holds assessment methods for QSPRModels"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from qsprpred.data import QSPRDataset
from qsprpred.data.sampling.splits import DataSplit
from qsprpred.logs import logger
from qsprpred.models.early_stopping import EarlyStoppingMode
from qsprpred.models.metrics import SklearnMetrics
from qsprpred.models.model import QSPRModel
from qsprpred.models.monitors import AssessorMonitor, BaseMonitor


class ModelAssessor(ABC):
    """Base class for assessment methods.

    Attributes:
        scoreFunc (Metric): scoring function to use, should match the output of the
                        evaluation method (e.g. if the evaluation methods returns
                        class probabilities, the scoring function support class
                        probabilities)
        monitor (AssessorMonitor): monitor to use for assessment, if None, a BaseMonitor
            is used
        useProba (bool): use probabilities for classification models
        mode (EarlyStoppingMode): early stopping mode for fitting
        splitMultitaskScores (bool): whether to split the scores per task for multitask models
    """

    def __init__(
        self,
        scoring: str | Callable[[Iterable, Iterable], float],
        monitor: AssessorMonitor | None = None,
        use_proba: bool = True,
        mode: EarlyStoppingMode | None = None,
        split_multitask_scores: bool = False,
    ):
        """Initialize the evaluation method class.

        Args:
            scoring: str | Callable[[Iterable, Iterable], float],
            monitor (AssessorMonitor): monitor to track the evaluation
            use_proba (bool): use probabilities for classification models
            mode (EarlyStoppingMode): early stopping mode for fitting
            split_multitask_scores (bool): whether to split the scores per task for multitask models
        """
        self.monitor = monitor
        self.useProba = use_proba
        self.mode = mode
        self.scoreFunc = (
            SklearnMetrics(scoring) if isinstance(scoring, str) else scoring
        )
        self.splitMultitaskScores = split_multitask_scores

    @abstractmethod
    def __call__(
        self,
        model: QSPRModel,
        ds: QSPRDataset,
        save: bool = True,
        parameters: dict | None = None,
        monitor: AssessorMonitor | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Evaluate the model.

        Args:
            model (QSPRModel): model to evaluate
            ds (QSPRDataset): dataset to evaluate on
            save (bool): save predictions to file
            parameters (dict): parameters to use for the evaluation
            monitor (AssessorMonitor): monitor to track the evaluation, overrides
                                       the monitor set in the constructor
            kwargs: additional arguments for fit function of the model

        Returns:
            np.ndarray: scores for the model. If splitMultitaskScores is True, each
            column represents a task and each row a fold. Otherwise, a 1D array is
            returned with the scores for each fold.
        """

    def predictionsToDataFrame(
        self,
        model: QSPRModel,
        y: np.array,
        predictions: np.ndarray | list[np.ndarray],
        index: pd.Series,
        extra_columns: dict[str, np.ndarray] | None = None,
    ) -> pd.DataFrame:
        """Create a dataframe with true values and predictions.

        Args:
            model (QSPRModel): model to evaluate
            y (np.array): target values
            predictions (np.ndarray | list[np.ndarray]): predictions
            index (pd.Series): index of the data set
            extra_columns (dict[str, np.ndarray]): extra columns to add to the output
        """
        # Create dataframe with true values
        df_out = pd.DataFrame(
            y.values, columns=y.add_suffix("_Label").columns, index=index
        )
        # Add predictions to dataframe
        for idx, prop in enumerate(model.targetProperties):
            if prop.task.isClassification() and self.useProba:
                # convert one-hot encoded predictions to class labels
                # and add to train and test
                df_out[f"{prop.name}_Prediction"] = np.argmax(predictions[idx], axis=1)
                # add probability columns to train and test set
                df_out = pd.concat(
                    [
                        df_out,
                        pd.DataFrame(predictions[idx], index=index).add_prefix(
                            f"{prop.name}_ProbabilityClass_"
                        ),
                    ],
                    axis=1,
                )
            else:
                df_out[f"{prop.name}_Prediction"] = predictions[:, idx]
        # Add extra columns to dataframe if given (such as fold indexes)
        if extra_columns is not None:
            for col_name, col_values in extra_columns.items():
                df_out[col_name] = col_values
        return df_out


class CrossValAssessor(ModelAssessor):
    """Perform cross validation on a model.

    Attributes:
        useProba (bool): use predictProba instead of predict for classification
        monitor (AssessorMonitor): monitor to use for assessment, if None, a BaseMonitor
            is used
        mode (EarlyStoppingMode): mode to use for early stopping
        split (DataSplit): split to use for cross validation (default: KFold, n_splits=5)
        round (int): number of decimal places to round predictions to (default: 5)
        splitMultitaskScores (bool): whether to split the scores per task for multitask models
    """

    def __init__(
        self,
        scoring: str | Callable[[Iterable, Iterable], float],
        split: DataSplit | None = None,
        monitor: AssessorMonitor | None = None,
        use_proba: bool = True,
        mode: EarlyStoppingMode | None = None,
        round: int = 5,
        split_multitask_scores: bool = False,
    ):
        super().__init__(scoring, monitor, use_proba, mode, split_multitask_scores)
        self.split = split
        if monitor is None:
            self.monitor = BaseMonitor()
        self.round = round

    def __call__(
        self,
        model: QSPRModel,
        ds: QSPRDataset,
        save: bool = True,
        parameters: dict | None = None,
        monitor: AssessorMonitor | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Perform cross validation on the model with the given parameters.

        Arguments:
            model (QSPRModel): model to assess
            ds (QSPRDataset): dataset to assess on
            scoring (str | Callable): scoring function to use
            save (bool): whether to save predictions to file
            parameters (dict): optional model parameters to use in assessment
            monitor (AssessorMonitor): optional, overrides monitor set in constructor
            **kwargs: additional keyword arguments for the fit function

        Returns:
            np.ndarray: scores for the validation sets. If splitMultitaskScores is True, each
            column represents a task and each row a fold. Otherwise, a 1D array is
            returned with the scores for each fold.
        """
        model.initFromDataset(ds)
        monitor = monitor or self.monitor
        split = self.split or KFold(
            n_splits=5, shuffle=True, random_state=model.randomState
        )
        evalparams = model.parameters if parameters is None else parameters
        X, _ = ds.getFeatures()
        y, _ = ds.getTargetPropertiesValues()
        monitor.onAssessmentStart(model, ds, self.__class__.__name__)
        # cross validation
        fold_counter = np.zeros(y.shape[0])
        predictions = []
        scores = []
        for i, (X_train, X_test, y_train, y_test, idx_train, idx_test) in enumerate(
            ds.iterFolds(split=split)
        ):
            logger.debug(
                "cross validation fold %s started: %s"
                % (i, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            monitor.onFoldStart(
                fold=i, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
            )
            # fit model
            crossval_estimator = model.loadEstimator(evalparams)
            model_fit = model.fit(
                X_train,
                y_train,
                crossval_estimator,
                self.mode,
                monitor=monitor,
                **kwargs,
            )
            # make predictions
            if model.task.isRegression() or not self.useProba:
                fold_predictions = model.predict(X_test, crossval_estimator)
            else:
                fold_predictions = model.predictProba(X_test, crossval_estimator)
            # score
            if model.isMultiTask and self.splitMultitaskScores:
                scores_tasks = []
                for idx, prop in enumerate(model.targetProperties):
                    if self.useProba and prop.task.isClassification():
                        prop_predictions = [fold_predictions[idx]]
                        scores_tasks.append(
                            self.scoreFunc(y.iloc[idx_test, idx], prop_predictions)
                        )
                    else:
                        scores_tasks.append(
                            self.scoreFunc(
                                y.iloc[idx_test, idx], fold_predictions[:, idx]
                            )
                        )
                scores.append(scores_tasks)
            else:
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
                pd.Series(y.index).iloc[idx_test],
                extra_columns={"Fold": fold_counter[idx_test]},
            )
            monitor.onFoldEnd(model_fit, fold_predictions_df)
            predictions.append(fold_predictions_df)
        # save results
        if save:
            pd.concat(predictions).round(self.round).to_csv(
                f"{model.outPrefix}.cv.tsv", sep="\t"
            )
        monitor.onAssessmentEnd(pd.concat(predictions))
        return np.array(scores)


class TestSetAssessor(ModelAssessor):
    """Assess a model on a test set.

    Attributes:+
        useProba (bool): use predictProba instead of predict for classification
        monitor (AssessorMonitor): monitor to use for assessment, if None, a BaseMonitor
            is used
        mode (EarlyStoppingMode): mode to use for early stopping
        round (int): number of decimal places to round predictions to (default: 3)
        splitMultitaskScores (bool): whether to split the scores per task for multitask models
    """

    def __init__(
        self,
        scoring: str | Callable[[Iterable, Iterable], float],
        monitor: AssessorMonitor | None = None,
        use_proba: bool = True,
        mode: EarlyStoppingMode | None = None,
        round: int = 5,
        split_multitask_scores: bool = False,
    ):
        super().__init__(scoring, monitor, use_proba, mode, split_multitask_scores)
        if monitor is None:
            self.monitor = BaseMonitor()
        self.round = round

    def __call__(
        self,
        model: QSPRModel,
        ds: QSPRDataset,
        save: bool = True,
        parameters: dict | None = None,
        monitor: AssessorMonitor | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Make predictions for independent test set.

        Arguments:
            model (QSPRModel): model to assess
            ds (QSPRDataset): dataset to assess on
            scoring (str | Callable): scoring function to use
            save (bool): whether to save predictions to file
            parameters (dict): optional model parameters to use in assessment
            use_proba (bool): use predictProba instead of predict for classification
            monitor (AssessorMonitor): optional, overrides monitor set in constructor
            **kwargs: additional keyword arguments for the fit function

        Returns:
            np.ndarray: scores for the test set. If splitMultitaskScores is True, each
            column represents a task. Otherwise, a 1D array is returned with the score
            for the test set.
        """
        model.initFromDataset(ds)
        monitor = monitor or self.monitor
        evalparams = model.parameters if parameters is None else parameters
        X, X_ind = ds.getFeatures()
        y, y_ind = ds.getTargetPropertiesValues()
        monitor.onAssessmentStart(model, ds, self.__class__.__name__)
        monitor.onFoldStart(fold=0, X_train=X, y_train=y, X_test=X_ind, y_test=y_ind)
        # fit model
        ind_estimator = model.loadEstimator(evalparams)
        ind_estimator = model.fit(
            X, y, ind_estimator, self.mode, monitor=monitor, **kwargs
        )
        # predict values for independent test set
        if model.task.isRegression() or not self.useProba:
            predictions = model.predict(X_ind, ind_estimator)
        else:
            predictions = model.predictProba(X_ind, ind_estimator)
        # score
        if model.isMultiTask and self.splitMultitaskScores:
            scores_tasks = []
            for idx, prop in enumerate(model.targetProperties):
                if self.useProba and prop.task.isClassification():
                    prop_predictions = [predictions[idx]]
                    scores_tasks.append(
                        self.scoreFunc(y_ind.iloc[:, idx], prop_predictions)
                    )
                else:
                    scores_tasks.append(
                        self.scoreFunc(y_ind.iloc[:, idx], predictions[:, idx])
                    )
            score = scores_tasks
        else:
            score = [self.scoreFunc(y_ind, predictions)]
        predictions_df = self.predictionsToDataFrame(
            model, y_ind, predictions, y_ind.index
        )
        monitor.onFoldEnd(ind_estimator, predictions_df)
        # predict values for independent test set and save results
        if save:
            predictions_df.round(self.round).to_csv(
                f"{model.outPrefix}.ind.tsv", sep="\t"
            )
        monitor.onAssessmentEnd(predictions_df)
        return np.array(score)
