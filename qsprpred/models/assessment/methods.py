"""This module holds assessment methods for QSPRModels"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ...data.sampling.splits import DataSplit
from ...data.tables.interfaces.qspr_data_set import QSPRDataSet
from ...logs import logger
from ...models.early_stopping import EarlyStoppingMode
from ...models.model import QSPRModel
from ...models.monitors import AssessorMonitor, BaseMonitor
from .metrics.scikit_learn import SklearnMetrics
from ...data.processing.pipeline import DatasetPipeline


class ModelAssessor(ABC):
    """Base class for assessment methods.

    Attributes:
        name (str): name of the assessment method
        scoreFunc (Metric): scoring function to use, should match the output of the
            evaluation method (e.g. if the evaluation methods returns class 
            probabilities, the scoring function support class probabilities)
        monitor (AssessorMonitor): monitor to use for assessment, if None, a BaseMonitor
            is used
        useProba (bool): wheter to use probabilities for classification models
        mode (EarlyStoppingMode): early stopping mode for fitting
        splitMultitaskScores (bool): whether to split the scores per task for multitask 
            models
        scores (np.ndarray): Scores returned by the scoring function for each fold 
        predictions (pd.Dataframe): Predictions returned by the model for each fold
    """
    def __init__(
        self,
        name: str,
        scoring: str | Callable[[Iterable, Iterable], float],
        monitor: AssessorMonitor | None = None,
        use_proba: bool = True,
        mode: EarlyStoppingMode | None = None,
        split_multitask_scores: bool = False,
    ):
        """Initialize the evaluation method class.

        Args:
            name (str): name of the evaluation method
            scoring: str | Callable[[Iterable, Iterable], float],
            monitor (AssessorMonitor): monitor to track the evaluation
            use_proba (bool): use probabilities for classification models
            mode (EarlyStoppingMode): early stopping mode for fitting
            split_multitask_scores (bool): whether to split the scores per task for multitask models
        """
        self.name = name
        self.scoreFunc = (
            SklearnMetrics(scoring) if isinstance(scoring, str) else scoring
        )
        self.monitor = monitor
        self.useProba = use_proba
        self.mode = mode
        self.splitMultitaskScores = split_multitask_scores
        self.scores = None
        self.predictions = None

    @abstractmethod
    def __call__(
        self,
        model: QSPRModel,
        ds: QSPRDataSet,
        save: bool = True,
        parameters: dict | None = None,
        monitor: AssessorMonitor | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Evaluate the model.

        Args:
            model (QSPRModel): model to evaluate
            ds (QSPRDataSet): dataset to evaluate on
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
        y_train: np.ndarray,
        y_test: np.ndarray,
        train_preds: np.ndarray | list[np.ndarray],
        test_preds: np.ndarray | list[np.ndarray],
        fold: int,
    ) -> pd.DataFrame:
        """Create a dataframe with true values and predictions.

        Args:
            model (QSPRModel): model to evaluate.
            dataset (QSPRDataSet): dataset to evaluate on.
            y_train (np.ndarray): training target values.
            y_test (np.ndarray): testing target values.
            train_preds (np.ndarray | list[np.ndarray]): training predictions.
            test_preds (np.ndarray | list[np.ndarray]): testing predictions.
            fold (int): current fold number.
            
        Returns:
            pd.DataFrame: dataframe with true values and predictions.
        """
        # Combine predictions
        if isinstance(train_preds, list):
            predictions = [
                np.concatenate((train_preds[idx], test_preds[idx]))
                for idx in range(len(train_preds))
            ]
        else:
            predictions = np.vstack([train_preds, test_preds])
        
        # Combine target values into dataframe
        y = pd.concat([y_train, y_test])
        df_out = y.add_suffix("_Label")

        # Add predictions to dataframe
        for idx, prop in enumerate(model.targetProperties):
            if prop.task.isClassification() and self.useProba:
                df_out[f"{prop.name}_Prediction"] = np.argmax(predictions[idx], axis=1)
                df_out = pd.concat(
                    [
                        df_out,
                        pd.DataFrame(predictions[idx], index=y.index
                                    ).add_prefix(f"{prop.name}_ProbabilityClass_"),
                    ],
                    axis=1,
                )
            else:
                df_out[f"{prop.name}_Prediction"] = predictions[:, idx]
                
        # Add set labels
        set_labels = ["Train"] * len(y_train) + ["Test"] * len(y_test)
        df_out["Set"] = set_labels
        
        # Add fold number
        df_out["Fold"] = fold

        return df_out


class Assessor(ModelAssessor):
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
        name: str,
        scoring: str | Callable[[Iterable, Iterable], float],
        split: DataSplit,
        monitor: AssessorMonitor | None = None,
        use_proba: bool = True,
        mode: EarlyStoppingMode | None = None,
        round: int = 5,
        split_multitask_scores: bool = False,
    ):
        super().__init__(name, scoring, monitor, use_proba, mode, split_multitask_scores)
        self.split = split
        if monitor is None:
            self.monitor = BaseMonitor()
        self.round = round

    def __call__(
        self,
        model: QSPRModel,
        ds: QSPRDataSet,
        pipeline: DatasetPipeline | None = None,
        parameters: dict | None = None,
        monitor: AssessorMonitor | None = None,
        save: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Perform cross validation on the model with the given parameters.

        Arguments:
            model (QSPRModel): model to assess
            ds (QSPRDataSet): dataset to assess on
            scoring (str | Callable): scoring function to use
            pipeline (DatasetPipeline): optional pipeline to apply to the dataset
            parameters (dict): optional model parameters to use in assessment
            monitor (AssessorMonitor): optional, overrides monitor set in constructor
            order (pd.Index): optional, order of the indices in the dataset
            **kwargs: additional keyword arguments for the fit function

        Returns:
            np.ndarray: scores for the validation sets. If splitMultitaskScores is True, 
            each column represents a task and each row a fold. Otherwise, a 1D array is
            returned with the scores for each fold.
        """
        monitor = monitor or self.monitor
        evalparams = model.parameters if parameters is None else parameters
        pipeline = pipeline if pipeline is not None else DatasetPipeline()
        monitor.onAssessmentStart(
            model, ds, pipeline, self.name, evalparams, self.split, 
        )
        # Assess model on each fold in the split
        self.scores = []
        self.predictions = []
        for i, (X_train, y_train, X_test, y_test) in enumerate(
            pipeline.apply(ds, self.split)
        ):
            logger.debug(
                "Model Assessment fold %s started: %s" %
                (i, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            monitor.onFoldStart(
                fold=i, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
            )
            # fit model
            model.initFromData(ds, pipeline)
            estimator = model.loadEstimator(evalparams)
            model_fit = model.fit(
                X_train,
                y_train,
                estimator,
                self.mode,
                monitor=monitor,
                **kwargs,
            )
            # make predictions
            if model.task.isRegression() or not self.useProba:
                test_preds = model.predict(X_test, estimator)
                train_preds = model.predict(X_train, estimator)
            else:
                test_preds = model.predictProba(X_test, estimator)
                train_preds = model.predictProba(X_train, estimator)
            # score
            if model.isMultiTask and self.splitMultitaskScores:
                scores_tasks = []
                for idx, prop in enumerate(model.targetProperties):
                    if self.useProba and prop.task.isClassification():
                        prop_predictions = [test_preds[idx]]
                        scores_tasks.append(
                            self.scoreFunc(y_test[prop.name], prop_predictions)
                        )
                    else:
                        scores_tasks.append(
                            self.scoreFunc(
                                y_test[prop.name], test_preds[:, idx]
                            )
                        )
                self.scores.append(scores_tasks)
            else:
                score = self.scoreFunc(y_test, test_preds)
                self.scores.append(score)
            # Combine predictions and log fold results
            logger.debug(
                "fold %s ended: %s" %
                (i, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            preds_df = self.predictionsToDataFrame(
                model, y_train, y_test, train_preds, test_preds, fold=i
            )
            monitor.onFoldEnd(model_fit, preds_df, self.scores[i])
            self.predictions.append(preds_df)
        monitor.onAssessmentEnd(pd.concat(self.predictions))
        if save:
            pd.concat(self.predictions).round(self.round
                ).to_csv(f"{model.outPrefix}_{self.name}.tsv", sep="\t")
        return np.array(self.scores)