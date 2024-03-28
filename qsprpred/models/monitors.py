import json
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

from .model import QSPRModel
from ..data.tables.qspr import QSPRDataset
from ..utils.serialization import JSONSerializable


class FitMonitor(JSONSerializable, ABC):
    """Base class for monitoring the fitting of a model."""

    @abstractmethod
    def onFitStart(
            self,
            model: QSPRModel,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
    ):
        """Called before the training has started.

        Args:
            model (QSPRModel): model to be fitted
            X_train (np.ndarray): training data
            y_train (np.ndarray): training targets
            X_val (np.ndarray | None): validation data, used for early stopping
            y_val (np.ndarray | None): validation targets, used for early stopping
        """

    @abstractmethod
    def onFitEnd(self, estimator: Any, best_epoch: int | None = None):
        """Called after the training has finished.

        Args:
            estimator (Any): estimator that was fitted
            best_epoch (int | None): index of the best epoch
        """

    @abstractmethod
    def onEpochStart(self, epoch: int):
        """Called before each epoch of the training.

        Args:
            epoch (int): index of the current epoch
        """

    @abstractmethod
    def onEpochEnd(self, epoch: int, train_loss: float, val_loss: float | None = None):
        """Called after each epoch of the training.

        Args:
            epoch (int): index of the current epoch
            train_loss (float): loss of the current epoch
            val_loss (float | None): validation loss of the current epoch
        """

    @abstractmethod
    def onBatchStart(self, batch: int):
        """Called before each batch of the training.

        Args:
            batch (int): index of the current batch
        """

    @abstractmethod
    def onBatchEnd(self, batch: int, loss: float):
        """Called after each batch of the training.

        Args:
            batch (int): index of the current batch
            loss (float): loss of the current batch
        """


class AssessorMonitor(FitMonitor):
    """Base class for monitoring the assessment of a model."""

    @abstractmethod
    def onAssessmentStart(
            self, model: QSPRModel, data: QSPRDataset, assesment_type: str
    ):
        """Called before the assessment has started.

        Args:
            model (QSPRModel): model to assess
            data (QSPRDataset): data set used in assessment
            assesment_type (str): type of assessment
        """

    @abstractmethod
    def onAssessmentEnd(self, predictions: pd.DataFrame):
        """Called after the assessment has finished.

        Args:
            predictions (pd.DataFrame): predictions of the assessment
        """

    @abstractmethod
    def onFoldStart(
            self,
            fold: int,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
    ):
        """Called before each fold of the assessment.

        Args:
            fold (int): index of the current fold
            X_train (np.ndarray): training data of the current fold
            y_train (np.ndarray): training targets of the current fold
            X_test (np.ndarray): test data of the current fold
            y_test (np.ndarray): test targets of the current fold
        """

    @abstractmethod
    def onFoldEnd(
            self, model_fit: Any | tuple[Any, int], fold_predictions: pd.DataFrame
    ):
        """Called after each fold of the assessment.

        Args:
            model_fit (Any|tuple[Any, int]): fitted estimator of the current fold, or
                                             tuple containing the fitted estimator and
                                             the number of epochs it was trained for
        """


class HyperparameterOptimizationMonitor(AssessorMonitor):
    """Base class for monitoring the hyperparameter optimization of a model."""

    @abstractmethod
    def onOptimizationStart(
            self, model: QSPRModel, data: QSPRDataset, config: dict,
            optimization_type: str
    ):
        """Called before the hyperparameter optimization has started.

        Args:
            model (QSPRModel): model to optimize
            data (QSPRDataset): data set used in optimization
            config (dict): configuration of the hyperparameter optimization
            optimization_type (str): type of hyperparameter optimization
        """

    @abstractmethod
    def onOptimizationEnd(self, best_score: float, best_parameters: dict):
        """Called after the hyperparameter optimization has finished.

        Args:
            best_score (float): best score found during optimization
            best_parameters (dict): best parameters found during optimization
        """

    @abstractmethod
    def onIterationStart(self, params: dict):
        """Called before each iteration of the hyperparameter optimization.

        Args:
            params (dict): parameters used for the current iteration
        """

    @abstractmethod
    def onIterationEnd(self, score: float, scores: list[float]):
        """Called after each iteration of the hyperparameter optimization.

        Args:
            score (float): (aggregated) score of the current iteration
            scores (list[float]): scores of the current iteration
                                  (e.g for cross-validation)
        """


class NullMonitor(HyperparameterOptimizationMonitor):
    """Monitor that does nothing."""

    def onFitStart(
            self,
            model: QSPRModel,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
    ):
        """Called before the training has started.

        Args:
            model (QSPRModel): model to be fitted
            X_train (np.ndarray): training data
            y_train (np.ndarray): training targets
            X_val (np.ndarray | None): validation data, used for early stopping
            y_val (np.ndarray | None): validation targets, used for early stopping
        """

    def onFitEnd(self, estimator: Any, best_epoch: int | None = None):
        """Called after the training has finished.

        Args:
            estimator (Any): estimator that was fitted
            best_epoch (int | None): index of the best epoch
        """

    def onEpochStart(self, epoch: int):
        """Called before each epoch of the training.

        Args:
            epoch (int): index of the current epoch
        """

    def onEpochEnd(self, epoch: int, train_loss: float, val_loss: float | None = None):
        """Called after each epoch of the training.

        Args:
            epoch (int): index of the current epoch
            train_loss (float): loss of the current epoch
            val_loss (float | None): validation loss of the current epoch
        """

    def onBatchStart(self, batch: int):
        """Called before each batch of the training.

        Args:
            batch (int): index of the current batch
        """

    def onBatchEnd(self, batch: int, loss: float):
        """Called after each batch of the training.

        Args:
            batch (int): index of the current batch
            loss (float): loss of the current batch
        """

    def onAssessmentStart(
            self, model: QSPRModel, data: QSPRDataset, assesment_type: str
    ):
        """Called before the assessment has started.

        Args:
            model (QSPRModel): model to assess
            data (QSPRDataset): data set used in assessment
            assesment_type (str): type of assessment
        """

    def onAssessmentEnd(self, predictions: pd.DataFrame):
        """Called after the assessment has finished.

        Args:
            predictions (pd.DataFrame): predictions of the assessment
        """

    def onFoldStart(
            self,
            fold: int,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
    ):
        """Called before each fold of the assessment.

        Args:
            fold (int): index of the current fold
            X_train (np.ndarray): training data of the current fold
            y_train (np.ndarray): training targets of the current fold
            X_test (np.ndarray): test data of the current fold
            y_test (np.ndarray): test targets of the current fold
        """

    def onFoldEnd(
            self, model_fit: Any | tuple[Any, int], fold_predictions: pd.DataFrame
    ):
        """Called after each fold of the assessment.

        Args:
            model_fit (Any|tuple[Any, int]): fitted estimator of the current fold, or
                                             tuple containing the fitted estimator and
                                             the number of epochs it was trained for
        """

    def onOptimizationStart(
            self, model: QSPRModel, data: QSPRDataset, config: dict,
            optimization_type: str
    ):
        """Called before the hyperparameter optimization has started.

        Args:
            model (QSPRModel): model to optimize
            data (QSPRDataset): data set used in optimization
            config (dict): configuration of the hyperparameter optimization
            optimization_type (str): type of hyperparameter optimization
        """

    def onOptimizationEnd(self, best_score: float, best_parameters: dict):
        """Called after the hyperparameter optimization has finished.

        Args:
            best_score (float): best score found during optimization
            best_parameters (dict): best parameters found during optimization
        """

    def onIterationStart(self, params: dict):
        """Called before each iteration of the hyperparameter optimization.

        Args:
            params (dict): parameters used for the current iteration
        """

    def onIterationEnd(self, score: float, scores: list[float]):
        """Called after each iteration of the hyperparameter optimization.

        Args:
            score (float): (aggregated) score of the current iteration
            scores (list[float]): scores of the current iteration
                                  (e.g for cross-validation)
        """


class ListMonitor(HyperparameterOptimizationMonitor):
    """Monitor that combines multiple monitors.

    Attributes:
        monitors (list[HyperparameterOptimizationMonitor]): list of monitors
    """

    def __init__(self, monitors: list[HyperparameterOptimizationMonitor]):
        """Initialize the monitor.

        Args:
            monitors (list[HyperparameterOptimizationMonitor]): list of monitors
        """
        self.monitors = monitors

    def onFitStart(
            self,
            model: QSPRModel,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
    ):
        """Called before the training has started.

        Args:
            model (QSPRModel): model to be fitted
            X_train (np.ndarray): training data
            y_train (np.ndarray): training targets
            X_val (np.ndarray | None): validation data, used for early stopping
            y_val (np.ndarray | None): validation targets, used for early stopping
        """
        for monitor in self.monitors:
            monitor.onFitStart(model, X_train, y_train, X_val, y_val)

    def onFitEnd(self, estimator: Any, best_epoch: int | None = None):
        """Called after the training has finished.

        Args:
            estimator (Any): estimator that was fitted
            best_epoch (int | None): index of the best epoch
        """
        for monitor in self.monitors:
            monitor.onFitEnd(estimator, best_epoch)

    def onEpochStart(self, epoch: int):
        """Called before each epoch of the training.

        Args:
            epoch (int): index of the current epoch
        """
        for monitor in self.monitors:
            monitor.onEpochStart(epoch)

    def onEpochEnd(self, epoch: int, train_loss: float, val_loss: float | None = None):
        """Called after each epoch of the training.

        Args:
            epoch (int): index of the current epoch
            train_loss (float): loss of the current epoch
            val_loss (float | None): validation loss of the current epoch
        """
        for monitor in self.monitors:
            monitor.onEpochEnd(epoch, train_loss, val_loss)

    def onBatchStart(self, batch: int):
        """Called before each batch of the training.

        Args:
            batch (int): index of the current batch
        """
        for monitor in self.monitors:
            monitor.onBatchStart(batch)

    def onBatchEnd(self, batch: int, loss: float):
        """Called after each batch of the training.

        Args:
            batch (int): index of the current batch
            loss (float): loss of the current batch
        """
        for monitor in self.monitors:
            monitor.onBatchEnd(batch, loss)

    def onAssessmentStart(
            self, model: QSPRModel, data: QSPRDataset, assesment_type: str
    ):
        """Called before the assessment has started.

        Args:
            model (QSPRModel): model to assess
            data (QSPRDataset): data set used in assessment
            assesment_type (str): type of assessment
        """
        for monitor in self.monitors:
            monitor.onAssessmentStart(model, data, assesment_type)

    def onAssessmentEnd(self, predictions: pd.DataFrame):
        """Called after the assessment has finished.

        Args:
            predictions (pd.DataFrame): predictions of the assessment
        """
        for monitor in self.monitors:
            monitor.onAssessmentEnd(predictions)

    def onFoldStart(
            self,
            fold: int,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
    ):
        """Called before each fold of the assessment.

        Args:
            fold (int): index of the current fold
            X_train (np.ndarray): training data of the current fold
            y_train (np.ndarray): training targets of the current fold
            X_test (np.ndarray): test data of the current fold
            y_test (np.ndarray): test targets of the current fold
        """
        for monitor in self.monitors:
            monitor.onFoldStart(fold, X_train, y_train, X_test, y_test)

    def onFoldEnd(
            self, model_fit: Any | tuple[Any, int], fold_predictions: pd.DataFrame
    ):
        """Called after each fold of the assessment.

        Args:
            model_fit (Any|tuple[Any, int]): fitted estimator of the current fold, or
                                             tuple containing the fitted estimator and
                                             the number of epochs it was trained for
        """
        for monitor in self.monitors:
            monitor.onFoldEnd(model_fit, fold_predictions)

    def onOptimizationStart(
            self, model: QSPRModel, data: QSPRDataset, config: dict,
            optimization_type: str
    ):
        """Called before the hyperparameter optimization has started.

        Args:
            model (QSPRModel): model to optimize
            data (QSPRDataset): data set used in optimization
            config (dict): configuration of the hyperparameter optimization
            optimization_type (str): type of hyperparameter optimization
        """
        for monitor in self.monitors:
            monitor.onOptimizationStart(model, data, config, optimization_type)

    def onOptimizationEnd(self, best_score: float, best_parameters: dict):
        """Called after the hyperparameter optimization has finished.

        Args:
            best_score (float): best score found during optimization
            best_parameters (dict): best parameters found during optimization
        """
        for monitor in self.monitors:
            monitor.onOptimizationEnd(best_score, best_parameters)

    def onIterationStart(self, params: dict):
        """Called before each iteration of the hyperparameter optimization.

        Args:
            params (dict): parameters used for the current iteration
        """
        for monitor in self.monitors:
            monitor.onIterationStart(params)

    def onIterationEnd(self, score: float, scores: list[float]):
        """Called after each iteration of the hyperparameter optimization.

        Args:
            score (float): (aggregated) score of the current iteration
            scores (list[float]): scores of the current iteration
                                  (e.g for cross-validation)
        """
        for monitor in self.monitors:
            monitor.onIterationEnd(score, scores)


class BaseMonitor(HyperparameterOptimizationMonitor):
    """Base monitoring the fitting, training and optimization of a model.

    Information about the fitting, training and optimization process is stored
    internally, but not logged. This class can be used as a base class for other
    other monitors that do log the information elsewhere.

    If used to monitor hyperparameter optimization, the information about the
    underlying assessments and fits is stored in the assessments and fits
    attributes, respectively. If used to monitor assessment, the information
    about the fits is stored in the fits attribute.

    Attributes:
        config (dict): configuration of the hyperparameter optimization
        bestScore (float): best score found during optimization
        bestParameters (dict): best parameters found during optimization
        assessments (dict): dictionary of assessments, keyed by the iteration number
            (each assessment includes: assessmentModel, assessmentDataset, foldData,
                predictions, estimators, fits)
        scores (pd.DataFrame): scores for each hyperparameter search iteration
        model (QSPRModel): model to optimize
        data (QSPRDataset): dataset used in optimization

        assessmentType (str): type of current assessment
        assessmentModel (QSPRModel): model to assess in current assessment
        assessmentDataset (QSPRDataset): data set used in current assessment
        foldData (dict): dictionary of input data, keyed by the fold index, of the
            current assessment
        predictions (pd.DataFrame): predictions for the dataset of the current assessment
        estimators (dict): dictionary of fitted estimators, keyed by the fold index of
            the current assessment
        currentFold (int): index of the current fold of the
            current assessment
        fits (dict): dictionary of fit data, keyed by the fold index of the current
            assessment (each fit includes: fitData, fitLog, batchLog, bestEstimator,
                bestEpoch)

        fitData (dict): dictionary of input data of the current fit of the current
            assessment
        fitModel (QSPRModel): model to fit in current fit of the current assessment
        fitLog (pd.DataFrame): log of the training process of the current fit of the
            current assessment
        batchLog (pd.DataFrame): log of the training process per batch of the current
            fit of the current assessment
        currentEpoch (int): index of the current epoch of the current fit of the current
            assessment
        currentBatch (int): index of the current batch of the current fit of the current
            assessment
        bestEstimator (Any): best estimator of the current fit of the current
            assessment
        bestEpoch (int): index of the best epoch of the current fit of the current
            assessment
    """

    def __init__(self):
        self.data = None
        self.model = None
        self.optimizationType = None
        # hyperparameter optimization data
        self.config = None
        self.bestScore = None
        self.bestParameters = None
        self.parameters = {}
        self.assessments = {}
        self.scores = pd.DataFrame(
            columns=["aggregated_score", "fold_scores"]
        ).rename_axis("Iteration")
        self.iteration = None

        # assessment data
        self.assessmentModel = None
        self.assessmentDataset = None
        self.foldData = {}
        self.predictions = None
        self.estimators = {}
        self.currentFold = None
        self.fits = {}

        # fit data
        self.fitModel = None
        self.fitData = None
        self.fitLog = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])
        self.batchLog = pd.DataFrame(columns=["epoch", "batch", "loss"])
        self.currentEpoch = None
        self.currentBatch = None
        self.bestEstimator = None
        self.bestEpoch = None

    def __getstate__(self):
        o_dict = super().__getstate__()
        # convert all data frames to dicts
        for key, value in o_dict.items():
            if isinstance(value, pd.DataFrame):
                o_dict[key] = {"pd.DataFrame": value.to_dict()}
        return o_dict

    def __setstate__(self, state):
        # convert all dicts to data frames
        for key, value in state.items():
            if isinstance(value, dict) and "pd.DataFrame" in value:
                state[key] = pd.DataFrame.from_dict(value["pd.DataFrame"])
        super().__setstate__(state)

    def onOptimizationStart(
            self, model: QSPRModel, data: QSPRDataset, config: dict,
            optimization_type: str
    ):
        """Called before the hyperparameter optimization has started.

        Args:
            model (QSPRModel): model to optimize
            data (QSPRDataset): data set used in optimization
            config (dict): configuration of the hyperparameter optimization
            optimization_type (str): type of hyperparameter optimization
        """
        self.optimizationType = optimization_type
        self.iteration = 0
        self.model = model
        self.data = data
        self.config = config

    def onOptimizationEnd(self, best_score: float, best_parameters: dict):
        """Called after the hyperparameter optimization has finished.

        Args:
            best_score (float): best score found during optimization
            best_parameters (dict): best parameters found during optimization
        """
        self.bestScore = best_score
        self.bestParameters = best_parameters

    def onIterationStart(self, params: dict):
        """Called before each iteration of the hyperparameter optimization.

        Args:
            params (dict): parameters used for the current iteration
        """
        self.parameters[self.iteration] = params

    def onIterationEnd(self, score: float, scores: list[float]):
        """Called after each iteration of the hyperparameter optimization.

        Args:
            score (float): (aggregated) score of the current iteration
            scores (list[float]): scores of the current iteration
                                  (e.g for cross-validation)
        """
        self.scores.loc[self.iteration, :] = [score, scores]
        self.assessments[self.iteration] = self._get_assessment()
        self._clear_assessment()
        self.iteration += 1

    def onAssessmentStart(
            self, model: QSPRModel, data: QSPRDataset, assesment_type: str
    ):
        """Called before the assessment has started.

        Args:
            model (QSPRModel): model to assess
            data (QSPRDataset): data set used in assessment
            assesment_type (str): type of assessment
        """
        self.assessmentModel = model
        self.assessmentDataset = data
        self.assessmentType = assesment_type

    def onAssessmentEnd(self, predictions: pd.DataFrame):
        """Called after the assessment has finished.

        Args:
            predictions (pd.DataFrame): predictions of the assessment
        """
        self.predictions = predictions

    def onFoldStart(
            self,
            fold: int,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
    ):
        """Called before each fold of the assessment.

        Args:
            fold (int): index of the current fold
            X_train (np.ndarray): training data of the current fold
            y_train (np.ndarray): training targets of the current fold
            X_test (np.ndarray): test data of the current fold
            y_test (np.ndarray): test targets of the current fold
        """
        self.currentFold = fold
        self.foldData[fold] = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

    def onFoldEnd(
            self, model_fit: Any | tuple[Any, int], fold_predictions: pd.DataFrame
    ):
        """Called after each fold of the assessment.

        Args:
            model_fit (Any|tuple[Any, int]): fitted estimator of the current fold, or
                                             tuple containing the fitted estimator and
                                             the number of epochs it was trained for
            fold_predictions (pd.DataFrame): predictions of the current fold
        """
        self.estimators[self.currentFold] = model_fit
        self.fits[self.currentFold] = self._getFit()
        self._clearFit()

    def _clear_assessment(self):
        """Clear the assessment data."""
        self.assessmentModel = None
        self.asssessmentDataset = None
        self.foldData = {}
        self.predictions = None
        self.estimators = {}
        self.fits = {}

    def _get_assessment(self) -> tuple[QSPRModel, QSPRDataset, pd.DataFrame, dict]:
        """Return the assessment data."""
        return {
            "assessmentModel": self.assessmentModel,
            "assessmentDataset": self.assessmentDataset,
            "foldData": self.foldData,
            "predictions": self.predictions,
            "estimators": self.estimators,
            "fits": self.fits,
        }

    def onFitStart(
            self,
            model: QSPRModel,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
    ):
        """Called before the training has started.

        Args:
            model (QSPRModel): model to be fitted
            data (QSPRDataset): data set used in training
            X_train (np.ndarray): training data
            y_train (np.ndarray): training targets
            X_val (np.ndarray | None): validation data, used for early stopping
            y_val (np.ndarray | None): validation targets, used for early stopping
        """
        self.fitModel = model
        self.fitData = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
        }
        self.currentEpoch = 0
        self.currentBatch = 0

    def onFitEnd(self, estimator: Any, best_epoch: int | None = None):
        """Called after the training has finished.

        Args:
            estimator (Any): estimator that was fitted
            best_epoch (int | None): index of the best epoch
        """
        self.bestEstimator = estimator
        self.bestEpoch = best_epoch

    def onEpochStart(self, epoch: int):
        """Called before each epoch of the training.

        Args:
            epoch (int): index of the current epoch
        """
        self.currentEpoch = epoch

    def onEpochEnd(self, epoch: int, train_loss: float, val_loss: float | None = None):
        """Called after each epoch of the training.

        Args:
            epoch (int): index of the current epoch
            train_loss (float): loss of the current epoch
            val_loss (float | None): validation loss of the current epoch
        """
        self.fitLog.loc[epoch] = [epoch, train_loss, val_loss]

    def onBatchStart(self, batch: int):
        """Called before each batch of the training.

        Args:
            batch (int): index of the current batch
        """
        self.currentBatch = batch

    def onBatchEnd(self, batch: int, loss: float):
        """Called after each batch of the training.

        Args:
            batch (int): index of the current batch
            loss (float): loss of the current batch
        """
        self.batchLog.loc[len(self.batchLog)] = [self.currentEpoch, batch, loss]

    def _clearFit(self):
        self.fitLog = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])
        self.batchLog = pd.DataFrame(columns=["epoch", "batch", "loss"])
        self.fitData = None
        self.currentEpoch = None
        self.currentBatch = None
        self.bestEstimator = None
        self.bestEpoch = None

    def _getFit(self) -> tuple[pd.DataFrame, pd.DataFrame, Any, int]:
        return {
            "fitData": self.fitData,
            "fitLog": self.fitLog,
            "batchLog": self.batchLog,
            "bestEstimator": self.bestEstimator,
            "bestEpoch": self.bestEpoch,
        }


class FileMonitor(BaseMonitor):
    def __init__(
            self,
            save_optimization: bool = True,
            save_assessments: bool = True,
            save_fits: bool = True,
    ):
        """Monitor hyperparameter optimization, assessment and fitting to files.

        Args:
            save_optimization (bool): whether to save the hyperparameter optimization
                scores
            save_assessments (bool): whether to save assessment predictions
            save_fits (bool): whether to save the fit log and batch log
        """
        super().__init__()
        self.saveOptimization = save_optimization
        self.saveAssessments = save_assessments
        self.saveFits = save_fits
        self.outDir = None

    def onOptimizationStart(
            self, model: QSPRModel, data: QSPRDataset, config: dict,
            optimization_type: str
    ):
        """Called before the hyperparameter optimization has started.

        Args:
            model (QSPRModel): model to optimize
            data (QSPRDataset): data set used in optimization
            config (dict): configuration of the hyperparameter optimization
            optimization_type (str): type of hyperparameter optimization
        """
        super().onOptimizationStart(model, data, config, optimization_type)
        self.outDir = self.outDir or model.outDir
        self.optimizationPath = f"{self.outDir}/{self.optimizationType}"

    def onIterationStart(self, params: dict):
        """Called before each iteration of the hyperparameter optimization.

        Args:
            params (dict): parameters used for the current iteration
        """
        super().onIterationStart(params)
        self.optimizationItPath = f"{self.optimizationPath}/iteration_{self.iteration}"

    def onIterationEnd(self, score: float, scores: list[float]):
        """Called after each iteration of the hyperparameter optimization.

        Args:
            score (float): (aggregated) score of the current iteration
            scores (list[float]): scores of the current iteration
                                  (e.g for cross-validation)
        """
        if self.saveAssessments:
            # save parameters to json
            with open(f"{self.optimizationItPath}/parameters.json", "w") as f:
                json.dump(self.parameters[self.iteration], f)
        super().onIterationEnd(score, scores)
        if self.saveOptimization:
            # add parameters to scores with separate columns
            savescores = pd.concat(
                [self.scores, pd.DataFrame(self.parameters).T], axis=1
            )
            savescores.to_csv(
                f"{self.optimizationPath}/{self.optimizationType}_scores.tsv",
                sep="\t",
                index=False,
            )

    def onAssessmentStart(
            self, model: QSPRModel, data: QSPRDataset, assesment_type: str
    ):
        """Called before the assessment has started.

        Args:
            model (QSPRModel): model to assess
            data (QSPRDataset): data set used in assessment
            assesment_type (str): type of assessment
        """
        super().onAssessmentStart(model, data, assesment_type)
        self.outDir = self.outDir or model.outDir
        if self.saveAssessments:
            if self.iteration is not None:
                self.assessmentPath = f"{self.optimizationItPath}/{self.assessmentType}"
            else:
                self.assessmentPath = f"{self.outDir}/{self.assessmentType}"
            os.makedirs(self.assessmentPath, exist_ok=True)

    def onAssessmentEnd(self, predictions: pd.DataFrame):
        """Called after the assessment has finished.

        Args:
            predictions (pd.DataFrame): predictions of the assessment
        """
        super().onAssessmentEnd(predictions)
        if self.saveAssessments:
            predictions.to_csv(
                f"{self.assessmentPath}/{self.assessmentType}_predictions.tsv", sep="\t"
            )

    def onFitStart(
            self,
            model: QSPRModel,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
    ):
        """Called before the training has started.

        Args:
            model (QSPRModel): model to be fitted
            X_train (np.ndarray): training data
            y_train (np.ndarray): training targets
            X_val (np.ndarray | None): validation data, used for early stopping
            y_val (np.ndarray | None): validation targets, used for early stopping
        """
        super().onFitStart(model, X_train, y_train, X_val, y_val)
        self.outDir = self.outDir or model.outDir
        self.fitPath = self.outDir
        if self.saveFits:
            if self.iteration is not None:
                self.fitPath = f"{self.optimizationItPath}"
            if self.currentFold is not None:
                self.fitPath = (
                    f"{self.fitPath}/{self.assessmentType}/fold_{self.currentFold}"
                )
            os.makedirs(self.fitPath, exist_ok=True)

    def onFitEnd(self, estimator: Any, best_epoch: int | None = None):
        """Called after the training has finished.

        Args:
            estimator (Any): estimator that was fitted
            best_epoch (int | None): index of the best epoch
        """
        super().onFitEnd(estimator, best_epoch)
        if self.saveFits:
            self.fitLog.to_csv(f"{self.fitPath}/fit_log.tsv", sep="\t")
            self.batchLog.to_csv(f"{self.fitPath}/batch_log.tsv", sep="\t")


class WandBMonitor(BaseMonitor):
    """Monitor hyperparameter optimization to weights and biases."""

    def __init__(self, project_name: str, **kwargs):
        """Monitor assessment to weights and biases.

        Args:
            project_name (str): name of the project to log to
            kwargs: additional keyword arguments for wandb.init
        """
        super().__init__()
        try:
            import wandb
        except ImportError:
            raise ImportError("WandBMonitor requires wandb to be installed.")
        self.wandb = wandb

        wandb.login()

        self.projectName = project_name
        self.kwargs = kwargs

    def onFoldStart(
            self,
            fold: int,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
    ):
        """Called before each fold of the assessment.

        Args:
            fold (int): index of the current fold
            X_train (np.ndarray): training data of the current fold
            y_train (np.ndarray): training targets of the current fold
            X_test (np.ndarray): test data of the current fold
            y_test (np.ndarray): test targets of the current fold
        """
        super().onFoldStart(fold, X_train, y_train, X_test, y_test)
        config = {
            "fold": fold,
            "model": self.assessmentModel.name,
            "assessmentType": self.assessmentType,
        }
        # add hyperparameter optimization parameters if available
        if hasattr(self, "optimizationType"):
            config["optimizationType"] = self.optimizationType
            config.update(self.parameters[self.iteration])
            config["hyperParamOpt_iteration"] = self.iteration
        else:
            config["optimizationType"] = None

        group = (
            f"{self.model.name}_{self.optimizationType}_{self.iteration}"
            if hasattr(self, "optimizationType")
            else f"{self.assessmentModel.name}"
        )
        name = f"{group}_{self.assessmentType}_{fold}"

        self.wandb.init(
            project=self.projectName,
            config=config,
            name=name,
            group=group,
            dir=f"{self.assessmentModel.outDir}",
            **self.kwargs,
        )

    def onFoldEnd(
            self, model_fit: Any | tuple[Any, int], fold_predictions: pd.DataFrame
    ):
        """Called after each fold of the assessment.

        Args:
            model_fit (Any |tuple[Any, int]):
                fitted estimator of the current fold
        """
        super().onFoldEnd(model_fit, fold_predictions)

        fold_predictions_copy = deepcopy(fold_predictions)

        # add smiles to fold predictions by merging on index
        dataset_smiles = self.assessmentDataset.getDF()[
            self.assessmentDataset.smilesCol
        ]
        fold_predictions_copy = fold_predictions_copy.merge(
            dataset_smiles, left_index=True, right_index=True
        )

        fold_predictions_copy["molecule"] = None
        for index, row in fold_predictions_copy.iterrows():
            mol = Chem.MolFromSmiles(row[self.assessmentDataset.smilesCol])
            if mol is not None:
                fold_predictions_copy.at[index, "molecule"] = self.wandb.Image(
                    Draw.MolToImage(mol, size=(200, 200))
                )

        wandbTable = self.wandb.Table(data=fold_predictions_copy)

        self.wandb.log({"Test Results": wandbTable})
        self.wandb.finish()

    def onFitStart(
            self,
            model: QSPRModel,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
    ):
        """Called before the training has started.

        Args:
            model (QSPRModel): model to train
        """
        super().onFitStart(model, X_train, y_train, X_val, y_val)
        # initialize wandb run if not already initialized
        if not self.wandb.run:
            self.wandb.init(
                project=self.projectName,
                config={"model": self.fitModel.name},
                name=f"{self.fitModel.name}_fit",
                group=self.fitModel.name,
                dir=f"{self.fitModel.outDir}",
                **self.kwargs,
            )

    def onFitEnd(self, estimator: Any, best_epoch: int | None = None):
        """Called after the training has finished.

        Args:
            estimator (Any): estimator that was fitted
            best_epoch (int | None): index of the best epoch
        """
        super().onFitEnd(estimator, best_epoch)
        self.wandb.log({"best_epoch": best_epoch})
        # finish wandb run if not already finished
        if not hasattr(self, "assessmentType"):
            self.wandb.finish()

    def onEpochEnd(self, epoch: int, train_loss: float, val_loss: float | None = None):
        """Called after each epoch of the training.

        Args:
            epoch (int): index of the current epoch
            train_loss (float): loss of the current epoch
            val_loss (float | None): validation loss of the current epoch
        """
        super().onEpochEnd(epoch, train_loss, val_loss)
        self.wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    def onBatchEnd(self, batch: int, loss: float):
        """Called after each batch of the training.

        Args:
            batch (int): index of the current batch
            loss (float): loss of the current batch
        """
        super().onBatchEnd(batch, loss)
        self.wandb.log({"batch": batch, "loss": loss})
