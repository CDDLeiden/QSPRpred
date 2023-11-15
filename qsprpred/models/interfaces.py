"""This module holds the base class for QSPRmodels, model types should be a subclass."""

import copy
import inspect
import json
import os
import shutil
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Iterable, List, Type, Union

import numpy as np
import pandas as pd

from ..data.data import MoleculeTable, QSPRDataset
from ..data.utils.feature_standardization import SKLearnStandardizer
from ..logs import logger
from ..models import SSPACE
from ..models.early_stopping import EarlyStopping, EarlyStoppingMode
from ..models.metrics import SklearnMetric
from ..models.tasks import ModelTasks
from ..serialization import JSONSerializable
from ..utils.inspect import import_class


class QSPRModel(JSONSerializable, ABC):
    """The definition of the common model interface for the package.

    The QSPRModel handles model initialization, fitting, predicting and saving.

    Attributes:
        name (str): name of the model
        data (QSPRDataset): data set used to train the model
        alg (Type): estimator class
        parameters (dict): dictionary of algorithm specific parameters
        estimator (Any):
            the underlying estimator instance of the type specified in `QSPRModel.alg`,
            if `QSPRModel.fit` or optimization was performed
        featureCalculators (MoleculeDescriptorsCalculator):
            feature calculator instance taken from the data set or
            deserialized from file if the model is loaded without data
        featureStandardizer (SKLearnStandardizer):
            feature standardizer instance taken from the data set
            or deserialized from file if the model is loaded without data
        baseDir (str):
            base directory of the model,
            the model files are stored in a subdirectory `{baseDir}/{outDir}/`
        earlyStopping (EarlyStopping):
            early stopping tracker for training of QSPRpred models that support
            early stopping (e.g. neural networks)
        randomState (int):
            Random state to use for all random operations for reproducibility.
    """

    _notJSON = ["alg", "data", "estimator"]

    @staticmethod
    def handleInvalidsInPredictions(
        mols: list[str],
        predictions: np.ndarray | list[np.ndarray],
        failed_mask: np.ndarray,
    ) -> np.ndarray:
        """Replace invalid predictions with None.

        Args:
            mols (MoleculeTable): molecules for which the predictions were made
            predictions (np.ndarray): predictions made by the model
            failed_mask (np.ndarray): boolean mask of failed predictions

        Returns:
            np.ndarray: predictions with invalids replaced by None
        """
        if any(failed_mask):
            if isinstance(predictions, list):
                predictions_with_invalids = [
                    np.full((len(mols), pred.shape[1]), None) for pred in predictions
                ]
                for i, pred in enumerate(predictions):
                    predictions_with_invalids[i][~failed_mask, :] = pred
            else:
                predictions_with_invalids = np.full(
                    (len(mols), predictions.shape[1]), None
                )
                predictions_with_invalids[~failed_mask, :] = predictions
            predictions = predictions_with_invalids
        return predictions

    @classmethod
    def getDefaultParamsGrid(cls) -> list:
        """Get the path to the file with default search grid parameter settings
        for some predefined estimator types.

        Returns:
            list: list of default parameter grids
        """
        return SSPACE

    @classmethod
    def loadParamsGrid(
        cls, fname: str, optim_type: str, model_types: str
    ) -> np.ndarray:
        """Load parameter grids for bayes or grid search parameter
        optimization from json file.

        Arguments:
            fname (str):
                file name of json file containing array with three columns
                containing modeltype, optimization type (grid or bayes) and model type
            optim_type (str): optimization type (`grid` or `bayes`)
            model_types (list of str):
                model type for hyperparameter optimization (e.g. RF)

        Returns:
            np.ndarray:
                array with three columns containing modeltype,
                optimization type (grid or bayes) and model type
        """
        if fname:
            try:
                with open(fname) as json_file:
                    optim_params = np.array(json.load(json_file), dtype=object)
            except FileNotFoundError:
                logger.error("Search space file (%s) not found" % fname)
                sys.exit()
        else:
            with open(cls.getDefaultParamsGrid()) as json_file:
                optim_params = np.array(json.load(json_file), dtype=object)
        # select either grid or bayes optimization parameters from param array
        optim_params = optim_params[optim_params[:, 2] == optim_type, :]
        # check all ModelTasks to be used have parameter grid
        model_types = [model_types] if isinstance(model_types, str) else model_types
        if not set(model_types).issubset(list(optim_params[:, 0])):
            logger.error(
                "model types %s missing from models in search space dict (%s)" %
                (model_types, optim_params[:, 0])
            )
            sys.exit()
        logger.info("search space loaded from file")
        return optim_params

    def __init__(
        self,
        base_dir: str,
        alg: Type | None = None,
        data: QSPRDataset | None = None,
        name: str | None = None,
        parameters: dict | None = None,
        autoload=True,
        random_state: int | None = None,
    ):
        """Initialize a QSPR model instance.

        If the model is loaded from file, the data set is not required.
        Note that the data set is required for fitting and optimization.

        Args:
            base_dir (str):
                base directory of the model,
                the model files are stored in a subdirectory `{baseDir}/{outDir}/`
            alg (Type): estimator class
            data (QSPRDataset): data set used to train the model
            name (str): name of the model
            parameters (dict): dictionary of algorithm specific parameters
            autoload (bool):
                if `True`, the estimator is loaded from the serialized file
                if it exists, otherwise a new instance of alg is created
            random_state (int):
                Random state to use for shuffling and other random operations.
        """
        self.name = name or (alg.__class__.__name__ if alg else None)
        if self.name is None:
            raise ValueError("Model name not specified.")
        self.baseDir = os.path.abspath(base_dir.rstrip("/"))
        # initialize settings from data
        self.data = None
        self.targetProperties = None
        self.nTargets = None
        self.featureCalculators = None
        self.featureStandardizer = None
        # initialize estimator
        self.earlyStopping = EarlyStopping() if self.supportsEarlyStopping else None
        if autoload and os.path.exists(self.metaFile):
            new = self.fromFile(self.metaFile)
            self.__dict__.update(new.__dict__)
            self.name = name or (alg.__class__.__name__ if alg else None)
            self.baseDir = os.path.abspath(base_dir.rstrip("/"))
            if data is not None:
                self.initFromData(data)
            if parameters:
                logger.warning(
                    f"Explicitly specified parameters ({parameters})"
                    f"will override model settings read from file: {self.parameters}."
                    f"Estimator will be reloaded with the new parameters "
                    f"and will have to be re-fitted if fitted previously."
                )
                self.parameters = parameters
                self.estimator = self.loadEstimator(self.parameters)
            if random_state:
                logger.warning(
                    f"Explicitly specified random state ({random_state})"
                    f"will override model settings read from file: {self.randomState}."
                )
                self.initRandomState(random_state)
        else:
            self.initFromData(data)
            self.parameters = parameters
            # initialize an estimator instance with the given parameters
            self.alg = alg
            # initialize random state
            self.randomState = None
            self.initRandomState(random_state)
            # load the estimator
            self.estimator = self.loadEstimator(self.parameters)
        assert self.estimator is not None, \
            "Estimator not initialized when it should be."
        assert self.alg is not None, \
            "Algorithm class not initialized when it should be."
        if not autoload and not self.checkForData(exception=False):
            raise ValueError(
                "No data set specified. Make sure you "
                "initialized this model with a 'QSPRDataset' "
                "instance to train on. Or if you want to load "
                "a model from file, set 'autoload' to True."
            )

    def __str__(self) -> str:
        """Return the name of the model and the underlying class as the identifier."""
        if self.estimator is not None:
            name = self.estimator.__class__.__name__
        elif self.alg is not None:
            name = self.alg.__name__
        else:
            name = "None"
        return f"{self.name} ({name})"

    def __setstate__(self, state):
        """Set state."""
        super().__setstate__(state)
        self.data = None
        self.alg = import_class(self.alg)
        self.estimator = self.loadEstimator(self.parameters)

    def initFromData(self, data: QSPRDataset | None):
        if data is not None:
            self.data = data
            self.targetProperties = self.data.targetProperties
            self.nTargets = len(self.targetProperties)
            self.featureCalculators = self.data.descriptorCalculators
            self.featureStandardizer = self.data.feature_standardizer
        else:
            self.data = None
            self.targetProperties = None
            self.nTargets = None
            self.featureCalculators = None
            self.featureStandardizer = None

    def initRandomState(self, random_state):
        """Set random state if applicable.
        Defaults to random state of dataset if no random state is provided,

        Args:
            random_state (int):
                Random state to use for shuffling and other random operations.
        """
        new_random_state = random_state or (
            self.data.randomState if self.data is not None else None
        )
        if new_random_state is None:
            self.randomState = int(np.random.randint(0, 2**32 - 1, dtype=np.int64))
            logger.warning(
                "No random state supplied, "
                "and could not find random state on the dataset."
            )
        self.randomState = new_random_state
        constructor_params = [
            name for name, _ in inspect.signature(self.alg.__init__).parameters.items()
        ]
        if "random_state" in constructor_params:
            if self.parameters:
                self.parameters.update({"random_state": new_random_state})
            else:
                self.parameters = {"random_state": new_random_state}
        elif random_state:
            logger.warning(
                f"Random state supplied, but alg {self.alg} does not support it."
                " Ignoring this setting."
            )

    @property
    def task(self) -> ModelTasks:
        """Return the task of the model, taken from the data set
        or deserialized from file if the model is loaded without data.

        Returns:
            ModelTasks: task of the model
        """
        return ModelTasks.getModelTask(self.targetProperties)

    @property
    def isMultiTask(self) -> bool:
        """Return if model is a multitask model, taken from the data set
        or deserialized from file if the model is loaded without data.

        Returns:
            bool: True if model is a multitask model
        """
        return self.task.isMultiTask()

    @property
    def classPath(self) -> str:
        """Return the fully classified path of the model.

        Returns:
            str: class path of the model
        """
        return self.__class__.__module__ + "." + self.__class__.__name__

    @property
    def outDir(self) -> str:
        """Return output directory of the model,
        the model files are stored in this directory (`{baseDir}/{name}`).

        Returns:
            str: output directory of the model
        """
        os.makedirs(f"{self.baseDir}/{self.name}", exist_ok=True)
        return f"{self.baseDir}/{self.name}"

    @property
    def outPrefix(self) -> str:
        """Return output prefix of the model files.

        The model files are stored with this prefix (i.e. `{outPrefix}_meta.json`).

        Returns:
            str: output prefix of the model files
        """
        return f"{self.outDir}/{self.name}"

    @property
    def metaFile(self) -> str:
        return f"{self.outPrefix}_meta.json"

    @property
    @abstractmethod
    def supportsEarlyStopping(self) -> bool:
        """Return if the model supports early stopping.

        Returns:
            bool: True if the model supports early stopping
        """

    @property
    def optimalEpochs(self) -> int | None:
        """Return the optimal number of epochs for early stopping.

        Returns:
            int | None: optimal number of epochs
        """
        return self._optimalEpochs

    @optimalEpochs.setter
    def optimalEpochs(self, value: int | None = None):
        """Set the optimal number of epochs for early stopping.

        Args:
            value (int | None, optional): optimal number of epochs
        """
        self._optimalEpochs = value

    def setParams(self, params: dict):
        """Set model parameters.

        Args:
            params (dict): dictionary of model parameters
        """
        if self.parameters is not None:
            self.parameters.update(params)
        else:
            self.parameters = params

    def checkForData(self, exception: bool = True) -> bool:
        """Check if the model has a data set.

        Args:
            exception (bool): if true, an exception is raised if no data is set

        Returns:
            bool: True if data is set, False otherwise (if exception is False)
        """
        has_data = self.data is not None
        if exception and not has_data:
            raise ValueError(
                "No data set specified. "
                "Make sure you initialized this model "
                "with a 'QSPRDataset' instance to train on."
            )
        return has_data

    def convertToNumpy(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        y: pd.DataFrame | np.ndarray | QSPRDataset | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Convert the given data matrix and target matrix to np.ndarray format.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix
            y (pd.DataFrame, np.ndarray, QSPRDataset): target matrix

        Returns:
                data matrix and/or target matrix in np.ndarray format
        """
        if isinstance(X, QSPRDataset):
            X = X.getFeatures(raw=True, concat=True)
        if isinstance(X, pd.DataFrame):
            X = X.values
        if y is not None:
            if isinstance(y, QSPRDataset):
                y = y.getTargetPropertiesValues(concat=True)
            if isinstance(y, pd.DataFrame):
                y = y.values
            return X, y
        else:
            return X

    def getParameters(self, new_parameters) -> dict | None:
        """Get the model parameters combined with the given parameters.

        If both the model and the given parameters contain the same key,
        the value from the given parameters is used.

        Args:
            new_parameters (dict): dictionary of new parameters to add

        Returns:
            dict: dictionary of model parameters
        """
        parameters_out = copy.deepcopy(self.parameters)
        if parameters_out is not None:
            parameters_out.update(new_parameters)
        else:
            parameters_out = new_parameters
        return parameters_out

    def createPredictionDatasetFromMols(
        self,
        mols: list[str],
        smiles_standardizer: str | Callable[[str], str] = "chembl",
        n_jobs: int = 1,
        fill_value: float = np.nan,
    ) -> tuple[QSPRDataset, np.ndarray]:
        """Create a `QSPRDataset` instance from a list of SMILES strings.

        Args:
            mols (list): list of SMILES strings
            smiles_standardizer (str, callable): smiles standardizer to use
            n_jobs (int): number of parallel jobs to use
            fill_value (float): value to fill for missing features

        Returns:
            tuple:
                a tuple containing the `QSPRDataset` instance and a boolean mask
                indicating which molecules failed to be processed
        """
        # make a molecule table first and add the target properties
        dataset = MoleculeTable.fromSMILES(
            f"{self.__class__.__name__}_{hash(self)}",
            mols,
            drop_invalids=False,
            n_jobs=n_jobs,
        )
        for targetproperty in self.targetProperties:
            dataset.addProperty(targetproperty.name, np.nan)
        # create the dataset and get failed molecules
        dataset = QSPRDataset.fromMolTable(
            dataset,
            self.targetProperties,
            drop_empty=False,
            drop_invalids=False,
            n_jobs=n_jobs,
        )
        dataset.standardizeSmiles(smiles_standardizer, drop_invalid=False)
        failed_mask = dataset.dropInvalids().values
        # prepare dataset and return it
        dataset.prepareDataset(
            smiles_standardizer=smiles_standardizer,
            feature_calculators=self.featureCalculators,
            feature_standardizer=self.featureStandardizer,
            feature_fill_value=fill_value,
            shuffle=False,
        )
        return dataset, failed_mask

    def predictDataset(self,
                       dataset: QSPRDataset,
                       use_probas: bool = False) -> np.ndarray | list[np.ndarray]:
        """
        Make predictions for the given dataset.

        Args:
            dataset: a `QSPRDataset` instance
            use_probas: use probabilities if this is a classification model

        Returns:
            np.ndarray | list[np.ndarray]:
                an array of predictions or a list of arrays of predictions
                (for classification models with use_probas=True)
        """
        if self.task.isRegression() or not use_probas:
            predictions = self.predict(dataset)
            # always return 2D array
            if self.task.isClassification():
                predictions = predictions.astype(int)
        else:
            # return a list of 2D arrays
            predictions = self.predictProba(dataset)
        return predictions

    def predictMols(
        self,
        mols: List[str],
        use_probas: bool = False,
        smiles_standardizer: Union[str, callable] = "chembl",
        n_jobs: int = 1,
        fill_value: float = np.nan,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Make predictions for the given molecules.

        Args:
            mols (List[str]): list of SMILES strings
            use_probas (bool): use probabilities for classification models
            smiles_standardizer:
                either `chembl`, `old`, or a partial function
                that reads and standardizes smiles.
            n_jobs: Number of jobs to use for parallel processing.
            fill_value: Value to use for missing values in the feature matrix.

        Returns:
            np.ndarray | list[np.ndarray]:
                an array of predictions or a list of arrays of predictions
                (for classification models with use_probas=True)
        """
        if not self.featureCalculators:
            raise ValueError("No feature calculator set on this instance.")
        # create data set from mols
        dataset, failed_mask = self.createPredictionDatasetFromMols(
            mols, smiles_standardizer, n_jobs, fill_value
        )
        # make predictions for the dataset
        predictions = self.predictDataset(dataset, use_probas)
        # handle invalids
        predictions = self.handleInvalidsInPredictions(mols, predictions, failed_mask)
        return predictions

    def cleanFiles(self):
        """Clean up the model files.

        Removes the model directory and all its contents.
        """
        if os.path.exists(self.outDir):
            shutil.rmtree(self.outDir)

    def fitAttached(self, monitor=None, mode=EarlyStoppingMode.OPTIMAL, **kwargs) -> str:
        """Train model on the whole attached data set.

        ** IMPORTANT ** For models that supportEarlyStopping, `CrossValAssessor`
        should be run first, so that the average number of epochs from the
        cross-validation with early stopping can be used for fitting the model.

        Args:
            monitor (FitMonitor): monitor for the fitting process, if None, the base
                monitor is used
            mode (EarlyStoppingMode): early stopping mode for models that support
                early stopping, by default fit the 'optimal' number of
                epochs previously stopped at in model assessment on train or test set,
                to avoid the use of extra data for a validation set.
            kwargs: additional arguments to pass to fit

        Returns:
            str: path to the saved model
        """
        # do some checks
        self.checkForData()
        # get data
        X_all = self.data.getFeatures(concat=True).values
        y_all = self.data.getTargetPropertiesValues(concat=True).values
        # load estimator
        self.estimator = self.loadEstimator(self.parameters)
        # fit model
        logger.info(
            "Model fit started: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        self.estimator = self.fit(
            X_all, y_all, mode=mode, monitor=monitor, **kwargs
        )
        logger.info(
            "Model fit ended: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        # save model and return path
        return self.save()

    def toJSON(self):
        o_dict = json.loads(super().toJSON())
        estimator_path = self.saveEstimator()
        estimator_path = estimator_path.replace(self.baseDir, ".")
        o_dict["py/state"]["baseDir"] = "."
        o_dict["py/state"]["estimator"] = estimator_path
        o_dict["py/state"]["alg"] = f"{self.alg.__module__}.{self.alg.__name__}"
        return json.dumps(o_dict, indent=4)

    def save(self):
        """Save model to file.

        Returns:
            str: absolute path to the saved model
        """
        os.makedirs(self.outDir, exist_ok=True)
        return self.toFile(self.metaFile)

    @classmethod
    def fromFile(cls, filename: str):
        ret = super().fromFile(filename)
        model_dir = os.path.dirname(filename)
        ret.baseDir = os.path.dirname(model_dir)
        ret.estimator = ret.loadEstimatorFromFile(ret.parameters)
        return ret

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        y: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any = None,
        mode: EarlyStoppingMode = EarlyStoppingMode.NOT_RECORDING,
        monitor: "FitMonitor" = None,
        **kwargs,
    ) -> Any | tuple[Any, int] | None:
        """Fit the model to the given data matrix or `QSPRDataset`.

        Note. convertToNumpy can be called here, to convert the input data to
            np.ndarray format.

        Note. if no estimator is given, the estimator instance of the model is used.

        Note. if a model supports early stopping, the fit function should have the
            `early_stopping` decorator and the mode argument should be used to set the
            early stopping mode. If the model does not support early stopping, the mode
            argument is ignored.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to fit
            y (pd.DataFrame, np.ndarray, QSPRDataset): target matrix to fit
            estimator (Any): estimator instance to use for fitting
            mode (EarlyStoppingMode): early stopping mode
            monitor (FitMonitor): monitor for the fitting process,
                if None, the base monitor is used
            kwargs: additional arguments to pass to the fit method of the estimator

        Returns:
            Any: fitted estimator instance
            int: in case of early stopping, the number of iterations
                after which the model stopped training
        """

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any = None
    ) -> np.ndarray:
        """Make predictions for the given data matrix or `QSPRDataset`.

        Note. convertToNumpy can be called here, to convert the input data to
        np.ndarray format.

        Note. if no estimator is given, the estimator instance of the model
              is used.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to predict
            estimator (Any): estimator instance to use for fitting

        Returns:
            np.ndarray:
                2D array containing the predictions, where each row corresponds
                to a sample in the data and each column to a target property
        """

    @abstractmethod
    def predictProba(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any = None
    ) -> list[np.ndarray]:
        """Make predictions for the given data matrix or `QSPRDataset`,
        but use probabilities for classification models. Does not work with
        regression models.

        Note. convertToNumpy can be called here, to convert the input data to
        np.ndarray format.

        Note. if no estimator is given, the estimator instance of the model
              is used.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to make predict
            estimator (Any): estimator instance to use for fitting

        Returns:
            list[np.ndarray]:
                a list of 2D arrays containing the probabilities for each class,
                where each array corresponds to a target property, each row
                to a sample in the data and each column to a class
        """

    @abstractmethod
    def loadEstimator(self, params: dict | None = None) -> object:
        """Initialize estimator instance with the given parameters.

        If `params` is `None`, the default parameters will be used.

        Arguments:
            params (dict): algorithm parameters

        Returns:
            object: initialized estimator instance
        """

    @abstractmethod
    def loadEstimatorFromFile(self, params: dict | None = None) -> object:
        """Load estimator instance from file and apply the given parameters.

        Args:
            params (dict): algorithm parameters

        Returns:
            object: initialized estimator instance
        """

    @abstractmethod
    def saveEstimator(self) -> str:
        """Save the underlying estimator to file.

        Returns:
            path (str): absolute path to the saved estimator
        """


class FitMonitor(ABC):
    """Base class for monitoring the fitting of a model."""
    @abstractmethod
    def onFitStart(self, model: QSPRModel):
        """Called before the training has started.

        Args:
            model (QSPRModel): model to be fitted
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
    def onEpochEnd(
        self, epoch: int, train_loss: float, val_loss: float | None = None
    ):
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
    def onAssessmentStart(self, model: QSPRModel, assesment_type: str):
        """Called before the assessment has started.

        Args:
            model (QSPRModel): model to assess
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
        X_train: np.array,
        y_train: np.array,
        X_test: np.array,
        y_test: np.array,
    ):
        """Called before each fold of the assessment.

        Args:
            fold (int): index of the current fold
            X_train (np.array): training data of the current fold
            y_train (np.array): training targets of the current fold
            X_test (np.array): test data of the current fold
            y_test (np.array): test targets of the current fold
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
            predictions (pd.DataFrame): predictions of the current fold
        """


class HyperparameterOptimizationMonitor(AssessorMonitor):
    """Base class for monitoring the hyperparameter optimization of a model."""
    @abstractmethod
    def onOptimizationStart(
        self, model: QSPRModel, config: dict, optimization_type: str
    ):
        """Called before the hyperparameter optimization has started.

        Args:
            model (QSPRModel): model to optimize
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
    """
    def __init__(
        self,
        scoring: str | Callable[[Iterable, Iterable], float],
        monitor: AssessorMonitor | None = None,
        use_proba: bool = True,
        mode: EarlyStoppingMode | None = None,
    ):
        """Initialize the evaluation method class.

        Args:
            scoring: str | Callable[[Iterable, Iterable], float],
            monitor (AssessorMonitor): monitor to track the evaluation
            use_proba (bool): use probabilities for classification models
            mode (EarlyStoppingMode): early stopping mode for fitting
        """
        self.monitor = monitor
        self.useProba = use_proba
        self.mode = mode
        self.scoreFunc = (
            SklearnMetric.getMetric(scoring) if isinstance(scoring, str) else scoring
        )

    @abstractmethod
    def __call__(
        self,
        model: QSPRModel,
        save: bool,
        parameters: dict | None,
        monitor: AssessorMonitor,
        **kwargs,
    ) -> list[float]:
        """Evaluate the model.

        Args:
            model (QSPRModel): model to evaluate
            save (bool): save predictions to file
            parameters (dict): parameters to use for the evaluation
            monitor (AssessorMonitor): monitor to track the evaluation, overrides
                                       the monitor set in the constructor
            kwargs: additional arguments for fit function of the model

        Returns:
            list[float]: scores of the model for each fold
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
        for idx, prop in enumerate(model.data.targetProperties):
            if prop.task.isClassification() and self.useProba:
                # convert one-hot encoded predictions to class labels
                # and add to train and test
                df_out[f"{prop.name}_Prediction"] = np.argmax(predictions[idx], axis=1)
                # add probability columns to train and test set
                df_out = pd.concat(
                    [
                        df_out,
                        pd.DataFrame(predictions[idx], index=index
                                    ).add_prefix(f"{prop.name}_ProbabilityClass_"),
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


class HyperparameterOptimization(ABC):
    """Base class for hyperparameter optimization.

    Attributes:
        runAssessment (ModelAssessor): evaluation method to use
        scoreAggregation (Callable[[Iterable], float]): function to aggregate scores
        paramGrid (dict): dictionary of parameters to optimize
        monitor (HyperparameterOptimizationMonitor): monitor to track the optimization
        bestScore (float): best score found during optimization
        bestParams (dict): best parameters found during optimization
    """
    def __init__(
        self,
        param_grid: dict,
        model_assessor: ModelAssessor,
        score_aggregation: Callable[[Iterable], float],
        monitor: HyperparameterOptimizationMonitor | None = None,
    ):
        """Initialize the hyperparameter optimization class.

        param_grid (dict):
            dictionary of parameters to optimize
        model_assessor (ModelAssessor):
            assessment method to use for determining the best parameters
        score_aggregation (Callable[[Iterable], float]): function to aggregate scores
        monitor (HyperparameterOptimizationMonitor): monitor to track the optimization,
            if None, a BaseMonitor is used
        """
        self.runAssessment = model_assessor
        self.scoreAggregation = score_aggregation
        self.paramGrid = param_grid
        self.bestScore = -np.inf
        self.bestParams = None
        self.monitor = monitor
        self.config = {
            "param_grid": param_grid,
            "model_assessor": model_assessor,
            "score_aggregation": score_aggregation
        }

    @abstractmethod
    def optimize(self, model: QSPRModel) -> dict:
        """Optimize the model hyperparameters.

        Args:
            model (QSPRModel): model to optimize

        Returns:
            dict: dictionary of best parameters
        """
