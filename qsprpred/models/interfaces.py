"""This module holds the base class for QSPRmodels, model types should be a subclass."""

import copy
import json
import os
import shutil
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Iterable, List, Type, Union

import numpy as np
import pandas as pd

from .. import VERSION
from ..data.data import MoleculeTable, QSPRDataset, TargetProperty
from ..data.utils.feature_standardization import SKLearnStandardizer
from ..logs import logger
from ..models import SSPACE
from ..models.metrics import SklearnMetric
from ..models.tasks import ModelTasks
from ..utils.inspect import import_class
from ..models.early_stopping import EarlyStopping, EarlyStoppingMode


class QSPRModel(ABC):
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
        metaInfo (dict):
            dictionary of metadata about the model,
            only available after the model is saved
        baseDir (str):
            base directory of the model,
            the model files are stored in a subdirectory `{baseDir}/{outDir}/`
        metaFile (str):
            absolute path to the metadata file of the model (`{outPrefix}_meta.json`)
    """
    @staticmethod
    def readStandardizer(path: str) -> "SKLearnStandardizer":
        """Read a feature standardizer from a JSON file.

        If the file does not exist, None is returned.

        Args:
            path (str): absolute path to the JSON file

        Returns:
            SKLearnStandardizer:
                feature standardizer instance or None if the file does not exist
        """
        if os.path.exists(path):
            return SKLearnStandardizer.fromFile(path)

    @staticmethod
    def readParams(path: str) -> dict:
        """Read model parameters from a JSON file.

        Args:
            path (str): absolute path to the JSON file

        Returns:
            dict: dictionary of model parameters
        """
        with open(path, "r", encoding="utf-8") as j:
            logger.info("loading model parameters from file: %s" % path)
            return json.loads(j.read())

    @staticmethod
    def readDescriptorCalculators(
        paths: list[str],
    ) -> list["DescriptorsCalculator"]:  # noqa: F821
        """Read a descriptor calculators from a JSON file.

        Args:
            paths (list[str]): absolute path to the JSON file

        Returns:
            list[DescriptorsCalculator]: descriptor calculator instance
            or None if the file does not exist
        """
        calcs = []
        for path in paths:
            if os.path.exists(path):
                data = json.load(open(path, "r", encoding="utf-8"))
                if "calculator" not in data:
                    calc_cls = (
                        "qsprpred.data.utils.descriptorcalculator."
                        "MoleculeDescriptorsCalculator"
                    )
                else:
                    calc_cls = data["calculator"]
                calc_cls = import_class(calc_cls)
                calc = calc_cls.fromFile(path)
                calcs.append(calc)
            else:
                raise FileNotFoundError(
                    f"Descriptor calculator file '{path}' not found."
                )
        return calcs

    @staticmethod
    def readEarlyStopping(path):
        """Read early stopping settings from file.

        If the file does not exist, None is returned.

        Args:
            path (str): absolute path to the JSON file

        Returns:
            EarlyStopping:
                Early stopping instance or None if the file does not exist
        """
        if os.path.exists(path):
            return EarlyStopping.fromFile(path)

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

    @classmethod
    def fromFile(cls, path: str) -> "QSPRModel":
        """Load a model from its meta file.

        Args:
            path (str): full path to the model meta file
        """
        meta = cls.readMetadata(path)
        model_class = import_class(meta["model_class"])
        model_name = meta["name"]
        base_dir = os.path.dirname(path).replace(f"{model_name}", "")
        return model_class(name=model_name, base_dir=base_dir)

    @classmethod
    def readMetadata(cls, path: str) -> dict:
        """Read model metadata from a JSON file.

        Args:
            path (str): absolute path to the JSON file

        Returns:
            dict: dictionary containing the model metadata

        Raises:
            FileNotFoundError: if the file does not exist
        """
        if os.path.exists(path):
            with open(path) as j:
                meta_info = json.loads(j.read())
                meta_info["target_properties"] = TargetProperty.fromList(
                    meta_info["target_properties"], task_from_str=True
                )
        else:
            raise FileNotFoundError(f"Metadata file '{path}' does not exist.")
        return meta_info

    def __init__(
        self,
        base_dir: str,
        alg: Type | None = None,
        data: QSPRDataset | None = None,
        name: str | None = None,
        parameters: dict | None = None,
        autoload=True
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
        """
        self.data = data
        self.name = name or alg.__class__.__name__
        self.baseDir = base_dir.rstrip("/")
        # load metadata and update parameters accordingly
        self.metaFile = f"{self.outPrefix}_meta.json"
        if autoload:
            try:
                self.metaInfo = self.readMetadata(self.metaFile)
            except FileNotFoundError:
                if not self.data:
                    raise FileNotFoundError(
                        f"Metadata file '{self.metaFile}' not found"
                    )
                self.metaInfo = {}
        else:
            self.metaInfo = {}
            if self.data is None:
                raise ValueError(
                    "No data set specified. Make sure you "
                    "initialized this model with a 'QSPRDataset' "
                    "instance to train on. Or if you want to load "
                    "a model from file, set 'autoload' to True."
                )
        # get parameters from metadata if available
        self.parameters = parameters
        if self.metaInfo and not self.parameters:
            self.parameters = self.readParams(
                os.path.join(self.baseDir, self.metaInfo["parameters_path"])
            )
        # initialize a feature calculator instance
        self.featureCalculators = (
            self.data.descriptorCalculators
            if self.data else self.readDescriptorCalculators(
                [
                    os.path.join(self.baseDir, path)
                    for path in self.metaInfo["feature_calculator_paths"]
                ]
            )
        )
        # initialize a standardizer instance
        self.featureStandardizer = (
            self.data.feature_standardizer if self.data else self.readStandardizer(
                os.path.join(self.baseDir, self.metaInfo["feature_standardizer_path"])
            ) if self.metaInfo["feature_standardizer_path"] else None
        )
        # Initialize early stopping tracker if model supports early stopping
        if self.supportsEarlyStopping:
            self.earlyStopping = (
                self.readEarlyStopping(
                    os.path.join(self.baseDir, self.metaInfo["early_stopping_path"])
                ) if "early_stopping_path" in self.metaInfo else EarlyStopping()
            )

        # initialize a estimator instance with the given parameters
        self.alg = alg
        if autoload:
            self.estimator = self.loadEstimatorFromFile(params=self.parameters)

    def __str__(self) -> str:
        """Return the name of the model and the underlying class as the identifier."""
        if self.estimator is not None:
            name = self.estimator.__class__.__name__
        elif self.alg is not None:
            name = self.alg.__name__
        else:
            name = "None"
        return f"{self.name} ({name})"

    @property
    def targetProperties(self) -> list[TargetProperty]:
        """Return the target properties of the model,
        taken from the data set or deserialized
        from file if the model is loaded without data.

        Returns:
            list[TargetProperty]: target properties of the model
        """
        return (
            self.data.targetProperties
            if self.data else self.metaInfo["target_properties"]
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
    def nTargets(self) -> int:
        """Return the number of target properties of the model, taken from the data set
        or deserialized from file if the model is loaded without data.

        Returns:
            int: number of target properties of the model
        """
        return (
            len(self.data.targetProperties)
            if self.data else len(self.metaInfo["target_properties"])
        )

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

    def saveParams(self, params: dict) -> str:
        """Save model parameters to a JSON file.

        Args:
            params (dict): dictionary of model parameters

        Returns:
            str: absolute path to the JSON file
        """
        path = f"{self.outDir}/{self.name}_params.json"
        self.setParams(params)
        # save parameters to file
        with open(path, "w", encoding="utf-8") as j:
            logger.info("saving model parameters to file: %s" % path)
            j.write(json.dumps(params, indent=4))
        return path

    def setParams(self, params):
        """Set model parameters.

        Args:
            params (dict): dictionary of model parameters
        """
        if self.parameters is not None:
            self.parameters.update(params)
        else:
            self.parameters = params

    def saveDescriptorCalculators(self) -> list[str]:
        """Save the current descriptor calculator to a JSON file.

        The file is stored in the output directory of the model.

        Returns:
            list[str]:
                absolute paths to the JSON files containing
                the saved descriptor calculators
        """
        paths = []
        for calc in self.featureCalculators:
            path = f"{self.outDir}/{self.name}_descriptor_calculator_{calc}.json"
            calc.toFile(path)
            paths.append(path)
        return paths

    def saveStandardizer(self) -> str:
        """Save the current feature standardizer to a JSON file.

        The file is stored in the output directory of the model.

        Returns:
            str:
                absolute path to the JSON file containing the saved feature standardizer
        """
        path = f"{self.outDir}/{self.name}_feature_standardizer.json"
        self.featureStandardizer.toFile(path)
        return path

    def saveEarlyStopping(self) -> str:
        path = f"{self.outDir}/{self.name}_early_stopping.json"
        self.earlyStopping.toFile(path)
        return path

    def saveMetadata(self) -> str:
        """Save model metadata to a JSON file.

        The file is stored in the output directory of the model.

        Returns:
            str: absolute path to the JSON file containing the model metadata
        """
        with open(self.metaFile, "w", encoding="utf-8") as j:
            logger.info("saving model metadata to file: %s" % self.metaFile)
            j.write(json.dumps(self.metaInfo, indent=4))
        return self.metaFile

    def save(self) -> str:
        """Save the model data, parameters, metadata and all other
         files to the output directory of the model.

        Returns:
            str: absolute path to the JSON file containing the model metadata
        """
        self.metaInfo["name"] = self.name
        self.metaInfo["version"] = VERSION
        self.metaInfo["model_class"] = self.classPath
        self.metaInfo["target_properties"] = TargetProperty.toList(
            copy.deepcopy(self.data.targetProperties), task_as_str=True
        )
        self.metaInfo["parameters_path"] = self.saveParams(
            self.parameters
        ).replace(f"{self.baseDir}/", "")
        self.metaInfo["feature_calculator_paths"] = (
            [
                x.replace(f"{self.baseDir}/", "")
                for x in self.saveDescriptorCalculators()
            ] if self.featureCalculators else None
        )
        self.metaInfo["feature_standardizer_path"] = (
            self.saveStandardizer().replace(f"{self.baseDir}/", "")
            if self.featureStandardizer else None
        )
        self.metaInfo["estimator_path"] = self.saveEstimator().replace(
            f"{self.baseDir}/", ""
        )
        if self.supportsEarlyStopping:
            self.metaInfo["early_stopping_path"] = self.saveEarlyStopping().replace(
                f"{self.baseDir}/", ""
            )
        return self.saveMetadata()

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
        y: pd.DataFrame | np.ndarray | QSPRDataset | None = None
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
            shuffle=False
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
            #return a list of 2D arrays
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

    def fitAttached(self, **kwargs) -> str:
        """Train model on the whole attached data set.

        ** IMPORTANT ** For models that supportEarlyStopping, `CrossValAssessor`
        should be run first, so that the average number of epochs from the
        cross-validation with early stopping can be used for fitting the model.

        Args:
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
        # Use early stopping false here, since we are fitting on the whole data set
        # the number of epochs is already determined from the cross-validation
        self.estimator = self.fit(
            X_all, y_all, mode=EarlyStoppingMode.OPTIMAL, **kwargs
        )
        logger.info(
            "Model fit ended: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        # save model and return path
        return self.save()

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | np.ndarray | QSPRDataset,
        y: pd.DataFrame | np.ndarray | QSPRDataset,
        estimator: Any = None,
        mode: EarlyStoppingMode = EarlyStoppingMode.NOT_RECORDING,
        **kwargs
    ) -> Any | tuple[Any, int] | None:
        """Fit the model to the given data matrix or `QSPRDataset`.

        Note. convertToNumpy can be called here, to convert the input data to
        np.ndarray format.

        Note. if no estimator is given, the estimator instance of the model
              is used.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to fit
            y (pd.DataFrame, np.ndarray, QSPRDataset): target matrix to fit
            estimator (Any): estimator instance to use for fitting
            mode (EarlyStoppingMode): early stopping mode

        Returns:
            Any: fitted estimator instance
            int]: in case of early stopping, the number of iterations
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
            path (str): path to the saved estimator
        """


class ModelAssessor(ABC):
    """Base class for assessment methods.

    Attributes:
        useProba (bool): use probabilities for classification models
    """
    def __init__(self, use_proba: bool = True):
        """Initialize the evaluation method class.

        Args:
            use_proba (bool): use probabilities for classification models
        """
        self.useProba = use_proba

    @abstractmethod
    def __call__(
        self, model: QSPRModel
    ) -> list[tuple[np.ndarray, np.ndarray | list[np.ndarray]]]:
        """Evaluate the model.

        Args:
            model (QSPRModel): model to evaluate

        Returns:
            list[tuple[np.ndarray, np.ndarray | list[np.ndarray]]:
                list of tuples containing the true values and the predictions, where
                each tuple corresponds a set of predictions such as different folds.
        """

    def savePredictionsToFile(
        self,
        model: QSPRModel,
        y: np.array,
        predictions: np.ndarray | list[np.ndarray],
        index: pd.Series,
        file_suffix: str,
        extra_columns: dict[str, np.ndarray] | None = None
    ):
        """Save predictions to file.

        Args:
            model (QSPRModel): model to evaluate
            y (np.array): target values
            predictions (np.ndarray | list[np.ndarray]): predictions
            index (pd.Series): index of the data set
            file_suffix (str): suffix to add to the file name
            extra_columns (dict[str, np.ndarray]): extra columns to add to the output
        """
        # Create dataframe with true values
        df_out = pd.DataFrame(
            y.values, columns=y.add_suffix("_Label").columns, index=index
        )
        # Add predictions to dataframe
        for idx, prop in enumerate(model.data.targetProperties):
            if prop.task.isClassification():
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
        df_out.to_csv(f"{model.outPrefix}.{file_suffix}.tsv", sep="\t")


class HyperParameterOptimization(ABC):
    """Base class for hyperparameter optimization.

    Attributes:
        scoreFunc (Metric): scoring function to use, should match the output of the
                            evaluation method (e.g. if the evaluation methods returns
                            class probabilities, the scoring function support class
                            probabilities)
        runAssessment (ModelAssessor): evaluation method to use
        scoreAggregation (Callable[[Iterable], float]): function to aggregate scores
        paramGrid (dict): dictionary of parameters to optimize
        bestScore (float): best score found during optimization
        bestParams (dict): best parameters found during optimization
    """
    def __init__(
        self, scoring: str | Callable[[Iterable, Iterable], float], param_grid: dict,
        model_assessor: ModelAssessor, score_aggregation: Callable[[Iterable], float]
    ):
        """Initialize the hyperparameter optimization class.

        scoring (str | Callable[[Iterable, Iterable], float]):
            Metric name from `sklearn.metrics` or user-defined scoring function.
        param_grid (dict):
            dictionary of parameters to optimize
        model_assessor (ModelAssessor):
            assessment method to use for determining the best parameters
        score_aggregation (Callable[[Iterable], float]): function to aggregate scores
        """
        self.scoreFunc = (
            SklearnMetric.getMetric(scoring) if type(scoring) == str else scoring
        )
        self.runAssessment = model_assessor
        self.scoreAggregation = score_aggregation
        self.paramGrid = param_grid
        self.bestScore = -np.inf
        self.bestParams = None

    @abstractmethod
    def optimize(self, model: QSPRModel) -> dict:
        """Optimize the model hyperparameters.

        Args:
            model (QSPRModel): model to optimize

        Returns:
            dict: dictionary of best parameters
        """
