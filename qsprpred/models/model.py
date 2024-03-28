"""This module holds the base class for QSPRmodels, model types should be a subclass."""

import copy
import inspect
import json
import os
import shutil
import sys
import typing
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, List, Type, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol

from ..data.tables.mol import MoleculeTable
from ..data.tables.qspr import QSPRDataset
from ..logs import logger
from ..models.early_stopping import EarlyStopping, EarlyStoppingMode
from ..tasks import ModelTasks
from ..utils.inspect import dynamic_import
from ..utils.serialization import JSONSerializable


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

    _notJSON: typing.ClassVar = ["estimator", *JSONSerializable._notJSON]

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
        try:
            with open(fname) as json_file:
                optim_params = np.array(json.load(json_file), dtype=object)
        except FileNotFoundError:
            logger.error("Search space file (%s) not found" % fname)
            sys.exit()
        # select either grid or bayes optimization parameters from param array
        optim_params = optim_params[optim_params[:, 2] == optim_type, :]
        # check all ModelTasks to be used have parameter grid
        model_types = [model_types] if isinstance(model_types, str) else model_types
        if not set(model_types).issubset(list(optim_params[:, 0])):
            logger.error(
                "model types %s missing from models in search space dict (%s)"
                % (model_types, optim_params[:, 0])
            )
            sys.exit()
        logger.info("search space loaded from file")
        return optim_params

    def __init__(
        self,
        base_dir: str,
        alg: Type | None = None,
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
            alg (Type):
                estimator class
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
            self.parameters = parameters
            # initialize an estimator instance with the given parameters
            self.alg = alg
            # initialize random state
            self.randomState = None
            self.initRandomState(random_state)
            # load the estimator
            self.estimator = self.loadEstimator(self.parameters)
        assert (
            self.estimator is not None
        ), "Estimator not initialized when it should be."
        assert (
            self.alg is not None
        ), "Algorithm class not initialized when it should be."

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
        if type(self.alg) is str:
            self.alg = dynamic_import(self.alg)
        self.estimator = self.loadEstimator(self.parameters)

    def initFromDataset(self, data: QSPRDataset | None):
        if data is not None:
            self.targetProperties = data.targetProperties
            self.nTargets = len(self.targetProperties)
            self.featureCalculators = data.descriptorSets
            self.featureStandardizer = data.featureStandardizer
            if self.randomState is None:
                self.initRandomState(data.randomState)
        else:
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
        if random_state is None:
            self.randomState = int(np.random.randint(0, 2**31-1, dtype=np.int64))
            logger.info(
                "No random state supplied."
                f"Setting random state to: {self.randomState}."
            )
        self.randomState = random_state
        constructor_params = [
            name for name, _ in inspect.signature(self.alg.__init__).parameters.items()
        ]
        common_params = ["random_state", "random_seed", "seed"]
        random_param = None
        for seed_param in common_params:
            if seed_param not in constructor_params:
                try:
                    if self.parameters:
                        params = {
                            k: v for k, v in self.parameters.items() if k != seed_param
                        }
                        params[seed_param] = self.randomState
                        self.alg(
                            **params,
                        )
                    else:
                        self.alg(**{seed_param: random_state})
                    random_param = seed_param
                    break
                except TypeError:
                    pass
            else:
                random_param = seed_param
                break
        if random_param is not None:
            if self.parameters:
                self.parameters.update({random_param: random_state})
            else:
                self.parameters = {random_param: random_state}
            self.estimator = self.loadEstimator(self.parameters)
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

    def setParams(self, params: dict | None, reset_estimator: bool = True):
        """Set model parameters. The estimator is also
        updated with the new parameters if 'reload_estimator' is `True`.

        Args:
            params (dict):
                dictionary of model parameters or `None` to reset the parameters
            reset_estimator (bool):
                if `True`, the estimator is reinitialized with the new parameters
        """
        if self.parameters is not None:
            self.parameters.update(params)
        else:
            self.parameters = params
        if reset_estimator:
            self.estimator = self.loadEstimator(self.parameters)

    def checkData(self, ds: QSPRDataset, exception: bool = True) -> bool:
        """Check if the model has a data set.

        Args:
            ds (QSPRDataset): data set to check
            exception (bool): if true, an exception is raised if no data is set

        Returns:
            bool: True if data is set, False otherwise (if exception is False)
        """
        has_data = ds is not None
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
            X = X.getFeatures(concat=True, refit_standardizer=False)
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
        mols: list[str | Mol],
        smiles_standardizer: str | Callable[[str], str] = "chembl",
        n_jobs: int = 1,
        fill_value: float = np.nan,
    ) -> tuple[QSPRDataset, np.ndarray]:
        """Create a `QSPRDataset` instance from a list of SMILES strings.

        Args:
            mols (list[str | Mol]): list of SMILES strings
            smiles_standardizer (str, callable): smiles standardizer to use
            n_jobs (int): number of parallel jobs to use
            fill_value (float): value to fill for missing features

        Returns:
            tuple:
                a tuple containing the `QSPRDataset` instance and a boolean mask
                indicating which molecules failed to be processed
        """
        # make a molecule table first and add the target properties
        if isinstance(mols[0], Mol):
            mols = [Chem.MolToSmiles(mol) for mol in mols]
        dataset = MoleculeTable.fromSMILES(
            f"{self.__class__.__name__}_{hash(self)}",
            mols,
            drop_invalids=False,
            n_jobs=n_jobs,
        )
        for target_property in self.targetProperties:
            target_property.imputer = None
            dataset.addProperty(target_property.name, np.nan)
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

    def predictDataset(
        self, dataset: QSPRDataset, use_probas: bool = False
    ) -> np.ndarray | list[np.ndarray]:
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
        mols: List[str | Mol],
        use_probas: bool = False,
        smiles_standardizer: Union[str, callable] = "chembl",
        n_jobs: int = 1,
        fill_value: float = np.nan,
        use_applicability_domain: bool = False,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Make predictions for the given molecules.

        Args:
            mols (List[str  | Mol]): list of SMILES strings
            use_probas (bool): use probabilities for classification models
            smiles_standardizer:
                either `chembl`, `old`, or a partial function
                that reads and standardizes smiles.
            n_jobs: Number of jobs to use for parallel processing.
            fill_value: Value to use for missing values in the feature matrix.
            use_applicability_domain: Use applicability domain to return if a
                molecule is within the applicability domain of the model.

        Returns:
            np.ndarray | list[np.ndarray]:
                an array of predictions or a list of arrays of predictions
                (for classification models with use_probas=True)
            np.ndarray[bool]: boolean mask indicating which molecules fall
                within the applicability domain of the model
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

        # return predictions and if mols are within applicability domain if requested
        if hasattr(self, "applicabilityDomain") and use_applicability_domain:
            in_domain = self.applicabilityDomain.contains(
                dataset.getFeatures(concat=True, ordered=True, refit_standardizer=False)
            ).values
            in_domain = self.handleInvalidsInPredictions(mols, in_domain, failed_mask)
            
            return predictions, in_domain

        return predictions

    def cleanFiles(self):
        """Clean up the model files.

        Removes the model directory and all its contents.
        """
        if os.path.exists(self.outDir):
            shutil.rmtree(self.outDir)

    def fitDataset(
        self,
        ds: QSPRDataset,
        monitor=None,
        mode=EarlyStoppingMode.OPTIMAL,
        save_model=True,
        save_data=False,
        **kwargs,
    ) -> str:
        """Train model on the whole attached data set.

        ** IMPORTANT ** For models that supportEarlyStopping, `CrossValAssessor`
        should be run first, so that the average number of epochs from the
        cross-validation with early stopping can be used for fitting the model.

        Args:
            ds (QSPRDataset): data set to fit this model on
            monitor (FitMonitor): monitor for the fitting process, if None, the base
                monitor is used
            mode (EarlyStoppingMode): early stopping mode for models that support
                early stopping, by default fit the 'optimal' number of
                epochs previously stopped at in model assessment on train or test set,
                to avoid the use of extra data for a validation set.
            save_model (bool): save the model to file
            save_data (bool): save the supplied dataset to file
            kwargs: additional arguments to pass to fit

        Returns:
            str: path to the saved model, if `save_model` is True
        """
        # do some checks
        self.checkData(ds)
        # init properties from data
        self.initFromDataset(ds)
        # get data
        X_all = ds.getFeatures(concat=True).values
        y_all = ds.getTargetPropertiesValues(concat=True).values
        # load estimator
        self.estimator = self.loadEstimator(self.parameters)
        # fit model
        logger.info(
            "Model fit started: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        self.estimator = self.fit(X_all, y_all, mode=mode, monitor=monitor, **kwargs)
        logger.info(
            "Model fit ended: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        if hasattr(ds, "applicabilityDomain") and ds.applicabilityDomain is not None:
            ds.applicabilityDomain.fit(X_all)
            self.applicabilityDomain = ds.applicabilityDomain
        if save_data:
            ds.save()
        # save model and return path
        if save_model:
            return self.save()

    def toJSON(self):
        o_dict = json.loads(super().toJSON())
        estimator_path = self.saveEstimator()
        estimator_path = estimator_path.replace(self.baseDir, ".")
        o_dict["py/state"]["baseDir"] = "."
        o_dict["py/state"]["estimator"] = estimator_path
        o_dict["py/state"]["alg"] = f"{self.alg.__module__}.{self.alg.__name__}"
        return json.dumps(o_dict, indent=4)

    def save(self, save_estimator=False):
        """Save model to file.

        Args:
            save_estimator (bool):
                Explicitly save the estimator to file, if `True`.
                Note that some models may save the estimator by default
                even if this argument is `False`.

        Returns:
            str:
                absolute path to the metafile of the saved model
            str:
                absolute path to the saved estimator, if `include_estimator` is `True`
        """
        os.makedirs(self.outDir, exist_ok=True)
        meta_path = self.toFile(self.metaFile)
        if save_estimator:
            est_path = self.saveEstimator()
            return meta_path, est_path
        else:
            return meta_path

    @classmethod
    def fromFile(cls, filename: str) -> "QSPRModel":
        ret = super().fromFile(filename)
        model_dir = os.path.dirname(filename)
        ret.baseDir = os.path.dirname(model_dir)
        ret.estimator = ret.loadEstimatorFromFile(ret.parameters)
        return ret

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.DataFrame | np.ndarray,
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
            X (pd.DataFrame, np.ndarray): data matrix to fit
            y (pd.DataFrame, np.ndarray): target matrix to fit
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
        self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
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
        self, X: pd.DataFrame | np.ndarray | QSPRDataset, estimator: Any = None
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
