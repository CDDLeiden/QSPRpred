"""This module holds the base class for QSPRmodels, model types should be a subclass."""
import copy
import json
import os
import shutil
import sys
from abc import ABC, abstractmethod
from typing import List, Type, Union

import numpy as np
import pandas as pd
from qsprpred.data.data import MoleculeTable, QSPRDataset, TargetProperty
from qsprpred.data.utils.descriptorcalculator import DescriptorsCalculator
from qsprpred.data.utils.feature_standardization import SKLearnStandardizer
from qsprpred.logs import logger
from qsprpred.models import SSPACE
from qsprpred.models.tasks import ModelTasks
from sklearn import metrics


class QSPRModel(ABC):
    """
    The definition of the common model interface for the package. Handles model initialization, fit, cross validation and hyperparameter optimization.

    Attributes:
        name (str): name of the model
        data (QSPRDataset): data set used to train the model
        alg (estimator): estimator instance or class
        parameters (dict): dictionary of algorithm specific parameters
        model (estimator): the underlying estimator instance, if `fit` or optimization is perforemed, this model instance gets updated accordingly
        featureCalculator (DescriptorsCalculator): feature calculator instance taken from the data set or deserialized from file if the model is loaded without data
        featureStandardizer (SKLearnStandardizer): feature standardizer instance taken from the data set or deserialized from file if the model is loaded without data
        metaInfo (dict): dictionary of metadata about the model, only available after the model is saved
        baseDir (str): base directory of the model, the model files are stored in a subdirectory `{baseDir}/{outDir}/`
        outDir (str): output directory of the model, the model files are stored in this directory (`{baseDir}/qspr/models/{name}`)
        outPrefix (str): output prefix of the model files, the model files are stored with this prefix (i.e. `{outPrefix}_meta.json`)
        metaFile (str): absolute path to the metadata file of the model (`{outPrefix}_meta.json`)
        task (ModelTasks): task of the model, taken from the data set or deserialized from file if the model is loaded without data
        targetProperty (str): target property of the model, taken from the data set or deserialized from file if the model is loaded without data
    """

    def __init__(self, base_dir: str, alg=None, data: QSPRDataset = None,
                 name: str = None, parameters: dict = None, autoload=True):
        """Initialize a QSPR model instance.

        If the model is loaded from file, the data set is not required.
        Note that the data set is required for fitting and optimization.

        Args:
            base_dir (str): base directory of the model, the model files are stored in a subdirectory `{baseDir}/{outDir}/`
            alg (estimator): estimator instance or class
            data (QSPRDataset): data set used to train the model
            name (str): name of the model
            parameters (dict): dictionary of algorithm specific parameters
            autoload (bool): if True, the model is loaded from the serialized file if it exists, otherwise a new model is created
        """
        self.data = data
        self.name = name or alg.__class__.__name__
        self.baseDir = base_dir.rstrip('/')

        # load metadata and update parameters accordingly
        self.metaFile = f"{self.outPrefix}_meta.json"
        try:
            self.metaInfo = self.readMetadata(self.metaFile)
        except FileNotFoundError:
            if not self.data:
                raise FileNotFoundError(f"Metadata file '{self.metaFile}' not found")
            else:
                self.metaInfo = {}

        # get parameters from metadata if available
        self.parameters = parameters
        if self.metaInfo and not self.parameters:
            self.parameters = self.readParams(os.path.join(self.baseDir, self.metaInfo['parameters_path']))

        # initialize a feature calculator instance
        self.featureCalculator = self.data.descriptorCalculator if self.data else self.readDescriptorCalculator(
            os.path.join(self.baseDir, self.metaInfo['feature_calculator_path']))

        # initialize a standardizer instance
        self.featureStandardizer = self.data.feature_standardizer if self.data else self.readStandardizer(os.path.join(
            self.baseDir, self.metaInfo['feature_standardizer_path'])) if self.metaInfo['feature_standardizer_path'] else None

        # initialize a model instance with the given parameters
        self.alg = alg
        if autoload:
            self.model = self.loadModel(alg=self.alg, params=self.parameters)

    def __str__(self):
        """Return the name of the model and the underlying class as the identifier."""
        return f"{self.name} ({self.model.__class__.__name__ if self.model else self.alg.__class__.__name__ if self.alg else 'None'})"

    # Adding scoring functions available for hyperparam optimization:
    @property
    def _needs_proba_to_score(self):
        if self.task.isClassification():
            return ['average_precision', 'neg_brier_score', 'neg_log_loss', 'roc_auc',
                    'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted']
        elif self.task.isRegresssion():
            return []

    @property
    def _needs_discrete_to_score(self):
        if self.task.isClassification():
            return ['accuracy', 'balanced_accuracy', 'top_k_accuracy', 'f1', 'f1_micro',
                    'f1_macro', 'f1_weighted', 'f1_samples', 'precision', 'precision_micro',
                    'precision_macro', 'precision_weighted', 'precision_samples', 'recall',
                    'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples']
        elif self.task.isRegresssion():
            return []

    @property
    def _supports_multitask(self):
        if self.task.isClassification():
            if self.task == ModelTasks.MULTITASK_SINGLECLASS:
                return ['accuracy', 'average_precision', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples',
                        'metrics.log_loss', 'precision', 'precision_micro', 'precision_macro', 'precision_weighted',
                        'precision_samples', 'recall', 'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples']
            elif self.task == ModelTasks.MULTITASK_MULTICLASS:
                return []
        elif self.task.isRegression:
            return self._supported_scoring()
        else:
            return []

    @property
    def _supported_scoring(self):
        if self.task.isClassification():
            return ['average_precision', 'neg_brier_score', 'neg_log_loss', 'roc_auc',
                    'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted'
                    'accuracy', 'balanced_accuracy', 'top_k_accuracy', 'f1', 'f1_micro',
                    'f1_macro', 'f1_weighted', 'f1_samples', 'precision', 'precision_micro',
                    'precision_macro', 'precision_weighted', 'precision_samples', 'recall',
                    'recall_micro', 'recall_macro', 'recall_weighted', 'recall_samples']
        elif self.task.isRegresssion():
            return ['explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error',
                    'neg_root_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error',
                    'r2', 'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance', 'neg_mean_absolute_percentage_error',
                    'd2_absolute_error_score', 'd2_pinball_score', 'd2_tweedie_scor']

    @property
    def targetProperties(self):
        """Return the target properties of the model, taken from the data set or deserialized from file if the model is loaded without data.

        Returns:
            str: target property of the model
        """
        return self.data.targetProperties if self.data else self.metaInfo['target_properties']

    @property
    def task(self):
        """Return the task of the model, taken from the data set or deserialized from file if the model is loaded without data.

        Returns:
            ModelType: task of the model
        """
        return ModelTasks.getModelTask(self.targetProperties)

    @property
    def isMultiTask(self):
        """Return if model is a multitask model, taken from the data set or deserialized from file if the model is loaded without data.

        Returns:
            bool: True if model is a multitask model
        """
        return self.task.isMultiTask()

    @property
    def nTargets(self):
        """Return the number of target properties of the model, taken from the data set or deserialized from file if the model is loaded without data.

        Returns:
            int: number of target properties of the model
        """
        return len(self.data.targetProperties) if self.data else len(self.metaInfo['target_properties'])

    @property
    def outDir(self):
        """Return output directory of the model, the model files are stored in this directory (`{baseDir}/qspr/models/{name}`).

        Returns:
            str: output directory of the model
        """
        os.makedirs(f'{self.baseDir}/qspr/models/{self.name}', exist_ok=True)
        return f'{self.baseDir}/qspr/models/{self.name}'

    @property
    def outPrefix(self):
        """Return output prefix of the model files, the model files are stored with this prefix (i.e. `{outPrefix}_meta.json`).

        Returns:
            str: output prefix of the model files
        """
        return f'{self.outDir}/{self.name}'

    @staticmethod
    def readParams(path):
        """Read model parameters from a JSON file.

        Args:
            path (str): absolute path to the JSON file
        """
        with open(path, "r", encoding="utf-8") as j:
            logger.info(
                'loading model parameters from file: %s' % path)
            return json.loads(j.read())

    def saveParams(self, params):
        """Save model parameters to a JSON file.

        Args:
            params (dict): dictionary of model parameters
        """
        path = f'{self.outDir}/{self.name}_params.json'
        with open(path, "w", encoding="utf-8") as j:
            logger.info(
                'saving model parameters to file: %s' % path)
            j.write(json.dumps(params, indent=4))

        return path

    @staticmethod
    def readDescriptorCalculator(path):
        """Read a descriptor calculator from a JSON file.

        Args:
            path (str): absolute path to the JSON file

        Returns:
            DescriptorsCalculator: descriptor calculator instance or None if the file does not exist
        """
        if os.path.exists(path):
            return DescriptorsCalculator.fromFile(path)

    def saveDescriptorCalculator(self):
        """Save the current descriptor calculator to a JSON file. The file is stored in the output directory of the model.

        Returns:
            str: absolute path to the JSON file containing the descriptor calculator
        """
        path = f'{self.outDir}/{self.name}_descriptor_calculator.json'
        self.featureCalculator.toFile(path)
        return path

    @staticmethod
    def readStandardizer(path):
        """Read a feature standardizer from a JSON file. If the file does not exist, None is returned.

        Args:
            path (str): absolute path to the JSON file

        Returns:
            SKLearnStandardizer: feature standardizer instance or None if the file does not exist
        """
        if os.path.exists(path):
            return SKLearnStandardizer.fromFile(path)

    def saveStandardizer(self):
        """Save the current feature standardizer to a JSON file. The file is stored in the output directory of the model.

        Returns:
            str: absolute path to the JSON file containing the saved feature standardizer
        """
        path = f'{self.outDir}/{self.name}_feature_standardizer.json'
        self.featureStandardizer.toFile(path)
        return path

    @classmethod
    def readMetadata(cls, path):
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
                metaInfo = json.loads(j.read())
                metaInfo["target_properties"] = TargetProperty.fromList(
                    metaInfo["target_properties"], task_from_str=True)
        else:
            raise FileNotFoundError(f'Metadata file "{path}" does not exist.')

        return metaInfo

    def saveMetadata(self):
        """Save model metadata to a JSON file. The file is stored in the output directory of the model.

        Returns:
            str: absolute path to the JSON file containing the model metadata
        """
        with open(self.metaFile, "w", encoding="utf-8") as j:
            logger.info(
                'saving model metadata to file: %s' % self.metaFile)
            j.write(json.dumps(self.metaInfo, indent=4))

        return self.metaFile

    def save(self):
        """Save the model data, parameters, metadata and all other files to the output directory of the model.

        Returns:
            dict: dictionary containing the model metadata that was saved
        """
        self.metaInfo['target_properties'] = TargetProperty.toList(
            copy.deepcopy(self.data.targetProperties), task_as_str=True)
        self.metaInfo['parameters_path'] = self.saveParams(self.parameters).replace(f"{self.baseDir}/", '')
        self.metaInfo['feature_calculator_path'] = self.saveDescriptorCalculator().replace(
            f"{self.baseDir}/", '') if self.featureCalculator else None
        self.metaInfo['feature_standardizer_path'] = self.saveStandardizer().replace(
            f"{self.baseDir}/", '') if self.featureStandardizer else None
        self.metaInfo['model_path'] = self.saveModel().replace(f"{self.baseDir}/", '')
        return self.saveMetadata()

    def checkForData(self, exception=True):
        """Check if the model has data set.

        Args:
            exception (bool): if true, an exception is raised if no data is set

        Returns:
            bool: True if data is set, False otherwise (if exception is False)
        """
        hasData = self.data is not None
        if exception and not hasData:
            raise ValueError(
                'No data set specified. Make sure you initialized this model with a "QSPRDataset" instance to train on.')

        return hasData

    @abstractmethod
    def fit(self):
        """Build estimator model from the whole associated data set."""
        pass

    @abstractmethod
    def evaluate(self, save=True):
        """Make predictions for crossvalidation and independent test set.

        If save is True, the predictions are saved to a file in the output directory.

        Arguments:
            save (bool): don't save predictions when used in bayesian optimization
        """
        pass

    @abstractmethod
    def gridSearch(self, search_space_gs):
        """
        Optimization of hyperparameters using gridSearch.

        Args:
            search_space_gs (dict): search space for the grid search
        """
        pass

    @abstractmethod
    def bayesOptimization(self, search_space_gs):
        """Bayesian optimization of hyperparameters using optuna.

        Arguments:
            search_space_gs (dict): search space for the grid search
        """
        pass

    @staticmethod
    def loadParamsGrid(fname, optim_type, model_types):
        """Load parameter grids for bayes or grid search parameter optimization from json file.

        Arguments:
            fname (str): file name of json file containing array with three columns containing modeltype,
                            optimization type (grid or bayes) and model type
            optim_type (str): optimization type ('grid' or 'bayes')
            model_types (list of str): model type for hyperparameter optimization (e.g. RF)
        """
        if fname:
            try:
                with open(fname) as json_file:
                    optim_params = np.array(json.load(json_file), dtype=object)
            except FileNotFoundError:
                logger.error("Search space file (%s) not found" % fname)
                sys.exit()
        else:
            with open(SSPACE) as json_file:
                optim_params = np.array(json.load(json_file), dtype=object)

        # select either grid or bayes optimization parameters from param array
        optim_params = optim_params[optim_params[:, 2] == optim_type, :]

        # check all ModelTasks to be used have parameter grid
        model_types = [model_types] if isinstance(
            model_types, str) else model_types

        if not set(list(model_types)).issubset(list(optim_params[:, 0])):
            logger.error("model types %s missing from models in search space dict (%s)" % (
                model_types, optim_params[:, 0]))
            sys.exit()
        logger.info("search space loaded from file")

        return optim_params

    @abstractmethod
    def loadModel(self, alg: Union[Type, object] = None, params: dict = None):
        """Initialize model instance with the given parameters. If no algorithm is given, the model is loaded from file based on available metadata. If no parameters are given, they are also loaded from the available file.

        Arguments:
            alg (object): algorithm class to instantiate
            params (dict): algorithm parameters
        """
        pass

    @abstractmethod
    def saveModel(self) -> str:
        """Save the underlying model to file.

        Returns:
            str: path to the saved model
        """
        pass

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray, QSPRDataset]):
        """Make predictions for the given data matrix or `QSPRDataset`.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to make predictions for

        Returns:
            np.ndarray: an array of predictions, can be a 1D array for single target models or a 2D/3D array for multi-target/multi-class models
        """
        pass

    @abstractmethod
    def predictProba(self, X: Union[pd.DataFrame, np.ndarray, QSPRDataset]):
        """Make predictions for the given data matrix or `QSPRDataset`, but use probabilities for classification models.

        Args:
            X (pd.DataFrame, np.ndarray, QSPRDataset): data matrix to make predictions for
        """
        pass

    def predictMols(self, mols: List[str], use_probas: bool = False):
        """Make predictions for the given molecules.

        Args:
            mols (List[str]): list of SMILES strings
            use_probas (bool): use probabilities for classification models
        """
        dataset = MoleculeTable.fromSMILES(f"{self.__class__.__name__}_{hash(self)}", mols, drop_invalids=False)
        for targetproperty in self.targetProperties:
            dataset.addProperty(targetproperty.name, np.nan)
        dataset = QSPRDataset.fromMolTable(dataset, self.targetProperties, drop_empty=False, drop_invalids=False)
        failed_mask = dataset.dropInvalids().values
        if not self.featureCalculator:
            raise ValueError("No feature calculator set on this instance.")
        dataset.prepareDataset(
            standardize=True,
            sanitize=True,
            feature_calculator=self.featureCalculator,
            feature_standardizer=self.featureStandardizer
        )
        if self.task.isRegression() or not use_probas:
            predictions = self.predict(dataset)
            # always return 2D array
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            if self.task.isClassification():
                predictions = predictions.astype(int)
        else:
            # NOTE: if a multiclass-multiouput, this will return a list of 2D arrays
            predictions = self.predictProba(dataset)

        if any(~failed_mask):
            # fill in the failed predictions with None
            if isinstance(predictions, list):
                predictions_with_invalids = [np.full((len(mols), pred.shape[1]), None) for pred in predictions]
                for i, pred in enumerate(predictions):
                    predictions_with_invalids[i][failed_mask, :] = pred
            else:
                predictions_with_invalids = np.full((len(mols), predictions.shape[1]), None)
                predictions_with_invalids[failed_mask, :] = predictions
            predictions = predictions_with_invalids
        return predictions

    @classmethod
    def fromFile(cls, path):
        """Load a model from its meta file.

        Args:
            path (str): full path to the model meta file
        """
        name = cls.readMetadata(path)['name']
        dir_name = os.path.dirname(path).replace(f"qspr/models/{name}", "")
        return cls(name=name, base_dir=dir_name)

    def cleanFiles(self):
        """Clean up the model files. Removes the model directory and all its contents."""
        if os.path.exists(self.outDir):
            shutil.rmtree(self.outDir)

    def get_scoring_func(self, scoring):
        """Get scoring function from sklearn.metrics.

        Args:
            scoring (Union[str, Callable]): metric name from sklearn.metrics or
                user-defined scoring function.

        Raises:
            ValueError: If the scoring function is currently not supported by
                GridSearch and BayesOptimization.

        Returns:
            score_func (Callable): scorer function from sklearn.metrics (`str` as input)
            or user-defined function (`callable` as input)
        """
        if all([scoring not in self._supported_scoring, isinstance(scoring, str)]):
            raise ValueError("Scoring function %s not supported. Supported scoring functions are: %s"
                             % (scoring, self._supported_scoring))
        elif callable(scoring):
            return scoring
        elif scoring is None:
            if self.task.isRegression():
                scorer = metrics.get_scorer('explained_variance')
            elif self.task in [ModelTasks.MULTICLASS, ModelTasks.MULTITASK_SINGLECLASS]:
                return lambda y_true, y_pred: metrics.roc_auc_score(
                    y_true, y_pred, multi_class='ovr', average='weighted')
            elif self.task in [ModelTasks.SINGLECLASS]:
                scorer = metrics.get_scorer('roc_auc')
            else:
                raise ValueError("No supported scoring function for task %s" % self.task)
        else:
            scorer = metrics.get_scorer(scoring)
        return scorer._score_func
