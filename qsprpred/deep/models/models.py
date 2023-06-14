"""
models

Created by: Martin Sicho
On: 12.05.23, 16:39
"""
import math
import os
import sys
from copy import deepcopy
from datetime import datetime
from typing import Callable, Type

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from ...data.data import QSPRDataset
from ...deep import DEFAULT_DEVICE, DEFAULT_GPUS, SSPACE
from ...deep.models.neural_network import STFullyConnected
from ...logs import logger
from ...models.interfaces import QSPRModel
from ...models.tasks import ModelTasks


class QSPRDNN(QSPRModel):
    """This class holds the methods for training and fitting a
    Deep Neural Net QSPR model initialization.

    Here the model instance is created and parameters can be defined.

    Attributes:
        name (str): name of the model
        data (QSPRDataset): data set used to train the model
        alg (estimator): estimator instance or class
        parameters (dict): dictionary of algorithm specific parameters
        estimator (object):
            the underlying estimator instance, if `fit` or optimization is perforemed,
            this model instance gets updated accordingly
        featureCalculators (MoleculeDescriptorsCalculator):
            feature calculator instance taken from the data set
            or deserialized from file if the model is loaded without data
        featureStandardizer (SKLearnStandardizer):
            feature standardizer instance taken from the data set
            or deserialized from file if the model is loaded without data
        metaInfo (dict):
            dictionary of metadata about the model, only available
            after the model is saved
        baseDir (str):
            base directory of the model, the model files
            are stored in a subdirectory `{baseDir}/{outDir}/`
        metaFile (str):
            absolute path to the metadata file of the model (`{outPrefix}_meta.json`)
        device (cuda device): cuda device
        gpus (int/ list of ints): gpu number(s) to use for model fitting
        patience (int):
            number of epochs to wait before early stop if no progress
            on validation set score
        tol (float):
            minimum absolute improvement of loss necessary to count as
            progress on best validation score
        nClass (int): number of classes
        nDim (int): number of features
        optimalEpochs (int): number of epochs to train the model for optimal performance
        device (torch.device): cuda device, cpu or gpu
        gpus (list[int]): gpu number(s) to use for model fitting
        patience (int):
            number of epochs to wait before early stop
            if no progress on validation set score
        optimalEpochs (int):
            number of epochs to train the model for optimal performance
    """
    def __init__(
        self,
        base_dir: str,
        alg: Type = STFullyConnected,
        data: QSPRDataset | None = None,
        name: str | None = None,
        parameters: dict | None = None,
        autoload: bool = True,
        scoring: str | Callable | None = None,
        device: torch.device = DEFAULT_DEVICE,
        gpus: list[int] = DEFAULT_GPUS,
        patience: int = 50,
        tol: float = 0,
    ):
        """Initialize a QSPRDNN model.

        Args:
            base_dir (str):
                base directory of the model, the model files are stored in
                a subdirectory `{baseDir}/{outDir}/`
            alg (Type, optional):
                model class or instance. Defaults to STFullyConnected.
            data (QSPRDataset, optional):
                data set used to train the model. Defaults to None.
            name (str, optional):
                name of the model. Defaults to None.
            parameters (dict, optional):
                dictionary of algorithm specific parameters. Defaults to None.
            autoload (bool, optional):
                whether to load the model from file or not. Defaults to True.
            scoring (str | Callable, optional):
                scoring function for the model. Defaults to `None`, in which case
                the default scoring function for the task is used.
            device (torch.device, optional):
                The cuda device. Defaults to `DEFAULT_DEVICE`.
            gpus (list[int], optional):
                gpu number(s) to use for model fitting. Defaults to `DEFAULT_GPUS`.
            patience (int, optional):
                number of epochs to wait before early stop if no progress
                on validation set score. Defaults to 50.
            tol (float, optional):
                minimum absolute improvement of loss necessary to count as progress
                on best validation score. Defaults to 0.
        """
        self.device = device
        self.gpus = gpus
        self.patience = patience
        self.tol = tol
        self.nClass = None
        self.nDim = None
        super().__init__(
            base_dir, alg, data, name, parameters, autoload=autoload, scoring=scoring
        )
        if self.task.isMultiTask():
            raise NotImplementedError(
                "Multitask modelling is not implemented for QSPRDNN models."
            )
        self.optimalEpochs = (
            self.parameters["n_epochs"]
            if self.parameters is not None and "n_epochs" in self.parameters else -1
        )

    @classmethod
    def getDefaultParamsGrid(cls) -> list[list]:
        return SSPACE

    def loadEstimator(self, params: dict | None = None) -> object:
        """Load model from file or initialize new model.

        Args:
            params (dict, optional): model parameters. Defaults to None.

        Returns:
            model (object): model instance
        """
        if self.task.isRegression():
            self.nClass = 1
        else:
            self.nClass = (
                self.data.targetProperties[0].nClasses
                if self.data else self.metaInfo["n_class"]
            )
        self.nDim = self.data.X.shape[1] if self.data else self.metaInfo["n_dim"]
        # initialize model
        estimator = self.alg(
            n_dim=self.nDim,
            n_class=self.nClass,
            device=self.device,
            gpus=self.gpus,
            is_reg=self.task == ModelTasks.REGRESSION,
            patience=self.patience,
            tol=self.tol,
        )
        # set parameters if available and return
        if params is not None:
            if self.parameters is not None:
                temp_params = deepcopy(self.parameters)
                temp_params.update(params)
                estimator.set_params(**temp_params)
            else:
                estimator.set_params(**params)
        elif self.parameters is not None:
            estimator.set_params(**self.parameters)
        return estimator

    def loadEstimatorFromFile(
        self, params: dict | None = None, fallback_load: bool = True
    ) -> object:
        """Load estimator from file.

        Args:
            params (dict): parameters
            fallback_load (bool):
                if `True`, init estimator from `alg` and `params` if no estimator
                found at path

        Returns:
            estimator (object): estimator instance
        """
        path = f"{self.outPrefix}_weights.pkg"
        estimator = self.loadEstimator(params)
        # load states if available
        if os.path.exists(path):
            estimator.load_state_dict(torch.load(path))
        elif not fallback_load:
            raise FileNotFoundError(
                f"No estimator found at {path}, "
                f"loading estimator weights from file failed."
            )
        return estimator

    def saveEstimator(self) -> str:
        """Save the QSPRDNN model.

        Returns:
            str: path to the saved model
        """
        path = f"{self.outPrefix}_weights.pkg"
        torch.save(self.estimator.state_dict(), path)
        return path

    def fit(self) -> str:
        """Train model on the training data,
        determine best model using test set, save best model.

        ** IMPORTANT ** The `evaluate` method should be run first,
        so that the average number of epochs from the cross-validation
        with early stopping can be used for fitting the model.
        """
        # do some checks
        if self.optimalEpochs == -1:
            logger.error(
                "Cannot fit final model without first determining "
                "the optimal number of epochs for fitting. \
                          Run the `evaluate` method first."
            )
            sys.exit()
        self.checkForData()
        # get data
        X_all = self.data.getFeatures(concat=True).values
        y_all = self.data.getTargetPropertiesValues(concat=True).values
        # load estimator
        if self.parameters is not None:
            self.parameters.update({"n_epochs": self.optimalEpochs})
        else:
            self.parameters = {"n_epochs": self.optimalEpochs}
        self.estimator = self.loadEstimator(self.parameters)
        # fit model
        logger.info(
            "Model fit started: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        logger.info("Logging model fit to: %s.log" % self.outPrefix)
        self.estimator.fit(X_all, y_all, log=True, log_prefix=self.outPrefix)
        logger.info(
            "Model fit ended: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        # save model and return path
        return self.save()

    def saveParams(self, params: dict) -> str:
        """Save model parameters to file.

        Args:
            params (dict): model parameters

        Returns:
            str: path to the saved parameters as json
        """
        return super().saveParams(
            {
                k: params[k]
                for k in params
                if not k.startswith("_") and k not in ["training", "device", "gpus"]
            }
        )

    def evaluate(
        self,
        save: bool = True,
        es_val_size: float = 0.1,
        parameters: dict | None = None,
    ) -> np.ndarray:
        """Make predictions for cross-validation and independent test set.

        Args:
            save (bool):
                whether to save the cross validation predictions
            es_val_size (float):
                validation set size for early stopping in CV
            parameters (dict):
                model parameters, if None, the parameters from the model are used

        Returns:
            np.ndarray:
                predictions for test set and cross-validation for further analysis
        """
        evalparams = self.parameters if parameters is None else parameters
        X, X_ind = self.data.getFeatures()
        y, y_ind = self.data.getTargetPropertiesValues()
        last_save_epochs = 0
        # prepare arrays to store molecule ids
        cvs_ids = np.array([None] * len(X))
        # create array for cross validation predictions
        # and for keeping track of the number of folds
        if self.task.isRegression():
            cvs = np.zeros((y.shape[0], 1))
        else:
            cvs = np.zeros((y.shape[0], self.data.targetProperties[0].nClasses))
        fold_counter = np.zeros(y.shape[0])
        # perform cross validation
        for i, (X_train, X_test, y_train, y_test, idx_train, idx_test) in enumerate(
            self.data.createFolds()
        ):
            crossvalmodel = self.loadEstimator(evalparams)
            y_train = y_train.reshape(-1, 1)
            # split cross validation fold train set into train
            # and validation set for early stopping
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                X_train, y_train, test_size=es_val_size
            )
            last_save_epoch = crossvalmodel.fit(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold
            )
            last_save_epochs += last_save_epoch
            logger.info(f"cross validation fold {i}: last save epoch {last_save_epoch}")
            cvs[idx_test] = crossvalmodel.predict(X_test)
            fold_counter[idx_test] = i
            cvs_ids[idx_test] = X.index.values[idx_test]
        # save cross validation predictions if specified
        if save:
            indmodel = self.loadEstimator(evalparams)
            n_folds = max(fold_counter) + 1
            # save the optimal number of epochs for fitting the model as the average
            # number of epochs from the cross-validation
            self.optimalEpochs = int(math.ceil(last_save_epochs / n_folds)) + 1
            indmodel = indmodel.set_params(n_epochs=self.optimalEpochs)
            indmodel.fit(X_train_fold, y_train_fold)
            inds = indmodel.predict(X_ind)
            inds_ids = X_ind.index.values
            # save cross validation predictions and independent test set predictions
            cvs_index = pd.Index(cvs_ids, name=self.data.getDF().index.name)
            ind_index = pd.Index(inds_ids, name=self.data.getDF().index.name)
            train, test = y.add_suffix("_Label"), y_ind.add_suffix("_Label")
            train, test = pd.DataFrame(
                train.values, columns=train.columns, index=cvs_index
            ), pd.DataFrame(test.values, columns=test.columns, index=ind_index)
            for idx, prop in enumerate(self.data.targetProperties):
                if prop.task.isClassification():
                    (
                        train[f"{prop.name}_Prediction"],
                        test[f"{prop.name}_Prediction"],
                    ) = np.argmax(cvs, axis=1), np.argmax(inds, axis=1)
                    # add probability columns to train and test set
                    # FIXME: this will not work for multiclass classification
                    train = pd.concat(
                        [
                            train,
                            pd.DataFrame(cvs, index=cvs_index
                                        ).add_prefix(f"{prop.name}_ProbabilityClass_"),
                        ],
                        axis=1,
                    )
                    test = pd.concat(
                        [
                            test,
                            pd.DataFrame(inds, index=ind_index
                                        ).add_prefix(f"{prop.name}_ProbabilityClass_"),
                        ],
                        axis=1,
                    )
                else:
                    (
                        train[f"{prop.name}_Prediction"],
                        test[f"{prop.name}_Prediction"],
                    ) = (cvs[:, idx], inds[:, idx])
            train["Fold"] = fold_counter
            train.to_csv(self.outPrefix + ".cv.tsv", sep="\t")
            test.to_csv(self.outPrefix + ".ind.tsv", sep="\t")
        # return predictions in the right format for scorer
        if self.task.isClassification():
            if self.scoreFunc.needsProbasToScore:
                if self.task == ModelTasks.SINGLECLASS:
                    # if binary classification,
                    # return only the scores for the positive class
                    return cvs[:, 1]
                else:
                    return cvs
            else:
                # if score function does not need probabilities,
                # return the class with the highest score
                return np.argmax(cvs, axis=1)
        else:
            return cvs

    def save(self) -> str:
        """Save the QSPRDNN model and meta information.

        Returns:
            str: path to the saved model
        """
        self.metaInfo["n_dim"] = self.nDim
        self.metaInfo["n_class"] = self.nClass
        return super().save()

    def predict(self, X: pd.DataFrame | np.ndarray | QSPRDataset) -> np.ndarray:
        """Predict the target property values for the given features.

        Args:
            X (pd.DataFrame | np.ndarray | QSPRDataset): features to predict

        Returns:
            np.ndarray: predicted target property values
        """
        scores = self.predictProba(X)
        if self.task.isClassification():
            return np.argmax(scores, axis=1)
        else:
            return scores.flatten()

    def predictProba(self, X: pd.DataFrame | np.ndarray | QSPRDataset) -> np.ndarray:
        """Predict the probability of target property values for the given features.

        Args:
            X (pd.DataFrame | np.ndarray | QSPRDataset): features to predict

        Returns:
            np.ndarray: predicted probability of target property values
        """
        if isinstance(X, QSPRDataset):
            X = X.getFeatures(raw=True, concat=True)
        if self.featureStandardizer:
            X = self.featureStandardizer(X)

        return self.estimator.predict(X)
