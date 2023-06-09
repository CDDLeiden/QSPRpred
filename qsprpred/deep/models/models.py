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
from typing import Optional, Type, Union

import numpy as np
import optuna
import pandas as pd
import torch

from qsprpred.deep import SSPACE
from qsprpred.data.data import QSPRDataset
from qsprpred.deep import DEFAULT_DEVICE, DEFAULT_GPUS
from qsprpred.logs import logger
from qsprpred.models.interfaces import QSPRModel
from qsprpred.deep.models.neural_network import STFullyConnected
from qsprpred.models.tasks import ModelTasks
from sklearn.model_selection import train_test_split


class QSPRDNN(QSPRModel):
    """This class holds the methods for training and fitting a Deep Neural Net QSPR model initialization.

    Here the model instance is created and parameters can be defined.

    Attributes:
        name (str): name of the model
        data (QSPRDataset): data set used to train the model
        alg (estimator): estimator instance or class
        parameters (dict): dictionary of algorithm specific parameters
        model (estimator): the underlying estimator instance, if `fit` or optimization is perforemed, this model instance gets updated accordingly
        featureCalculators (MoleculeDescriptorsCalculator): feature calculator instance taken from the data set or deserialized from file if the model is loaded without data
        featureStandardizer (SKLearnStandardizer): feature standardizer instance taken from the data set or deserialized from file if the model is loaded without data
        metaInfo (dict): dictionary of metadata about the model, only available after the model is saved
        baseDir (str): base directory of the model, the model files are stored in a subdirectory `{baseDir}/{outDir}/`
        outDir (str): output directory of the model, the model files are stored in this directory (`{baseDir}/qspr/models/{name}`)
        outPrefix (str): output prefix of the model files, the model files are stored with this prefix (i.e. `{outPrefix}_meta.json`)
        metaFile (str): absolute path to the metadata file of the model (`{outPrefix}_meta.json`)
        task (ModelTasks): task of the model, taken from the data set or deserialized from file if the model is loaded without data
        targetProperties (list(targetProperty)): target property of the model, taken from the data set or deserialized from file if the model is loaded without data
        device (cuda device): cuda device
        gpus (int/ list of ints): gpu number(s) to use for model fitting
        patience (int): number of epochs to wait before early stop if no progress on validiation set score
        tol (float): minimum absolute improvement of loss necessary to count as progress on best validation score
        n_class (int): number of classes
        n_dim (int): number of features
        optimal_epochs (int): number of epochs to train the model for optimal performance
    """

    def __init__(self,
                 base_dir: str,
                 alg: STFullyConnected | Type = STFullyConnected,
                 data: Optional[QSPRDataset | None] = None,
                 name: str | None = None,
                 parameters: dict | None = None,
                 autoload: bool = True,
                 scoring: str | callable | None = None,
                 device=DEFAULT_DEVICE,
                 gpus=DEFAULT_GPUS,
                 patience=50, tol=0
                 ):
        """Initialize a QSPRDNN model.

        Args:
            base_dir (str): base directory of the model, the model files are stored in a subdirectory `{baseDir}/{outDir}/`
            alg (Union[STFullyConnected, Type], optional): model class or instance. Defaults to STFullyConnected.
            data (QSPRDataset, optional): data set used to train the model. Defaults to None.
            name (str, optional): name of the model. Defaults to None.
            parameters (dict, optional): dictionary of algorithm specific parameters. Defaults to None.
            autoload (bool, optional): whether to load the model from file or not. Defaults to True.
            scoring (Optional[str, callable], optional): scoring function for the model. Defaults to None, in which case the default scoring function for the task is used.
            device (cuda device, optional): cuda device. Defaults to DEFAULT_DEVICE.
            gpu (int/ list of ints, optional): gpu number(s) to use for model fitting. Defaults to DEFAULT_GPUS.
            patience (int, optional): number of epochs to wait before early stop if no progress on validiation set score. Defaults to 50.
            tol (float, optional): minimum absolute improvement of loss necessary to count as progress on best validation score. Defaults to 0.
        """
        self.device = device
        self.gpus = gpus
        self.patience = patience
        self.tol = tol
        
        super().__init__(base_dir, alg, data, name, parameters, autoload=autoload, scoring=scoring)

        if self.task.isMultiTask():
            raise NotImplementedError(
                'Multitask modelling is not implemented for QSPRDNN models.')

        self.optimal_epochs = self.parameters['n_epochs'] if self.parameters is not None and 'n_epochs' in self.parameters else -1

    @classmethod
    def getDefaultParamsGrid(cls):
        return SSPACE

    def loadEstimator(self, params: dict = None):
        """
        Load model from file or initialize new model.

        Args:
            alg (Union[Type, object], optional): model class or instance. Defaults to None.
            params (dict, optional): model parameters. Defaults to None.

        Returns:
            model (object): model instance
        """
        if self.task.isRegression():
            self.n_class = 1
        else:
            self.n_class = self.data.targetProperties[0].nClasses if self.data else self.metaInfo['n_class']
        self.n_dim = self.data.X.shape[1] if self.data else self.metaInfo['n_dim']
        
        # initialize model
        estimator = self.alg(
            n_dim=self.n_dim,
            n_class=self.n_class,
            device=self.device,
            gpus=self.gpus,
            is_reg=self.task == ModelTasks.REGRESSION,
            patience=self.patience,
            tol=self.tol
        )
        
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
    
    def loadEstimatorFromFile(self, params: dict = None, fallback_load: bool = True):
        """Load estimator from file.
        
        Args:
            params (dict): parameters
            fallback_load (bool): if True, init estimator from alg and params if no estimator found at path
        """
        path = f'{self.outPrefix}_weights.pkg'
        estimator = self.loadEstimator(params)
        # load states if available
        if os.path.exists(path):
            estimator.load_state_dict(torch.load(path))
        elif not fallback_load:
            raise FileNotFoundError(f'No estimator found at {path}, loading estimator weights from file failed.')
        
        return estimator

    def saveEstimator(self) -> str:
        """Save the QSPRDNN model.

        Returns:
            str: path to the saved model
        """
        path = f'{self.outPrefix}_weights.pkg'
        torch.save(self.estimator.state_dict(), path)
        return path

    def fit(self):
        """Train model on the training data, determine best model using test set, save best model.

        ** IMPORTANT: evaluate should be run first, so that the average number of epochs from the cross-validation
                        with early stopping can be used for fitting the model.
        """
        if self.optimal_epochs == -1:
            logger.error('Cannot fit final model without first determining the optimal number of epochs for fitting. \
                          first run evaluate.')
            sys.exit()

        self.checkForData()

        X_all = self.data.getFeatures(concat=True).values
        y_all = self.data.getTargetPropertiesValues(concat=True).values

        if self.parameters is not None:
            self.parameters.update({"n_epochs": self.optimal_epochs})
        else:
            self.parameters = {"n_epochs": self.optimal_epochs}
        self.estimator = self.loadEstimator(self.parameters)

        logger.info('Model fit started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logger.info('Logging model fit to: %s.log' % self.outPrefix)
        self.estimator.fit(X_all, y_all, log=True, log_prefix=self.outPrefix)
        logger.info('Model fit ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        self.save()

    def saveParams(self, params):
        """Save model parameters to file.

        Args:
            params (dict): model parameters

        Returns:
            params (dict): model parameters
        """
        return super().saveParams(
            {
                k: params[k] for k in params
                if not k.startswith('_')
                and k not in ['training', 'device', 'gpus']
            }
        )

    def evaluate(self, save=True, ES_val_size=0.1, parameters=None):
        """Make predictions for crossvalidation and independent test set.

        Args:
            save (bool): wether to save the cross validation predictions
            ES_val_size (float): validation set size for early stopping in CV
            parameters (dict): model parameters, if None, the parameters from the model are used
        """
        evalparams = self.parameters if parameters is None else parameters
        X, X_ind = self.data.getFeatures()
        y, y_ind = self.data.getTargetPropertiesValues()
        last_save_epochs = 0

        # prepare arrays to store molecule ids
        cvs_ids = np.array([None] * len(X))
        inds_ids = X_ind.index.to_numpy()

        # create array for cross validation predictions and for keeping track of the number of folds
        if self.task.isRegression():
            cvs = np.zeros((y.shape[0], 1))
        else:
            cvs = np.zeros((y.shape[0], self.data.targetProperties[0].nClasses))
        fold_counter = np.zeros(y.shape[0])

        for i, (X_train, X_test, y_train, y_test, idx_train, idx_test) in enumerate(self.data.createFolds()):
            crossvalmodel = self.loadEstimator(evalparams)
            y_train = y_train.reshape(-1, 1)

            # split cross validation fold train set into train and validation set for early stopping
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                X_train, y_train, test_size=ES_val_size)
            last_save_epoch = crossvalmodel.fit(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            last_save_epochs += last_save_epoch
            logger.info(f'cross validation fold {i}: last save epoch {last_save_epoch}')
            cvs[idx_test] = crossvalmodel.predict(X_test)

            fold_counter[idx_test] = i
            cvs_ids[idx_test] = X.index.values[idx_test]

        if save:
            indmodel = self.loadEstimator(evalparams)
            n_folds = max(fold_counter) + 1

            # save the optimal number of epochs for fitting the model as the average
            # number of epochs from the cross-validation
            self.optimal_epochs = int(math.ceil(last_save_epochs / n_folds)) + 1
            indmodel = indmodel.set_params(**{"n_epochs": self.optimal_epochs})
            indmodel.fit(X_train_fold, y_train_fold)
            inds = indmodel.predict(X_ind)
            inds_ids = X_ind.index.values

            # save cross validation predictions and independent test set predictions
            cvs_index = pd.Index(cvs_ids, name=self.data.getDF().index.name)
            ind_index = pd.Index(inds_ids, name=self.data.getDF().index.name)
            train, test = y.add_suffix('_Label'), y_ind.add_suffix('_Label')
            train, test = pd.DataFrame(
                train.values, columns=train.columns, index=cvs_index), pd.DataFrame(
                test.values, columns=test.columns, index=ind_index)
            for idx, prop in enumerate(self.data.targetProperties):
                if prop.task.isClassification():
                    train[f'{prop.name}_Prediction'], test[f'{prop.name}_Prediction'] = np.argmax(cvs, axis=1), np.argmax(inds, axis=1)
                    # add probability columns to train and test set
                    # FIXME: this will not work for multiclass classification
                    train = pd.concat([train, pd.DataFrame(
                        cvs, index=cvs_index).add_prefix(f'{prop.name}_ProbabilityClass_')], axis=1)
                    test = pd.concat([test, pd.DataFrame(inds, index=ind_index).add_prefix(
                        f'{prop.name}_ProbabilityClass_')], axis=1)
                else:
                    train[f'{prop.name}_Prediction'], test[f'{prop.name}_Prediction'] = cvs[:, idx], inds[:, idx]
            train['Fold'] = fold_counter
            train.to_csv(self.outPrefix + '.cv.tsv', sep='\t')
            test.to_csv(self.outPrefix + '.ind.tsv', sep='\t')

        # Return predictions in the right format for scorer
        if self.task.isClassification():
            if self.scoreFunc.needsProbasToScore:
                if self.task == ModelTasks.SINGLECLASS:
                    return cvs[:, 1]  # if binary classification, return only the scores for the positive class
                else:
                    return cvs
            else:
                # if score function does not need probabilities, return the class with the highest score
                return np.argmax(cvs, axis=1)
        else:
            return cvs

    def objective(self, trial, scoring, th, search_space_bs):
        """Objective for bayesian optimization.

        Arguments:
            trial (int): current trial number
            scoring (Optional[str]): scoring function for the objective.
            th (float): threshold for scoring if `scoring in self._needs_discrete_to_score`.
            search_space_bs (dict): search space for bayes optimization

        Returns:
            score (float): score of the current trial
        """
        bayesian_params = {}

        for key, value in search_space_bs.items():
            if value[0] == 'categorical':
                bayesian_params[key] = trial.suggest_categorical(key, value[1])
            elif value[0] == 'discrete_uniform':
                bayesian_params[key] = trial.suggest_discrete_uniform(key, value[1], value[2], value[3])
            elif value[0] == 'float':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2])
            elif value[0] == 'int':
                bayesian_params[key] = trial.suggest_int(key, value[1], value[2])
            elif value[0] == 'loguniform':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2], log=True)
            elif value[0] == 'uniform':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2])

        y, y_ind = self.data.getTargetPropertiesValues()
        if scoring is not None and scoring.needs_discrete_to_score:
            y = np.where(y > th, 1, 0)

        score_func = self.getScoringFunction(scoring)
        score = score_func(y, self.evaluate(save=False, parameters=bayesian_params))
        return score

    def save(self):
        """Save the QSPRDNN model and meta information.

        Returns:
            str: path to the saved model
        """
        self.metaInfo['n_dim'] = self.n_dim
        self.metaInfo['n_class'] = self.n_class
        return super().save()

    def predict(self, X: Union[pd.DataFrame, np.ndarray, QSPRDataset]):
        """Predict the target property values for the given features.

        Args:
            X (Union[pd.DataFrame, np.ndarray, QSPRDataset]): features

        Returns:
            np.ndarray: predicted target property values
        """
        scores = self.predictProba(X)
        if self.task.isClassification():
            return np.argmax(scores, axis=1)
        else:
            return scores.flatten()

    def predictProba(self, X: Union[pd.DataFrame, np.ndarray, QSPRDataset]):
        """Predict the probability of target property values for the given features.

        Args:
            X (Union[pd.DataFrame, np.ndarray, QSPRDataset]): features

        Returns:
            np.ndarray: predicted probability of target property values
        """
        if isinstance(X, QSPRDataset):
            X = X.getFeatures(raw=True, concat=True)
        if self.featureStandardizer:
            X = self.featureStandardizer(X)

        return self.estimator.predict(X)
