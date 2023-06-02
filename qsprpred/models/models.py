"""Here the QSPRmodel classes can be found.

At the moment there is a class for sklearn type models
and one for a keras DNN model. To add more types a model class should be added, which
is a subclass of the QSPRModel type.
"""
import os
from copy import deepcopy
from datetime import datetime
from inspect import isclass
from typing import Type, Union

import numpy as np
import optuna
import pandas as pd
import sklearn_json as skljson
from qsprpred.data.data import QSPRDataset
from qsprpred.logs import logger
from qsprpred.models.interfaces import QSPRModel
from qsprpred.models.tasks import ModelTasks
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR


class QSPRsklearn(QSPRModel):
    """QSPRModel class for sklearn type models.

    This class is a subclass of the QSPRModel type. It is used to create
    QSPR models based on sklearn type models.
    
    Attributes:
        estimator (sklearn model): sklearn estimator
        parameters (dict): dictionary with parameters for the sklearn model
        task (ModelTasks): task of the model
        targetProperties (list): list of target properties
        nTargets (int): number of target properties
    """

    def __init__(self, base_dir: str, alg=None, data: QSPRDataset = None,
                 name: str = None, parameters: dict = None, autoload: bool = True):
        super().__init__(base_dir, alg, data, name, parameters, autoload)

        if self.task == ModelTasks.MULTITASK_MIXED:
            raise ValueError(
                'MultiTask with a mix of classification and regression tasks is not supported for sklearn models.')

        if self.task == ModelTasks.MULTITASK_MULTICLASS:
            raise NotImplementedError(
                'At the moment there are no supported metrics for multi-task multi-class/mix multi-and-single class classification.')

        # initialize models with defined parameters
        if (type(self.estimator) in [SVC, SVR]):
            logger.warning("parameter max_iter set to 10000 to avoid training getting stuck. \
                            Manually set this parameter if this is not desired.")
            if self.parameters:
                self.parameters.update({'max_iter': 10000})
            else:
                self.parameters = {'max_iter': 10000}

        if self.parameters not in [None, {}] and hasattr(self, "estimator"):
            self.estimator.set_params(**self.parameters)

        logger.info('parameters: %s' % self.parameters)
        logger.debug(f'Model "{self.name}" initialized in: "{self.baseDir}"')

    def fit(self):
        """Build estimator model from entire data set."""
        # check if data is available
        self.checkForData()

        X_all = self.data.getFeatures(concat=True).values
        y_all = self.data.getTargetPropertiesValues(concat=True)
        if not self.task.isMultiTask:
            y_all = y_all.values.ravel()

        fit_set = {'X': X_all}

        if type(self.estimator).__name__ == 'PLSRegression':
            fit_set['Y'] = y_all
        else:
            fit_set['y'] = y_all

        logger.info('Model fit started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.estimator.fit(**fit_set)
        logger.info('Model fit ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.save()

    def evaluate(self, save=True, parameters=None):
        """Make predictions for crossvalidation and independent test set.

        arguments:
            save (bool): don't save predictions when used in bayesian optimization
            parameters (dict): model parameters, if None, the parameters from the model are used
        """
        evalparams = self.parameters if parameters is None else parameters
        
        # check if data is available
        self.checkForData()

        folds = self.data.createFolds()
        X, X_ind = self.data.getFeatures()
        y, y_ind = self.data.getTargetPropertiesValues()

        # prepare arrays to store molecule ids
        cvs_ids = np.array([None] * len(X))
        inds_ids = X_ind.index.to_numpy()

        # cvs and inds are used to store the predictions for the cross validation and the independent test set
        if self.task.isRegression():
            cvs = np.zeros((y.shape[0], self.nTargets))
        else:
            # cvs, inds need to be lists of arrays for multiclass-multitask classification
            cvs = [np.zeros((y.shape[0], prop.nClasses)) for prop in self.targetProperties]
            inds = [np.zeros((y_ind.shape[0], prop.nClasses)) for prop in self.targetProperties]

        fold_counter = np.zeros(y.shape[0])

        # cross validation
        for i, (X_train, X_test, y_train, y_test, idx_train, idx_test) in enumerate(folds):
            crossvalmodel = self.loadEstimator(evalparams)
            
            logger.info('cross validation fold %s started: %s' % (i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            fold_counter[idx_test] = i

            fit_set = {'X': X_train}

            # self.data.createFolds() returns numpy arrays by default so we don't call `.values` here
            if (isclass(self.alg) and self.alg.__name__ == 'PLSRegression') or (
                    type(self.alg).__name__ == 'PLSRegression'):
                fit_set['Y'] = y_train.ravel()
            else:
                if self.isMultiTask:
                    fit_set['y'] = y_train
                else:
                    fit_set['y'] = y_train.ravel()
            crossvalmodel.fit(**fit_set)

            if self.task.isRegression():
                preds = crossvalmodel.predict(X_test)
                # some sklearn regression models return 1d arrays and others 2d arrays (e.g. PLSRegression)
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                cvs[idx_test] = preds
            else:
                # for the multiclass-multitask case predict_proba returns a list of
                # arrays, otherwise a single array is returned
                preds = crossvalmodel.predict_proba(X_test)
                for idx in range(len(self.data.targetProperties)):
                    if len(self.data.targetProperties) == 1:
                        cvs[idx][idx_test] = preds
                    else:
                        cvs[idx][idx_test] = preds[idx]
            cvs_ids[idx_test] = X.iloc[idx_test].index.to_numpy()

            logger.info('cross validation fold %s ended: %s' % (i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        # fitting on whole trainingset and predicting on test set
        fit_set = {'X': X}

        if type(self.estimator).__name__ == 'PLSRegression':
            fit_set['Y'] = y.values.ravel()
        else:
            if self.data.isMultiTask:
                fit_set['y'] = y
            else:
                fit_set['y'] = y.values.ravel()

        indmodel = self.loadEstimator(evalparams)
        indmodel.fit(**fit_set)

        if X_ind.shape[0] > 0:
            if self.task.isRegression():
                preds = indmodel.predict(X_ind)
                # some sklearn regression models return 1d arrays and others 2d arrays (e.g. PLSRegression)
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                inds = preds
            else:
                # for the multiclass-multitask case predict_proba returns a list of
                # arrays, otherwise a single array is returned
                preds = indmodel.predict_proba(X_ind)
                for idx in range(self.nTargets):
                    if self.nTargets == 1:
                        inds[idx] = preds
                    else:
                        inds[idx] = preds[idx]
        else:
            logger.warning('No independent test set available. Skipping prediction on independent test set.')
            
        # save crossvalidation results
        if save:
            index_name = self.data.getDF().index.name
            ind_index = pd.Index(inds_ids, name=index_name)
            cvs_index = pd.Index(cvs_ids, name=index_name)
            train, test = y.add_suffix('_Label'), y_ind.add_suffix('_Label')
            train, test = pd.DataFrame(
                train.values, columns=train.columns, index=cvs_index), pd.DataFrame(
                test.values, columns=test.columns, index=ind_index)
            for idx, prop in enumerate(self.data.targetProperties):
                if prop.task.isClassification():
                    # convert one-hot encoded predictions to class labels and add to train and test
                    train[f'{prop.name}_Prediction'], test[f'{prop.name}_Prediction'] = np.argmax(
                        cvs[idx], axis=1), np.argmax(inds[idx], axis=1)
                    # add probability columns to train and test set
                    train = pd.concat([train, pd.DataFrame(
                        cvs[idx], index=cvs_index).add_prefix(f'{prop.name}_ProbabilityClass_')], axis=1)
                    test = pd.concat([test, pd.DataFrame(inds[idx], index=ind_index).add_prefix(
                        f'{prop.name}_ProbabilityClass_')], axis=1)
                else:
                    train[f'{prop.name}_Prediction'], test[f'{prop.name}_Prediction'] = cvs[:, idx], inds[:, idx]
            train['Fold'] = fold_counter
            train.to_csv(self.outPrefix + '.cv.tsv', sep='\t')
            test.to_csv(self.outPrefix + '.ind.tsv', sep='\t')

        # Return predictions in the right format for scorer
        if self.task.isRegression():
            return cvs
        else:
            if self.score_func.needs_proba_to_score:
                if self.task in [ModelTasks.SINGLECLASS, ModelTasks.MULTITASK_SINGLECLASS]:
                    return np.transpose([y_pred[:, 1] for y_pred in cvs])
                else:
                    if self.task.isMultiTask():
                        return cvs
                    else:
                        return cvs[0]
            else:
                return np.transpose([np.argmax(y_pred, axis=1) for y_pred in cvs])

    def loadEstimator(self, params: dict = None):
        """Load estimator from alg and params."""
        if params:
            if self.parameters is not None:
                temp_params = deepcopy(self.parameters)
                temp_params.update(params)
                return self.alg(**temp_params)
            else:
                return self.alg(**params)
        elif self.parameters is not None:
            return self.alg(**self.parameters)
        else:
            return self.alg()
    
    def loadEstimatorFromFile(self, params: dict = None, fallback_load: bool = True):
        """Load estimator from file.
        
        Args:
            params (dict): parameters
            fallback_load (bool): if True, init estimator from alg and params if no estimator found at path
        """
        path = f'{self.outPrefix}.json'
        if os.path.isfile(path):
            estimator = skljson.from_json(path)
            self.alg = estimator.__class__
            if params:
                return estimator.set_params(**params)
            else:
                return estimator
        elif fallback_load:
            return self.loadEstimator(params)
        else:
            raise FileNotFoundError(f'No estimator found at {path}, loading estimator from file failed.')

    def saveEstimator(self) -> str:
        estimator_path = f'{self.outPrefix}.json'
        skljson.to_json(self.estimator, estimator_path)
        return estimator_path

    def predict(self, X: Union[pd.DataFrame, np.ndarray, QSPRDataset]):
        if isinstance(X, QSPRDataset):
            X = X.getFeatures(raw=True, concat=True)
        if self.featureStandardizer:
            X = self.featureStandardizer(X)
        return self.estimator.predict(X)

    def predictProba(self, X: Union[pd.DataFrame, np.ndarray, QSPRDataset]):
        if isinstance(X, QSPRDataset):
            X = X.getFeatures(raw=True, concat=True)
        if self.featureStandardizer:
            X = self.featureStandardizer(X)
        return self.estimator.predict_proba(X)
