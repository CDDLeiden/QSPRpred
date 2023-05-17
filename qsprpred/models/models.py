"""Here the QSPRmodel classes can be found.

At the moment there is a class for sklearn type models
and one for a keras DNN model. To add more types a model class should be added, which
is a subclass of the QSPRModel type.
"""
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
        if (type(self.model) in [SVC, SVR]):
            logger.warning("parameter max_iter set to 10000 to avoid training getting stuck. \
                            Manually set this parameter if this is not desired.")
            if self.parameters:
                self.parameters.update({'max_iter': 10000})
            else:
                self.parameters = {'max_iter': 10000}

        if self.parameters not in [None, {}]:
            self.model.set_params(**self.parameters)

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

        if type(self.model).__name__ == 'PLSRegression':
            fit_set['Y'] = y_all
        else:
            fit_set['y'] = y_all

        logger.info('Model fit started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.model.fit(**fit_set)
        logger.info('Model fit ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.save()

    def evaluate(self, save=True):
        """Make predictions for crossvalidation and independent test set.

        arguments:
            scoring (SklearnMetric): scoring metric
            save (bool): don't save predictions when used in bayesian optimization
        """
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
            self.model.fit(**fit_set)

            if self.task.isRegression():
                preds = self.model.predict(X_test)
                # some sklearn regression models return 1d arrays and others 2d arrays (e.g. PLSRegression)
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                cvs[idx_test] = preds
            else:
                # for the multiclass-multitask case predict_proba returns a list of
                # arrays, otherwise a single array is returned
                preds = self.model.predict_proba(X_test)
                for idx in range(len(self.data.targetProperties)):
                    if len(self.data.targetProperties) == 1:
                        cvs[idx][idx_test] = preds
                    else:
                        cvs[idx][idx_test] = preds[idx]
            cvs_ids[idx_test] = X.iloc[idx_test].index.to_numpy()

            logger.info('cross validation fold %s ended: %s' % (i, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        # fitting on whole trainingset and predicting on test set
        fit_set = {'X': X}

        if type(self.model).__name__ == 'PLSRegression':
            fit_set['Y'] = y.values.ravel()
        else:
            if self.data.isMultiTask:
                fit_set['y'] = y
            else:
                fit_set['y'] = y.values.ravel()

        self.model.fit(**fit_set)

        if X_ind.shape[0] > 0:
            if self.task.isRegression():
                preds = self.model.predict(X_ind)
                # some sklearn regression models return 1d arrays and others 2d arrays (e.g. PLSRegression)
                if preds.ndim == 1:
                    preds = preds.reshape(-1, 1)
                inds = preds
            else:
                # for the multiclass-multitask case predict_proba returns a list of
                # arrays, otherwise a single array is returned
                preds = self.model.predict_proba(X_ind)
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

    def gridSearch(self, search_space_gs, n_jobs=1):
        """Optimization of hyperparameters using gridSearch.

        Arguments:
            search_space_gs (dict): search space for the grid search
            scoring (Optional[str, Callable]): scoring function for the grid search.
            n_jobs (int): number of jobs for hyperparameter optimization

        Note: Default `scoring=None` will use explained_variance for regression,
        roc_auc_ovr_weighted for multiclass, and roc_auc for binary classification.
        For a list of the available scoring functions see:
        https://scikit-learn.org/stable/modules/model_evaluation.html
        """
        grid = GridSearchCV(self.model, search_space_gs, n_jobs=n_jobs, verbose=1, cv=(
            (x[4], x[5]) for x in self.data.createFolds()), scoring=self.score_func.scorer, refit=False)

        X, X_ind = self.data.getFeatures()
        y, y_ind = self.data.getTargetPropertiesValues()
        fit_set = {'X': X, 'y': y.iloc[:, 0].values.ravel()}
        logger.info('Grid search started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        grid.fit(**fit_set)
        logger.info('Grid search ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        logger.info('Grid search best parameters: %s' % grid.best_params_)
        self.parameters = grid.best_params_
        self.model = self.model.set_params(**grid.best_params_)
        self.save()

    def bayesOptimization(self, search_space_bs, n_trials, th=0.5, n_jobs=1):
        """Bayesian optimization of hyperparameters using optuna.

        Args:
            search_space_gs (dict): search space for the grid search
            n_trials (int): number of trials for bayes optimization
            scoring (Optional[str, Callable]): scoring function for the optimization.
            th (float): threshold for scoring if `scoring in self._needs_discrete_to_score`.
            n_jobs (int): the number of parallel trials

        Example of search_space_bs for scikit-learn's MLPClassifier:
        >>> model = QSPRsklearn(base_dir='.', data=dataset,
        >>>                     alg = MLPClassifier(), alg_name="MLP")
        >>>  search_space_bs = {
        >>>    'learning_rate_init': ['float', 1e-5, 1e-3,],
        >>>    'power_t' : ['discrete_uniform', 0.2, 0.8, 0.1],
        >>>    'momentum': ['float', 0.0, 1.0],
        >>> }
        >>> model.bayesOptimization(search_space_bs=search_space_bs, n_trials=10)

        Avaliable suggestion types:
        ['categorical', 'discrete_uniform', 'float', 'int', 'loguniform', 'uniform']
        """
        print('Bayesian optimization can take a while for some hyperparameter combinations')
        if n_jobs > 1:
            logger.warning("At the moment n_jobs>1 not available for bayesoptimization. n_jobs set to 1")
            n_jobs = 1

        study = optuna.create_study(direction='maximize')
        logger.info('Bayesian optimization started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        study.optimize(lambda trial: self.objective(trial, search_space_bs), n_trials, n_jobs=n_jobs)
        logger.info('Bayesian optimization ended: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        trial = study.best_trial

        logger.info('Bayesian optimization best params: %s' % trial.params)
        self.parameters = trial.params
        self.model = self.model.set_params(**trial.params)
        self.save()

    def objective(self, trial, search_space_bs):
        """Objective for bayesian optimization.

        Arguments:
            trial (int): current trial number
            th (float): threshold for scoring if `scoring in self._needs_discrete_to_score`.
            search_space_bs (dict): search space for bayes optimization
        """
        bayesian_params = {}

        for key, value in search_space_bs.items():
            if value[0] == 'categorical':
                bayesian_params[key] = trial.suggest_categorical(key, value[1])
            elif value[0] == 'discrete_uniform':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2], step=value[3])
            elif value[0] == 'float':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2])
            elif value[0] == 'int':
                bayesian_params[key] = trial.suggest_int(key, value[1], value[2])
            elif value[0] == 'loguniform':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2], log=True)
            elif value[0] == 'uniform':
                bayesian_params[key] = trial.suggest_float(key, value[1], value[2])

        print(bayesian_params)
        self.model.set_params(**bayesian_params)

        y, y_ind = self.data.getTargetPropertiesValues()
        score = self.score_func(y, self.evaluate(save=False))
        return score

    def loadModel(self, alg: Union[Type, BaseEstimator] = None, params: dict = None):
        if alg is not None and isinstance(alg, BaseEstimator):
            if params:
                return alg.set_params(**params)
            else:
                return alg
        elif isclass(alg):
            if params:
                return alg(**params)
            else:
                return alg()
        else:
            model_path = f'{self.outDir}/{self.name}.json'
            model = skljson.from_json(model_path)
            self.alg = model.__class__
            self.model = model
            return model

    def saveModel(self) -> str:
        model_path = f'{self.outDir}/{self.name}.json'
        skljson.to_json(self.model, model_path)
        return model_path

    def predict(self, X: Union[pd.DataFrame, np.ndarray, QSPRDataset]):
        if isinstance(X, QSPRDataset):
            X = X.getFeatures(raw=True, concat=True)
        if self.featureStandardizer:
            X = self.featureStandardizer(X)
        return self.model.predict(X)

    def predictProba(self, X: Union[pd.DataFrame, np.ndarray, QSPRDataset]):
        if isinstance(X, QSPRDataset):
            X = X.getFeatures(raw=True, concat=True)
        if self.featureStandardizer:
            X = self.featureStandardizer(X)
        return self.model.predict_proba(X)
