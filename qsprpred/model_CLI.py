#!/usr/bin/env python

import argparse
import json
import os
import os.path
import random
import sys
from datetime import datetime

import numpy as np
import optuna
import torch
from qsprpred.data.data import QSPRDataset
from qsprpred.logs.utils import backUpFiles, commit_hash, enable_file_logger
from qsprpred.models.models import QSPRDNN, QSPRModel, QSPRsklearn
from qsprpred.models.tasks import ModelTasks
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor


def QSPRArgParser(txt=None):
    """Define and read command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # base arguments
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-ran', '--random_state', type=int, default=1, help="Seed for the random state")
    parser.add_argument('-ncpu', '--ncpu', type=int, default=8, help="Number of CPUs")
    parser.add_argument('-gpus', '--gpus', nargs="*", default=['0'], help="List of GPUs")

    # model target arguments
    parser.add_argument('-pr', '--properties', type=str, nargs='+', action='append',
                        help="properties to be predicted identifiers. Add this argument for each model to be trained \
                              e.g. for one multi-task model for CL and Fu and one single task for CL do:\
                              -pr CL Fu -pr CL")

    # model type arguments
    parser.add_argument('-m', '--model_types', type=str, nargs='*',
                        choices=['RF', 'XGB', 'SVM', 'PLS', 'NB', 'KNN', 'DNN'],
                        default=['RF', 'XGB', 'SVM', 'PLS', 'NB', 'KNN', 'DNN'],
                        help="Modeltype, defaults to run all modeltypes, choose from: 'RF', 'XGB', 'DNN', 'SVM',\
                             'PLS' (only with REG), 'NB' (only with CLS) 'KNN'")
    parser.add_argument('-r', '--regression', type=str, default=None,
                        help="If True, only regression model, if False, only classification, default both")

    # model settings
    parser.add_argument('-p', '--parameters', type=str, default=None,
                        help="file name of json file with non-default parameter settings \
                             (base_dir/qspr/models/[-p]_params.json). NB. If json file with name \
                             {model_type}_{REG/CLS}_{property}_params.json) present in qspr/models folder those settings will also be used, \
                             but if the same parameter is present in both files the settings from (base_dir/[-p]_params.json) will be used.")
    parser.add_argument('-sw', '--sample_weighing', action='store_true',
                        help='Sets balanced class weights.')
    parser.add_argument('-pat', '--patience', type=int, default=50, help="for DNN, number of epochs for early stopping")
    parser.add_argument('-tol', '--tolerance', type=float, default=0.01,
                        help="for DNN, minimum absolute change of loss to count as progress")

    # model training procedure
    parser.add_argument('-s', '--save_model', action='store_true',
                        help="If included then the model will be trained on all data and saved")
    parser.add_argument('-o', '--optimization', type=str, default=None,
                        help="Hyperparameter optimization, if 'None' no optimization, if 'grid' gridsearch, \
                            if 'bayes' bayesian optimization")
    parser.add_argument('-ss', '--search_space', type=str, default=None,
                        help="search_space hyperparameter optimization json file location (base_dir/[name].json), \
                              if None default qsprpred.models.search_space.json used")
    parser.add_argument('-nt', '--n_trials', type=int, default=20, help="number of trials for bayes optimization")
    parser.add_argument('-me', '--model_evaluation', action='store_true',
                        help='If on, model evaluation through cross validation and independent test set is performed.')

    # other
    parser.add_argument('-ng', '--no_git', action='store_true', help="If on, git hash is not retrieved")

    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()

    for props in args.properties:
        if len(props) > 1:
            sys.exit("Multitask not yet implemented")

    # If no regression argument, does both regression and classification
    if args.regression is None:
        args.regression = [True, False]
    elif args.regression.lower() in ['true', 'reg', 'regression']:
        args.regression = [True]
    elif args.regression.lower() in ['false', 'cls', 'classification']:
        args.regression = [False]
    else:
        sys.exit("invalid regression arg given")

    return args


def QSPR_modelling(args):
    """Optimize, evaluate and train estimators."""
    if not os.path.exists(args.base_dir + '/qspr/models'):
        os.makedirs(args.base_dir + '/qspr/models')

    # read in file with specified parameters for model fitting
    parameters = None
    if args.parameters:
        try:
            with open(f'{args.base_dir}/{args.parameters}.json') as json_file:
                par_dicts = np.array(json.load(json_file))
        except FileNotFoundError:
            log.error(
                "Parameter settings file (%s/%s.json) not found." % (args.base_dir, args.parameters))
            sys.exit()

    if args.optimization in ['grid', 'bayes']:
        if args.search_space:
            grid_params = QSPRModel.loadParamsGrid(
                f'{args.base_dir}/{args.search_space}.json',
                args.optimization,
                args.model_types)
        else:
            grid_params = QSPRModel.loadParamsGrid(
                None, args.optimization, args.model_types)

    for reg in args.regression:
        task = ModelTasks.REGRESSION if reg else ModelTasks.CLASSIFICATION
        reg_abbr = 'REG' if reg else 'CLS'
        for property in args.properties:
            log.info(f"Property: {property[0]}")

            mydataset = QSPRDataset.fromFile(f'{args.base_dir}/qspr/data/{property[0]}_{reg_abbr}_QSPRdata.pkl')
            mydataset.reload()
            mydataset.createFolds(n_folds=mydataset.n_folds)

            for model_type in args.model_types:
                print(model_type)
                log.info(f'Model: {model_type} {reg_abbr}')

                if model_type not in ['RF', 'XGB', 'DNN', 'SVM', 'PLS', 'NB', 'KNN']:
                    log.warning(f'Model type {model_type} does not exist')
                    continue
                if model_type == 'NB' and reg:
                    log.warning("NB with regression invalid, skipped.")
                    continue
                if model_type == 'PLS' and not reg:
                    log.warning("PLS with classification invalid, skipped.")
                    continue

                if args.parameters:
                    try:
                        parameters = par_dicts[par_dicts[:, 0] == model_type, 1][0]
                        if not model_type in ["NB", "PLS", "SVM", "DNN"]:
                            parameters = parameters.update({"n_jobs": args.ncpu})
                    except BaseException:
                        log.warning(f'Model type {model_type} not in parameter file, default parameter settings used.')
                        parameters = None if model_type in ["NB", "PLS", "SVM", "DNN"] else {"n_jobs": args.ncpu}
                else:
                    parameters = None if model_type in ["NB", "PLS", "SVM", "DNN"] else {"n_jobs": args.ncpu}

                if not reg:
                    class_weight = 'balanced' if args.sample_weighing else None
                    counts = mydataset.y.value_counts()
                    scale_pos_weight = counts[0] / counts[1] if args.sample_weighing else 1
                alg_dict = {
                    'RF': RandomForestRegressor() if reg else RandomForestClassifier(class_weight=class_weight),
                    'XGB': XGBRegressor(objective='reg:squarederror') if reg else
                    XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss',
                                  scale_pos_weight=scale_pos_weight),
                    'SVM': SVR() if reg else SVC(probability=True, class_weight=class_weight),
                    'PLS': PLSRegression(),
                    'NB': GaussianNB(),
                    'KNN': KNeighborsRegressor() if reg else KNeighborsClassifier()}

                # Create QSPR model object
                if model_type == 'DNN':
                    QSPRmodel = QSPRDNN(
                        base_dir=args.base_dir,
                        data=mydataset,
                        parameters=parameters,
                        gpus=args.gpus,
                        patience=args.patience,
                        tol=args.tolerance)
                else:
                    QSPRmodel = QSPRsklearn(
                        args.base_dir,
                        data=mydataset,
                        alg=alg_dict[model_type],
                        alg_name=model_type,
                        parameters=parameters)

                # if desired run parameter optimization
                if args.optimization == 'grid':
                    search_space_gs = grid_params[grid_params[:, 0] ==
                                                  model_type, 1][0]
                    log.info(search_space_gs)
                    QSPRmodel.gridSearch(search_space_gs)
                elif args.optimization == 'bayes':
                    search_space_bs = grid_params[grid_params[:, 0] ==
                                                  model_type, 1][0]
                    log.info(search_space_bs)
                    if reg and model_type == "RF":
                        if mydataset.y.min()[0] < 0 or mydataset.y_ind.min()[0] < 0:
                            search_space_bs.update(
                                {'criterion': ['categorical', ['squared_error']]})
                        else:
                            search_space_bs.update(
                                {'criterion': ['categorical', ['squared_error', 'poisson']]})
                    elif model_type == "RF":
                        search_space_bs.update(
                            {'criterion': ['categorical', ['gini', 'entropy']]})
                    QSPRmodel.bayesOptimization(search_space_bs, args.n_trials)

                # initialize models from saved or default parameters

                if args.model_evaluation:
                    QSPRmodel.evaluate()

                if args.save_model:
                    if (model_type == 'DNN') and not (args.model_evaluation):
                        log.warning(
                            "Fit skipped: DNN can only be fitted after cross-validation for determining \
                                     optimal number of epochs to stop training")
                    else:
                        QSPRmodel.fit()


if __name__ == '__main__':
    args = QSPRArgParser()

    # Set random seeds
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    os.environ['TF_DETERMINISTIC_OPS'] = str(args.random_state)

    # Backup files
    tasks = ['REG' if reg == True else 'CLS' for reg in args.regression]
    file_prefixes = [
        f'{alg}_{task}_{property}'
        for alg in args.model_types
        for task in tasks for property in args.properties]
    backup_msg = backUpFiles(
        args.base_dir,
        'qspr/models',
        tuple(file_prefixes),
        cp_suffix='_params')

    if not os.path.exists(f'{args.base_dir}/qspr/models'):
        os.makedirs(f'{args.base_dir}/qspr/models')

    logSettings = enable_file_logger(
        os.path.join(args.base_dir, 'qspr/models'),
        'QSPRmodel.log', args.debug, __name__,
        commit_hash(os.path.dirname(os.path.realpath(__file__)))
        if not args.no_git else None, vars(args),
        disable_existing_loggers=False)

    log = logSettings.log
    log.info(backup_msg)

    # Add optuna logging
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    # Stop showing logs in sys.stderr.
    optuna.logging.disable_default_handler()
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # Create json log file with used commandline arguments
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(f'{args.base_dir}/qspr/models/QSPRmodel.json', 'w') as f:
        json.dump(vars(args), f)

    # Optimize, evaluate and train estimators according to QSPR arguments
    log.info('QSPR modelling started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    QSPR_modelling(args)

    log.info('QSPR modelling completed: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
