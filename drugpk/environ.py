#!/usr/bin/env python

from locale import D_FMT
import os
import os.path
from pyexpat import model
import sys
import json
import random
import optuna
import argparse

import numpy as np
import pandas as pd

from xgboost import XGBRegressor, XGBClassifier

import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR

from drugpk import DEFAULT_GPUS
from drugpk.logs.utils import backUpFiles, enable_file_logger, commit_hash
from drugpk.environment.data import QSKRDataset
from drugpk.environment.models import QSKRModel, QSKRDNN, QSKRsklearn
import pickle

def EnvironmentArgParser(txt=None):
    """ 
        Define and read command line arguments
    """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-ran', '--random_state', type=int, default=1, help="Seed for the random state")
    parser.add_argument('-i', '--input', type=str, default='dataset.tsv',
                        help="tsv file name that contains SMILES, property value column")
    parser.add_argument('-m', '--model_types', type=str, nargs='*', default=['RF', 'XGB', 'SVM', 'PLS', 'NB', 'KNN'],
                        help="Modeltype, defaults to run all modeltypes, choose from: 'RF', 'XGB', 'DNN', 'SVM', 'PLS' (only with REG), 'NB' (only with CLS) 'KNN' or 'MT_DNN'") 
    parser.add_argument('-pr', '--properties', type=str, nargs='*', default=['CL'],
                        help="property to be predicted identifier") 
    parser.add_argument('-r', '--regression', type=str, default=None,
                        help="If True, only regression model, if False, only classification, default both")
    parser.add_argument('-a', '--threshold', type=float, default=6.5,
                        help="threshold on predicted property for classification")
    parser.add_argument('-n', '--test_size', type=str, default="0.1",
                        help="Random test split fraction if float is given and absolute size if int is given, used when no temporal split given.")
    parser.add_argument('-y', '--year', type=int, default=None,
                        help="Temporal split limit, default to random split (see test_size)")
    parser.add_argument('-s', '--save_model', action='store_true',
                        help="If included then the model will be trained on all data and saved")   
    parser.add_argument('-p', '--parameters', type=str, default=None,
                        help="file name of json file with non-default parameter settings (base_dir/envs/[-p]_params.json). NB. If json file with name \
                             {model_type}_{REG/CLS}_{valuecol}_params.json) present in envs folder those settings will also be used, \
                             but if the same parameter is present in both files the settings from (base_dir/[-p]_params.json) will be used.")
    parser.add_argument('-o', '--optimization', type=str, default=None,
                        help="Hyperparameter optimization, if 'None' no optimization, if 'grid' gridsearch, if 'bayes' bayesian optimization")
    parser.add_argument('-ss', '--search_space', type=str, default=None,
                        help="search_space hyperparameter optimization json file location (base_dir/[name].json), \
                              if None default drugpk.environment.search_space.json used")                  
    parser.add_argument('-nt', '--n_trials', type=int, default=20, help="number of trials for bayes optimization")
    parser.add_argument('-c', '--model_evaluation', action='store_true',
                        help='If on, model evaluation through cross validation and independent test set is performed.')
    parser.add_argument('-ncpu', '--ncpu', type=int, default=8,
                        help="Number of CPUs")
    parser.add_argument('-gpus', '--gpus', nargs="*",  default=['0'],
                        help="List of GPUs")
    parser.add_argument('-pat', '--patience', type=int, default=50,
                        help="for DNN, number of epochs for early stopping")
    parser.add_argument('-tol', '--tolerance', type=float, default=0.01,
                        help="for DNN, minimum absolute change of loss to count as progress")       
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved")
    
    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()

    # If no regression argument, does both regression and classification
    if args.regression is None: 
        args.regression = [True, False]
    elif args.regression.lower() in ['true', 'reg', 'regression']:
        args.regression = [True]
    elif args.regression.lower() in ['false', 'cls', 'classification']:
        args.regression = [False]
    else:
        sys.exit("invalid regression arg given")

    if '.' in args.test_size:
        args.test_size = float(args.test_size) 
    else: 
        args.test_size = int(args.test_size)

    return args


def Environ(args):
    """
        Optimize, evaluate and train estimators
    """
    
    if not os.path.exists(args.base_dir + '/envs'):
        os.makedirs(args.base_dir + '/envs') 
    
    # read in file with specified parameters for model fitting
    parameters=None
    if args.parameters:
            try:
                with open(f'{args.base_dir}/envs/{args.parameters}_params.json') as json_file:
                    par_dicts = np.array(json.load(json_file))
            except:
                log.error("Parameter settings file (%s/%s.json) not found." % args.base_dir/args.parameters)
                sys.exit()


    if args.optimization in ['grid', 'bayes']:
        if args.search_space:
            grid_params = QSKRModel.loadParamsGrid(f'{args.base_dir}/{args.search_space}.json', args.optimization, args.model_types)
        else:
            grid_params = QSKRModel.loadParamsGrid(None, args.optimization, args.model_types)

    for reg in args.regression:
        reg_abbr = 'REG' if reg else 'CLS'
        for property in args.properties:
            try:
                df = pd.read_csv(f'{args.base_dir}/data/{args.input}', sep='\t')
            except:
                log.error(f'Dataset file ({args.base_dir}/data/{args.input}) not found')
                sys.exit()
        
            #prepare dataset for training QSKR model
            mydataset = QSKRDataset(df, property, reg = reg, timesplit=args.year,
                                    test_size=args.test_size, th = args.threshold)
            mydataset.splitDataset()

            # save dataset object
            mydataset.folds = None
            pickle.dump(mydataset, open(f'{args.base_dir}/envs/{property}_{reg_abbr}_QSKRdata.pkg', 'bw'))
            mydataset.createFolds()
            
            for model_type in args.model_types:
                print(model_type)
                log.info(f'Model: {model_type} {reg_abbr}')

                if model_type == 'MT_DNN':
                    log.warning('MT DNN is not implemented yet')
                    continue
                elif model_type not in ['RF', 'XGB', 'DNN', 'SVM', 'PLS', 'NB', 'KNN']: 
                    log.warning(f'Model type {model_type} does not exist')
                    continue
                if model_type == 'NB' and reg:
                    log.warning("NB with regression invalid, skipped.")
                    continue
                if model_type == 'PLS' and not reg:
                    log.warning("PLS with classification invalid, skipped.")
                    continue

                if model_type != "RF":
                    mydataset.X, mydataset.X_ind = mydataset.dataStandardization(mydataset.X, mydataset.X_ind)

                if args.parameters:
                    try:
                        parameters = par_dicts[par_dicts[:,0]==model_type,1][0]
                    except:
                        log.warning(f'Model type {model_type} not in parameter file, default parameter settings used.')
                        parameters = None

                alg_dict = {
                    'RF' : RandomForestRegressor() if reg else RandomForestClassifier(),
                    'XGB': XGBRegressor(objective='reg:squarederror') if reg else \
                           XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss'),
                    'SVM': SVR() if reg else SVC(probability=True),
                    'PLS': PLSRegression(),
                    'NB' : GaussianNB(),
                    'KNN': KNeighborsRegressor() if reg else KNeighborsClassifier()
                }

                # Create QSKR model object
                if model_type == 'DNN':
                    qsKrmodel = QSKRDNN(base_dir = args.base_dir, data=mydataset, parameters=parameters, gpus=args.gpus,
                                        patience = args.patience, tol = args.tolerance)
                else:
                    qsKrmodel = QSKRsklearn(args.base_dir, data=mydataset, alg=alg_dict[model_type],
                                            alg_name=model_type, n_jobs=args.ncpu, parameters=parameters)

                # if desired run parameter optimization
                if args.optimization == 'grid':
                    search_space_gs = grid_params[grid_params[:,0] == model_type,1][0]
                    log.info(search_space_gs)
                    qsKrmodel.gridSearch(search_space_gs, args.save_model)
                elif args.optimization == 'bayes':
                    search_space_bs = grid_params[grid_params[:,0] == model_type,1][0]
                    log.info(search_space_bs)
                    if reg and model_type == "RF":
                        search_space_bs.update({'criterion' : ['categorical', ['squared_error', 'poisson']]})
                    elif model_type == "RF":
                        search_space_bs.update({'criterion' : ['categorical', ['gini', 'entropy']]})
                    qsKrmodel.bayesOptimization(search_space_bs, args.n_trials, args.save_model)
                
                # initialize models from saved or default parameters

                if args.optimization is None and args.save_model:
                    qsKrmodel.fit()
                
                if args.model_evaluation:
                    qsKrmodel.evaluate()

               
if __name__ == '__main__':
    args = EnvironmentArgParser()

    #Set random seeds
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    os.environ['TF_DETERMINISTIC_OPS'] = str(args.random_state)

    # Backup files
    tasks = [ 'REG' if reg == True else 'CLS' for reg in args.regression ]
    file_prefixes = [ f'{alg}_{task}_{property}' for alg in args.model_types for task in tasks for property in args.properties]
    backup_msg = backUpFiles(args.base_dir, 'envs', tuple(file_prefixes), cp_suffix='_params')

    if not os.path.exists(f'{args.base_dir}/envs'):
        os.mkdir(f'{args.base_dir}/envs')

    logSettings = enable_file_logger(
        os.path.join(args.base_dir, 'envs'),
        'environ.log',
        args.debug,
        __name__,
        commit_hash(os.path.dirname(os.path.realpath(__file__))) if not args.no_git else None,
        vars(args),
        disable_existing_loggers=False
    )   

    log = logSettings.log
    log.info(backup_msg)

    #Add optuna logging
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # Create json log file with used commandline arguments 
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(f'{args.base_dir}/envs/environ.json', 'w') as f:
        json.dump(vars(args), f)
    
    #Optimize, evaluate and train estimators according to environment arguments
    Environ(args)