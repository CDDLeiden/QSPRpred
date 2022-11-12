#!/usr/bin/env python

import argparse
import json
import os
import os.path
import pickle
import random
import sys

import numpy as np
import optuna
import pandas as pd
import torch
from qsprpred.data.data import QSPRDataset
from qsprpred.data.utils.datafilters import papyrusLowQualityFilter
from qsprpred.data.utils.datasplitters import randomsplit, scaffoldsplit, temporalsplit
from qsprpred.data.utils.descriptorcalculator import (
    descriptorsCalculator,
    get_descriptor,
)
from qsprpred.data.utils.descriptorsets import (
    DrugExPhyschem,
    Mordred,
    MorganFP,
    rdkit_descs,
)
from qsprpred.data.utils.featurefilters import (
    BorutaFilter,
    highCorrelationFilter,
    lowVarianceFilter,
)
from qsprpred.logs.utils import backUpFiles, commit_hash, enable_file_logger
from qsprpred.models.models import QSPRDNN, QSPRModel, QSPRsklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor


def QSPRArgParser(txt=None):
    """Define and read command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #base arguments
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-ran', '--random_state', type=int, default=1, help="Seed for the random state")
    parser.add_argument('-i', '--input', type=str, default='dataset.tsv',
                        help="tsv file name that contains SMILES and property value column")
    parser.add_argument('-ncpu', '--ncpu', type=int, default=8,
                        help="Number of CPUs")
    parser.add_argument('-gpus', '--gpus', nargs="*",  default=['0'],
                        help="List of GPUs")
    
    # model target arguments
    parser.add_argument('-sm', '--smilescol', type=str, default='SMILES', help="Name of the column in the dataset\
                        containing the smiles.")
    parser.add_argument('-pr', '--properties', type=str, nargs='+', action='append',
                        help="properties to be predicted identifiers. Add this argument for each model to be trained \
                              e.g. for one multi-task model for CL and Fu and one single task for CL do:\
                              -pr CL Fu -pr CL")

    # model type arguments
    parser.add_argument('-m', '--model_types', type=str, nargs='*', choices=['RF', 'XGB', 'SVM', 'PLS', 'NB', 'KNN', 'DNN'],
                        default=['RF', 'XGB', 'SVM', 'PLS', 'NB', 'KNN', 'DNN'],
                        help="Modeltype, defaults to run all modeltypes, choose from: 'RF', 'XGB', 'DNN', 'SVM',\
                             'PLS' (only with REG), 'NB' (only with CLS) 'KNN'") 
    parser.add_argument('-r', '--regression', type=str, default=None,
                        help="If True, only regression model, if False, only classification, default both")
    parser.add_argument('-th', '--threshold', type=json.loads,
                        help='Threshold on predicted property for classification. if len th larger than 1,\
                              these values will used for multiclass classification (then lower and upper boundary \
                              need to be included, e.g. for three classes [0,1],[1,2],[2,3]: 0,1,2,3)\
                              This needs to be given for each property included in any of the models as follows, e.g.\
                              -th \'{"CL":[6.5],"fu":[0,1,2,3,4]}\'. Note. no spaces and surround by single quotes')

    # model settings
    parser.add_argument('-p', '--parameters', type=str, default=None,
                        help="file name of json file with non-default parameter settings (base_dir/qsprmodels/[-p]_params.json). NB. If json file with name \
                             {model_type}_{REG/CLS}_{property}_params.json) present in qsprmodels folder those settings will also be used, \
                             but if the same parameter is present in both files the settings from (base_dir/[-p]_params.json) will be used.")
    parser.add_argument('-pat', '--patience', type=int, default=50,
                        help="for DNN, number of epochs for early stopping")
    parser.add_argument('-tol', '--tolerance', type=float, default=0.01,
                        help="for DNN, minimum absolute change of loss to count as progress")       

    # Data pre-processing arguments
    parser.add_argument('-lq', "--low_quality", action='store_true', help="If lq, than low quality data will be \
                        should be a column 'Quality' where all 'Low' will be removed")
    parser.add_argument('-lt', '--log_transform', type=json.loads,
                        help='For each property if its values need to be log-tranformed. This arg only has an effect \
                              when mode is regression, otherwise will be ignored!\
                              This needs to be given for each property included in any of the models as follows, e.g.\
                              -th \'{"CL":True,"fu":False}\'. Note. no spaces and surround by single quotes')

    # Data set split arguments
    parser.add_argument('-sp', '--split', type=str, choices=['random', 'time', 'scaffold'], default='random')
    parser.add_argument('-sf', '--split_fraction', type=float, default=0.1,
                        help="Fraction of the dataset used as test set. Used for randomsplit and scaffoldsplit")
    parser.add_argument('-st', '--split_time', type=float, default=2015,
                        help="Temporal split limit. Used for temporal split.")
    parser.add_argument('-stc', '--split_timecolumn', type=str, default="Year",
                        help="Temporal split time column. Used for temporal split.")

    # features to calculate
    parser.add_argument('-fe', '--features', type=str, choices=['Morgan', 'RDkit', 'Mordred', 'DrugEx'], nargs='*')

    # feature filters
    parser.add_argument('-lv', '--low_variability', type=float, default=None, help="low variability threshold\
                        for feature removal.")
    parser.add_argument('-hc', '--high_correlation', type=float, default=None, help="high correlation threshold\
                        for feature removal.")
    parser.add_argument('-bf', '--boruta_filter', action='store_true', help="boruta filter with random forest")

    # model training procedure
    parser.add_argument('-s', '--save_model', action='store_true',
                        help="If included then the model will be trained on all data and saved")   
    parser.add_argument('-o', '--optimization', type=str, default=None,
                        help="Hyperparameter optimization, if 'None' no optimization, if 'grid' gridsearch, if 'bayes' bayesian optimization")
    parser.add_argument('-ss', '--search_space', type=str, default=None,
                        help="search_space hyperparameter optimization json file location (base_dir/[name].json), \
                              if None default qsprpred.models.search_space.json used")                  
    parser.add_argument('-nt', '--n_trials', type=int, default=20, help="number of trials for bayes optimization")
    parser.add_argument('-me', '--model_evaluation', action='store_true',
                        help='If on, model evaluation through cross validation and independent test set is performed.')

    # other
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved")
    
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


def QSPR(args):
    """Optimize, evaluate and train estimators.""" 
    if not os.path.exists(args.base_dir + '/qsprmodels'):
        os.makedirs(args.base_dir + '/qsprmodels') 
    
    # read in file with specified parameters for model fitting
    parameters=None
    if args.parameters:
            try:
                with open(f'{args.base_dir}/qsprmodels/{args.parameters}_params.json') as json_file:
                    par_dicts = np.array(json.load(json_file))
            except:
                log.error("Parameter settings file (%s/%s.json) not found." % args.base_dir/args.parameters)
                sys.exit()

    if args.optimization in ['grid', 'bayes']:
        if args.search_space:
            grid_params = QSPRModel.loadParamsGrid(f'{args.base_dir}/{args.search_space}.json', args.optimization, args.model_types)
        else:
            grid_params = QSPRModel.loadParamsGrid(None, args.optimization, args.model_types)

    for reg in args.regression:
        reg_abbr = 'REG' if reg else 'CLS'
        for property in args.properties:
            log.info(f"Property: {property[0]}")
            try:
                df = pd.read_csv(f'{args.base_dir}/data/{args.input}', sep='\t')
            except:
                log.error(f'Dataset file ({args.base_dir}/data/{args.input}) not found')
                sys.exit()
        
            #prepare dataset for training QSPR model
            th = args.threshold[property[0]] if args.threshold else {}
            log_transform = args.log_transform[property[0]] if args.log_transform else {}
            mydataset = QSPRDataset(df, smilescol=args.smilescol, property=property[0],
                                    reg=reg, th=th, log=log_transform)
            
            # data filters
            datafilters = []
            if args.low_quality:
                datafilters.append(papyrusLowQualityFilter())

            # data splitter
            if args.split == 'scaffold':
                split=scaffoldsplit(test_fraction=args.split_fraction)
            elif args.split == 'temporal':
                split=temporalsplit(timesplit=args.split_time, timecol=args.split_timecolumn)
            else:
                split=randomsplit(test_fraction=args.split_fraction)

            # feature calculator
            descriptorsets = []
            if 'Morgan' in args.features:
                descriptorsets.append(MorganFP(3, nBits=2048))
            if 'RDkit' in args.features:
                descriptorsets.append(rdkit_descs())
            if 'Mordred' in args.features:
                descriptorsets.append(Mordred())
            if 'DrugEx' in args.features:
                descriptorsets.append(DrugExPhyschem())

            # feature filters
            featurefilters=[]
            if args.low_variability:
                featurefilters.append(lowVarianceFilter(th=args.low_variability))
            if args.high_correlation:
                featurefilters.append(highCorrelationFilter(th=args.high_correlation))
            if args.boruta_filter:
                if args.regression:
                    featurefilters.append(BorutaFilter())
                else:
                     featurefilters.append(BorutaFilter(estimator = RandomForestClassifier(n_jobs=5)))


            mydataset.prepareDataset(fname=f"{args.base_dir}/qsprmodels/{reg_abbr}_{property[0]}_DescCalc.json", 
                                     feature_calculators=descriptorsCalculator(descriptorsets),
                                     datafilters=datafilters, split=split, featurefilters=featurefilters)

            # save dataset object
            mydataset.folds = None
            pickle.dump(mydataset, open(f'{args.base_dir}/qsprmodels/{property[0]}_{reg_abbr}_QSPRdata.pkg', 'bw'))
            mydataset.createFolds()

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

                # Create QSPR model object
                if model_type == 'DNN':
                    QSPRmodel = QSPRDNN(base_dir = args.base_dir, data=mydataset, parameters=parameters, gpus=args.gpus,
                                        patience = args.patience, tol=args.tolerance)
                else:
                    QSPRmodel = QSPRsklearn(args.base_dir, data=mydataset, alg=alg_dict[model_type],
                                            alg_name=model_type, n_jobs=args.ncpu, parameters=parameters)

                # if desired run parameter optimization
                if args.optimization == 'grid':
                    search_space_gs = grid_params[grid_params[:,0] == model_type,1][0]
                    log.info(search_space_gs)
                    QSPRmodel.gridSearch(search_space_gs)
                elif args.optimization == 'bayes':
                    search_space_bs = grid_params[grid_params[:,0] == model_type,1][0]
                    log.info(search_space_bs)
                    if reg and model_type == "RF":
                        if log_transform:
                            search_space_bs.update({'criterion' : ['categorical', ['squared_error']]})
                        else:
                            search_space_bs.update({'criterion' : ['categorical', ['squared_error', 'poisson']]})
                    elif model_type == "RF":
                        search_space_bs.update({'criterion' : ['categorical', ['gini', 'entropy']]})
                    QSPRmodel.bayesOptimization(search_space_bs, args.n_trials)
                
                # initialize models from saved or default parameters

                if args.model_evaluation:
                    QSPRmodel.evaluate()

                if args.save_model:
                    if (model_type == 'DNN') and not (args.model_evaluation):
                        log.warning("Fit skipped: DNN can only be fitted after cross-validation for determining \
                                     optimal number of epochs to stop training")
                    else:
                        QSPRmodel.fit()
         
if __name__ == '__main__':
    args = QSPRArgParser()

    #Set random seeds
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    os.environ['TF_DETERMINISTIC_OPS'] = str(args.random_state)

    # Backup files
    tasks = [ 'REG' if reg == True else 'CLS' for reg in args.regression ]
    file_prefixes = [ f'{alg}_{task}_{property}' for alg in args.model_types for task in tasks for property in args.properties]
    backup_msg = backUpFiles(args.base_dir, 'qsprmodels', tuple(file_prefixes), cp_suffix='_params')

    if not os.path.exists(f'{args.base_dir}/qsprmodels'):
        os.mkdir(f'{args.base_dir}/qsprmodels')

    logSettings = enable_file_logger(
        os.path.join(args.base_dir, 'qsprmodels'),
        'QSPR.log',
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
    with open(f'{args.base_dir}/qsprmodels/QSPR.json', 'w') as f:
        json.dump(vars(args), f)
    
    #Optimize, evaluate and train estimators according to QSPR arguments
    QSPR(args)