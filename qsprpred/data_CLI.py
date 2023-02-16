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
import pandas as pd
import torch
from qsprpred.data.data import QSPRDataset
from qsprpred.data.utils.datafilters import papyrusLowQualityFilter
from qsprpred.data.utils.datasplitters import randomsplit, scaffoldsplit, temporalsplit
from qsprpred.data.utils.descriptorcalculator import DescriptorsCalculator
from qsprpred.data.utils.descriptorsets import (
    DrugExPhyschem,
    FingerprintSet,
    Mold2,
    Mordred,
    PaDEL,
    PredictorDesc,
    rdkit_descs,
)
from qsprpred.data.utils.featurefilters import (
    BorutaFilter,
    highCorrelationFilter,
    lowVarianceFilter,
)
from qsprpred.data.utils.scaffolds import Murcko
from qsprpred.logs.utils import backUpFiles, commit_hash, enable_file_logger
from qsprpred.models.models import QSPRDNN, QSPRsklearn
from qsprpred.models.tasks import ModelTasks
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def QSPRArgParser(txt=None):
    """Define and read command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # base arguments
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-ran', '--random_state', type=int, default=1, help="Seed for the random state")
    parser.add_argument('-i', '--input', type=str, default='dataset.tsv',
                        help="tsv file name that contains SMILES and property value column")
    parser.add_argument('-ncpu', '--ncpu', type=int, default=8,
                        help="Number of CPUs")

    # model target arguments
    parser.add_argument('-sm', '--smilescol', type=str, default='SMILES', help="Name of the column in the dataset\
                        containing the smiles.")
    parser.add_argument('-pr', '--properties', type=str, nargs='+', action='append',
                        help="properties to be predicted identifiers. Add this argument for each model to be trained \
                              e.g. for one multi-task model for CL and Fu and one single task for CL do:\
                              -pr CL Fu -pr CL")

    # model type arguments
    parser.add_argument('-r', '--regression', type=str, default=None,
                        help="If True, only regression model, if False, only classification, default both")
    parser.add_argument('-th', '--threshold', type=json.loads,
                        help='Threshold on predicted property for classification. if len th larger than 1,\
                              these values will used for multiclass classification (then lower and upper boundary \
                              need to be included, e.g. for three classes [0,1],[1,2],[2,3]: 0,1,2,3)\
                              This needs to be given for each property included in any of the models as follows, e.g.\
                              -th \'{"CL":[6.5],"fu":[0,1,2,3,4]}\'. Note. no spaces and surround by single quotes')

    # Data pre-processing arguments
    parser.add_argument('-lq', "--low_quality", action='store_true', help="If lq, than low quality data will be \
                        should be a column 'Quality' where all 'Low' will be removed")
    parser.add_argument('-lt', '--log_transform', type=json.loads,
                        help='For each property if its values need to be log-tranformed. This arg only has an effect \
                              when mode is regression, otherwise will be ignored!\
                              This needs to be given for each property included in any of the models as follows, e.g.\
                              -lt \'{"CL":True,"fu":False}\'. Note. no spaces and surround by single quotes')

    # Data set split arguments
    parser.add_argument('-sp', '--split', type=str, choices=['random', 'time', 'scaffold'], default='random')
    parser.add_argument('-sf', '--split_fraction', type=float, default=0.1,
                        help="Fraction of the dataset used as test set. Used for randomsplit and scaffoldsplit")
    parser.add_argument('-st', '--split_time', type=float, default=2015,
                        help="Temporal split limit. Used for temporal split.")
    parser.add_argument('-stc', '--split_timecolumn', type=str, default="Year",
                        help="Temporal split time column. Used for temporal split.")

    # features to calculate
    parser.add_argument('-fe', '--features', type=str, choices=['Morgan', 'RDkit', 'Mordred', 'Mold2',
                                                                'PaDEL', 'DrugEx'],
                        nargs='*')
    parser.add_argument('-pd', '--predictor_descs', type=str, nargs='+',
                        help="It is also possible to use a QSPRpred model(s) as molecular feature(s). Give\
                        the path(s) to the metadata of the model(s) relative to the base_directory.")

    # feature filters
    parser.add_argument('-lv', '--low_variability', type=float, default=None, help="low variability threshold\
                        for feature removal.")
    parser.add_argument('-hc', '--high_correlation', type=float, default=None, help="high correlation threshold\
                        for feature removal.")
    parser.add_argument('-bf', '--boruta_filter', action='store_true', help="boruta filter with random forest")

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


def QSPR_dataprep(args):
    """Optimize, evaluate and train estimators."""
    if not os.path.exists(args.base_dir + '/qspr/data'):
        os.makedirs(args.base_dir + '/qspr/data')

    for reg in args.regression:
        task = ModelTasks.REGRESSION if reg else ModelTasks.CLASSIFICATION
        reg_abbr = 'REGRESSION' if reg else 'CLASSIFICATION'
        for property in args.properties:
            log.info(f"Property: {property[0]} {reg_abbr}")
            try:
                df = pd.read_csv(f'{args.base_dir}/data/{args.input}', sep='\t')
            except BaseException:
                log.error(f'Dataset file ({args.base_dir}/data/{args.input}) not found')
                sys.exit()

            # prepare dataset for training QSPR model
            th = args.threshold[property[0]] if args.threshold else None
            if task == ModelTasks.REGRESSION and th:
                log.warning("Threshold argument specified with regression. Threshold will be ignored.")
                th = None
            log_transform = np.log if args.log_transform and args.log_transform[property[0]] else None
            mydataset = QSPRDataset(
                f"{reg_abbr}_{property[0]}",
                target_prop=property[0],
                df=df,
                smilescol=args.smilescol,
                task=task,
                th=th,
                n_jobs=args.ncpu,
                target_transformer=log_transform,
                store_dir=f"{args.base_dir}/qspr/data/",
                overwrite=True)

            # data filters
            datafilters = []
            if args.low_quality:
                datafilters.append(papyrusLowQualityFilter())

            # data splitter
            if args.split == 'scaffold':
                split = scaffoldsplit(test_fraction=args.split_fraction, scaffold=Murcko(), dataset=mydataset)
            elif args.split == 'temporal':
                split = temporalsplit(timesplit=args.split_time, timeprop=args.split_timecolumn, dataset=mydataset)
            else:
                split = randomsplit(test_fraction=args.split_fraction)

            # feature calculator
            descriptorsets = []
            if 'Morgan' in args.features:
                descriptorsets.append(FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=2048))
            if 'RDkit' in args.features:
                descriptorsets.append(rdkit_descs())
            if 'Mordred' in args.features:
                descriptorsets.append(Mordred())
            if 'Mold2' in args.features:
                descriptorsets.append(Mold2())
            if 'PaDEL' in args.features:
                descriptorsets.append(PaDEL())
            if 'DrugEx' in args.features:
                descriptorsets.append(DrugExPhyschem())
            if args.predictor_descs:
                for predictor_path in args.predictor_descs:
                    # load in predictor from files
                    if "DNN" in predictor_path:
                        descriptorsets.append(PredictorDesc(QSPRDNN.fromFile(predictor_path)))
                    else:
                        descriptorsets.append(PredictorDesc(QSPRsklearn.fromFile(predictor_path)))

            # feature filters
            featurefilters = []
            if args.low_variability:
                featurefilters.append(lowVarianceFilter(th=args.low_variability))
            if args.high_correlation:
                featurefilters.append(highCorrelationFilter(th=args.high_correlation))
            if args.boruta_filter:
                if args.regression:
                    featurefilters.append(BorutaFilter(estimator=RandomForestRegressor(n_jobs=args.ncpu)))
                else:
                    featurefilters.append(BorutaFilter(estimator=RandomForestClassifier(n_jobs=args.ncpu)))

            # prepare dataset for modelling
            mydataset.prepareDataset(feature_calculator=DescriptorsCalculator(descriptorsets),
                                     datafilters=datafilters, split=split, feature_filters=featurefilters,
                                     feature_standardizer=StandardScaler())

            # save dataset files and fingerprints
            mydataset.save()


if __name__ == '__main__':
    args = QSPRArgParser()

    # Set random seeds
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    os.environ['TF_DETERMINISTIC_OPS'] = str(args.random_state)

    # Backup files
    tasks = ['REG' if reg == True else 'CLS' for reg in args.regression]
    file_prefixes = [f'{property}_{task}' for task in tasks for property in args.properties]
    backup_msg = backUpFiles(
        args.base_dir,
        'qspr/data',
        tuple(file_prefixes),
        cp_suffix=[
            'calculators',
            'standardizer',
            'meta'])

    if not os.path.exists(f'{args.base_dir}/qspr/data'):
        os.makedirs(f'{args.base_dir}/qspr/data')

    logSettings = enable_file_logger(
        os.path.join(args.base_dir, 'qspr/data'),
        'QSPRdata.log',
        args.debug,
        __name__,
        commit_hash(os.path.dirname(os.path.realpath(__file__))) if not args.no_git else None,
        vars(args),
        disable_existing_loggers=False
    )

    log = logSettings.log
    log.info(backup_msg)

    # Add optuna logging
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # Create json log file with used commandline arguments
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(f'{args.base_dir}/qspr/data/QSPRdata.json', 'w') as f:
        json.dump(vars(args), f)

    # Optimize, evaluate and train estimators according to QSPR arguments
    log.info('Data preparation started: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    QSPR_dataprep(args)

    log.info('Data preparation completed: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
