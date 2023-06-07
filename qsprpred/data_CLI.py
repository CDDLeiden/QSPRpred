#!/usr/bin/env python

import argparse
import json
import os
import os.path
import random
import sys
from datetime import datetime
from importlib.util import find_spec

import numpy as np
import optuna
import pandas as pd
from qsprpred.data.data import QSPRDataset
from qsprpred.data.utils.datafilters import papyrusLowQualityFilter
from qsprpred.data.utils.datasplitters import (
    ManualSplit,
    randomsplit,
    scaffoldsplit,
    temporalsplit,
)
from qsprpred.data.utils.descriptorcalculator import MoleculeDescriptorsCalculator
from qsprpred.data.utils.descriptorsets import (
    DrugExPhyschem,
    FingerprintSet,
    PredictorDesc,
    rdkit_descs,
)
from qsprpred.extra.data.utils.descriptorsets import Mordred, Mold2, PaDEL, ExtendedValenceSignature
from qsprpred.data.utils.featurefilters import (
    BorutaFilter,
    highCorrelationFilter,
    lowVarianceFilter,
)
from qsprpred.data.utils.scaffolds import Murcko
from qsprpred.deep.models.models import QSPRDNN
from qsprpred.extra.data.utils.descriptorsets import Mold2, Mordred, PaDEL
from qsprpred.logs.utils import backUpFiles, commit_hash, enable_file_logger
from qsprpred.models.models import QSPRsklearn
from qsprpred.models.tasks import TargetTasks
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def QSPRArgParser(txt=None):
    """Define and read command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # base arguments
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-de', '--debug', action='store_true')
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
    parser.add_argument('-im', '--imputation', type=str, choices=['mean', 'median', 'most_frequent'])

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
    parser.add_argument('-sp', '--split', type=str, choices=['random', 'time', 'scaffold', 'manual'], default='random',
                        help="Split type (Randomsplit, scaffoldsplit, manualsplit and temporalsplit)."
                        "If manualsplit is chosen, a column 'datasplit' with values 'train' and 'test' is required."
                        "If scaffoldsplit or temporalsplit is chosen, you can specify the split_fraction with -sf."
                        "If temporalsplit is chosen, you can specify the split_time with -st and the split_timecolumn with -stc.")
    parser.add_argument('-sf', '--split_fraction', type=float, default=0.1,
                        help="Fraction of the dataset used as test set. Used for randomsplit and scaffoldsplit")
    parser.add_argument('-st', '--split_time', type=float, default=2015,
                        help="Temporal split limit. Used for temporal split.")
    parser.add_argument('-stc', '--split_timecolumn', type=str, default="Year",
                        help="Temporal split time column. Used for temporal split.")

    # features to calculate
    parser.add_argument('-fe', '--features', type=str, choices=['Morgan', 'RDkit', 'Mordred', 'Mold2',
                                                                'PaDEL', 'DrugEx', 'Signature'],
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
    parser.add_argument('-fv', '--fill_value', type=float, default=np.nan,
                        help="Fill value for missing values in the calculated features")
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

    return args


def QSPR_dataprep(args):
    """Optimize, evaluate and train estimators."""
    if not os.path.exists(args.base_dir + '/qspr/data'):
        os.makedirs(args.base_dir + '/qspr/data')

    for reg in args.regression:
        for props in args.properties:
            props_name = '_'.join(props)
            log.info(f"Property: {props_name}")
            try:
                df = pd.read_csv(f'{args.base_dir}/data/{args.input}', sep='\t')
            except BaseException:
                log.error(f'Dataset file ({args.base_dir}/data/{args.input}) not found')
                sys.exit()

            # prepare dataset for training QSPR model
            target_props = []
            for prop in props:
                th = args.threshold[prop] if args.threshold else None
                if reg:
                    task = TargetTasks.REGRESSION
                else:
                    if th is None:
                        task = TargetTasks.MULTICLASS if len(
                            df[prop].dropna().unique()) > 2 else TargetTasks.SINGLECLASS
                        th = 'precomputed'
                    else:
                        task = TargetTasks.SINGLECLASS if len(th) == 1 else TargetTasks.MULTICLASS
                if task == TargetTasks.REGRESSION and th:
                    log.warning("Threshold argument specified with regression. Threshold will be ignored.")
                    th = None
                log_transform = np.log if args.log_transform and args.log_transform[prop] else None
                target_props.append({"name": prop, "task": task, "th": th, "transformer": log_transform})

            # missing value imputation
            if args.imputation is not None:
                if args.imputation == 'mean':
                    imputer = SimpleImputer(strategy='mean')
                elif args.imputation == 'median':
                    imputer = SimpleImputer(strategy='median')
                elif args.imputation == 'most_frequent':
                    imputer = SimpleImputer(strategy='most_frequent')
                else:
                    sys.exit("invalid impute arg given")

            mydataset = QSPRDataset(
                f"{props_name}_{task}",
                target_props=target_props,
                df=df,
                smilescol=args.smilescol,
                n_jobs=args.ncpu,
                store_dir=f"{args.base_dir}/qspr/data/",
                overwrite=True,
                target_imputer=imputer if args.imputation is not None else None)

            # data filters
            datafilters = []
            if args.low_quality:
                datafilters.append(papyrusLowQualityFilter())

            # data splitter
            if args.split == 'scaffold':
                split = scaffoldsplit(test_fraction=args.split_fraction, scaffold=Murcko(), dataset=mydataset)
            elif args.split == 'time':
                split = temporalsplit(timesplit=args.split_time, timeprop=args.split_timecolumn, dataset=mydataset)
            elif args.split == 'manual':
                if "datasplit" not in df.columns:
                    raise ValueError("No datasplit column found in dataset."
                                     "Please add a column 'datasplit' with values 'train' and 'test' if using manual split.")
                split = ManualSplit(splitcol=df["datasplit"], trainval="train", testval="test")
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
            if 'Signature' in args.features:
                descriptorsets.append(ExtendedValenceSignature(depth=1))
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
            mydataset.prepareDataset(feature_calculators=[MoleculeDescriptorsCalculator(descriptorsets)],
                                     datafilters=datafilters, split=split, feature_filters=featurefilters,
                                     feature_standardizer=StandardScaler(), feature_fill_value=0.0)

            # save dataset files and fingerprints
            mydataset.save()


if __name__ == '__main__':
    args = QSPRArgParser()

    # Set random seeds
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    if find_spec('torch') is not None:
        import torch
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
