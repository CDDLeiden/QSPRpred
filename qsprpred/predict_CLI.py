#!/usr/bin/env python

import argparse
import json
import os
import os.path
import random
import sys

import numpy as np
import optuna
import pandas as pd
import torch
from qsprpred.data.utils.smiles_standardization import (
    chembl_smi_standardizer,
    sanitize_smiles,
)
from qsprpred.logs.utils import commit_hash, enable_file_logger
from qsprpred.models.models import QSPRDNN, QSPRsklearn
from rdkit import Chem


def QSPRArgParser(txt=None):
    """Define and read command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # base arguments
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input file")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-ran', '--random_state', type=int, default=1, help="Seed for the random state")
    parser.add_argument('-i', '--input', type=str, default='dataset.tsv',
                        help="tsv file name that contains SMILES")
    parser.add_argument('-o', '--output', type=str, default='predictions',
                        help="tsv output file name that contains SMILES and predictions")
    parser.add_argument('-ncpu', '--ncpu', type=int, default=8,
                        help="Number of CPUs")
    parser.add_argument('-gpus', '--gpus', nargs="*", default=['0'],
                        help="List of GPUs")

    # model predictions arguments
    parser.add_argument('-sm', '--smilescol', type=str, default='SMILES', help="Name of the column in the dataset\
                        containing the smiles.")
    parser.add_argument('-pr', '--properties', type=str, nargs='+', action='append',
                        help="properties to be predicted identifiers. Add this argument \
                              for each trained model in the base_dir/qspr/data directory \
                              you want to use to make predicictions with \
                              e.g. for one multi-task model for CL and Fu and one single task for CL do:\
                              -pr CL Fu -pr CL")
    parser.add_argument('-r', '--regression', type=str, default=None,
                        help="If True, only regression model, if False, only classification, default both")
    parser.add_argument('-m', '--model_types', type=str, nargs='*',
                        choices=['RF', 'XGB', 'SVM', 'PLS', 'NB', 'KNN', 'DNN'],
                        default=['RF', 'XGB', 'SVM', 'PLS', 'NB', 'KNN', 'DNN'],
                        help="Modeltype, defaults to run all modeltypes, choose from: 'RF', 'XGB', 'DNN', 'SVM',\
                             'PLS' (only with REG), 'NB' (only with CLS) 'KNN'")
    parser.add_argument('-np', '--no_preprocessing', action='store_true',
                        help="If included do not standardize and sanitize SMILES.")

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


def QSPR_predict(args):
    """Make predictions with pre-trained QSPR models for a set of smiles."""
    try:
        df = pd.read_csv(f'{args.base_dir}/data/{args.input}', sep='\t')
    except FileNotFoundError:
        log.error(f'Dataset file ({args.base_dir}/data/{args.input}) not found')
        sys.exit()

    # standardize and sanitize smiles
    smiles_list = df[args.smilescol]

    # drop invalid smiles
    smiles_list = []
    mols = []
    for smiles in df[args.smilescol]:
        try:
            if not args.no_preprocessing:
                smiles = sanitize_smiles(chembl_smi_standardizer(smiles)[0])
            mol = Chem.MolFromSmiles(smiles)
            smiles_list.append(smiles)
            if mol:
                mols.append(mol)
            else:
                raise Exception
        except BaseException:
            log.info(
                f"Dropped invalid Smiles: {smiles}"
            )

    results = {"SMILES": smiles_list}
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    for reg in args.regression:
        reg_abbr = 'REGRESSION' if reg else 'CLASSIFICATION'
        for property in args.properties:
            for model_type in args.model_types:
                print(model_type)
                log.info(f'Model: {model_type} {reg_abbr} {property[0]}')

                metadata_path = f'{args.base_dir}/qspr/models/{model_type}_{reg_abbr}/{model_type}_{reg_abbr}_meta.json'
                if not os.path.exists(metadata_path):
                    log.warning(f"{metadata_path} does not exist. Model skipped.")
                    continue

                if "DNN" == model_type:
                    predictor = QSPRDNN.fromFile(metadata_path)
                else:
                    predictor = QSPRsklearn.fromFile(metadata_path)
                predictions = predictor.predictMols(smiles_list)
                results.update({f"preds_{model_type}_{reg_abbr}_{property[0]}": predictions})

    pred_path = f"{args.base_dir}/qspr/predictions/{args.output}.tsv"
    pd.DataFrame(results).to_csv(pred_path, sep="\t", index=False)
    log.info(f"Predictions saved to {pred_path}")


if __name__ == '__main__':
    args = QSPRArgParser()

    # Set random seeds
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    os.environ['TF_DETERMINISTIC_OPS'] = str(args.random_state)

    # Backup files
    tasks = ['REG' if reg == True else 'CLS' for reg in args.regression]
    file_prefixes = [f'{alg}_{task}_{property}' for alg in args.model_types for task in tasks
                     for property in args.properties]
    #backup_msg = backUpFiles(args.base_dir, 'qspr/predictions', tuple(file_prefixes), cp_suffix='_params')

    if not os.path.exists(f'{args.base_dir}/qspr/predictions'):
        os.makedirs(f'{args.base_dir}/qspr/predictions')

    logSettings = enable_file_logger(
        os.path.join(args.base_dir, 'qspr/predictions'),
        'QSPRpredict.log',
        args.debug,
        __name__,
        commit_hash(os.path.dirname(os.path.realpath(__file__))) if not args.no_git else None,
        vars(args),
        disable_existing_loggers=False
    )

    log = logSettings.log
    # log.info(backup_msg)

    # Add optuna logging
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # Create json log file with used commandline arguments
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(f'{args.base_dir}/qspr/predictions/QSPRpredict.json', 'w') as f:
        json.dump(vars(args), f)

    # Optimize, evaluate and train estimators according to QSPR arguments
    QSPR_predict(args)
