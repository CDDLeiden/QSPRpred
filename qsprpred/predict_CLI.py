#!/usr/bin/env python

import argparse
import json
import os
import os.path
import random
import sys
from importlib.util import find_spec

import numpy as np
import optuna
import pandas as pd

from .logs.utils import backup_files, commit_hash, enable_file_logger
from .models.interfaces import QSPRModel


def QSPRArgParser(txt=None):
    """Define and read command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # base arguments
    parser.add_argument(
        "-b",
        "--base_dir",
        type=str,
        default=".",
        help="Base directory which contains a folder 'data' with input file",
    )
    parser.add_argument("-de", "--debug", action="store_true")
    parser.add_argument(
        "-ran", "--random_state", type=int, default=1, help="Seed for the random state"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="dataset.tsv",
        help="tsv file name that contains SMILES",
    )
    parser.add_argument(
        "-sm",
        "--smiles_col",
        type=str,
        default="SMILES",
        help="SMILES column name in input file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="predictions",
        help="tsv output file name that contains SMILES and predictions",
    )
    parser.add_argument("-ncpu", "--ncpu", type=int, default=8, help="Number of CPUs")
    parser.add_argument(
        "-gpus", "--gpus", nargs="*", default=["0"], help="List of GPUs"
    )
    parser.add_argument(
        "-pr",
        "--use_probas",
        action="store_true",
        help=(
            "If included use probabilities instead of predictions "
            "for classification tasks."
        ),
    )

    # model predictions arguments
    parser.add_argument(
        "-mp",
        "--metadata_paths",
        nargs="*",
        help="Path to metadata json file for each model to be used.",
    )

    # other
    parser.add_argument(
        "-ng", "--no_git", action="store_true", help="If on, git hash is not retrieved"
    )
    parser.add_argument(
        "-fv",
        "--fill_value",
        type=float,
        default=np.nan,
        help="Fill value for missing values in the calculated features",
    )

    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()

    return args


def QSPR_predict(args):
    """Make predictions with pre-trained QSPR models for a set of smiles."""
    try:
        df = pd.read_csv(f"{args.base_dir}/data/{args.input}", sep="\t")
    except FileNotFoundError:
        log.error(f"Dataset file ({args.base_dir}/data/{args.input}) not found")
        sys.exit()

    # standardize and sanitize smiles
    smiles_list = df[args.smiles_col].tolist()

    results = {"SMILES": smiles_list}
    for metadata_path in args.metadata_paths:
        metafile = f"{args.base_dir}/{metadata_path}"
        if not os.path.exists(metafile):
            log.warning(
                f"{args.base_dir}/{metadata_path} does not exist. Model skipped."
            )
            continue

        predictor = QSPRModel.fromFile(metafile)

        predictions = predictor.predictMols(
            smiles_list, use_probas=args.use_probas, fill_value=args.fill_value
        )
        # if predictions 2d array with more than 1 column, add as separate columns
        for idx, target in enumerate(predictor.targetProperties):
            if args.use_probas:
                if isinstance(predictions, list):
                    for i in range(predictions[idx].shape[1]):
                        results.update(
                            {
                                f"preds_{predictor.name}_{target.name}_class_{i}":
                                    predictions[idx][:, i].flatten()
                            }
                        )
                else:
                    for i in range(predictions.shape[1]):
                        results.update(
                            {
                                f"preds_{predictor.name}_{target.name}_class_{i}":
                                    predictions[:, i].flatten()
                            }
                        )
            else:
                results.update(
                    {
                        f"preds_{predictor.name}_{target.name}":
                            predictions[:, idx].flatten()
                    }
                )

    pred_path = f"{args.base_dir}/qspr/predictions/{args.output}.tsv"
    pd.DataFrame(results).to_csv(pred_path, sep="\t", index=False)
    log.info(f"Predictions saved to {pred_path}")


if __name__ == "__main__":
    args = QSPRArgParser()

    # Set random seeds
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    if find_spec("torch") is not None:
        import torch

        torch.manual_seed(args.random_state)
    os.environ["TF_DETERMINISTIC_OPS"] = str(args.random_state)

    # Backup files
    backup_msg = backup_files(
        args.base_dir, "qspr/predictions", tuple(args.output), cp_suffix="_params"
    )

    if not os.path.exists(f"{args.base_dir}/qspr/predictions"):
        os.makedirs(f"{args.base_dir}/qspr/predictions")

    logSettings = enable_file_logger(
        os.path.join(args.base_dir, "qspr/predictions"),
        "QSPRpredict.log",
        args.debug,
        __name__,
        commit_hash(os.path.dirname(os.path.realpath(__file__)))
        if not args.no_git else None,
        vars(args),
        disable_existing_loggers=False,
    )

    log = logSettings.log
    log.info(backup_msg)

    # Add optuna logging
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # Create json log file with used commandline arguments
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(f"{args.base_dir}/qspr/predictions/QSPRpredict.json", "w") as f:
        json.dump(vars(args), f)

    # Optimize, evaluate and train estimators according to QSPR arguments
    QSPR_predict(args)
