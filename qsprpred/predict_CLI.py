#!/usr/bin/env python

import argparse
import json
import os
import os.path
import sys

import numpy as np
import optuna
import pandas as pd

from .logs.utils import backup_files, enable_file_logger
from .models.model import QSPRModel


def QSPRArgParser(txt=None):
    """Define and read command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # base arguments
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./predictions.tsv",
        help="Output path to save results",
    )
    parser.add_argument(
        "-sb",
        "--skip_backup",
        action="store_true",
        help="Skip backup of files. WARNING: this may overwrite "
        "previous results, use with caution.",
    )
    parser.add_argument("-de", "--debug", action="store_true")
    parser.add_argument(
        "-ran", "--random_state", type=int, default=1, help="Seed for the random state"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="./dataset.tsv",
        help="path to tsv file name that contains SMILES",
    )
    parser.add_argument(
        "-sm",
        "--smiles_col",
        type=str,
        default="SMILES",
        help="SMILES column name in input file.",
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
        "--model_paths",
        nargs="*",
        help="Path to model meta file for each model to be used.",
    )

    # other
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
        df = pd.read_csv(args.input, sep="\t")
    except FileNotFoundError:
        log.error(f"Dataset file ({args.input}) not found")
        sys.exit()

    smiles_list = df[args.smiles_col].tolist()

    results = {"SMILES": smiles_list}
    for model_path in args.model_paths:
        if not os.path.exists(model_path):
            log.warning(f"{model_path} does not exist. Model skipped.")
            continue

        predictor = QSPRModel.fromFile(model_path)

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
                                f"preds_{predictor.name}_{target.name}_class_{i}": predictions[
                                    idx
                                ][
                                    :, i
                                ].flatten()
                            }
                        )
                else:
                    for i in range(predictions.shape[1]):
                        results.update(
                            {
                                f"preds_{predictor.name}_{target.name}_class_{i}": predictions[
                                    :, i
                                ].flatten()
                            }
                        )
            else:
                results.update(
                    {
                        f"preds_{predictor.name}_{target.name}": predictions[
                            :, idx
                        ].flatten()
                    }
                )

    pd.DataFrame(results).to_csv(args.output_path, sep="\t", index=False)
    log.info(f"Predictions saved to {args.output_path}")


if __name__ == "__main__":
    args = QSPRArgParser()

    # Backup files
    if not args.skip_backup:
        backup_msg = backup_files(
            os.path.dirname(args.output_path),
            (os.path.basename(args.output_path)),
            cp_suffix="_params",
        )

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    logSettings = enable_file_logger(
        os.path.join(os.path.dirname(args.output_path)),
        "QSPRpredict.log",
        args.debug,
        __name__,
        vars(args),
        disable_existing_loggers=False,
    )

    log = logSettings.log
    if not args.skip_backup:
        log.info(backup_msg)

    # Add optuna logging
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # Create json log file with used commandline arguments
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(f"{os.path.dirname(args.output_path)}/QSPRpredict.json", "w") as f:
        json.dump(vars(args), f)

    # Optimize, evaluate and train estimators according to QSPR arguments
    QSPR_predict(args)
