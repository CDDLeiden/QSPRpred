#!/usr/bin/env python

import argparse
import json
import os.path
import sys
from datetime import datetime

import numpy as np
import optuna
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from qsprpred.data.tables.qspr import QSPRDataset
from qsprpred.models.assessment.methods import CrossValAssessor, TestSetAssessor
from qsprpred.tasks import TargetTasks
from .extra.gpu.models.dnn import DNNModel
from .logs.utils import backup_files, enable_file_logger
from .models.early_stopping import EarlyStoppingMode
from .models.hyperparam_optimization import GridSearchOptimization, OptunaOptimization
from .models.scikit_learn import QSPRModel, SklearnModel


def QSPRArgParser(txt=None):
    """Define and read command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # base arguments
    parser.add_argument(
        "-dp",
        "--data_paths",
        type=str,
        nargs="*",
        help=(
            "Each data file path to be used as input for the model, "
            "e.g ./target1_MULTICLASS_df.pkl"
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help="Directory to write the output model files to",
    )
    parser.add_argument("-de", "--debug", action="store_true")
    parser.add_argument(
        "-sb",
        "--skip_backup",
        action="store_true",
        help="Skip backup of files. WARNING: this may overwrite "
        "previous results, use with caution.",
    )
    parser.add_argument(
        "-ran", "--random_state", type=int, default=1, help="Seed for the random state"
    )
    parser.add_argument("-ncpu", "--ncpu", type=int, default=8, help="Number of CPUs")
    parser.add_argument(
        "-gpus", "--gpus", nargs="*", default=["0"], help="List of GPUs"
    )
    # model arguments
    parser.add_argument(
        "-ms",
        "--model_suffix",
        type=str,
        default=None,
        help="Suffix to add to model name",
    )
    # model type arguments
    parser.add_argument(
        "-mt",
        "--model_types",
        type=str,
        nargs="*",
        choices=["RF", "XGB", "SVM", "PLS", "NB", "KNN", "DNN"],
        default=["RF", "XGB", "SVM", "PLS", "NB", "KNN", "DNN"],
        help=(
            "Modeltype, defaults to run all ModelTasks, choose from: 'RF', 'XGB', "
            "'DNN', 'SVM', 'PLS' (only with REG), 'NB' (only with CLS) 'KNN'"
        ),
    )

    # model settings
    parser.add_argument(
        "-p",
        "--parameters",
        type=str,
        default=None,
        help=(
            "file path of json file with non-default parameter settings, "
            "e.g. ./parameters.json"
        ),
    )
    parser.add_argument(
        "-sw",
        "--sample_weighing",
        action="store_true",
        help="Sets balanced class weights.",
    )
    parser.add_argument(
        "-pat",
        "--patience",
        type=int,
        default=50,
        help="for DNN, number of epochs for early stopping",
    )
    parser.add_argument(
        "-tol",
        "--tolerance",
        type=float,
        default=0.01,
        help="for DNN, minimum absolute change of loss to count as progress",
    )

    # model training procedure
    parser.add_argument(
        "-s",
        "--save_model",
        action="store_true",
        help="If included then the model will be trained on all data and saved",
    )
    parser.add_argument(
        "-op",
        "--optimization",
        type=str,
        default=None,
        help="Hyperparameter optimization, if 'None' no optimization, if 'grid' gridsearch, \
                            if 'bayes' bayesian optimization",
    )
    parser.add_argument(
        "-ss",
        "--search_space",
        type=str,
        default=None,
        help=(
            "search_space hyperparameter optimization json file location "
            "(./my_search_space.json)."
        ),
    )
    parser.add_argument(
        "-nj",
        "--n_jobs",
        type=int,
        default=1,
        help=(
            "number of parallel trials for hyperparameter optimization, "
            "warning this increase the number of CPU's used (ncpu x n_jobs)"
        ),
    )
    parser.add_argument(
        "-nt",
        "--n_trials",
        type=int,
        default=20,
        help="number of trials for bayes optimization",
    )
    parser.add_argument(
        "-me",
        "--model_evaluation",
        action="store_true",
        help=(
            "If on, model evaluation through cross validation and "
            "independent test set is performed."
        ),
    )

    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()

    return args


def QSPR_modelling(args):
    """Optimize, evaluate and train estimators."""

    # read in file with specified parameters for model fitting
    parameters = None
    if args.parameters:
        try:
            with open(f"{args.parameters}") as json_file:
                par_dicts = np.array(json.load(json_file))
        except FileNotFoundError:
            log.error(f"Parameter settings file ({args.parameters}) not found.")
            sys.exit()

    if args.optimization in ["grid", "bayes"]:
        if args.search_space:
            grid_params = QSPRModel.loadParamsGrid(
                args.search_space,
                args.optimization,
                args.model_types,
            )
        else:
            log.error(
                "Please specify a search_space file for hyperparameter optimization."
            )
            sys.exit()

    for dataset in args.datasets:
        log.info(f"Dataset: {dataset.name}")

        tasks = [prop.task for prop in dataset.targetProperties]
        if all(TargetTasks.REGRESSION == task for task in tasks):
            reg = True
        elif all(task.isClassification() for task in tasks):
            reg = False
        else:
            raise ValueError("Mixed tasks not supported")
        reg_abbr = "regression" if reg else "classification"

        for model_type in args.model_types:
            log.info(f"Model: {model_type} {reg_abbr}")

            if model_type not in ["RF", "XGB", "DNN", "SVM", "PLS", "NB", "KNN"]:
                log.warning(f"Model type {model_type} does not exist")
                continue
            if model_type == "NB" and reg:
                log.warning("NB with regression invalid, skipped.")
                continue
            if model_type == "PLS" and not reg:
                log.warning("PLS with classification invalid, skipped.")
                continue

            alg_dict = {
                "RF": RandomForestRegressor if reg else RandomForestClassifier,
                "XGB": XGBRegressor if reg else XGBClassifier,
                "SVM": SVR if reg else SVC,
                "PLS": PLSRegression,
                "NB": GaussianNB,
                "KNN": KNeighborsRegressor if reg else KNeighborsClassifier,
                "DNN": None,
            }

            # setting some default parameters
            parameters = {}
            if alg_dict[model_type] == XGBRegressor:
                parameters["objective"] = "reg:squarederror"
            elif alg_dict[model_type] == XGBClassifier:
                parameters["objective"] = "binary:logistic"
                parameters["use_label_encoder"] = False
                parameters["eval_metric"] = "logloss"
            if alg_dict[model_type] == SVC:
                parameters["probability"] = True
            if model_type not in ["NB", "PLS", "SVM", "DNN"]:
                parameters["n_jobs"] = args.ncpu

            # class_weight and scale_pos_weight are only used for RF, XGB and SVM
            if not reg:
                class_weight = "balanced" if args.sample_weighing else None
                if alg_dict[model_type] in [RandomForestClassifier, SVC]:
                    parameters["class_weight"] = class_weight
                counts = dataset.y.value_counts()
                scale_pos_weight = (
                    counts[0] / counts[1]
                    if (
                        args.sample_weighing
                        and len(tasks) == 1
                        and not tasks[0].isMultiClass()
                    )
                    else 1
                )
                if alg_dict[model_type] == XGBClassifier:
                    parameters["scale_pos_weight"] = scale_pos_weight

            # set parameters from file
            if args.parameters:
                try:
                    parameters = par_dicts[par_dicts[:, 0] == model_type, 1][0]
                except BaseException:
                    log.warning(
                        f"Model type {model_type} not in parameter file, "
                        "default parameter settings used."
                    )

            # Create QSPR model object
            model_name = (
                f"{model_type}_{dataset.name}"
                if not args.model_suffix
                else f"{model_type}_{dataset.name}_{args.model_suffix}"
            )
            if model_type == "DNN":
                qspr_model = DNNModel(
                    base_dir=f"{args.output_dir}",
                    parameters=parameters,
                    name=model_name,
                    gpus=args.gpus,
                    patience=args.patience,
                    tol=args.tolerance,
                    random_state=args.random_state,
                )
            else:
                qspr_model = SklearnModel(
                    base_dir=f"{args.output_dir}",
                    alg=alg_dict[model_type],
                    name=model_name,
                    parameters=parameters,
                    random_state=args.random_state,
                )

            # if desired run parameter optimization
            score_func = (
                "r2"
                if dataset.targetProperties[0].task.isRegression()
                else "roc_auc_ovr"
            )
            best_params = None
            if args.optimization == "grid":
                search_space_gs = grid_params[grid_params[:, 0] == model_type, 1][0]
                log.info(search_space_gs)
                gridsearcher = GridSearchOptimization(
                    model_assessor=CrossValAssessor(scoring=score_func),
                    param_grid=search_space_gs,
                )
                best_params = gridsearcher.optimize(qspr_model, dataset)
            elif args.optimization == "bayes":
                search_space_bs = grid_params[grid_params[:, 0] == model_type, 1][0]
                log.info(search_space_bs)
                if reg and model_type == "RF":
                    if dataset.y.min()[0] < 0 or dataset.y_ind.min()[0] < 0:
                        search_space_bs.update(
                            {"criterion": ["categorical", ["squared_error"]]}
                        )
                    else:
                        search_space_bs.update(
                            {"criterion": ["categorical", ["squared_error", "poisson"]]}
                        )
                elif model_type == "RF":
                    search_space_bs.update(
                        {"criterion": ["categorical", ["gini", "entropy"]]}
                    )
                bayesoptimizer = OptunaOptimization(
                    model_assessor=CrossValAssessor(scoring=score_func),
                    param_grid=search_space_bs,
                    n_trials=args.n_trials,
                    n_jobs=args.n_jobs,
                )
                best_params = bayesoptimizer.optimize(qspr_model, dataset)
            if best_params is not None:
                qspr_model.setParams(best_params)

            if args.model_evaluation:
                CrossValAssessor(mode=EarlyStoppingMode.RECORDING, scoring=score_func)(
                    qspr_model,
                    dataset,
                )
                TestSetAssessor(
                    mode=EarlyStoppingMode.NOT_RECORDING, scoring=score_func
                )(qspr_model, dataset)

            if args.save_model:
                if (model_type == "DNN") and not (args.model_evaluation):
                    log.warning(
                        "Fit skipped: DNN can only be fitted after cross-validation "
                        "for determining optimal number of epochs to stop training."
                    )
                else:
                    qspr_model.fitDataset(dataset)


if __name__ == "__main__":
    args = QSPRArgParser()

    # Backup files
    datasets = [QSPRDataset.fromFile(data_file) for data_file in args.data_paths]
    file_prefixes = [
        f"{alg}_{dataset.name}" for alg in args.model_types for dataset in datasets
    ]
    if args.model_suffix:
        file_prefixes = [f"{prefix}_{args.model_suffix}" for prefix in file_prefixes]

    if not args.skip_backup:
        backup_msg = backup_files(
            args.output_dir, tuple(file_prefixes), cp_suffix="_params"
        )

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    logSettings = enable_file_logger(
        args.output_dir,
        "QSPRmodel.log",
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
    # Stop showing logs in sys.stderr.
    optuna.logging.disable_default_handler()
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # Create json log file with used commandline arguments
    with open(f"{args.output_dir}/QSPRmodel.json", "w") as f:
        json.dump(vars(args), f)
    log.info(f"Command line arguments written to {args.output_dir}/QSPRmodel.json")

    # Optimize, evaluate and train estimators according to QSPR arguments
    log.info(
        "QSPR modelling started: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    args.datasets = datasets
    QSPR_modelling(args)

    log.info(
        "QSPR modelling completed: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
