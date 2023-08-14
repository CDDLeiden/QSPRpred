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
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from .data.data import QSPRDataset
from .deep.models.models import QSPRDNN
from .logs.utils import backup_files, enable_file_logger
from .models.assessment_methods import CrossValAssessor, TestSetAssessor
from .models.hyperparam_optimization import GridSearchOptimization, OptunaOptimization
from .models.metrics import SklearnMetric
from .models.models import QSPRModel, QSPRsklearn
from .models.tasks import TargetTasks
from .models.early_stopping import EarlyStoppingMode


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
        help="Base directory which contains a folder 'data' with input files",
    )
    parser.add_argument("-de", "--debug", action="store_true")
    parser.add_argument(
        "-ran", "--random_state", type=int, default=1, help="Seed for the random state"
    )
    parser.add_argument("-ncpu", "--ncpu", type=int, default=8, help="Number of CPUs")
    parser.add_argument(
        "-gpus", "--gpus", nargs="*", default=["0"], help="List of GPUs"
    )

    # model target arguments
    parser.add_argument(
        "-dp",
        "--data_prefixes",
        type=str,
        nargs="*",
        help=(
            "Prefix of each data file to be used as input for the model, "
            "e.g. target1_MULTICLASS for a file named target1_MULTICLASS_df.pkl"
        ),
    )
    parser.add_argument(
        "-ms", "--model_suffix", type=str, help="Suffix of the model to be saved"
    )
    parser.add_argument(
        "-pr",
        "--properties",
        type=str,
        nargs="+",
        action="append",
        help=(
            "properties to be predicted identifiers. Add this argument for each model "
            "to be trained e.g. for one multi-task model for CL and Fu and one single "
            "task for CL do -pr CL Fu -pr CL"
        ),
    )
    parser.add_argument(
        "-lt",
        "--log_transform",
        type=json.loads,
        help=(
            "For each property if its values need to be log-transformed. This arg only "
            "has an effect when mode is regression, otherwise will be ignored! This "
            "needs to be given for each property included in any of the models as "
            "follows, e.g. -lt \"{'CL':True,'fu':False}\". Note. no spaces and "
            "surround by single quotes"
        ),
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
            "file name of json file with non-default parameter settings "
            "(base_dir/qspr/models/[-p]_params.json). NB. If json file with name "
            "{model_type}_{REG/CLS}_{property}_params.json) present in "
            "qspr/models folder those settings will also be used, but if the same "
            "parameter is present in both files the settings from "
            "(base_dir/[-p]_params.json) will be used."
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
        "-o",
        "--optimization",
        type=str,
        default=None,
        help=
        "Hyperparameter optimization, if 'None' no optimization, if 'grid' gridsearch, \
                            if 'bayes' bayesian optimization",
    )
    parser.add_argument(
        "-ss",
        "--search_space",
        type=str,
        default=None,
        help=(
            "search_space hyperparameter optimization json file location "
            "(base_dir/[name].json). If None, default "
            "qsprpred.models.search_space.json used."
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

    # other
    parser.add_argument(
        "-ng", "--no_git", action="store_true", help="If on, git hash is not retrieved"
    )

    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()

    return args


def QSPR_modelling(args):
    """Optimize, evaluate and train estimators."""
    if not os.path.exists(args.base_dir + "/qspr/models"):
        os.makedirs(args.base_dir + "/qspr/models")

    # read in file with specified parameters for model fitting
    parameters = None
    if args.parameters:
        try:
            with open(f"{args.base_dir}/{args.parameters}.json") as json_file:
                par_dicts = np.array(json.load(json_file))
        except FileNotFoundError:
            log.error(
                "Parameter settings file (%s/%s.json) not found." %
                (args.base_dir, args.parameters)
            )
            sys.exit()

    if args.optimization in ["grid", "bayes"]:
        if args.search_space:
            grid_params = QSPRModel.loadParamsGrid(
                f"{args.base_dir}/{args.search_space}.json",
                args.optimization,
                args.model_types,
            )
        else:
            grid_params = QSPRModel.loadParamsGrid(
                None, args.optimization, args.model_types
            )

    for data_prefix in args.data_prefixes:
        log.info(f"Data file: {data_prefix}_df.pkl")

        mydataset = QSPRDataset.fromFile(
            f"{args.base_dir}/qspr/data/{data_prefix}_df.pkl"
        )

        tasks = [prop.task for prop in mydataset.targetProperties]
        if all(TargetTasks.REGRESSION == task for task in tasks):
            reg = True
        elif all(task.isClassification() for task in tasks):
            reg = False
        else:
            raise ValueError("Mixed tasks not supported")
        reg_abbr = "regression" if reg else "classification"

        for model_type in args.model_types:
            print(model_type)
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
                counts = mydataset.y.value_counts()
                scale_pos_weight = (
                    counts[0] / counts[1] if (
                        args.sample_weighing and len(tasks) == 1 and
                        not tasks[0].isMultiClass()
                    ) else 1
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
            if model_type == "DNN":
                QSPRmodel = QSPRDNN(
                    base_dir=f"{args.base_dir}/qspr/models/",
                    data=mydataset,
                    parameters=parameters,
                    name=f"{model_type}_{data_prefix}",
                    gpus=args.gpus,
                    patience=args.patience,
                    tol=args.tolerance,
                )
            else:
                QSPRmodel = QSPRsklearn(
                    base_dir=f"{args.base_dir}/qspr/models/",
                    data=mydataset,
                    alg=alg_dict[model_type],
                    name=f"{model_type}_{data_prefix}",
                    parameters=parameters,
                )

            # if desired run parameter optimization
            score_func = SklearnMetric.getDefaultMetric(QSPRmodel.task)
            if args.optimization == "grid":
                search_space_gs = grid_params[grid_params[:, 0] == model_type, 1][0]
                log.info(search_space_gs)
                gridsearcher = GridSearchOptimization(
                    scoring=score_func, param_grid=search_space_gs
                )
                best_params = gridsearcher.optimize(QSPRmodel)
                QSPRmodel.saveParams(best_params)
            elif args.optimization == "bayes":
                search_space_bs = grid_params[grid_params[:, 0] == model_type, 1][0]
                log.info(search_space_bs)
                if reg and model_type == "RF":
                    if mydataset.y.min()[0] < 0 or mydataset.y_ind.min()[0] < 0:
                        search_space_bs.update(
                            {"criterion": ["categorical", ["squared_error"]]}
                        )
                    else:
                        search_space_bs.update(
                            {
                                "criterion":
                                    ["categorical", ["squared_error", "poisson"]]
                            }
                        )
                elif model_type == "RF":
                    search_space_bs.update(
                        {"criterion": ["categorical", ["gini", "entropy"]]}
                    )
                bayesoptimizer = OptunaOptimization(
                    scoring=score_func,
                    param_grid=search_space_bs,
                    n_trials=args.n_trials,
                    n_jobs=args.n_jobs,
                )
                best_params = bayesoptimizer.optimize(QSPRmodel)

            # initialize models from saved or default parameters

            if args.model_evaluation:
                CrossValAssessor(mode=EarlyStoppingMode.RECORDING)(QSPRmodel)
                TestSetAssessor(mode=EarlyStoppingMode.NOT_RECORDING)(QSPRmodel)

            if args.save_model:
                if (model_type == "DNN") and not (args.model_evaluation):
                    log.warning(
                        "Fit skipped: DNN can only be fitted after cross-validation "
                        "for determining optimal number of epochs to stop training."
                    )
                else:
                    QSPRmodel.fitAttached()


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
    file_prefixes = [
        f"{alg}_{data_prefix}" for alg in args.model_types
        for data_prefix in args.data_prefixes
    ]
    backup_msg = backup_files(
        args.base_dir, "qspr/models", tuple(file_prefixes), cp_suffix="_params"
    )

    if not os.path.exists(f"{args.base_dir}/qspr/models"):
        os.makedirs(f"{args.base_dir}/qspr/models")

    logSettings = enable_file_logger(
        os.path.join(args.base_dir, "qspr/models"),
        "QSPRmodel.log",
        args.debug,
        __name__,
        vars(args),
        disable_existing_loggers=False,
    )

    log = logSettings.log
    log.info(backup_msg)

    # Add optuna logging
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    # Stop showing logs in sys.stderr.
    optuna.logging.disable_default_handler()
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    # Create json log file with used commandline arguments
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open(f"{args.base_dir}/qspr/models/QSPRmodel.json", "w") as f:
        json.dump(vars(args), f)

    # Optimize, evaluate and train estimators according to QSPR arguments
    log.info(
        "QSPR modelling started: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    QSPR_modelling(args)

    log.info(
        "QSPR modelling completed: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
