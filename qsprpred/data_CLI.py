#!/usr/bin/env python

import argparse
import json
import os.path
import sys
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
from boruta import BorutaPy
from rdkit.Chem.rdFingerprintGenerator import TopologicalTorsionFP
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from qsprpred.data.chem.clustering import (
    FPSimilarityMaxMinClusters,
    FPSimilarityLeaderPickerClusters,
)
from qsprpred.data.descriptors.fingerprints import (
    MorganFP,
    RDKitMACCSFP,
    AtomPairFP,
    LayeredFP,
    PatternFP,
    RDKitFP,
    AvalonFP,
)
from qsprpred.data.descriptors.sets import (
    DrugExPhyschem,
    PredictorDesc,
    RDKitDescs,
    SmilesDesc,
)
from qsprpred.data.processing.data_filters import papyrusLowQualityFilter
from qsprpred.data.processing.feature_filters import (
    BorutaFilter,
    HighCorrelationFilter,
    LowVarianceFilter,
)
from qsprpred.data.sampling.splits import (
    ClusterSplit,
    ManualSplit,
    RandomSplit,
    ScaffoldSplit,
    TemporalSplit,
)
from qsprpred.data.tables.qspr import QSPRDataset
from qsprpred.tasks import TargetTasks
from .data.chem.scaffolds import BemisMurckoRDKit
from .extra.gpu.models.dnn import DNNModel
from .logs.utils import backup_files, enable_file_logger
from .models.scikit_learn import SklearnModel


def QSPRArgParser(txt=None):
    """Define and read command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # base arguments
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="./dataset.tsv",
        help="path to tsv file that contains SMILES and property value column",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help="Directory to write output files to.",
    )
    parser.add_argument(
        "-ds",
        "--data_suffix",
        type=str,
        default=None,
        help="Suffix to add to the dataset name.",
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
    # model target arguments
    parser.add_argument(
        "-sm",
        "--smiles_col",
        type=str,
        default="SMILES",
        help="Name of the column in the dataset containing the smiles.",
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
            "task for CL do: -pr CL Fu -pr CL"
        ),
    )
    parser.add_argument(
        "-im",
        "--imputation",
        type=json.loads,
        help=(
            "Imputation method for missing values. Specify the imputation method as a "
            "dictionary with the property name as key and the imputation method as "
            "value, e.g. -im \"{'CL':'mean','fu':'median'}\". Note: no spaces and "
            "surrounded by single quotes. Choose from 'mean', 'median', 'most_frequent'"
        ),
        default={},
    )
    # model type arguments
    parser.add_argument(
        "-r",
        "--regression",
        type=str,
        default=None,
        help="If True, only regression model, if False, only classification, default both",
    )
    parser.add_argument(
        "-th",
        "--threshold",
        type=json.loads,
        help=(
            "Threshold on predicted property for classification. if len th larger "
            "than 1, these values will used for multiclass classification (then "
            "lower and upper boundary need to be included, e.g. for three classes "
            "[0,1],[1,2],[2,3]: 0,1,2,3) This needs to be given for each property "
            "included in any of the models as follows, e.g. -th "
            "\"{'CL':[6.5],'fu':[0,1,2,3,4]}\". Note: no spaces and surround "
            "by single quotes"
        ),
    )
    # Data pre-processing arguments
    parser.add_argument(
        "-lq",
        "--low_quality",
        action="store_true",
        help="If lq, than low quality data will be should be a column 'Quality' where "
        "all 'Low' will be removed",
    )
    parser.add_argument(
        "-tr",
        "--transform_data",
        type=json.loads,
        help=(
            "Transformation of the output property. This arg only has an effect when"
            "task is regression, otherwise will be ignored! Specify the transformation "
            "as a dictionary with the property name as key and the transformation as "
            "value, e.g. -tr \"{'CL':'log10','fu':'sqrt'}\". Note: no spaces and "
            "surrounded by single quotes. Choose from 'log10', 'log2', 'log', 'sqrt',"
            "'cbrt', 'exp', 'square', 'cube', 'reciprocal'"
        ),
        default={},
    )
    # Data set split arguments
    parser.add_argument(
        "-sp",
        "--split",
        type=str,
        choices=[
            "random",
            "cluster",
            "scaffold",
            "time",
            "manual",
        ],
        default="random",
        help=(
            "Split type. If 'random', 'cluster' or 'scaffold', you can specify the"
            "split_fraction with -sf. If 'cluster', you can specify the clustering"
            "method with -scm. If 'time', you can specify the split_time with -st and "
            " the split_timecolumn with -stc. If 'manual', a column 'datasplit' with "
            "values 'train' and 'test' is required."
        ),
    )
    parser.add_argument(
        "-sf",
        "--split_fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of the dataset used as test set. "
            "Used for RandomSplit, ClusterSplit and ScaffoldSplit"
        ),
    )
    parser.add_argument(
        "-st",
        "--split_time",
        type=float,
        default=2015,
        help="Temporal split limit. Used for TemporalSplit.",
    )
    parser.add_argument(
        "-stc",
        "--split_timecolumn",
        type=str,
        default="Year",
        help="Temporal split time column. Used for TemporalSplit.",
    )
    parser.add_argument(
        "-scm",
        "--split_cluster_method",
        type=str,
        choices=["MaxMin", "LeaderPicker"],
        default="MaxMin",
        help="Cluster method. Used for ClusterSplit.",
    )
    # features to calculate
    parser.add_argument(
        "-fe",
        "--features",
        type=str,
        choices=[
            "Morgan",
            "RDkit",
            "Mordred",
            "Mold2",
            "PaDEL",
            "DrugEx",
            "Signature",
            "MaccsFP",
            "AvalonFP",
            "TopologicalFP",
            "AtomPairFP",
            "RDKitFP",
            "PatternFP",
            "LayeredFP",
            "Smiles",
        ],
        nargs="*",
    )
    parser.add_argument(
        "-pd",
        "--predictor_descs",
        type=str,
        nargs="+",
        help=(
            "It is also possible to use a QSPRpred model(s) as molecular feature(s). "
            "Give the path(s) to the metadata of the model(s) relative to the "
            "base_directory."
        ),
    )
    # feature filters
    parser.add_argument(
        "-lv",
        "--low_variability",
        type=float,
        default=None,
        help="low variability threshold for feature removal.",
    )
    parser.add_argument(
        "-hc",
        "--high_correlation",
        type=float,
        default=None,
        help="high correlation threshold for feature removal.",
    )
    parser.add_argument(
        "-bf",
        "--boruta_filter",
        type=float,
        default=None,
        help="Boruta filter with random forest estimator, value between 0 and 100 "
        "for percentile threshold for comparison between shadow and real features"
        "see https://github.com/scikit-learn-contrib/boruta_py for more info.",
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
    # If no regression argument, does both regression and classification
    if args.regression is None:
        args.regression = [True, False]
    elif args.regression.lower() in ["true", "reg", "regression"]:
        args.regression = [True]
    elif args.regression.lower() in ["false", "cls", "classification"]:
        args.regression = [False]
    else:
        sys.exit("invalid regression arg given")
    return args


def QSPR_dataprep(args):
    """Optimize, evaluate and train estimators."""
    for reg in args.regression:
        for props in args.properties:
            props_name = "_".join(props)
            log.info(f"Property: {props_name}")
            try:
                df = pd.read_csv(args.input, sep="\t")
            except FileNotFoundError:
                log.error(f"Dataset file ({args.input}) not found")
                sys.exit()
            # prepare dataset for training QSPR model
            target_props = []
            for prop in props:
                th = args.threshold[prop] if args.threshold else None
                if reg:
                    task = TargetTasks.REGRESSION
                elif th is None:
                    task = (
                        TargetTasks.MULTICLASS
                        if len(df[prop].dropna().unique()) > 2  # noqa: PLR2004
                        else TargetTasks.SINGLECLASS
                    )
                    th = "precomputed"
                else:
                    task = (
                        TargetTasks.SINGLECLASS
                        if len(th) == 1
                        else TargetTasks.MULTICLASS
                    )
                if task == TargetTasks.REGRESSION and th:
                    log.warning(
                        "Threshold argument specified with regression. "
                        "Threshold will be ignored."
                    )
                    th = None
                transform_dict = {
                    "log10": lambda x: (__import__("numpy").log10(x)),
                    "log2": lambda x: (__import__("numpy").log2(x)),
                    "log": lambda x: (__import__("numpy").log(x)),
                    "sqrt": lambda x: (__import__("numpy").sqrt(x)),
                    "cbrt": lambda x: (__import__("numpy").cbrt(x)),
                    "exp": lambda x: (__import__("numpy").exp(x)),
                    "square": lambda x: __import__("numpy").power(x, 2),
                    "cube": lambda x: __import__("numpy").power(x, 3),
                    "reciprocal": lambda x: __import__("numpy").reciprocal(x),
                }
                target_props.append(
                    {
                        "name": prop,
                        "task": task,
                        "th": th,
                        "transformer": transform_dict[args.transform_data[prop]]
                        if prop in args.transform_data
                        else None,
                        "imputer": SimpleImputer(strategy=args.imputation[prop])
                        if prop in args.imputation
                        else None,
                    }
                )
            dataset_name = (
                f"{props_name}_{task}_{args.data_suffix}"
                if args.data_suffix
                else f"{props_name}_{task}"
            )
            mydataset = QSPRDataset(
                dataset_name,
                target_props=target_props,
                df=df,
                smiles_col=args.smiles_col,
                n_jobs=args.ncpu,
                store_dir=args.output_dir,
                overwrite=True,
                random_state=args.random_state
                if args.random_state is not None
                else None,
            )
            # data filters
            data_filters = []
            if args.low_quality:
                data_filters.append(papyrusLowQualityFilter())
            # data splitter
            if args.split == "scaffold":
                split = ScaffoldSplit(
                    test_fraction=args.split_fraction,
                    scaffold=BemisMurckoRDKit(),
                    dataset=mydataset,
                )
            elif args.split == "time":
                split = TemporalSplit(
                    timesplit=args.split_time,
                    timeprop=args.split_timecolumn,
                    dataset=mydataset,
                )
            elif args.split == "manual":
                if "datasplit" not in df.columns:
                    raise ValueError(
                        "No datasplit column found in dataset. Please add a column "
                        "'datasplit' with values 'train' and 'test' if using manual "
                        "split."
                    )
                split = ManualSplit(
                    splitcol=df["datasplit"], trainval="train", testval="test"
                )
            elif args.split == "cluster":
                if args.split_cluster_method == "MaxMin":
                    clustering = FPSimilarityMaxMinClusters()
                elif args.split_cluster_method == "LeaderPicker":
                    clustering = FPSimilarityLeaderPickerClusters()
                split = ClusterSplit(
                    test_fraction=args.split_fraction,
                    clustering=clustering,
                    dataset=mydataset,
                )
            else:
                split = RandomSplit(
                    test_fraction=args.split_fraction, dataset=mydataset
                )
            # feature calculator
            descriptorsets = []
            # Avoid importing optional dependencies if not needed
            f_arr = np.array(args.features)
            if np.isin(["Mordred", "Mold2", "PaDEL", "Signature"], f_arr).any():
                from qsprpred.extra.data.descriptors.sets import (
                    ExtendedValenceSignature,
                    Mold2,
                    Mordred,
                    PaDEL,
                )
            if "Morgan" in args.features:
                descriptorsets.append(MorganFP(radius=3, nBits=2048))
            if "RDkit" in args.features:
                descriptorsets.append(RDKitDescs())
            if "Mordred" in args.features:
                descriptorsets.append(Mordred())
            if "Mold2" in args.features:
                descriptorsets.append(Mold2())
            if "PaDEL" in args.features:
                descriptorsets.append(PaDEL())
            if "DrugEx" in args.features:
                descriptorsets.append(DrugExPhyschem())
            if "Signature" in args.features:
                descriptorsets.append(ExtendedValenceSignature(depth=1))
            if "MaccsFP" in args.features:
                descriptorsets.append(RDKitMACCSFP())
            if "AtomPairFP" in args.features:
                descriptorsets.append(AtomPairFP())
            if "TopologicalFP" in args.features:
                descriptorsets.append(TopologicalTorsionFP())
            if "AvalonFP" in args.features:
                descriptorsets.append(AvalonFP())
            if "RDKitFP" in args.features:
                descriptorsets.append(RDKitFP())
            if "PatternFP" in args.features:
                descriptorsets.append(PatternFP())
            if "LayeredFP" in args.features:
                descriptorsets.append(LayeredFP())
            if "Smiles" in args.features:
                descriptorsets.append(SmilesDesc())
            if args.predictor_descs:
                for predictor_path in args.predictor_descs:
                    # load in predictor from files
                    if "DNN" in predictor_path:
                        descriptorsets.append(
                            PredictorDesc(DNNModel.fromFile(predictor_path))
                        )
                    else:
                        descriptorsets.append(
                            PredictorDesc(SklearnModel.fromFile(predictor_path))
                        )
            # feature filters
            featurefilters = []
            if args.low_variability:
                featurefilters.append(LowVarianceFilter(th=args.low_variability))
            if args.high_correlation:
                featurefilters.append(HighCorrelationFilter(th=args.high_correlation))
            if args.boruta_filter:
                # boruta filter can not be used for multi-task models
                if len(props) > 1:
                    raise ValueError(
                        "Boruta filter can not be used for multi-task models"
                    )
                boruta_estimator = (
                    RandomForestRegressor(n_jobs=args.ncpu)
                    if args.regression
                    else RandomForestClassifier(n_jobs=args.ncpu)
                )
                featurefilters.append(
                    BorutaFilter(
                        BorutaPy(estimator=boruta_estimator, perc=args.boruta_filter),
                        args.random_state,
                    )
                )
            # prepare dataset for modelling
            mydataset.prepareDataset(
                feature_calculators=descriptorsets,
                data_filters=data_filters,
                split=split,
                feature_filters=featurefilters,
                feature_standardizer=StandardScaler()
                if "Smiles" not in args.features
                else None,
                feature_fill_value=args.fill_value,
            )

            # save dataset files and fingerprints
            mydataset.save()


if __name__ == "__main__":
    args = QSPRArgParser()

    # check input file and output directory exist
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file {args.input} not found.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # get a list of all the folders in the output directory
    folders = [
        f
        for f in os.listdir(args.output_dir)
        if os.path.isdir(f"{args.output_dir}/{f}")
    ]

    # remove folders that start with backup
    folders = [f for f in folders if not f.startswith("backup")]

    if not args.skip_backup:
        backup_msg = backup_files(
            args.output_dir,
            tuple(folders),
            cp_suffix=["calculators", "standardizer", "meta"],
        )

    logSettings = enable_file_logger(
        args.output_dir,
        "QSPRdata.log",
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
    with open(f"{args.output_dir}/QSPRdata.json", "w") as f:
        json.dump(vars(args), f)

    # Optimize, evaluate and train estimators according to QSPR arguments
    log.info(
        "Data preparation started: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    QSPR_dataprep(args)

    log.info(
        "Data preparation completed: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
