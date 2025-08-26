import os

import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor

from qsprpred import TargetSpec, TargetTasks
from qsprpred.benchmarks import BenchmarkSettings, BenchmarkRunner
from qsprpred.data import MoleculeTable, RandomSplit
from qsprpred.data.descriptors.fingerprints import MorganFP
from qsprpred.data.descriptors.sets import RDKitDescs
from qsprpred.data.processing.pipeline import DatasetPipeline
from qsprpred.data.processing.feature_filters import LowVarianceFilter
from qsprpred.data.sources import DataSource
from qsprpred.models import SklearnModel, Assessor
from qsprpred.utils.parallel import MultiprocessingJITGenerator

BASE_DIR = "./data/"
os.makedirs(BASE_DIR, exist_ok=True)
SEED = 42


class DataSourceTesting(DataSource):
    """
    Just a simple wrapper around our tutorial data set.
    """

    def __init__(self, name: str, store_dir: str):
        self.name = name
        self.storeDir = store_dir

    def getData(self, name: str | None = None, **kwargs) -> MoleculeTable:
        """We just need to fetch a simple `MoleculeTable`.
        Defining target properties is not necessary. We just need to
        make sure that the data set contains the target properties we
        want to use for benchmarking later.

        To make things faster we will sample only 100 molecules each time.
        This code could also be simplified so that reloading of the file is not necessary.
        """
        name = name or self.name
        mt = MoleculeTable.fromDF(
            df=pd.read_table("../../tutorials/tutorial_data/A2A_LIGANDS.tsv").sample(
                300, random_state=SEED
            ),
            name=name,
            path=self.storeDir,
            **kwargs,
        )
        mt.storeFormat = "csv"
        return mt

if __name__ == "__main__":
    # run classification
    source = DataSourceTesting("ConsistencyChecks", f"{BASE_DIR}/data")
    settings = BenchmarkSettings(
        name="ConsistencyChecksCLS",
        n_replicas=1,
        random_seed=SEED,
        data_sources=[source],
        descriptors=[
            [
                MorganFP(radius=2, nBits=256),
                RDKitDescs(),
            ],
        ],
        target_props=[
            # one or more properties to model
            [
                TargetSpec.fromDict(
                    {
                        "name": "pchembl_value_Mean",
                        "task": TargetTasks.SINGLECLASS,
                        "th": [6.5],
                    }
                )
            ],
        ],
        pipelines = [
            DatasetPipeline(
                steps = {
                    "benchmarkfilter": LowVarianceFilter(0.05),
                    "scaler": StandardScaler()
                }
            )
        ],
        models=[
            SklearnModel(
                name="ExtraTreesClassifier",
                alg=ExtraTreesClassifier,
                base_dir=f"{BASE_DIR}/models",
            ),
            SklearnModel(
                name="XGBClassifier",
                alg=XGBClassifier,
                base_dir=f"{BASE_DIR}/models",
            ),
            SklearnModel(
                name="GaussianNB",
                alg=GaussianNB,
                base_dir=f"{BASE_DIR}/models",
            ),
        ],
        assessors=[
            Assessor(
                name="crossval_roc_auc",
                scoring="roc_auc",
                split=KFold(n_splits=5, shuffle=True),
            ),
            Assessor(
                name="crossval_matthews_corrcoef",
                scoring="matthews_corrcoef",
                split=KFold(n_splits=5, shuffle=True),
                use_proba=False
            ),
            Assessor(
                name="test_roc_auc",
                scoring="roc_auc",
                split=RandomSplit(test_fraction=0.2),
            ),
            Assessor(
                name="test_matthews_corrcoef",
                scoring="matthews_corrcoef",
                split=RandomSplit(test_fraction=0.2),
                use_proba=False
            ),
        ],
        subsets={
            # apply cross-validation only to the training set
            "crossval_roc_auc": (RandomSplit(test_fraction=0.2), "Train", 0),
            "crossval_matthews_corrcoef": (RandomSplit(test_fraction=0.2), "Train", 0),
        },
        optimizers=[],
    )
    runner = BenchmarkRunner(
        settings,
        data_dir=f"{BASE_DIR}/CLS",
        parallel_generator_cpu = MultiprocessingJITGenerator(1)
    )
    runner.run(raise_errors=True)

    # run regression
    settings.name = "ConsistencyChecksREG"
    settings.target_props = [
        # one or more properties to model
        [
            TargetSpec.fromDict(
                {
                    "name": "pchembl_value_Mean",
                    "task": TargetTasks.REGRESSION,
                }
            )
        ],
    ]
    settings.assessors = [
        Assessor(
            name="crossval_r2",
            scoring="r2",
            split=KFold(n_splits=5, shuffle=True),
        ),
        Assessor(
            name="crossval_neg_root_mean_squared_error",
            scoring="neg_root_mean_squared_error",
            split=KFold(n_splits=5, shuffle=True),
        ),
        Assessor(
            name="test_r2",
            scoring="r2",
            split=RandomSplit(test_fraction=0.2),
        ),
        Assessor(
            name="test_neg_root_mean_squared_error",
            scoring="neg_root_mean_squared_error",
            split=RandomSplit(test_fraction=0.2),
        ),
    ]
    settings.subsets = {
        # apply cross-validation only to the training set
        "crossval_r2": (RandomSplit(test_fraction=0.2), "Train", 0),
        "crossval_neg_root_mean_squared_error": (RandomSplit(test_fraction=0.2), "Train", 0)
    }
    settings.models = [
        SklearnModel(
            name="ExtraTreesRegressor",
            alg=ExtraTreesRegressor,
            base_dir=f"{BASE_DIR}/models",
        ),
        SklearnModel(
            name="XGBRegressor",
            alg=XGBRegressor,
            base_dir=f"{BASE_DIR}/models",
        ),
        SklearnModel(
            name="PLSRegression",
            alg=PLSRegression,
            base_dir=f"{BASE_DIR}/models",
        ),
    ]
    runner = BenchmarkRunner(
        settings,
        data_dir=f"{BASE_DIR}/REG",
        parallel_generator_cpu = MultiprocessingJITGenerator(5)
    )
    runner.run(raise_errors=True)
