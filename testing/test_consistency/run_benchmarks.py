import os

import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from qsprpred import TargetProperty, TargetTasks
from qsprpred.benchmarks import BenchmarkSettings, DataPrepSettings, BenchmarkRunner
from qsprpred.data import MoleculeTable, RandomSplit
from qsprpred.data.descriptors.fingerprints import MorganFP
from qsprpred.data.descriptors.sets import RDKitDescs
from qsprpred.data.processing.feature_filters import LowVarianceFilter
from qsprpred.data.sources import DataSource
from qsprpred.models import SklearnModel, TestSetAssessor, CrossValAssessor

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
        return MoleculeTable(
            df=pd.read_table("../../tutorials/tutorial_data/A2A_LIGANDS.tsv").sample(
                300, random_state=SEED
            ),
            name=name,
            store_dir=self.storeDir,
            store_format="csv",
            **kwargs,
        )


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
            TargetProperty.fromDict(
                {
                    "name": "pchembl_value_Mean",
                    "task": TargetTasks.SINGLECLASS,
                    "th": [6.5],
                }
            )
        ],
    ],
    prep_settings=[
        DataPrepSettings(
            split=RandomSplit(test_fraction=0.2),  # random split
            feature_filters=[LowVarianceFilter(0.05)],
            feature_standardizer=StandardScaler(),
        ),
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
        CrossValAssessor(scoring="roc_auc"),
        CrossValAssessor(scoring="matthews_corrcoef", use_proba=False),
        TestSetAssessor(scoring="roc_auc"),
        TestSetAssessor(scoring="matthews_corrcoef", use_proba=False),
    ],
    optimizers=[],
)
runner = BenchmarkRunner(settings, data_dir=f"{BASE_DIR}/CLS")
runner.run(raise_errors=True)


# run regression
settings.name = "ConsistencyChecksREG"
settings.target_props = [
    # one or more properties to model
    [
        TargetProperty.fromDict(
            {
                "name": "pchembl_value_Mean",
                "task": TargetTasks.REGRESSION,
            }
        )
    ],
]
settings.assessors = [
    CrossValAssessor(scoring="r2"),
    CrossValAssessor(scoring="neg_root_mean_squared_error"),
    TestSetAssessor(scoring="r2"),
    TestSetAssessor(scoring="neg_root_mean_squared_error"),
]
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
runner = BenchmarkRunner(settings, data_dir=f"{BASE_DIR}/REG")
runner.run(raise_errors=True)
