from unittest import TestCase

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from . import BenchmarkRunner
from . import BenchmarkSettings, DataPrepSettings
from ..data.data import MoleculeTable, TargetProperty, QSPRDataset
from ..data.sources.data_source import DataSource
from ..data.tests import DataSetsMixIn
from ..models.assessment_methods import TestSetAssessor
from ..models.sklearn import SklearnModel
from ..models.tasks import TargetTasks
from ..utils.stringops import get_random_string


class DataSourceTesting(DataSetsMixIn, DataSource):

    def __init__(self):
        super().__init__()
        self.setUp()

    def getData(
            self,
            name: str | None = "TestDataSet",
            **kwargs
    ) -> MoleculeTable:
        return self.createLargeTestDataSet(name)

    def getDataSet(
            self,
            target_props: list[TargetProperty | dict],
            name: str | None = "TestDataSet",
            **kwargs
    ) -> QSPRDataset:
        return self.createLargeTestDataSet(name, target_props=target_props)


class TestBenchmarking(DataSetsMixIn, TestCase):

    def setUp(self):
        super().setUp()
        prep = self.getDefaultPrep()
        descriptors = prep['feature_calculators'][0].descSets
        del prep['feature_calculators']
        self.settings = BenchmarkSettings(
            name=get_random_string(prefix=self.__class__.__name__ + "_"),
            n_replicas=2,
            random_seed=42,
            data_sources=[DataSourceTesting(), DataSourceTesting()],
            descriptors=[descriptors],
            target_props=[
                [TargetProperty.fromDict({
                    "name": "CL",
                    "task": TargetTasks.SINGLECLASS,
                    "th": [10]
                })],
                [TargetProperty.fromDict({
                    "name": "fu",
                    "task": TargetTasks.SINGLECLASS,
                    "th": [0.3]
                })],
            ],
            prep_settings=[DataPrepSettings(**prep)],
            models=[
                SklearnModel(
                    name="GaussianNB",
                    alg=GaussianNB,
                    base_dir=f"{self.generatedPath}/models",
                ),
                SklearnModel(
                    name="MLPClassifier",
                    alg=MLPClassifier,
                    base_dir=f"{self.generatedPath}/models",
                ),
            ],
            assessors=[
                TestSetAssessor(scoring="roc_auc"),
                TestSetAssessor(scoring="matthews_corrcoef", use_proba=False),
            ],
            optimizers=[]
        )
        self.benchmark = BenchmarkRunner(
            self.settings,
            data_dir=f"{self.generatedPath}/benchmarks",
            results_file=f"{self.generatedPath}/benchmarks/results.tsv"
        )

    def test_benchmarking(self):
        results = self.benchmark.run(raise_errors=True)
        self.assertEqual(
            self.benchmark.n_runs * len(self.settings.assessors),
            len(results)
        )

