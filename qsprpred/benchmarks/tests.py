from unittest import TestCase

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from .. import TargetProperty, TargetTasks
from ..data import MoleculeTable, QSPRDataset
from ..data.sources.data_source import DataSource
from ..data.tests import DataSetsMixIn
from ..models.assessment_methods import TestSetAssessor
from ..models.sklearn import SklearnModel
from ..utils.stringops import get_random_string
from . import BenchmarkRunner, BenchmarkSettings, DataPrepSettings


class DataSourceTesting(DataSetsMixIn, DataSource):
    """Data source for testing purposes. Simply prepares the default
    data set from`DataSetsMixIn`.
    """
    def __init__(self):
        super().__init__()
        self.setUp()

    def getData(self, name: str | None = "TestDataSet", **kwargs) -> MoleculeTable:
        return self.createLargeTestDataSet(name)

    def getDataSet(
        self,
        target_props: list[TargetProperty | dict],
        name: str | None = "TestDataSet",
        **kwargs,
    ) -> QSPRDataset:
        return self.createLargeTestDataSet(name, target_props=target_props)


class BenchmarkingTest(DataSetsMixIn, TestCase):
    """Test benchmarking functionality on the test data set.

    Attributes:
        settings (BenchmarkSettings):
            Benchmark settings.
        benchmark (BenchmarkRunner):
            Benchmark runner.
    """
    def setUp(self):
        super().setUp()
        prep = self.getDefaultPrep()
        descriptors = prep["feature_calculators"][0].descSets
        del prep["feature_calculators"]
        self.settings = BenchmarkSettings(
            name=get_random_string(prefix=self.__class__.__name__ + "_"),
            n_replicas=2,
            random_seed=42,
            data_sources=[DataSourceTesting(), DataSourceTesting()],
            descriptors=[descriptors],
            target_props=[
                [
                    TargetProperty.fromDict(
                        {
                            "name": "CL",
                            "task": TargetTasks.SINGLECLASS,
                            "th": [10]
                        }
                    )
                ],
                [
                    TargetProperty.fromDict(
                        {
                            "name": "fu",
                            "task": TargetTasks.SINGLECLASS,
                            "th": [0.3]
                        }
                    )
                ],
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
            optimizers=[],
        )
        self.benchmark = BenchmarkRunner(
            self.settings,
            data_dir=f"{self.generatedPath}/benchmarks",
            results_file=f"{self.generatedPath}/benchmarks/results.tsv",
        )

    def test(self):
        """Run the test benchmark."""
        results = self.benchmark.run(raise_errors=True)
        self.assertEqual(
            self.benchmark.nRuns * len(self.settings.assessors), len(results)
        )
