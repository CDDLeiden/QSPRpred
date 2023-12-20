from unittest import TestCase

from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from . import BenchmarkRunner, BenchmarkSettings, DataPrepSettings
from .. import TargetProperty, TargetTasks
from ..data import MoleculeTable, QSPRDataset
from ..data.descriptors.sets import RDKitDescs
from ..data.sources.data_source import DataSource
from ..data.tests import DataSetsMixIn
from ..models.assessment_methods import CrossValAssessor, TestSetAssessor
from ..models.scikit_learn import SklearnModel
from ..utils.stringops import get_random_string


class DataSourceTesting(DataSetsMixIn, DataSource):
    """Data source for testing purposes. Simply prepares the default
    data set from`DataSetsMixIn`.
    """

    def __init__(self, name):
        super().__init__()
        self.setUp()
        self.name = name

    def getData(self, name: str | None = None, **kwargs) -> MoleculeTable:
        name = name or self.name
        return self.createLargeTestDataSet(name)

    def getDataSet(
        self,
        target_props: list[TargetProperty | dict],
        name: str | None = None,
        **kwargs,
    ) -> QSPRDataset:
        name = name or self.name
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
        descriptors.append(RDKitDescs())
        self.seed = 42
        self.nFolds = 3
        self.settings = BenchmarkSettings(
            name=get_random_string(prefix=self.__class__.__name__ + "_"),
            n_replicas=2,
            random_seed=self.seed,
            data_sources=[
                DataSourceTesting("TestData_1"),
                DataSourceTesting("TestData_2"),
            ],
            descriptors=[
                descriptors,
                [descriptors[0]],
                [descriptors[1]],
            ],
            target_props=[
                [
                    TargetProperty.fromDict(
                        {"name": "CL", "task": TargetTasks.SINGLECLASS, "th": [10]}
                    )
                ],
                [
                    TargetProperty.fromDict(
                        {"name": "fu", "task": TargetTasks.SINGLECLASS, "th": [0.3]}
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
                CrossValAssessor(
                    scoring="roc_auc",
                    split=KFold(
                        n_splits=self.nFolds, shuffle=True, random_state=self.seed
                    ),
                ),
                CrossValAssessor(
                    scoring="matthews_corrcoef",
                    split=KFold(
                        n_splits=self.nFolds, shuffle=True, random_state=self.seed
                    ),
                    use_proba=False,
                ),
                TestSetAssessor(scoring="roc_auc"),
                TestSetAssessor(scoring="matthews_corrcoef", use_proba=False),
            ],
            optimizers=[],  # FIXME: needs to be tested
        )
        self.benchmark = BenchmarkRunner(
            self.settings,
            data_dir=f"{self.generatedPath}/benchmarks",
            results_file=f"{self.generatedPath}/benchmarks/results.tsv",
        )
        self.nCVs = len(
            [a for a in self.settings.assessors if isinstance(a, CrossValAssessor)]
        )
        self.nAssessments = (
            self.nCVs * self.nFolds + len(self.settings.assessors) - self.nCVs
        )

    def test(self):
        """Run the test benchmark."""
        results = self.benchmark.run(raise_errors=True)
        self.assertEqual(
            self.benchmark.nRuns * self.nAssessments,
            len(results),
        )
