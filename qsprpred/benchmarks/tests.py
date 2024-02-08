from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier

from qsprpred.models.assessment.methods import CrossValAssessor, TestSetAssessor
from . import BenchmarkRunner, BenchmarkSettings, DataPrepSettings
from .. import TargetProperty, TargetTasks
from ..data import MoleculeTable, QSPRDataset
from ..data.descriptors.sets import RDKitDescs
from ..data.sources.data_source import DataSource
from ..models.scikit_learn import SklearnModel
from ..utils.stringops import get_random_string
from ..utils.testing.base import QSPRTestCase
from ..utils.testing.path_mixins import DataSetsPathMixIn


class DataSourceTesting(DataSetsPathMixIn, DataSource):
    """Data source for testing purposes. Simply prepares the default
    data set from`DataSetsPathMixIn`.
    """

    def __init__(self, name):
        super().__init__()
        self.setUpPaths()
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


class BenchmarkingTest(DataSetsPathMixIn, QSPRTestCase):
    """Test benchmarking functionality on the test data set.

    Attributes:
        settings (BenchmarkSettings):
            Benchmark settings.
        benchmark (BenchmarkRunner):
            Benchmark runner.
    """

    def setUp(self):
        super().setUp()
        self.setUpPaths()
        prep = self.getDefaultPrep()
        descriptors = prep["feature_calculators"]
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

    def checkRunResults(self, results):
        # TODO: more checks should be done here in the future
        for tps in self.settings.target_props:
            for assessor in self.settings.assessors:
                score = assessor.scoreFunc.name
                score_results = results[
                    (results["ScoreFunc"] == score)
                    & (results["Assessor"] == assessor.__class__.__name__)
                    & (results["TargetProperty"].isin([tp.name for tp in tps]))
                ]
                self.assertTrue(len(score_results) > 0)

    def checkSettings(self):
        self.assertTrue(len(self.settings.data_sources) > 0)
        self.assertTrue(len(self.settings.descriptors) > 0)
        self.assertTrue(len(self.settings.target_props) > 0)
        self.assertTrue(len(self.settings.prep_settings) > 0)
        self.assertTrue(len(self.settings.models) > 0)
        self.assertTrue(len(self.settings.assessors) > 0)
        self.settings.toFile(f"{self.generatedPath}/benchmarks/settings.json")
        settings = BenchmarkSettings.fromFile(
            f"{self.generatedPath}/benchmarks/settings.json"
        )
        self.assertEqual(len(self.settings.data_sources), len(settings.data_sources))
        self.assertEqual(len(self.settings.descriptors), len(settings.descriptors))
        self.assertEqual(len(self.settings.target_props), len(settings.target_props))
        self.assertEqual(len(self.settings.prep_settings), len(settings.prep_settings))
        self.assertEqual(len(self.settings.models), len(settings.models))
        self.assertEqual(len(self.settings.assessors), len(settings.assessors))

    def testSingleTaskCLS(self):
        """Run single task tests for classification."""
        self.checkSettings()
        results = self.benchmark.run(raise_errors=True)
        self.checkRunResults(results)
        self.checkSettings()

    def testSingleTaskREG(self):
        self.settings.target_props = [
            [
                TargetProperty.fromDict(
                    {
                        "name": "CL",
                        "task": TargetTasks.REGRESSION,
                    }
                )
            ]
        ]
        self.settings.models = [
            SklearnModel(
                name="RandomForestRegressor",
                alg=RandomForestRegressor,
                base_dir=f"{self.generatedPath}/models",
            ),
            SklearnModel(
                name="KNeighborsRegressor",
                alg=KNeighborsRegressor,
                base_dir=f"{self.generatedPath}/models",
            ),
        ]
        self.settings.assessors = [
            CrossValAssessor(
                scoring="r2",
                split=KFold(n_splits=self.nFolds, shuffle=True, random_state=self.seed),
            ),
            CrossValAssessor(
                scoring="neg_mean_squared_error",
                split=KFold(n_splits=self.nFolds, shuffle=True, random_state=self.seed),
            ),
            TestSetAssessor(scoring="r2"),
            TestSetAssessor(scoring="neg_mean_squared_error"),
        ]
        self.checkSettings()
        results = self.benchmark.run(raise_errors=True)
        self.checkRunResults(results)
        self.checkSettings()

    def testMultiTaskCLS(self):
        """Run the test benchmark."""
        self.settings.target_props = [
            [
                TargetProperty.fromDict(
                    {
                        "name": "CL",
                        "task": TargetTasks.SINGLECLASS,
                        "th": [10],
                        "imputer": SimpleImputer(strategy="mean"),
                    }
                ),
                TargetProperty.fromDict(
                    {
                        "name": "fu",
                        "task": TargetTasks.SINGLECLASS,
                        "th": [0.3],
                        "imputer": SimpleImputer(strategy="mean"),
                    }
                ),
            ]
        ]
        self.settings.models = [
            SklearnModel(
                name="RandomForestClassifier",
                alg=RandomForestClassifier,
                base_dir=f"{self.generatedPath}/models",
            ),
            SklearnModel(
                name="KNeighborsClassifier",
                alg=KNeighborsClassifier,
                base_dir=f"{self.generatedPath}/models",
            ),
        ]
        self.settings.assessors = [
            CrossValAssessor(
                scoring="roc_auc",
                split=KFold(n_splits=self.nFolds, shuffle=True, random_state=self.seed),
                split_multitask_scores=True,
            ),
            CrossValAssessor(
                scoring="matthews_corrcoef",
                split=KFold(n_splits=self.nFolds, shuffle=True, random_state=self.seed),
                use_proba=False,
                split_multitask_scores=True,
            ),
            TestSetAssessor(scoring="roc_auc", split_multitask_scores=True),
            TestSetAssessor(
                scoring="matthews_corrcoef",
                use_proba=False,
                split_multitask_scores=True,
            ),
        ]
        self.checkSettings()
        results = self.benchmark.run(raise_errors=True)
        self.checkRunResults(results)
        self.checkSettings()

    def testMultiTaskREG(self):
        self.settings.target_props = [
            [
                TargetProperty.fromDict(
                    {
                        "name": "CL",
                        "task": TargetTasks.REGRESSION,
                        "imputer": SimpleImputer(strategy="mean"),
                    }
                ),
                TargetProperty.fromDict(
                    {
                        "name": "fu",
                        "task": TargetTasks.REGRESSION,
                        "imputer": SimpleImputer(strategy="mean"),
                    }
                ),
            ]
        ]
        self.settings.models = [
            SklearnModel(
                name="RandomForestRegressor",
                alg=RandomForestRegressor,
                base_dir=f"{self.generatedPath}/models",
            ),
            SklearnModel(
                name="KNeighborsRegressor",
                alg=KNeighborsRegressor,
                base_dir=f"{self.generatedPath}/models",
            ),
        ]
        self.settings.assessors = [
            CrossValAssessor(
                scoring="r2",
                split=KFold(n_splits=self.nFolds, shuffle=True, random_state=self.seed),
                split_multitask_scores=True,
            ),
            CrossValAssessor(
                scoring="neg_mean_squared_error",
                split=KFold(n_splits=self.nFolds, shuffle=True, random_state=self.seed),
                split_multitask_scores=True,
            ),
            TestSetAssessor(scoring="r2", split_multitask_scores=True),
            TestSetAssessor(
                scoring="neg_mean_squared_error", split_multitask_scores=True
            ),
        ]
        self.checkSettings()
        results = self.benchmark.run(raise_errors=True)
        self.checkRunResults(results)
        self.checkSettings()
