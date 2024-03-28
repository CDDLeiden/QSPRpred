"""
Test module for testing extra models.

"""

from typing import Type

from parameterized import parameterized
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor

from qsprpred.extra.data.descriptors.sets import ProDec
from qsprpred.tasks import TargetProperty, TargetTasks
from .random import ScipyDistributionAlgorithm, RandomModel, RatioDistributionAlgorithm, \
    MedianDistributionAlgorithm
from ..data.utils.testing.path_mixins import DataSetsMixInExtras
from ..models.pcm import SklearnPCMModel
from ...utils.testing.base import QSPRTestCase
from ...utils.testing.check_mixins import ModelCheckMixIn
from ...utils.testing.path_mixins import ModelDataSetsPathMixIn


class ModelDataSetsMixInExtras(ModelDataSetsPathMixIn, DataSetsMixInExtras):
    """This class holds the tests for testing models in extras."""


class TestPCM(ModelDataSetsMixInExtras, ModelCheckMixIn, QSPRTestCase):
    """Test class for testing PCM models."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def getModel(
        self,
        name: str,
        alg: Type | None = None,
        parameters: dict | None = None,
        random_state: int | None = None,
    ):
        """Initialize dataset and model.

        Args:
            name (str): Name of the model.
            alg (Type | None): Algorithm class.
            parameters (dict | None): Parameters to use.
            random_state (int | None): Random seed to use.

        Returns:
            SklearnPCMModel: Initialized model.
        """
        return SklearnPCMModel(
            base_dir=self.generatedModelsPath,
            alg=alg,
            name=name,
            parameters=parameters,
            random_state=random_state,
        )

    @parameterized.expand(
        [
            (
                alg_name,
                [{"name": "pchembl_value_Median", "task": TargetTasks.REGRESSION}],
                alg_name,
                alg,
                random_state,
            )
            for alg, alg_name in ((XGBRegressor, "XGBR"),)
            for random_state in ([None], [1, 42], [42, 42])
        ]
        + [
            (
                alg_name,
                [{"name": "pchembl_value_Median", "task": TargetTasks.REGRESSION}],
                alg_name,
                alg,
                [None],
            )
            for alg, alg_name in ((PLSRegression, "PLSR"),)
        ]
        + [
            (
                alg_name,
                [
                    {
                        "name": "pchembl_value_Median",
                        "task": TargetTasks.SINGLECLASS,
                        "th": [6.5],
                    }
                ],
                alg_name,
                alg,
                random_state,
            )
            for alg, alg_name in ((XGBClassifier, "XGBC"),)
            for random_state in ([None], [1, 42], [42, 42])
        ]
    )
    def testRegressionBasicFitPCM(
        self,
        _,
        props: list[TargetProperty | dict],
        model_name: str,
        model_class: Type,
        random_state: list[int | None],
    ):
        """Test model training for regression models.

        Args:
            _: Name of the test.
            props (list[TargetProperty | dict]): List of target properties.
            model_name (str): Name of the model.
            model_class (Type): Class of the model.

        """
        if model_name not in ["SVR", "PLSR"]:
            parameters = {"n_jobs": self.nCPU}
        else:
            parameters = None

        # initialize dataset
        prep = self.getDefaultPrep()
        prep["feature_calculators"] = prep["feature_calculators"] + [
            ProDec(["Sneath"], self.getMSAProvider(self.generatedDataPath))
        ]
        dataset = self.createPCMDataSet(
            name=f"{model_name}_{props[0]['task']}_pcm",
            target_props=props,
            preparation_settings=prep,
            random_state=random_state[0],
        )
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_{props[0]['task']}",
            alg=model_class,
            parameters=parameters,
            random_state=random_state[0],
        )
        self.fitTest(model, dataset)
        predictor = SklearnPCMModel(
            name=f"{model_name}_{props[0]['task']}", base_dir=model.baseDir
        )

        # make predictions with the trained model and check if the results are (not)
        # equal if the random state is the (not) same and at the same time
        # check if the output is the same before and after saving and loading
        for protein_id in sorted(set(dataset.getProperty(dataset.proteinCol))):
            subset = dataset.searchOnProperty(
                dataset.proteinCol, [protein_id], exact=True
            )
            if random_state[0] is not None:
                comparison_model = self.getModel(
                    name=f"{model_name}_{props[0]['task']}",
                    alg=model_class,
                    parameters=parameters,
                    random_state=random_state[1],
                )
                self.fitTest(comparison_model, dataset)
                self.predictorTest(
                    predictor, # model loaded from file
                    subset,
                    comparison_model=comparison_model, # model not loaded from file
                    protein_id=protein_id,
                    expect_equal_result=random_state[0] == random_state[1],
                )
            else:
                self.predictorTest(
                    predictor,
                    subset,
                    protein_id=protein_id,
                )


class RandomBaseModelTestCase(ModelDataSetsMixInExtras, ModelCheckMixIn, QSPRTestCase):
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def getModel(
        self,
        name: str,
        alg: ScipyDistributionAlgorithm | RatioDistributionAlgorithm = ScipyDistributionAlgorithm,
        parameters: dict | None = None,
        random_state: int | None = None,
    ):
        """Initialize dataset and model.

        Args:
            name (str): Name of the model.
            alg (Type | None): Algorithm class.
            parameters (dict | None): Parameters to use.
            random_state (int | None): Random seed to use.

        Returns:
            RandomModel: Initialized model.
        """
        return RandomModel(
            base_dir=self.generatedModelsPath,
            name=name,
            alg=alg,
            parameters=parameters,
            random_state=random_state,
        )


class TestRandomModelRegression(RandomBaseModelTestCase):
    """Test the RandomModel class for regression models."""

    @parameterized.expand(
        [
            ("RandomModel", TargetTasks.REGRESSION, random_state)
            for random_state in ([None], [1, 42], [42, 42])
        ]
    )
    def testRegressionBasicFit(self, model_name, task, random_state):
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            target_props=[{"name": "CL", "task": task}],
            preparation_settings=self.getDefaultPrep(),
        )
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_{task}",
            random_state=random_state[0],
        )
        self.fitTest(model, dataset)
        # load model from file
        predictor = RandomModel(
            name=f"{model_name}_{task}",
            base_dir=model.baseDir,
            alg=MedianDistributionAlgorithm
        )
        self.predictorTest(predictor, dataset)

        # test if the results are (not) equal if the random state is the (not) same
        # and check if the output is the same before and after saving and loading
        if random_state[0] is not None:
            comparison_model = self.getModel(
                name=f"{model_name}_{task}",
                random_state=random_state[1],
            )
            self.fitTest(comparison_model, dataset)
            self.predictorTest(
                predictor, # model loaded from file
                dataset,
                comparison_model, # model not loaded from file
                expect_equal_result=random_state[0] == random_state[1],
            )

    @parameterized.expand(
        [
            ("RandomModel", random_state)
            for random_state in ([None], [1, 42], [42, 42])
        ]
    )
    def testRegressionMultiTaskFit(self, model_name, random_state: list[int | None]):
        """Test model training for multitask regression models."""
        # initialize dataset
        dataset = self.createLargeTestDataSet(
            target_props=[
                {
                    "name": "fu",
                    "task": TargetTasks.REGRESSION,
                    "imputer": SimpleImputer(strategy="mean"),
                },
                {
                    "name": "CL",
                    "task": TargetTasks.REGRESSION,
                    "imputer": SimpleImputer(strategy="mean"),
                },
            ],
            preparation_settings=self.getDefaultPrep(),
        )
        # test classifier
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_multitask_regression",
            random_state=random_state[0],
        )
        self.fitTest(model, dataset)
        predictor = RandomModel(
            name=f"{model_name}_multitask_regression",
            base_dir=model.baseDir,
            alg=MedianDistributionAlgorithm
        )
        self.predictorTest(predictor, dataset)
        
        # test if the results are (not) equal if the random state is the (not) same
        # and check if the output is the same before and after saving and loading
        if random_state[0] is not None:
            comparison_model = self.getModel(
                name=f"{model_name}_multitask_regression",
                random_state=random_state[1],
            )
            self.fitTest(comparison_model, dataset)
            self.predictorTest(
                predictor,
                dataset,
                comparison_model,
                expect_equal_result=random_state[0] == random_state[1],
            )


class TestRandomModelClassification(RandomBaseModelTestCase):
    """Test the RandomModel class for regression models."""
    @parameterized.expand(
        [
            (f"{alg_name}_{task}", task, th, alg_name, alg, random_state)
            for alg, alg_name in (
                (RatioDistributionAlgorithm, "RandomModel"),
            )
            for task, th in (
                (TargetTasks.SINGLECLASS, [6.5]),
                (TargetTasks.MULTICLASS, [0, 2, 10, 1100]),
            )
            for random_state in ([None], [42, 42])
        ]
    )
    def testClassificationBasicFit(
        self, _, task, th, model_name, model_class, random_state
    ):
        """Test model training for classification models."""
        parameters = None

        # initialize dataset
        dataset = self.createLargeTestDataSet(
            target_props=[{"name": "CL", "task": task, "th": th}],
            preparation_settings=self.getDefaultPrep(),
        )
        # test classifier
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_{task}",
            alg=model_class,
            parameters=parameters,
            random_state=random_state[0],
        )
        self.fitTest(model, dataset)
        predictor = RandomModel(
            name=f"{model_name}_{task}",
            base_dir=model.baseDir,
            alg=RatioDistributionAlgorithm
        )

        self.predictorTest(predictor, dataset)
        
        # test if the results are (not) equal if the random state is the (not) same
        # and check if the output is the same before and after saving and loading
        if random_state[0] is not None:
            comparison_model = self.getModel(
                name=f"{model_name}_{task}",
                alg=model_class,
                parameters=parameters,
                random_state=random_state[1],
            )
            self.fitTest(comparison_model, dataset)
            self.predictorTest(
                predictor, # model loaded from file
                dataset,
                comparison_model, # model not loaded from file
                expect_equal_result=random_state[0] == random_state[1],
            )


class TestRandomModelClassificationMultiTask(RandomBaseModelTestCase):
    """Test the SklearnModel class for multi-task classification models."""

    @parameterized.expand(
        [
            (alg_name, alg_name, alg, random_state)
            for alg, alg_name in ((RatioDistributionAlgorithm, "RandomModel"),)
            for random_state in ([None], [42, 42])
        ]
    )
    def testClassificationMultiTaskFit(self, _, model_name, model_class, random_state):
        """Test model training for multitask classification models."""
        parameters = {}

        # initialize dataset
        dataset = self.createLargeTestDataSet(
            target_props=[
                {
                    "name": "fu",
                    "task": TargetTasks.SINGLECLASS,
                    "th": [0.3],
                    "imputer": SimpleImputer(strategy="mean"),
                },
                {
                    "name": "CL",
                    "task": TargetTasks.SINGLECLASS,
                    "th": [6.5],
                    "imputer": SimpleImputer(strategy="mean"),
                },
            ],
            preparation_settings=self.getDefaultPrep(),
        )
        # test classifier
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_multitask_classification",
            alg=model_class,
            parameters=parameters,
            random_state=random_state[0],
        )
        self.fitTest(model, dataset)
        predictor = RandomModel(
            name=f"{model_name}_multitask_classification",
            base_dir=model.baseDir,
            alg=RatioDistributionAlgorithm
        )

        self.predictorTest(predictor, dataset)
        
        # test if the results are (not) equal if the random state is the (not) same
        # and check if the output is the same before and after saving and loading
        if random_state[0] is not None:
            comparison_model = self.getModel(
                name=f"{model_name}_multitask_classification",
                alg=model_class,
                parameters=parameters,
                random_state=random_state[1],
            )
            self.fitTest(comparison_model, dataset)
            self.predictorTest(
                predictor, # model loaded from file
                dataset,
                comparison_model, # model not loaded from file
                expect_equal_result=random_state[0] == random_state[1],
            )
