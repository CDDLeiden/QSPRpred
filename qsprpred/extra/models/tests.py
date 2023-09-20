"""
Test module for testing extra models.

"""

from typing import Type
from unittest import TestCase

from parameterized import parameterized
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from xgboost import XGBClassifier, XGBRegressor

from ...data.data import TargetProperty
from ...models.tasks import TargetTasks
from ...models.tests import N_CPUS, ModelDataSetsMixIn, ModelTestMixIn
from ..data.data import PCMDataSet
from ..data.tests import DataSetsMixInExtras
from ..data.utils.descriptor_utils.msa_calculator import ClustalMSA
from ..data.utils.descriptorcalculator import ProteinDescriptorCalculator
from ..data.utils.descriptorsets import ProDec
from ..models.pcm import SklearnPCMModel


class ModelDataSetsMixInExtras(ModelDataSetsMixIn, DataSetsMixInExtras):
    """This class holds the tests for testing models in extras."""


class TestPCM(ModelDataSetsMixInExtras, ModelTestMixIn, TestCase):
    def getModel(
        self,
        name: str,
        alg: Type | None = None,
        dataset: PCMDataSet | None = None,
        parameters: dict | None = None,
    ):
        """Initialize dataset and model.

        Args:
            name (str): Name of the model.
            alg (Type | None): Algorithm class.
            dataset (PCMDataSet | None): Dataset to use.
            parameters (dict | None): Parameters to use.

        Returns:
            SklearnPCMModel: Initialized model.
        """
        return SklearnPCMModel(
            base_dir=self.generatedModelsPath,
            alg=alg,
            data=dataset,
            name=name,
            parameters=parameters,
        )

    @parameterized.expand(
        [
            (
                alg_name,
                [{
                    "name": "pchembl_value_Median",
                    "task": TargetTasks.REGRESSION
                }],
                alg_name,
                alg,
            ) for alg, alg_name in (
                (PLSRegression, "PLSR"),
                (SVR, "SVR"),
                (XGBRegressor, "XGBR"),
            )
        ] + [
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
            ) for alg, alg_name in (
                (RandomForestClassifier, "RFC"),
                (XGBClassifier, "XGBC"),
            )
        ]
    )
    def testRegressionBasicFitPCM(
        self, _, props: list[TargetProperty | dict], model_name: str, model_class: Type
    ):
        """Test model training for regression models.

        Args:
            _: Name of the test.
            props (list[TargetProperty | dict]): List of target properties.
            model_name (str): Name of the model.
            model_class (Type): Class of the model.

        """
        if model_name not in ["SVR", "PLSR"]:
            parameters = {"n_jobs": N_CPUS}
        else:
            parameters = None
        # initialize dataset
        prep = self.getDefaultPrep()
        prep["feature_calculators"] = prep["feature_calculators"] + [
            ProteinDescriptorCalculator(
                desc_sets=[ProDec(sets=["Sneath"])],
                msa_provider=ClustalMSA(self.generatedDataPath),
            )
        ]
        dataset = self.createPCMDataSet(
            name=f"{model_name}_{props[0]['task']}_pcm",
            target_props=props,
            preparation_settings=prep,
        )
        # initialize model for training from class
        model = self.getModel(
            name=f"{model_name}_{props[0]['task']}",
            alg=model_class,
            dataset=dataset,
            parameters=parameters,
        )
        self.fitTest(model)
        predictor = SklearnPCMModel(
            name=f"{model_name}_{props[0]['task']}", base_dir=model.baseDir
        )
        self.predictorTest(predictor, protein_id=dataset.getDF()["accession"].iloc[0])
