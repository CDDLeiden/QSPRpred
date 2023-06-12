"""
tests

Created by: Martin Sicho
On: 12.05.23, 18:33
"""
import os
from unittest import TestCase

from parameterized import parameterized
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from xgboost import XGBClassifier, XGBRegressor

from ...models.tasks import TargetTasks
from ...models.tests import N_CPUS, ModelDataSetsMixIn, ModelTestMixIn
from ..data.tests import DataSetsMixInExtras
from ..data.utils.descriptor_utils.msa_calculator import ClustalMSA
from ..data.utils.descriptorcalculator import ProteinDescriptorCalculator
from ..data.utils.descriptorsets import ProDecDescriptorSet
from ..models.pcm import QSPRsklearnPCM


class ModelDataSetsMixInExtras(ModelDataSetsMixIn, DataSetsMixInExtras):
    """This class holds the tests for testing models in extras."""

    qsprModelsPath = f"{os.path.dirname(__file__)}/test_files/qspr/models"


class TestPCM(ModelDataSetsMixInExtras, ModelTestMixIn, TestCase):
    @staticmethod
    def get_model(name, alg=None, dataset=None, parameters=None):
        """Initialize dataset and model."""
        return QSPRsklearnPCM(
            base_dir=f"{os.path.dirname(__file__)}/test_files/",
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
    def test_regression_basic_fit_pcm(self, _, props, model_name, model_class):
        """Test model training for regression models."""
        if model_name not in ["SVR", "PLSR"]:
            parameters = {"n_jobs": N_CPUS}
        else:
            parameters = None

        # initialize dataset
        prep = self.get_default_prep()
        prep["feature_calculators"] = prep["feature_calculators"] + [
            ProteinDescriptorCalculator(
                desc_sets=[ProDecDescriptorSet(sets=["Sneath"])],
                msa_provider=ClustalMSA(self.qsprdatapath),
            )
        ]
        dataset = self.create_pcm_dataset(
            name=f"{model_name}_{props[0]['task']}_pcm",
            target_props=props,
            preparation_settings=prep,
        )

        # initialize model for training from class
        model = self.get_model(
            name=f"{model_name}_{props[0]['task']}",
            alg=model_class,
            dataset=dataset,
            parameters=parameters,
        )
        self.fitTest(model)
        predictor = QSPRsklearnPCM(
            name=f"{model_name}_{props[0]['task']}", base_dir=model.baseDir
        )
        self.predictorTest(predictor, protein_id=dataset.getDF()["accession"].iloc[0])
