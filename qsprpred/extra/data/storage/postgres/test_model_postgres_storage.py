import os

import numpy as np
import pytest
from dotenv import load_dotenv

import qsprpred.models.model as model_module
from qsprpred.models.model import QSPRModel
from qsprpred.utils.interfaces.randomized import Randomized
from .chem_store import PostgresChemStore

load_dotenv()


class FakeDataset:
    def __init__(self, mol_table):
        self.mol_table = mol_table
        self.prepare_called = False

    def prepareDataset(self, *args, **kwargs):
        self.prepare_called = True


class DummyQSPRModel(QSPRModel, Randomized):
    @property
    def randomState(self) -> int:
        return 42

    @property
    def supportsEarlyStopping(self):
        return False

    def fit(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        return np.array([[1.0]])

    def predictProba(self, *args, **kwargs):
        return [np.array([[1.0]])]

    def loadEstimator(self, params=None):
        return None

    def loadEstimatorFromFile(self, params=None):
        return None

    def saveEstimator(self):
        return ""


@pytest.fixture()
def postgres_store():
    dsn = os.getenv("QSPR_POSTGRES_DSN")
    if not dsn:
        pytest.skip("QSPR_POSTGRES_DSN is not set")

    store = PostgresChemStore(
        name="model_prediction_test",
        connection_string=dsn,
        table_name="test_model_prediction_storage",
        schema=os.getenv("QSPR_POSTGRES_SCHEMA", "public"),
        use_rdkit_cartridge=True,
        create=True,
    )
    store.clear()
    return store


def make_dummy_qspr_model():
    model = DummyQSPRModel.__new__(DummyQSPRModel)
    model.baseDir = "."
    model.chemStandardizer = None
    model.targetProperties = []
    model.featureCalculators = []
    model.featureStandardizer = None
    return model


def test_qspr_model_prediction_dataset_uses_postgres_storage(
        postgres_store,
        monkeypatch,
):
    captured = {}

    def fake_from_mol_table(mol_table, target_props, drop_empty_target_props=False):
        captured["mol_table"] = mol_table
        captured["target_props"] = target_props
        return FakeDataset(mol_table)

    monkeypatch.setattr(
        model_module.QSPRTable,
        "fromMolTable",
        staticmethod(fake_from_mol_table),
    )

    model = make_dummy_qspr_model()

    dataset, failed_mask = model.createPredictionDatasetFromMols(
        mols=["CCO", "CCN", "c1ccccc1"],
        storage=postgres_store,
    )

    assert postgres_store.getMolCount() == 3
    assert set(postgres_store.getDF()["SMILES"]) == {"CCO", "CCN", "c1ccccc1"}

    assert captured["mol_table"].storage is postgres_store
    assert captured["target_props"] == []
    assert dataset.prepare_called is True
    assert failed_mask.tolist() == [False, False, False]
