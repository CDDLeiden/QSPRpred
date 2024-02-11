import numpy as np
from parameterized import parameterized
from rdkit.Chem import Descriptors

from .fingerprints import MorganFP
from .sets import (
    DrugExPhyschem,
    PredictorDesc,
    TanimotoDistances,
    RDKitDescs,
    SmilesDesc,
)
from ... import TargetTasks
from ...data import RandomSplit
from ...data.processing.feature_filters import LowVarianceFilter, HighCorrelationFilter
from ...models import SklearnModel
from ...utils.testing.base import QSPRTestCase
from ...utils.testing.check_mixins import DescriptorInDataCheckMixIn
from ...utils.testing.path_mixins import DataSetsPathMixIn


class TestDescriptorCalculation(DataSetsPathMixIn, QSPRTestCase):
    """Test the calculation of descriptors."""

    def setUp(self):
        """Set up the test Dataframe."""
        super().setUp()
        self.setUpPaths()

    @parameterized.expand([(None, None), (1, None), (2, None), (4, 50)])
    def testSwitching(self, n_cpu, chunk_size):
        """Test if the feature calculator can be switched to a new dataset."""
        dataset = self.createLargeTestDataSet(
            "TestSwitching", n_jobs=n_cpu, chunk_size=chunk_size
        )
        feature_calculators = [
            MorganFP(radius=3, nBits=256),
            DrugExPhyschem(),
        ]
        split = RandomSplit(test_fraction=0.1)
        lv = LowVarianceFilter(0.05)
        hc = HighCorrelationFilter(0.9)
        dataset.prepareDataset(
            split=split,
            feature_calculators=feature_calculators,
            feature_filters=[lv, hc],
            recalculate_features=True,
            feature_fill_value=np.nan,
        )
        # create new dataset with the same calculator
        dataset_next = self.createLargeTestDataSet(self.__class__.__name__)
        dataset_next.prepareDataset(
            split=split,
            feature_calculators=feature_calculators,
            feature_filters=[lv, hc],
            recalculate_features=True,
            feature_fill_value=np.nan,
        )
        self.assertEqual(dataset.X.shape, dataset_next.X.shape)


class TestDescriptorSets(DataSetsPathMixIn, QSPRTestCase):
    """Test the descriptor sets."""

    def setUp(self):
        """Create the test Dataframe."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createLargeTestDataSet(self.__class__.__name__)
        self.dataset.nJobs = self.nCPU
        self.dataset.chunkSize = None
        self.dataset.shuffle()

    def testPredictorDescriptor(self):
        """Test the PredictorDesc descriptor set."""
        # give path to saved model parameters
        meta_path = (
            f"{self.inputBasePath}/test_predictor/"
            f"RFC_SINGLECLASS/RFC_SINGLECLASS_meta.json"
        )
        model = SklearnModel.fromFile(meta_path)
        desc_calc = PredictorDesc(model)
        self.dataset.addDescriptors([desc_calc])
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 1))
        self.assertTrue(self.dataset.X.any().any())
        # test from file instantiation
        desc_calc.toFile(f"{self.generatedDataPath}/test_calc.json")
        desc_calc_file = desc_calc.fromFile(f"{self.generatedDataPath}/test_calc.json")
        self.dataset.addDescriptors([desc_calc_file], recalculate=True)
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 1))
        self.assertTrue(self.dataset.X.any().any())

    def testFingerprintSet(self):
        """Test the fingerprint set descriptor calculator."""
        desc_calc = MorganFP(radius=3, nBits=128)
        self.dataset.addDescriptors([desc_calc])
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 128))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    def testTanimotoDistances(self):
        """Test the Tanimoto distances descriptor calculator, which calculates the
        Tanimoto distances between a list of SMILES."""
        list_of_smiles = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC"]
        desc_calc = [
            TanimotoDistances(
                list_of_smiles=list_of_smiles,
                fingerprint_type=MorganFP(radius=3, nBits=128),
            )
        ]
        self.dataset.addDescriptors(desc_calc)

    def testDrugExPhyschem(self):
        """Test the DrugExPhyschem descriptor calculator."""
        desc_calc = [DrugExPhyschem()]
        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 19))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    def testRDKitDescs(self):
        """Test the rdkit descriptors calculator."""
        desc_calc = [RDKitDescs()]
        self.dataset.addDescriptors(desc_calc)
        rdkit_desc_count = len(set(Descriptors._descList))
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), rdkit_desc_count))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)
        # with 3D
        desc_calc = [RDKitDescs(include_3d=True)]
        self.dataset.addDescriptors(desc_calc, recalculate=True)
        self.assertEqual(
            self.dataset.X.shape, (len(self.dataset), rdkit_desc_count + 10)
        )

    def testSmilesDesc(self):
        """Test the smiles descriptors calculator."""
        desc_calc = [SmilesDesc()]
        self.dataset.addDescriptors(desc_calc)

        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 1))
        self.assertTrue(self.dataset.X.any().any())

    def testConsistency(self):
        """Test if the descriptor calculator is consistent with the dataset."""
        len_prev = len(self.dataset)
        desc_calc = [MorganFP(radius=3, nBits=128)]
        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(len_prev, len(self.dataset))
        self.assertEqual(len_prev, len(self.dataset.getDescriptors()))
        self.assertEqual(len_prev, len(self.dataset.X))
        self.assertEqual(128, self.dataset.getDescriptors().shape[1])
        self.assertEqual(128, self.dataset.X.shape[1])
        self.assertEqual(128, self.dataset.X_ind.shape[1])
        self.assertEqual(128, self.dataset.getFeatures(concat=True).shape[1])
        self.assertEqual(len_prev, self.dataset.getFeatures(concat=True).shape[0])


class TestDescriptorsAll(DataSetsPathMixIn, DescriptorInDataCheckMixIn, QSPRTestCase):
    """Test all descriptor sets in all data sets."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    @parameterized.expand([
        (
            f"{desc_set}_{TargetTasks.REGRESSION}",
            desc_set,
            [{"name": "CL", "task": TargetTasks.REGRESSION}],
        )
        for desc_set in DataSetsPathMixIn.getAllDescriptors()
    ])
    def testDescriptorsAll(self, _, desc_set, target_props):
        """Tests all available descriptor sets.

        Note that they are not checked with all possible settings and all possible
        preparations, but only with the default settings provided by
        `DataSetsPathMixIn.getDefaultPrep()`. The list itself is defined and configured by
        `DataSetsPathMixIn.getAllDescriptors()`, so if you need a specific descriptor
        tested, add it there.
        """
        np.random.seed(42)
        dataset = self.createLargeTestDataSet(
            name=self.getDatSetName(desc_set, target_props),
            target_props=target_props,
            n_jobs=self.nCPU,
            chunk_size=None,
        )
        self.checkDataSetContainsDescriptorSet(
            dataset, desc_set, self.getDefaultPrep(), target_props
        )
