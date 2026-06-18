import numpy as np
from parameterized import parameterized
from rdkit.Chem import Descriptors

from .fingerprints import MorganFP
from .sets import (
    DrugExPhyschem,
    PredictorDesc,
    RDKitDescs,
    SmilesDesc,
    TanimotoDistances,
    RandomDescs,
)
from ..processing.feature_filters import LowVarianceFilter, HighCorrelationFilter
from ..processing.pipeline import DatasetPipeline
from ...data import RandomSplit
from ...models import SklearnModel
from ...utils.testing.base import QSPRTestCase
from ...utils.testing.path_mixins import DataSetsPathMixIn


class TestDescriptorCalculation(DataSetsPathMixIn, QSPRTestCase):
    """Test the calculation of descriptors."""

    def setUp(self):
        """Set up the test Dataframe."""
        super().setUp()
        self.setUpPaths()

    @staticmethod
    def getDescList():
        return [MorganFP(radius=3, nBits=256), DrugExPhyschem()]

    def testDropping(self):
        """Test dropping of descriptors from data sets."""
        dataset = self.createLargeTestDataSet("TestDropping")
        # test dropping of all sets
        dataset.addDescriptors(self.getDescList())
        full_len = sum(len(x) for x in dataset.descriptorSets)
        self.assertTrue(dataset.getDescriptors().shape[1] == full_len)
        dataset.dropDescriptorSets(dataset.descriptorSets)
        self.assertEqual(dataset.getDescriptors().shape[1], 0)
        dataset.dropDescriptorSets(dataset.descriptorSets, full_removal=True)
        self.assertEqual(len(dataset.descriptors), 0)
        dataset.addDescriptors(self.getDescList())
        dataset.dropDescriptorSets([str(x) for x in self.getDescList()])
        self.assertEqual(dataset.getDescriptors().shape[1], 0)
        dataset.dropDescriptorSets(
            [str(x) for x in self.getDescList()], full_removal=True
        )
        self.assertEqual(len(dataset.descriptors), 0)
        # test dropping of single set
        dataset.addDescriptors(self.getDescList())
        self.assertTrue(dataset.getDescriptors().shape[1] == full_len)
        dataset.dropDescriptorSets([dataset.descriptorSets[0]])
        self.assertEqual(
            dataset.getDescriptors().shape[1], len(self.getDescList()[1])
        )
        dataset.dropDescriptorSets(dataset.descriptorSets, full_removal=True)
        dataset.addDescriptors(self.getDescList())
        dataset.dropDescriptorSets([str(dataset.descriptorSets[0])], full_removal=True)
        self.assertEqual(
            dataset.getDescriptors().shape[1], len(self.getDescList()[1])
        )
        # test restoring of dropped sets
        dataset.addDescriptors(self.getDescList())
        self.assertTrue(dataset.getDescriptors().shape[1] == full_len)
        dataset.dropDescriptorSets(dataset.descriptorSets, full_removal=False)
        self.assertEqual(dataset.getDescriptors().shape[1], 0)
        dataset.restoreDescriptorSets(dataset.descriptorSets)
        self.assertTrue(dataset.getDescriptors().shape[1] == full_len)

    @parameterized.expand([(1, None), (2, None), (1, 17), (2, 17)])
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
        pipeline = DatasetPipeline(
            feature_calculators=feature_calculators,
            steps={
                "low_var_filter": LowVarianceFilter(0.05),
                "high_corr_filter": HighCorrelationFilter(0.75),
            }
        )
        X_train, y_train, X_test, y_test = next(pipeline.applyOnDataSet(dataset, split))
        # create new dataset with the same calculator
        dataset_next = self.createLargeTestDataSet(
            "TestSwitching", n_jobs=1, chunk_size=None
        )
        X_train_next, y_train_next, X_test_next, y_test_next = next(
            pipeline.applyOnDataSet(dataset_next, split)
        )
        # check if all matrices are identical
        if not np.array_equal(X_train, X_train_next):
            # check if IDs are the same in train and train_next
            check = set([x.split("_")[-1] for x in X_train.index.tolist()]) - set(
                [x.split("_")[-1] for x in X_train_next.index.tolist()])
            print(f"X_train is not equal: {check}")
        self.assertTrue(np.array_equal(X_train, X_train_next))
        self.assertTrue(np.array_equal(y_train, y_train_next))
        self.assertTrue(np.array_equal(X_test, X_test_next))
        self.assertTrue(np.array_equal(y_test, y_test_next))


class TestDescriptorSets(DataSetsPathMixIn, QSPRTestCase):
    """Test the descriptor sets."""

    def setUp(self):
        """Create the test Dataframe."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createLargeTestDataSet(self.__class__.__name__)
        self.dataset.nJobs = self.nCPU
        self.dataset.chunkSize = None

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
        self.assertEqual(self.dataset.getDescriptors().shape, (len(self.dataset), 1))
        self.assertTrue(self.dataset.getDescriptors().any().any())
        # test from file instantiation
        desc_calc.toFile(f"{self.generatedDataPath}/test_calc.json")
        desc_calc_file = desc_calc.fromFile(f"{self.generatedDataPath}/test_calc.json")
        self.dataset.addDescriptors([desc_calc_file], recalculate=True)
        self.assertEqual(self.dataset.getDescriptors().shape, (len(self.dataset), 1))
        self.assertTrue(self.dataset.getDescriptors().any().any())

    def testFingerprintSet(self):
        """Test the fingerprint set descriptor calculator."""
        desc_calc = MorganFP(radius=3, nBits=128)
        self.dataset.addDescriptors([desc_calc])
        self.assertEqual(self.dataset.getDescriptors().shape, (len(self.dataset), 128))
        self.assertTrue(self.dataset.getDescriptors().any().any())
        self.assertTrue(self.dataset.getDescriptors().any().sum() > 1)

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
        self.assertEqual(self.dataset.getDescriptors().shape, (len(self.dataset), 7))

    def testDrugExPhyschem(self):
        """Test the DrugExPhyschem descriptor calculator."""
        desc_calc = [DrugExPhyschem()]
        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(self.dataset.getDescriptors().shape, (len(self.dataset), 19))
        self.assertTrue(self.dataset.getDescriptors().any().any())
        self.assertTrue(self.dataset.getDescriptors().any().sum() > 1)

    def testRDKitDescs(self):
        """Test the rdkit descriptors calculator."""
        desc_calc = [RDKitDescs()]
        self.dataset.addDescriptors(desc_calc)
        rdkit_desc_count = len(set(Descriptors._descList))
        self.assertEqual(self.dataset.getDescriptors().shape,
                         (len(self.dataset), rdkit_desc_count))
        self.assertTrue(self.dataset.getDescriptors().any().any())
        self.assertTrue(self.dataset.getDescriptors().any().sum() > 1)
        # with 3D
        desc_calc = [RDKitDescs(include_3d=True)]
        self.dataset.addDescriptors(desc_calc, recalculate=True)
        self.assertEqual(
            self.dataset.getDescriptors().shape,
            (len(self.dataset), rdkit_desc_count + 10)
        )

    def testSmilesDesc(self):
        """Test the smiles descriptors calculator."""
        desc_calc = [SmilesDesc()]
        self.dataset.addDescriptors(desc_calc)

        self.assertEqual(self.dataset.getDescriptors().shape, (len(self.dataset), 1))
        self.assertTrue(self.dataset.getDescriptors().any().any())

    def testRandomDescs(self):
        """Test the random descriptors calculator."""
        desc_calc = [RandomDescs(n=10)]
        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(self.dataset.getDescriptors().shape, (len(self.dataset), 10))
        self.assertTrue(self.dataset.getDescriptors().any().any())
        self.assertFalse(self.dataset.getDescriptors().isna().any().any())
        self.dataset.dropDescriptorSets(desc_calc, full_removal=True)

        # test setting n
        desc_calc = [RandomDescs(n=100)]
        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(self.dataset.getDescriptors().shape, (len(self.dataset), 100))
        self.assertTrue(self.dataset.getDescriptors().any().any())
        self.dataset.dropDescriptorSets(desc_calc, full_removal=True)

        # test setting randomseed
        desc_calc = [RandomDescs(n=10, seed=42)]
        self.dataset.addDescriptors(desc_calc)
        descriptors_42 = self.dataset.getDescriptors()
        self.dataset.dropDescriptorSets(desc_calc, full_removal=True)

        desc_calc = [RandomDescs(n=10, seed=42)]
        self.dataset.addDescriptors(desc_calc)
        self.assertTrue(np.array_equal(self.dataset.getDescriptors(), descriptors_42))
        self.dataset.dropDescriptorSets(desc_calc, full_removal=True)

        desc_calc = [RandomDescs(n=10, seed=1)]
        self.dataset.addDescriptors(desc_calc)
        self.assertFalse(np.array_equal(self.dataset.getDescriptors(), descriptors_42))
        self.dataset.dropDescriptorSets(desc_calc, full_removal=True)

        # test add missing values
        desc_calc = [RandomDescs(n=10, missing=0.1)]
        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(self.dataset.getDescriptors().shape, (len(self.dataset), 10))
        self.assertTrue(self.dataset.getDescriptors().any().any())
        n_missing = 10 * len(self.dataset) * 0.1
        self.assertEqual(self.dataset.getDescriptors().isna().sum().sum(), n_missing)
        self.dataset.dropDescriptorSets(desc_calc, full_removal=True)

        desc_calc = [RandomDescs(n=10, missing=4)]
        self.dataset.addDescriptors(desc_calc)
        self.assertEqual(self.dataset.getDescriptors().shape, (len(self.dataset), 10))
        self.assertTrue(self.dataset.getDescriptors().any().any())
        self.assertEqual(self.dataset.getDescriptors().isna().sum().sum(),
                         (len(self.dataset) * 4))

# class TestDescriptorsAll(DataSetsPathMixIn, DescriptorInDataCheckMixIn, QSPRTestCase):
#     """Test all descriptor sets in all data sets."""
#     def setUp(self):
#         super().setUp()
#         self.setUpPaths()

#     @parameterized.expand(
#         [
#             (
#                 f"{desc_set}_{TargetTasks.REGRESSION}",
#                 desc_set,
#                 [{
#                     "name": "CL",
#                     "task": TargetTasks.REGRESSION
#                 }],
#             ) for desc_set in DataSetsPathMixIn.getAllDescriptors()
#         ]
#     )
#     def testDescriptorsAll(self, _, desc_set, target_props):
#         """Tests all available descriptor sets.

#         Note that they are not checked with all possible settings and all possible
#         preparations, but only with the default settings provided by
#         `DataSetsPathMixIn.getDefaultPrep()`. The list itself is defined and configured by
#         `DataSetsPathMixIn.getAllDescriptors()`, so if you need a specific descriptor
#         tested, add it there.
#         """
#         dataset = self.createLargeTestDataSet(
#             name=self.getDataSetName(desc_set, target_props),
#             target_props=target_props,
#             n_jobs=self.nCPU,
#             chunk_size=None,
#         )
#         self.checkDataSetContainsDescriptorSet(
#             dataset, desc_set, self.getDefaultPrep(), target_props
#         )
