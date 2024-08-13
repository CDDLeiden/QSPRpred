import numpy as np
from parameterized import parameterized
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ..descriptors.fingerprints import MorganFP
from ...data import (
    RandomSplit,
    TemporalSplit,
    BootstrapSplit,
    ScaffoldSplit,
    ClusterSplit,
    QSPRDataset,
)
from ...data.chem.clustering import (
    FPSimilarityLeaderPickerClusters,
    FPSimilarityMaxMinClusters,
)
from ...data.chem.scaffolds import BemisMurckoRDKit, BemisMurcko
from ...data.sampling.folds import FoldsFromDataSplit
from ...data.sampling.splits import ManualSplit
from ...utils.testing.base import QSPRTestCase
from ...utils.testing.path_mixins import DataSetsPathMixIn


class TestDataSplitters(DataSetsPathMixIn, QSPRTestCase):
    """Small tests to only check if the data splitters work on their own.

    The tests here should be used to check for all their specific parameters and edge
    cases."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()
        self.splitYear = 2000

    def testManualSplit(self):
        """Test the manual split function, where the split is done manually."""
        dataset = self.createLargeTestDataSet()
        dataset.nJobs = self.nCPU
        dataset.chunkSize = self.chunkSize
        # Add extra column to the data frame to use for splitting
        dataset.df["split"] = "train"
        dataset.df.loc[dataset.df.sample(frac=0.1).index, "split"] = "test"
        split = ManualSplit(dataset.df["split"], "train", "test")
        dataset.prepareDataset(split=split)
        self.validate_split(dataset)

    @parameterized.expand(
        [
            (False,),
            (True,),
        ]
    )
    def testRandomSplit(self, multitask):
        """Test the random split function."""
        if multitask:
            dataset = self.createLargeMultitaskDataSet()
        else:
            dataset = self.createLargeTestDataSet()
        dataset = self.createLargeTestDataSet()
        dataset.prepareDataset(split=RandomSplit(test_fraction=0.1))
        self.validate_split(dataset)

    @parameterized.expand(
        [
            (False,),
            (True,),
        ]
    )
    def testTemporalSplit(self, multitask):
        """Test the temporal split function, where the split is done based on a time
        property."""
        if multitask:
            dataset = self.createLargeMultitaskDataSet()
        else:
            dataset = self.createLargeTestDataSet()
        split = TemporalSplit(
            timesplit=self.splitYear,
            timeprop="Year of first disclosure",
        )
        # prepare and validate the split
        dataset.prepareDataset(split=split)
        self.validate_split(dataset)
        # test if dates higher than 2000 are in test set
        test_set = dataset.getFeatures()[1]
        years = dataset.getDF().loc[test_set.index, "Year of first disclosure"]
        self.assertTrue(all(years > self.splitYear))
        # test bootstrapping
        if multitask:
            dataset = self.createLargeMultitaskDataSet(
                name="TemporalSplit_bootstrap_mt"
            )
        else:
            dataset = self.createLargeTestDataSet(name="TemporalSplit_bootstrap")
        split = TemporalSplit(
            timesplit=[self.splitYear - 1, self.splitYear, self.splitYear + 1],
            timeprop="Year of first disclosure",
        )
        bootstrap_split = BootstrapSplit(
            split=split,
            n_bootstraps=10,
        )
        for time, fold_info in zip(
            split.timeSplit, list(dataset.iterFolds(bootstrap_split))
        ):
            years = dataset.getDF().loc[fold_info[1].index, "Year of first disclosure"]
            self.assertTrue(all(years > time))

    @parameterized.expand(
        [
            (False, BemisMurckoRDKit(), None),
            (
                False,
                BemisMurcko(use_csk=True),
                ["ScaffoldSplit_000", "ScaffoldSplit_001"],
            ),
            (True, BemisMurckoRDKit(), None),
        ]
    )
    def testScaffoldSplit(self, multitask, scaffold, custom_test_list):
        """Test the scaffold split function."""
        if multitask:
            dataset = self.createLargeMultitaskDataSet(name="ScaffoldSplit")
        else:
            dataset = self.createLargeTestDataSet(name="ScaffoldSplit")
        split = ScaffoldSplit(
            scaffold=scaffold,
            custom_test_list=custom_test_list,
        )
        dataset.prepareDataset(split=split)
        self.validate_split(dataset)
        # check that smiles in custom_test_list are in the test set
        if custom_test_list:
            self.assertTrue(
                all(mol_id in dataset.X_ind.index for mol_id in custom_test_list)
            )
        # check folding by scaffold
        if multitask:
            dataset = self.createLargeMultitaskDataSet(name="ScaffoldSplit_folding_mt")
        else:
            dataset = self.createLargeTestDataSet(name="ScaffoldSplit_folding")
        n_folds = 5
        split = ScaffoldSplit(
            scaffold=scaffold,
            custom_test_list=custom_test_list,
            n_folds=n_folds,
        )
        test_index_all = []
        for k, (X_train, X_test, y_train, y_test, train_index, test_index) in enumerate(
            dataset.iterFolds(split)
        ):
            self.assertTrue(all(x not in test_index_all for x in test_index))
            self.assertTrue(len(X_train) > len(X_test))
            test_index_all.extend(X_test.index.tolist())
        self.assertEqual(k, n_folds - 1)
        self.assertEqual(len(test_index_all), len(dataset.getFeatures(concat=True)))

    @parameterized.expand(
        [
            (
                False,
                FPSimilarityLeaderPickerClusters(
                    fp_calculator=MorganFP(radius=2, nBits=128)
                ),
                None,
            ),
            (
                False,
                FPSimilarityMaxMinClusters(fp_calculator=MorganFP(radius=2, nBits=128)),
                ["ClusterSplit_000", "ClusterSplit_001"],
            ),
            (
                True,
                FPSimilarityMaxMinClusters(fp_calculator=MorganFP(radius=2, nBits=128)),
                None,
            ),
            (
                True,
                FPSimilarityLeaderPickerClusters(
                    fp_calculator=MorganFP(radius=2, nBits=128)
                ),
                ["ClusterSplit_000", "ClusterSplit_001"],
            ),
        ]
    )
    def testClusterSplit(self, multitask, clustering_algorithm, custom_test_list):
        """Test the cluster split function."""
        if multitask:
            dataset = self.createLargeMultitaskDataSet(name="ClusterSplit")
        else:
            dataset = self.createLargeTestDataSet(name="ClusterSplit")
        split = ClusterSplit(
            clustering=clustering_algorithm,
            custom_test_list=custom_test_list,
            time_limit_seconds=10,
        )
        dataset.prepareDataset(split=split)
        self.validate_split(dataset)
        # check that smiles in custom_test_list are in the test set
        if custom_test_list:
            self.assertTrue(
                all(mol_id in dataset.X_ind.index for mol_id in custom_test_list)
            )

    def testSerialization(self):
        """Test the serialization of dataset with datasplit."""
        dataset = self.createLargeTestDataSet()
        split = ScaffoldSplit()
        n_bits = 128
        dataset.prepareDataset(
            split=split,
            feature_calculators=[MorganFP(radius=3, nBits=n_bits)],
            feature_standardizer=StandardScaler(),
        )
        self.validate_split(dataset)
        test_ids = dataset.X_ind.index.values
        train_ids = dataset.y_ind.index.values
        dataset.save()
        dataset_new = QSPRDataset.fromFile(dataset.metaFile)
        self.validate_split(dataset_new)
        self.assertTrue(dataset_new.descriptorSets)
        self.assertTrue(dataset_new.featureStandardizer)
        self.assertTrue(len(dataset_new.featureNames) == n_bits)
        self.assertTrue(all(mol_id in dataset_new.X_ind.index for mol_id in test_ids))
        self.assertTrue(all(mol_id in dataset_new.y_ind.index for mol_id in train_ids))
        dataset_new.clearFiles()


class TestFoldSplitters(DataSetsPathMixIn, QSPRTestCase):
    """Small tests to only check if the fold splitters work on their own.

    The tests here should be used to check for all their specific parameters and
    edge cases."""

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def validateFolds(self, folds, more=None):
        """Check if the folds have the data they should have after splitting."""
        k = 0
        tested_indices = []
        for (
            X_train,
            X_test,
            y_train,
            y_test,
            train_index,
            test_index,
        ) in folds:
            k += 1
            self.assertEqual(len(X_train), len(y_train))
            self.assertEqual(len(X_test), len(y_test))
            self.assertEqual(len(train_index), len(y_train))
            self.assertEqual(len(test_index), len(y_test))
            tested_indices.extend(X_test.index.tolist())
            if more:
                more(X_train, X_test, y_train, y_test, train_index, test_index)
        return k, tested_indices

    def testStandardFolds(self):
        """Test the default fold generator, which is a 5-fold cross validation."""
        # test default settings with regression
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors([MorganFP(radius=3, nBits=128)])
        fold = KFold(5, shuffle=True, random_state=dataset.randomState)
        generator = FoldsFromDataSplit(fold)
        k, indices = self.validateFolds(generator.iterFolds(dataset))
        self.assertEqual(k, 5)
        self.assertFalse(set(dataset.df.index) - set(indices))
        # test directly on data set
        k, indices = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        self.assertFalse(set(dataset.df.index) - set(indices))
        # test default settings with classification
        dataset.makeClassification("CL", th=[20])
        fold = StratifiedKFold(5, shuffle=True, random_state=dataset.randomState)
        k, indices = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        self.assertFalse(set(dataset.df.index) - set(indices))
        # test with a standarizer
        MAX_VAL = 2
        MIN_VAL = 1
        scaler = MinMaxScaler(feature_range=(MIN_VAL, MAX_VAL))
        dataset.prepareDataset(feature_standardizer=scaler)
        k, indices = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        self.assertFalse(set(dataset.df.index) - set(indices))

        def check_min_max(X_train, X_test, *args, **kwargs):
            self.assertTrue(np.max(X_train.values) == MAX_VAL)
            self.assertTrue(np.min(X_train.values) == MIN_VAL)
            self.assertTrue(np.max(X_test.values) == MAX_VAL)
            self.assertTrue(np.min(X_test.values) == MIN_VAL)

        self.validateFolds(dataset.iterFolds(fold), more=check_min_max)
        k, indices = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        self.assertFalse(set(dataset.df.index) - set(indices))

        # try with a split data set
        dataset.split(RandomSplit(test_fraction=0.1))
        k, indices = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        self.assertFalse(set(dataset.X.index) - set(indices))

    def testBootstrappedFold(self):
        dataset = self.createLargeTestDataSet(random_state=None)
        dataset.addDescriptors([MorganFP(radius=3, nBits=128)])
        split = RandomSplit(0.2)
        fold = BootstrapSplit(split, n_bootstraps=5)
        k, indices = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        # check if the indices are the same if we do the same split again
        split = RandomSplit(0.2)
        fold = BootstrapSplit(split, n_bootstraps=5, seed=dataset.randomState)
        k, indices_second = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        self.assertListEqual(indices, indices_second)
        # check if the indices are different if we do a different split
        split = RandomSplit(0.2)
        fold = BootstrapSplit(split, n_bootstraps=5, seed=42)
        k, indices_third = self.validateFolds(dataset.iterFolds(fold))
        self.assertEqual(k, 5)
        self.assertEqual(split.getSeed(), None)
        self.assertNotEqual(indices, indices_third)
