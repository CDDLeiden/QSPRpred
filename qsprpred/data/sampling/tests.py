import numpy as np
from parameterized import parameterized
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from ...data import (
    BootstrapSplit,
    ClusterSplit,
    QSPRTable,
    RandomSplit,
    ScaffoldSplit,
    TemporalSplit,
)
from ...data.chem.clustering import (
    FPSimilarityLeaderPickerClusters,
    FPSimilarityMaxMinClusters,
)
from ...data.chem.scaffolds import BemisMurcko, BemisMurckoRDKit
from ...data.sampling.splits import ManualSplit
from ...utils.testing.base import QSPRTestCase
from ...utils.testing.check_mixins import DataPrepCheckMixIn
from ...utils.testing.path_mixins import DataSetsPathMixIn
from ..descriptors.fingerprints import MorganFP
from ..processing.pipeline import DatasetPipeline


class TestDataSplitters(DataSetsPathMixIn, QSPRTestCase, DataPrepCheckMixIn):
    """Small tests to only check if the data splitters work on their own.

    The tests here should be used to check for all their specific parameters and edge
    cases."""
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def testManualSplit(self):
        """Test the manual split function, where the split is done manually."""
        dataset = self.createLargeTestDataSet()
        dataset.nJobs = self.nCPU
        dataset.chunkSize = self.chunkSize

        # Add extra column to the data frame to use for splitting
        df = dataset.getDF()
        test_ids = df.sample(frac=0.1, random_state=42).index
        train_ids = df.index.difference(test_ids)
        dataset.addProperty("split", "test", ids=test_ids)
        dataset.addProperty("split", "train", ids=train_ids)
        split = ManualSplit("split", "train", "test")
        dataset.addSplit(split, name="split")

        # check if the split is correctly stored
        self.checkSplit(dataset, "split")

        # test if the split corresponds to the manually selected ids
        train_ids_split, test_ids_split = dataset.getSplit("split", as_type="ids")[0]
        self.assertTrue(all(train_ids.sort_values() == train_ids_split))
        self.assertTrue(all(test_ids.sort_values() == test_ids_split))

        #Test if also works for multiple splits
        # Add extra column to the data frame to use for splitting
        df = dataset.getDF()
        test_ids2 = df.sample(frac=0.1, random_state=1).index
        train_ids2 = df.index.difference(test_ids2)
        dataset.addProperty("split2", "test", ids=test_ids2)
        dataset.addProperty("split2", "train", ids=train_ids2)
        double_split = ManualSplit(["split", "split2"], "train", "test")
        dataset.addSplit(double_split, name="double_split")

        # check if the split is correctly stored
        self.checkSplit(dataset, "double_split")

        # test if the split corresponds to the manually selected ids
        double_split_iterator = dataset.getSplit("double_split", as_type="ids")
        self.assertTrue(len(double_split_iterator) == 2)
        train_ids_split, test_ids_split = double_split_iterator[1]
        self.assertTrue(all(train_ids2.sort_values() == train_ids_split))
        self.assertTrue(all(test_ids2.sort_values() == test_ids_split))

    @parameterized.expand([
        (False, ),
        (True, ),
    ])
    def testRandomSplit(self, multitask):
        """Test the random split function."""
        if multitask:
            dataset = self.createLargeMultitaskDataSet()
        else:
            dataset = self.createLargeTestDataSet()

        dataset.addSplit(RandomSplit(test_fraction=0.1), name="RandomSplit")
        self.checkSplit(dataset, "RandomSplit")

    @parameterized.expand([
        (False, ),
        (True, ),
    ])
    def testTemporalSplit(self, multitask):
        """Test the temporal split function, where the split is done based on a time
        property."""
        if multitask:
            dataset = self.createLargeMultitaskDataSet()
        else:
            dataset = self.createLargeTestDataSet()
        split_year = 2000
        split = TemporalSplit(
            timesplit=split_year,
            timeprop="Year of first disclosure",
        )
        dataset.addSplit(split, name="temp_split")
        self.checkSplit(dataset, "temp_split")

        # test if dates higher than 2000 are in test set
        train_ids, test_ids = dataset.getSplit("temp_split", as_type="ids")[0]
        years = dataset.getDF().loc[test_ids, "Year of first disclosure"]
        self.assertTrue(all(years > split_year))

        # test bootstrapping
        if multitask:
            dataset = self.createLargeMultitaskDataSet(
                name="TemporalSplit_bootstrap_mt"
            )
        else:
            dataset = self.createLargeTestDataSet(name="TemporalSplit_bootstrap")
        split = TemporalSplit(
            timesplit=[split_year - 1, split_year, split_year + 1],
            timeprop="Year of first disclosure",
        )
        bootstrap_split = BootstrapSplit(
            split=split,
            n_bootstraps=10,
        )
        for time, fold_info in zip(
            split.timeSplit, list(dataset.split(bootstrap_split))
        ):
            years = dataset.getDF().loc[fold_info[1], "Year of first disclosure"]
            self.assertTrue(all(years > time))

    @parameterized.expand(
        [
            (False, BemisMurckoRDKit(), None),
            (
                False,
                BemisMurcko(use_csk=True),
                [
                    "ScaffoldSplit_storage_library_000",
                    "ScaffoldSplit_storage_library_001",
                ],
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
        dataset.addSplit(split, name="scaffold_split")
        self.checkSplit(dataset, "scaffold_split")
        # check that smiles in custom_test_list are in the test set
        if custom_test_list:
            test_index = dataset.getSplit("scaffold_split", as_type="ids")[0][1]
            self.assertTrue(all(mol_id in test_index for mol_id in custom_test_list))
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
        for k, (train_index, test_index) in enumerate(dataset.split(split)):
            self.assertTrue(all(x not in test_index_all for x in test_index))
            self.assertTrue(len(train_index) > len(test_index))
            test_index_all.extend(test_index.tolist())
        self.assertEqual(k, n_folds - 1)
        self.assertEqual(len(test_index_all), len(dataset.getDescriptors()))

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
                [
                    "ClusterSplit_storage_library_000",
                    "ClusterSplit_storage_library_001",
                ],
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
                [
                    "ClusterSplit_storage_library_000",
                    "ClusterSplit_storage_library_001",
                ],
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
        dataset.addSplit(split, name="cluster_split")
        self.checkSplit(dataset, "cluster_split")
        # check that smiles in custom_test_list are in the test set
        if custom_test_list:
            test_index = dataset.getSplit("cluster_split", as_type="ids")[0][1]
            self.assertTrue(all(mol_id in test_index for mol_id in custom_test_list))

    def testSerialization(self):
        """Test the serialization of dataset with datasplit."""
        dataset = self.createLargeTestDataSet()
        split = ScaffoldSplit()
        dataset.addSplit(split, name="scaffold_split")
        self.checkSplit(dataset, "scaffold_split")
        train_ids, test_ids = dataset.getSplit("scaffold_split", as_type="ids")[0]
        dataset.save()
        dataset_new = QSPRTable.fromFile(dataset.metaFile)
        self.checkSplit(dataset_new, "scaffold_split")
        train_ids_new, test_ids_new = dataset_new.getSplit(
            "scaffold_split", as_type="ids"
        )[0]
        self.assertTrue(all(mol_id in train_ids_new for mol_id in train_ids))
        self.assertTrue(all(mol_id in test_ids_new for mol_id in test_ids))
        dataset_new.clear()


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
            y_train,
            X_test,
            y_test,
        ) in folds:
            k += 1
            self.assertEqual(len(X_train), len(y_train))
            self.assertEqual(len(X_test), len(y_test))
            tested_indices.extend(X_test.index.tolist())
            if more:
                more(X_train, X_test, y_train, y_test)
        return k, tested_indices

    def testStandardFolds(self):
        """Test the default fold generator, which is a 5-fold cross validation."""
        # test default settings with regression
        dataset = self.createLargeTestDataSet()
        dataset.addDescriptors([MorganFP(radius=3, nBits=128)])
        fold_split = KFold(5, shuffle=True, random_state=dataset.randomState)
        dataset.addSplit(fold_split, name="fold_split")
        k, indices = self.validateFolds(dataset.iterSplit("fold_split", as_type="pandas"))
        self.assertEqual(k, 5)
        df = dataset.getDF()
        self.assertFalse(set(df.index) - set(indices))
        # test default settings with classification
        dataset.makeClassification("CL", th=[20])
        fold_split = StratifiedKFold(5, shuffle=True, random_state=dataset.randomState)
        dataset.addSplit(fold_split, name="fold_split")
        k, indices = self.validateFolds(dataset.iterSplit("fold_split", as_type="pandas"))
        self.assertEqual(k, 5)
        self.assertFalse(set(df.index) - set(indices))

        # test in a pipeline (with a standarizer)
        MAX_VAL = 2
        MIN_VAL = 1
        scaler = MinMaxScaler(feature_range=(MIN_VAL, MAX_VAL))
        pipeline=DatasetPipeline(steps={"standardizer": scaler})

        def check_min_max(X_train, X_test, *args, **kwargs):
            self.assertTrue(np.max(X_train.values) == MAX_VAL)
            self.assertTrue(np.min(X_train.values) == MIN_VAL)
            self.assertTrue(np.max(X_test.values) == MAX_VAL)
            self.assertTrue(np.min(X_test.values) == MIN_VAL)

        self.validateFolds(pipeline.apply(dataset, "fold_split"), check_min_max)
        k, indices = self.validateFolds(dataset.iterSplit("fold_split", as_type="pandas"))
        self.assertEqual(k, 5)
        self.assertFalse(set(df.index) - set(indices))

        # try with a split data set
        train_ids, _ = next(dataset.split(RandomSplit(test_fraction=0.1)))
        train_set = dataset[train_ids]
        train_set.addSplit(fold_split, name="fold_split")
        k, indices = self.validateFolds(train_set.iterSplit("fold_split", as_type="pandas"))
        self.assertEqual(k, 5)
        self.assertFalse(set(train_ids) - set(indices))

    def testBootstrappedFold(self):
        dataset = self.createLargeTestDataSet(random_state=1)
        dataset.addDescriptors([MorganFP(radius=3, nBits=128)])
        split = RandomSplit(0.2)
        fold = BootstrapSplit(split, n_bootstraps=5)
        dataset.addSplit(fold, name="fold_split")
        k, indices = self.validateFolds(dataset.iterSplit("fold_split", as_type="pandas"))
        self.assertEqual(k, 5)
        # check if the indices are the same if we do the same split again
        split = RandomSplit(0.2)
        fold = BootstrapSplit(split, n_bootstraps=5, seed=dataset.randomState)
        dataset.addSplit(fold, name="fold_split2")
        k, indices_second = self.validateFolds(dataset.iterSplit("fold_split2", as_type="pandas"))
        self.assertEqual(k, 5)
        self.assertListEqual(indices, indices_second)
        # check if the indices are different if we do a different split
        split = RandomSplit(0.2)
        fold = BootstrapSplit(split, n_bootstraps=5, seed=42)
        dataset.addSplit(fold, name="fold_split3")
        k, indices_third = self.validateFolds(dataset.iterSplit("fold_split3", as_type="pandas"))
        self.assertEqual(k, 5)
        self.assertEqual(split.randomState, None)
        self.assertNotEqual(indices, indices_third)
