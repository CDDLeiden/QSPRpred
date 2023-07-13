"""Different splitters to create train and tests for evalutating QSPR model performance.

To add a new data splitter:
* Add a DataSplit subclass for your new splitter
"""
from collections import defaultdict
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd
from gbmtsplits import GloballyBalancedSplit
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from ...logs import logger
from ..data import QSPRDataset
from ..interfaces import DataSetDependant, DataSplit
from .data_clustering import (
    FPSimilarityLeaderPickerClusters,
    FPSimilarityMaxMinClusters,
    MurckoScaffoldClusters,
    RandomClusters,
)
from .descriptorsets import FingerprintSet
from .scaffolds import Murcko, Scaffold

# TODO: Check if the order of molecules in (X,y) is the same as in the dataframe from dataset


class ManualSplit(DataSplit):
    """Splits dataset in train and test subsets based on a column in the dataframe.

    Attributes:
        splitCol (pd.Series): pandas series with split information
        trainVal (str): value in splitcol that will be used for training
        testVal (str): value in splitcol that will be used for testing

    Raises:
        ValueError: if there are more values in splitcol than trainval and testval
    """
    def __init__(self, splitcol: pd.Series, trainval: str, testval: str) -> None:
        """Initialize the ManualSplit object with the splitcol, trainval and testval
        attributes.

        Args:
            splitCol (pd.Series): pandas series with split information
            trainVal (str): value in splitcol that will be used for training
            testVal (str): value in splitcol that will be used for testing

        Raises:
            ValueError: if there are more values in splitcol than trainval and testval
        """
        self.splitCol = splitcol.reset_index(drop=True)
        self.trainVal = trainval
        self.testVal = testval
        # check if only trainval and testval are present in splitcol
        if not set(splitcol.unique()).issubset({trainval, testval}):
            raise ValueError(
                "There are more values in splitcol than trainval and testval"
            )

    def split(self, X, y):
        """
        Split the given data into one or multiple train/test subsets based on the
        predefined splitcol.

        Args:
            X (np.ndarray | pd.DataFrame): the input data matrix
            y (np.ndarray | pd.DataFrame | pd.Series): the target variable(s)

        Returns:
            an iterator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix
        """
        train = self.splitCol[self.splitCol == self.trainVal].index.values
        test = self.splitCol[self.splitCol == self.testVal].index.values
        return iter([(train, test)])

class RandomSplit(DataSplit):
    """Splits dataset in random train and test subsets.

    Attributes:
        dataSet (QSPRDataset): dataset that this splitter will be acting on
        testFraction (float): fraction of total dataset to testset
        seed (int): random seed for reproducibility
        nInitialClusters (int): number of initial random clusters used by
        the multitask splitter
    """
    def __init__(
            self,
            dataset: QSPRDataset | None= None,
            test_fraction: float = 0.1,
            seed: int = 42,
            n_initial_clusters: int | None = None,
    ) -> None:
        super().__init__(dataset)
        self.testFraction = test_fraction
        self.seed = seed
        self.nInitialClusters = n_initial_clusters

    def _singletask_split(self) -> Iterable[tuple[list[int], list[int]]]:
        """
        Split single-task dataset in two subsets with shuffled random sampling.
        In the case classification, stratified sampling is used.

        Returns:
            an iterator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix
        """
        task = self.tasks[0].task
        if task.isRegression():
            splitter = ShuffleSplit(
                1, test_size=self.testFraction, random_state=self.seed
            )
        elif task.isClassification():
            splitter = StratifiedShuffleSplit(
                1, test_size=self.testFraction, random_state=self.seed
            )

        return splitter.split(self.X, self.y)

    def _multitask_split(self) -> Iterable[tuple[list[int], list[int]]]:
        """
        Split multi-task dataset in two subsets with globally balanced random
        sampling from https://github.com/sohviluukkonen/gbmt-splits

        Returns:
            an iterator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix
        """

        # Dataframe and columns
        smiles_col = self.getDataSet().smilesCol
        target_cols = [TargetProperty.name for TargetProperty in self.tasks]

        # Initial random clustering
        clustering = RandomClusters(seed=self.seed, n_clusters=self.nInitialClusters)
        clusters = clustering.get_clusters(self.df[smiles_col].tolist())

        # Split dataset
        splitter = GloballyBalancedSplit(
            sizes=[1 - self.testFraction, self.testFraction],
            clusters=clusters,
            clustering_method=None,
            time_limit_seconds=60 * len(self.tasks),
        )
        df = splitter(self.df, smiles_col, target_cols)

        # Get indices
        train = df[df["Split"] == 0].index.values
        test = df[df["Split"] == 1].index.values

        # Check that all indices are used
        assert len(train) + len(test) == len(df),\
            "Not all samples were assigned to a split"

        return iter([(train, test)])


class TemporalSplit(DataSplit, DataSetDependant):
    """Splits dataset train and test subsets based on a threshold in time.

    Attributes:
        dataSet (QSPRDataset): dataset that this splitter will be acting on
        timeSplit(float): time point after which sample to test set
        timeCol (str): name of the column within the dataframe with timepoints
    """
    def __init__(
            self,timesplit: float, timeprop: str, dataset: QSPRDataset | None = None
    ) -> None:
        """Initialize a TemporalSplit object.

        Args:
            dataset (QSPRDataset): dataset that this splitter will be acting on
            timesplit(float): time point after which sample to test set
            timeprop (str): name of the column within the dataframe with timepoints
        """
        super().__init__(dataset=dataset)
        self.timeSplit = timesplit
        self.timeCol = timeprop

    def _singletask_split(self):
        """
        Split single-task dataset based on a time threshold.

        Returns:
            an iterator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix
        """

        indices = np.array(list(range(len(self.df))))
        mask = self.df[self.timeCol] > self.timeSplit
        mask = mask.values
        test = indices[mask]
        assert len(test) > 0, "No test samples found"
        train = indices[~mask]
        return iter([(train, test)])

    def _multitask_split(self):
        """
        Split multi-task dataset based on a time threshold.

        Returns:
            an iterator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix
        """
        logger.warning(
            "The TemporalSplit is not recommended for multitask\
            or PCM datasets might lead to very unbalanced subsets\
            for some tasks"
        )

        indices = np.array(list(range(len(self.df))))
        mask = self.df[self.timeCol] > self.timeSplit
        mask = mask.values
        test = indices[mask]

        # Check if there are any test samples for each task
        for task in self.tasks:
            if len(self.df[mask][task.name]) == 0:
                raise ValueError(f"No test samples found for task {task.name}")
            elif len(self.df[~mask][task.name]) == 0:
                raise ValueError(f"No train samples found for task {task.name}")

        train = indices[~mask]

        assert len(test) + len(train) == len(self.df), \
            "Not all samples were assigned to a split"

        return iter([(train, test)])


class ScaffoldSplit(DataSplit, DataSetDependant):
    """Splits dataset in train and test subsets based on their Murcko scaffold.

    Attributes:
        dataSet: QSPRDataset object.
        scaffold (qsprpred.data.utils.scaffolds.Scaffold()): `Murcko()` and
            `BemisMurcko()` are currently available, other types can be added through
            the abstract class `Scaffold`. Defaults to Murcko().
        testFraction (float): fraction of the test set. Defaults to 0.1.
        shuffle (bool): whether to shuffle the data or not. Defaults to True.
        customTestList (list): list of molecule indexes to force in test set. If
            forced test contains the totality of the molecules in the dataset, the
            custom_test_list reverts to default None.
    """
    def __init__(
        self,
        scaffold: Scaffold = Murcko(),
        test_fraction: float = 0.1,
        shuffle: bool = True,
        custom_test_list: list | None = None,
        dataset: QSPRDataset | None = None,
    ) -> None:
        super().__init__(dataset)
        self.scaffold = scaffold
        self.testFraction = test_fraction
        self.shuffle = shuffle
        self.customTestList = custom_test_list

    def _singletask_split(self) -> Iterable[tuple[list[int], list[int]]]:
        """
        Single-task dataset split based on scaffold.

        Returns:
            an iterator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix
        """
        # Add scaffolds to dataset
        dataset = self.getDataSet()
        dataset.addScaffolds([self.scaffold])
        # Shuffle dataset
        if self.shuffle:
            dataset.shuffle()
        # Get dataframe
        df = dataset.getDF()
        # Get scaffolds
        scaffold_list = df[[f"Scaffold_{self.scaffold}"]]
        scaffolds = defaultdict(list)
        # Get invalid scaffolds
        invalid_idx = []
        for idx, scaffold in scaffold_list.itertuples():
            if scaffold:
                scaffolds[scaffold].append(idx)
            else:
                logger.warning(
                    f"Invalid scaffold skipped for molecule with index: {idx}"
                )
                invalid_idx.append(idx)
        # Fill test set with groups of smiles with the same scaffold
        max_in_test = np.ceil(len(df) * self.testFraction)
        test_idx = []
        # Start filling test with scaffold groups that contain the smiles in input list
        if self.customTestList is not None:
            for _, scaffold_idx in scaffolds.items():
                if bool(set(scaffold_idx) & set(self.customTestList)):
                    test_idx.extend(scaffold_idx)
        # Revert to default scaffold grouping if all molecules are placed in test set
        if len(test_idx) > max_in_test:
            logger.warning(
                "Warning: Test set includes all molecules in custom_test_list but is "
                "now bigger than specified fraction"
            )
        try:
            assert len(test_idx) + len(invalid_idx) < len(
                df
            ), "Test set cannot contain the totality of the data"
        except AssertionError:
            logger.warning(
                "Warning: Test set cannot contain the totality of the data. "
                "Ignoring custom_test_list input."
            )
            test_idx = []
        # Continue filling until the test fraction is reached
        for _, scaffold_idx in scaffolds.items():
            if len(test_idx) < max_in_test:
                test_idx.extend(scaffold_idx)
            else:
                break
        # Get train and test set indices
        train_idx = [df.index.get_loc(x) for x in df.index if x not in test_idx]
        test_idx = [df.index.get_loc(x) for x in test_idx]
        return iter([(train_idx, test_idx)])

    def _multitask_split(self) -> Iterable[tuple[list[int], list[int]]]:
        """
        Split multi-task dataset in two subsets with globally balanced scaffold-based
        split from https://github.com/sohviluukkonen/gbmt-splits

        Returns:
            an iterator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix
        """

        if not isinstance(self.scaffold, Murcko):
            raise NotImplementedError("Multitask scaffold split only supports Murcko()")

        if self.customTestList is not None:
            raise NotImplementedError(
                "Multitask scaffold split does not support custom_test_list"
            )

        # Dataframe and columns
        df = self.getDataSet().getDF().copy()
        smiles_col = self.getDataSet().smilesCol
        target_cols = [TargetProperty.name for TargetProperty in self.tasks]

        # Initial clusters
        clustering = MurckoScaffoldClusters()
        clusters = clustering.get_clusters(df[smiles_col].tolist())

        # Split dataset
        splitter = GloballyBalancedSplit(
            sizes=[1 - self.testFraction, self.testFraction],
            clusters=clusters,
            clustering_method=None,
            time_limit_seconds=60 * len(self.tasks),
        )
        df = splitter(df, smiles_col, target_cols)

        # Get indices
        train_indices = df[df["Split"] == 0].index.values
        test_indices = df[df["Split"] == 1].index.values

        assert len(train_indices) + len(test_indices) == len(df), \
            "Not all samples were assigned to a split"

        return iter([(train_indices, test_indices)])


class ClusterSplit(DataSplit, DataSetDependant):
    """
    Splits dataset in train and test subsets based on MaxMin clustering of the features.

    Attributes:
        dataSet: QSPRDataset object.
        testFraction (float): fraction of the test set. Defaults to 0.1.
        fpCalculator (FingerprintSet): fingerprint calculator.
            Defaults to MorganFP with radius 3 and 2048 bits.
        customTestList (list): list of molecule indexes to force in test set.
            If forced test contains the totality of the molecules in the dataset,
            the custom_test_list reverts to default None.
        nInitialClusters (int): number of initial clusters. Defaults to None.
        seed (int): random seed. Used in case of 'MaxMin' clustering. Defaults to 42.
        similarityThreshold (float): similarity threshold for clustering. Used in case
            of 'LeaderPicker' clustering. Defaults to 0.7.
    """
    def __init__(
        self,
        test_fraction: float = 0.1,
        custom_test_list: list[str] | None = None,
        dataset: QSPRDataset = None,
        fp_calculator: FingerprintSet = FingerprintSet(
            fingerprint_type="MorganFP", radius=3, nBits=2048
        ),
        n_initial_clusters: int | None = None,
        seed: int = 42,
        similarity_threshold: float = 0.7,
        clustering_algorithm: Literal["MaxMin", "LeaderPicker"] = "MaxMin",
    ) -> None:
        super().__init__(dataset)
        self.testFraction = test_fraction
        self.customTestList = custom_test_list
        self.fpCalculator = fp_calculator
        self.nInitialClusters = n_initial_clusters
        self.seed = seed
        self.similarityThreshold = similarity_threshold
        self.clusteringAlgorithm = clustering_algorithm

    def _cluster_molecules(self):
        """
        Cluster molecules based on fingerprints.

        Returns:
            dict: dictionary of clusters. Keys are cluster indexes and values are lists
                of molecule indexes.
        """

        if self.clusteringAlgorithm == "MaxMin":
            clustering = FPSimilarityMaxMinClusters(
                n_clusters=self.nInitialClusters,
                seed=self.seed,
                fp_calculator=self.fpCalculator,
                initial_centroids=self.customTestList,
            )
        elif self.clusteringAlgorithm == "LeaderPicker":
            clustering = FPSimilarityLeaderPickerClusters(
                fp_calculator=self.fpCalculator,
                similarity_threshold=self.similarityThreshold,
            )
        else:
            raise ValueError(
                f"clustering_algorithm must be either 'MaxMin' \
                    or 'LeaderPicker', got {self.clusteringAlgorithm}"
            )
        clusters = clustering.get_clusters(self.df[self.dataset.smilesCol].tolist())

        return clusters

    def _singletask_split(self) -> Iterable[tuple[list[int], list[int]]]:
        """
        Split single-task dataset in two subsets based on clusters of fingerprints.

        Returns:
            an iterator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix
        """

        self.df.reset_index(drop=True, inplace=True)

        if self.customTestList is not None:
            assert self.clusteringAlgorithm == "MaxMin", \
                "custom_test_list only supported for MaxMin clustering"

            assert set(self.customTestList).issubset(self.df.QSPRID),\
                "custom_test_list contains invalid indexes"

            if self.nInitialClusters is None:
                self.nInitialClusters = len(self.df) // 100
            assert len(self.customTestList) < self.nInitialClusters, \
                "Number of molecules in custom_test_list must be less than \
                    n_initial_clusters. Try increasing n_initial_clusters."

            # Convert QSPRID indices into numerical indices, which are used
            # internally by the clustering algorithm
            self.customTestList = [
                int(self.df[self.df.QSPRID == x].index[0]) for x in self.customTestList
            ]

        # Initial clusters
        clusters = self._cluster_molecules()

        # Fill the test set with molecules from the clusters until the test fraction
        # is reached. Iterating clusters in order to ensure that the custom_test_list
        # molecules are included in the test set.
        max_in_test = np.ceil(len(self.df) * self.testFraction)
        test_idx = []
        for i, cluster in clusters.items():
            test_idx.extend(cluster)
            if len(test_idx) > max_in_test:
                if self.customTestList is None:
                    # Stop filling test set if it is bigger than specified fraction
                    break
                elif i >= len(self.customTestList):
                    # Stop filling test set if it is bigger than specified fraction
                    break
                else:
                    # Except, keep filling test set if still molecules from
                    # custom_test_list to be added
                    logger.warning(
                        "Warning: Test set includes all molecules in custom_test_list \
                            but is now bigger than specified fraction"
                    )

        # Get train set indices
        train_idx = list(set(self.df.index) - set(test_idx))

        # Reset index back to QSPRID
        self.df.set_index("QSPRID", inplace=True)

        return iter([(train_idx, test_idx)])

    def _multitask_split(self,) -> Iterable[tuple[list[int], list[int]]]:
        """
        Split multi-task dataset in two subsets based on clusters of fingerprints
        with globally balanced split from https://github.com/sohviluukkonen/gbmt-splits

        Returns:
            an iterator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix
        """

        if self.customTestList is not None:
            raise NotImplementedError(
                "Multitask cluster split does not support custom_test_list"
            )

        # Initial clusters
        clusters = self._cluster_molecules()

        # Split dataset
        splitter = GloballyBalancedSplit(
            sizes=[1 - self.testFraction, self.testFraction],
            clusters=clusters,
            clustering_method=None,
            time_limit_seconds=60 * len(self.tasks),
        )
        df = splitter(
            self.df,
            self.dataset.smilesCol,
            [TargetProperty.name for TargetProperty in self.tasks],
        )

        # Get indices
        train_indices = df[df["Split"] == 0].index.values
        test_indices = df[df["Split"] == 1].index.values

        assert len(train_indices) + len(test_indices) == len(df), \
            "Not all samples were assigned to a split"

        return iter([(train_indices, test_indices)])
