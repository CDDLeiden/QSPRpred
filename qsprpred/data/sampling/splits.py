"""Different splitters to create train and tests for evalutating QSPR model performance.

To add a new data splitter:
* Add a DataSplit subclass for your new splitter
"""
import platform
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import pandas as pd
from gbmtsplits import GloballyBalancedSplit
from sklearn.model_selection import ShuffleSplit

from ...data.chem.clustering import (
    FPSimilarityMaxMinClusters,
    MoleculeClusters,
    RandomClusters,
    ScaffoldClusters,
)
from ...data.chem.scaffolds import BemisMurckoRDKit, Scaffold
from ...data.tables.base import MoleculeDataTable, DataSetDependant
from ...data.tables.qspr import QSPRDataset
from ...logs import logger
from ...utils.interfaces.randomized import Randomized


class DataSplit(DataSetDependant, ABC):
    """
    Defines a function split a dataframe into train and test set.

    Attributes:
        dataset (MoleculeDataTable): The dataset to split.
    """

    def __init__(self, dataset: MoleculeDataTable | None = None) -> None:
        super().__init__(dataset)

    @abstractmethod
    def split(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame | pd.Series
    ) -> Iterable[tuple[list[int], list[int]]]:
        """Split the given data into one or multiple train/test subsets.

        These classes handle partitioning of a feature matrix
        by returning an generator of train
        and test indices. It is compatible with the approach taken
        in the `sklearn` package (see `sklearn.model_selection._BaseKFold`).
        This can be used for both cross-validation or a one time train/test split.

        Args:
            X (np.ndarray | pd.DataFrame): the input data matrix
            y (np.ndarray | pd.DataFrame | pd.Series): the target variable(s)

        Returns:
            an generator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix X (note that these are integer indices, rather than a
            pandas index!)
        """

    def splitDataset(self, dataset: "QSPRDataset"):
        return self.split(
            dataset.getFeatures(concat=True),
            dataset.getTargetPropertiesValues(concat=True),
        )


class RandomSplit(DataSplit, Randomized):
    """Splits dataset in random train and test subsets.

    Attributes:
        testFraction (float):
            fraction of total dataset to testset
        seed (int):
            Random state to use for shuffling and other random operations.
    """

    def __init__(
        self,
        test_fraction=0.1,
        dataset: QSPRDataset | None = None,
        seed: int | None = None,
    ) -> None:
        DataSplit.__init__(self, dataset)
        Randomized.__init__(self, seed)
        self.testFraction = test_fraction
        self.setSeed(seed or (dataset.randomState if self.hasDataSet else None))

    def split(self, X, y):
        if self.seed is None:
            self.seed = self.setSeed(
                self.getDataSet().randomState if self.hasDataSet else None
            )
        if self.seed is None:
            logger.info(
                "No random state supplied, "
                "and could not find random state on the dataset."
                "Random seed will be set randomly."
            )
        return ShuffleSplit(
            1, test_size=self.testFraction, random_state=self.seed
        ).split(X, y)


class BootstrapSplit(DataSplit, Randomized):
    """Splits dataset in random train and test subsets (bootstraps). Unlike
    cross-validation, bootstrapping allows for repeated samples in the test set.

    Attributes:
        nBootstraps (int):
            number of bootstraps to perform
        seed (int):
            Random state to use for shuffling and other random operations.
    """

    def __init__(self, split: DataSplit, n_bootstraps=5, seed=None):
        """Initialize a BootstrapSplit object.

        Args:
            split (DataSplit): the splitter to use for the bootstraps
            n_bootstraps (int): number of bootstraps to perform
            seed (int): random seed to use for random operations
        """
        Randomized.__init__(self, seed)
        self._split = split
        self._original_split_seed = split.seed if hasattr(split, "seed") else None
        self.nBootstraps = n_bootstraps
        self._current = 0

    def split(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame | pd.Series
    ) -> Iterable[tuple[list[int], list[int]]]:
        """Split the given data into `nBootstraps` training and test sets.

        Args:
            X (np.ndarray | pd.DataFrame): the input data matrix
            y (np.ndarray | pd.DataFrame | pd.Series): the target variable(s)

        Returns:
            an generator over `nBootstraps` tuples generated by the underlying splitter
        """
        if hasattr(self._split, "setDataSet") and self.hasDataSet:
            self._split.setDataSet(self.getDataSet())
        for i in range(self.nBootstraps):
            if hasattr(self._split, "setSeed") and self.seed is not None:
                self._split.setSeed(self.seed + self._current)
            yield from self._split.split(X, y)
            self._current += 1
        if hasattr(self._split, "setSeed"):
            self._split.setSeed(self._original_split_seed)
        self._current = 0


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
        super().__init__()
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
            an generator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix
        """
        train = self.splitCol[self.splitCol == self.trainVal].index.values
        test = self.splitCol[self.splitCol == self.testVal].index.values
        return iter([(train, test)])


class TemporalSplit(DataSplit):
    """
    Splits dataset train and test subsets based on a threshold in time.

    Attributes:
        dataset (QSPRDataset): dataset that this splitter will be acting on
        timeSplit(float): time point after which sample to test set
        timeCol (str): name of the column within the dataframe with timepoints
    """

    def __init__(
        self,
        timesplit: float | list[float],
        timeprop: str,
        dataset: QSPRDataset | None = None,
    ):
        """Initialize a TemporalSplit object.

        Args:
            dataset (QSPRDataset):
                dataset that this splitter will be acting on
            timesplit (float | list[float]):
                time point after which sample is moved to test set. If a list is
                provided, the splitter will split the dataset into multiple subsets
                based on the timepoints in the list.
            timeprop (str): name of the column within the dataframe with timepoints
        """
        super().__init__(dataset=dataset)
        self.timeSplit = timesplit
        self.timeCol = timeprop

    def split(self, X, y):
        """
        Split single-task dataset based on a time threshold.

        Returns:
            an generator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix
        """
        timesplits = (
            self.timeSplit
            if isinstance(self.timeSplit, list)
            else [
                self.timeSplit,
            ]
        )
        for timesplit in timesplits:
            # Get dataset, dataframe and tasks
            ds = self.getDataSet()
            df = ds.getDF().loc[X.index, :].copy()
            task_names = [TargetProperty.name for TargetProperty in ds.targetProperties]

            assert len(task_names) > 0, "No target properties found."
            assert len(X) == len(
                df
            ), "X and the current data in the dataset must have same length"

            if len(task_names) > 1:
                logger.warning(
                    "The TemporalSplit is not recommended for multitask "
                    "or PCM datasets might lead to very unbalanced subsets "
                    "for some tasks"
                )

            indices = np.array(list(range(len(df))))
            mask = df[self.timeCol] > timesplit
            mask = mask.values
            test = indices[mask]
            # Check if there are any test samples for each task
            for task in task_names:
                if len(df[mask][task]) == 0:
                    raise ValueError(f"No test samples found for task {task.name}")
                elif len(df[~mask][task]) == 0:
                    raise ValueError(f"No train samples found for task {task.name}")

            train = indices[~mask]

            yield train, test


class GBMTDataSplit(DataSplit):
    """
    Splits dataset into balanced train and test subsets
    based on an initial clustering algorithm. If `nFolds` is specified,
    the determined clusters will be split into `nFolds` groups of approximately
    equal size, and the splits will be generated by leaving out one group at a time.

    Attributes:
        dataset (QSPRDataset):
            dataset that this splitter will be acting on
        clustering (MoleculeClusters):
            clustering algorithm to use
        testFraction (float):
            fraction of total dataset to testset
        nFolds (int):
            number of folds to split the dataset into
            (this overrides `testFraction` and `customTestList`)
        customTestList (list):
            list of molecule indexes to force in test set
        split_kwargs (dict):
            additional arguments to be passed to the GloballyBalancedSplit
    """

    def __init__(
        self,
        dataset: QSPRDataset = None,
        clustering: MoleculeClusters = FPSimilarityMaxMinClusters(),
        test_fraction: float = 0.1,
        n_folds: int = 1,
        custom_test_list: list[str] | None = None,
        **split_kwargs,
    ):
        super().__init__(dataset)
        self.testFraction = test_fraction
        self.customTestList = custom_test_list
        self.clustering = clustering
        self.split_kwargs = split_kwargs if split_kwargs else {}
        self.nFolds = n_folds
        if self.nFolds > 1:
            self.testFraction = None
            self.customTestList = None

    def setDataSet(self, dataset: MoleculeDataTable):
        super().setDataSet(dataset)
        if self.nFolds > 1:
            self.testFraction = (len(dataset) / self.nFolds) / len(dataset)

    def split(
        self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.DataFrame | pd.Series
    ) -> Iterable[tuple[list[int], list[int]]]:
        """
        Split dataset into balanced train and test subsets
        based on an initial clustering algorithm.

        Args:
            X (np.ndarray | pd.DataFrame): the input data matrix
            y (np.ndarray | pd.DataFrame | pd.Series): the target variable(s)

        Returns:
            an generator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix
        """
        # if we are on Windows, raise an error
        if platform.system() == "Windows":
            logger.warning(
                "The GBMTDataSplit currently has a problem on Windows:"
                "https://github.com/coin-or/pulp/issues/671 and might hang up..."
            )
        # Get dataset, dataframe and tasks
        ds = self.getDataSet()
        df = ds.getDF().copy()  # need numeric index splits
        df = df.loc[X.index, :]
        df.reset_index(drop=True, inplace=True)
        task_names = [TargetProperty.name for TargetProperty in ds.targetProperties]
        assert len(task_names) > 0, "No target properties found."
        assert len(X) == len(
            df
        ), "X and the current data in the dataset must have same length"
        # Get clusters
        clusters = self.clustering.get_clusters(df[ds.smilesCol].tolist())
        # Pre-assign smiles of custom_test_list to test set
        preassigned_smiles = (
            {
                df.loc[df.QSPRID == qspridx][ds.smilesCol].values[0]: 1
                for qspridx in self.customTestList
            }
            if self.customTestList
            else None
        )
        logger.debug(f"Split arguments: {self.split_kwargs}")
        # Split dataset
        if self.nFolds == 1:
            sizes = [1 - self.testFraction, self.testFraction]
        else:
            sizes = [self.testFraction] * self.nFolds
        splitter = GloballyBalancedSplit(
            sizes=sizes,
            clusters=clusters,
            clustering_method=None,  # As precomputed clusters are provided
            **self.split_kwargs,
        )
        df_split = splitter(
            df,
            ds.smilesCol,
            task_names,
            preassigned_smiles=preassigned_smiles,
        )
        # Get indices
        for split in (
            df_split["Split"].unique()
            if self.nFolds > 1
            else [
                1,
            ]
        ):
            split = int(split)
            test_indices = df_split[df_split["Split"] == split].index.values
            train_indices = df_split[df_split["Split"] != split].index.values
            assert len(train_indices) + len(test_indices) == len(
                df
            ), "Not all samples were assigned to a split"
            # Reset index back to QSPRID
            df.set_index(ds.indexCols, inplace=True, drop=False)
            yield train_indices, test_indices


class GBMTRandomSplit(GBMTDataSplit):
    """
    Splits dataset into balanced random train and test subsets.

    Attributes:
        dataset (QSPRDataset):
            dataset that this splitter will be acting on
        testFraction (float):
            fraction of total dataset to testset
        seed (int):
            Random state to use for shuffling and other random operations.
        customTestList (list):
            list of molecule indexes to force in test set
        split_kwargs (dict):
            additional arguments to be passed to the GloballyBalancedSplit
    """

    def __init__(
        self,
        dataset: QSPRDataset | None = None,
        test_fraction: float = 0.1,
        n_folds: int = 1,
        seed: int | None = None,
        n_initial_clusters: int | None = None,
        custom_test_list: list[str] | None = None,
        **split_kwargs,
    ) -> None:
        seed = seed or (dataset.randomState if dataset is not None else None)
        if seed is None:
            logger.info(
                "No random state supplied, "
                "and could not find random state on the dataset."
                "Random seed will be set randomly."
            )

        super().__init__(
            dataset,
            RandomClusters(seed, n_initial_clusters),
            test_fraction,
            n_folds,
            custom_test_list,
            **split_kwargs,
        )


class ScaffoldSplit(GBMTDataSplit):
    """
    Splits dataset into balanced train and test subsets based on molecular scaffolds.

    Attributes:
        dataset (QSPRDataset):
            dataset that this splitter will be acting on
        testFraction (float):
            fraction of total dataset to testset
        customTestList (list):
            list of molecule indexes to force in test set
        split_kwargs (dict):
            additional arguments to be passed to the GloballyBalancedSplit
    """

    def __init__(
        self,
        dataset: QSPRDataset | None = None,
        scaffold: Scaffold = BemisMurckoRDKit(),
        test_fraction: float = 0.1,
        n_folds: int = 1,
        custom_test_list: list | None = None,
        **split_kwargs,
    ) -> None:
        super().__init__(
            dataset,
            ScaffoldClusters(scaffold),
            test_fraction,
            n_folds,
            custom_test_list,
            **split_kwargs,
        )


class ClusterSplit(GBMTDataSplit):
    """
    Splits dataset into balanced train and test subsets based on clusters of similar
    molecules.

    Attributes:
        dataset (QSPRDataset):
            dataset that this splitter will be acting on
        testFraction (float):
            fraction of total dataset to testset
        customTestList (list):
            list of molecule indexes to force in test set
        seed (int):
            Random state to use for shuffling and other random operations.
        split_kwargs (dict):
            additional arguments to be passed to the GloballyBalancedSplit
    """

    def __init__(
        self,
        dataset: QSPRDataset = None,
        test_fraction: float = 0.1,
        n_folds: int = 1,
        custom_test_list: list[str] | None = None,
        seed: int | None = None,
        clustering: MoleculeClusters | None = None,
        **split_kwargs,
    ) -> None:
        seed = seed or (dataset.randomState if dataset is not None else None)
        if seed is None:
            logger.info(
                "No random state supplied, "
                "and could not find random state on the dataset."
                "Random seed will be set randomly."
            )

        clustering = (
            clustering
            if clustering is not None
            else FPSimilarityMaxMinClusters(seed=seed)
        )
        super().__init__(
            dataset,
            clustering,
            test_fraction,
            n_folds,
            custom_test_list,
            **split_kwargs,
        )

    def setSeed(self, seed: int | None):
        """Set the seed for this instance.

        Args:
            seed (int):
                Random state to use for shuffling and other random operations.
        """
        self.seed = seed
        if hasattr(self.clustering, "seed"):
            self.clustering.seed = seed

    def getSeed(self):
        """Get the seed for this instance.

        Returns:
            int: the seed for this instance or None if no seed is set.
        """
        if hasattr(self, "seed"):
            return self.seed
        else:
            return None
