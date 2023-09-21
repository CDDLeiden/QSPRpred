"""Different splitters to create train and tests for evalutating QSPR model performance.

To add a new data splitter:
* Add a DataSplit subclass for your new splitter
"""
from typing import Iterable
import numpy as np
import pandas as pd
from gbmtsplits import GloballyBalancedSplit
from sklearn.model_selection import ShuffleSplit

from ...logs import logger
from ..data import QSPRDataset
from ..interfaces import DataSplit
from .data_clustering import (
    MoleculeClusters,
    FPSimilarityMaxMinClusters,
    ScaffoldClusters,
    RandomClusters,
)
from .scaffolds import Murcko, Scaffold

#TODO: reintroduce random seed in randomsplit below?
class RandomSplit(DataSplit):
    """Splits dataset in random train and test subsets.

    Attributes:
        testFraction (float): fraction of total dataset to testset
    """
    def __init__(self, test_fraction=0.1) -> None:
        self.testFraction = test_fraction

    def split(self, X, y):
        return ShuffleSplit(1, test_size=self.testFraction).split(X, y)

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
        timesplit: float,
        timeprop: str,
        dataset: QSPRDataset | None = None
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

    def split(self, X, y):
        """
        Split single-task dataset based on a time threshold.

        Returns:
            an iterator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix
        """

        # Get dataset, dataframe and tasks
        ds = self.getDataSet()
        df = ds.getDF().copy()
        task_names = [TargetProperty.name for TargetProperty in ds.targetProperties]

        assert len(task_names) > 0, "No target properties found."
        assert len(X) == len(df),\
            "X and the current data in the dataset must have same length"

        if len(task_names) > 1:
            logger.warning(
                "The TemporalSplit is not recommended for multitask\
                or PCM datasets might lead to very unbalanced subsets\
                for some tasks"
            )

        indices = np.array(list(range(len(df))))
        mask = df[self.timeCol] > self.timeSplit
        mask = mask.values
        test = indices[mask]

        # Check if there are any test samples for each task
        for task in task_names:
            if len(df[mask][task]) == 0:
                raise ValueError(f"No test samples found for task {task.name}")
            elif len(df[~mask][task]) == 0:
                raise ValueError(f"No train samples found for task {task.name}")

        train = indices[~mask]

        return iter([(train, test)])

class GBMTDataSplit(DataSplit):
    """
    Splits dataset into balanced train and test subsets
    based on an initial clustering algorithm.

    Attributes:
        dataset (QSPRDataset): dataset that this splitter will be acting on
        clustering (MoleculeClusters): clustering algorithm to use
        testFraction (float): fraction of total dataset to testset
        customTestList (list): list of molecule indexes to force in test set
        split_kwargs (dict): additional arguments to be passed to the GloballyBalancedSplit
    """

    def __init__(
        self,
        dataset: QSPRDataset = None,
        clustering : MoleculeClusters = FPSimilarityMaxMinClusters(),
        test_fraction: float = 0.1,
        custom_test_list: list[str] | None = None,
        **split_kwargs,
    ):
        super().__init__(dataset)
        self.testFraction = test_fraction
        self.customTestList = custom_test_list
        self.clustering = clustering
        self.split_kwargs = split_kwargs if split_kwargs else {}

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
            an iterator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix
        """

        # Get dataset, dataframe and tasks
        ds = self.getDataSet()
        df = ds.getDF().copy().reset_index(drop=True) # need numeric index splits
        task_names = [TargetProperty.name for TargetProperty in ds.targetProperties]

        assert len(task_names) > 0, "No target properties found."
        assert len(X) == len(df),\
            "X and the current data in the dataset must have same length"

        # Get clusters
        clusters = self.clustering.get_clusters(df[ds.smilesCol].tolist())

        # Pre-assign smiles of custom_test_list to test set
        preassigned_smiles = (
            {
                df.loc[df.QSPRID == qspridx][ds.smilesCol].values[0]: 1
                for qspridx in self.customTestList
            } if self.customTestList else None
        )

        print(self.split_kwargs)
        # Split dataset
        splitter = GloballyBalancedSplit(
            sizes=[1 - self.testFraction, self.testFraction],
            clusters=clusters,
            clustering_method=None, # As precomputed clusters are provided
            **self.split_kwargs,
        )
        df_split = splitter(
            df,
            ds.smilesCol,
            task_names,
            preassigned_smiles=preassigned_smiles,
        )

        # Get indices
        train_indices = df_split[df_split["Split"] == 0].index.values
        test_indices = df_split[df_split["Split"] == 1].index.values

        assert len(train_indices) + len(test_indices) == len(df), \
            "Not all samples were assigned to a split"

        # Reset index back to QSPRID
        df.set_index(ds.indexCols, inplace=True, drop=False)

        return iter([(train_indices, test_indices)])

class GBMTRandomSplit(GBMTDataSplit):
    """
    Splits dataset into balanced random train and test subsets.

    Attributes:
        dataset (QSPRDataset): dataset that this splitter will be acting on
        testFraction (float): fraction of total dataset to testset
        customTestList (list): list of molecule indexes to force in test set
        split_kwargs (dict): additional arguments to be passed to the GloballyBalancedSplit
    """
    def __init__(
        self,
        dataset: QSPRDataset | None = None,
        test_fraction: float = 0.1,
        seed: int | None = None,
        n_initial_clusters: int | None = None,
        custom_test_list: list[str] | None = None,
        **split_kwargs,
    ) -> None:
        seed = seed or (dataset.randomState if dataset is not None else None)
        super().__init__(
            dataset,
            RandomClusters(seed, n_initial_clusters),
            test_fraction,
            custom_test_list,
            **split_kwargs,
        )

class ScaffoldSplit(GBMTDataSplit):
    """
    Splits dataset into balanced train and test subsets based on molecular scaffolds.

    Attributes:
        dataset (QSPRDataset): dataset that this splitter will be acting on
        testFraction (float): fraction of total dataset to testset
        customTestList (list): list of molecule indexes to force in test set
        split_kwargs (dict): additional arguments to be passed to the GloballyBalancedSplit
    """
    def __init__(
        self,
        dataset: QSPRDataset | None = None,
        scaffold: Scaffold = Murcko(),
        test_fraction: float = 0.1,
        custom_test_list: list | None = None,
        **split_kwargs,
    ) -> None:
        super().__init__(
            dataset,
            ScaffoldClusters(scaffold),
            test_fraction,
            custom_test_list,
            **split_kwargs,
        )

class ClusterSplit(GBMTDataSplit):
    """
    Splits dataset into balanced train and test subsets based on clusters of similar
    molecules.

    Attributes:
        dataset (QSPRDataset): dataset that this splitter will be acting on
        testFraction (float): fraction of total dataset to testset
        customTestList (list): list of molecule indexes to force in test set
        split_kwargs (dict): additional arguments to be passed to the GloballyBalancedSplit
    """
    def __init__(
        self,
        dataset: QSPRDataset = None,
        test_fraction: float = 0.1,
        custom_test_list: list[str] | None = None,
        seed: int | None = None,
        clustering : MoleculeClusters | None = None,
        **split_kwargs,
    ) -> None:
        seed = seed or (dataset.randomState if dataset is not None else None)
        clustering = clustering if clustering is not None else FPSimilarityMaxMinClusters(seed=seed)
        super().__init__(
            dataset,
            clustering,
            test_fraction,
            custom_test_list,
            **split_kwargs,
        )
