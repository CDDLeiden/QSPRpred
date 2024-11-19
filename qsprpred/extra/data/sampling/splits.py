"""Different splitters to create train and tests for evalutating QSPR model performance.

To add a new data splitter:
* Add a datasplit subclass for your new splitter
"""

from typing import Iterable

import numpy as np

from qsprpred.data.sampling.splits import (
    DataSplit,
    ClusterSplit,
    GBMTDataSplit,
    GBMTRandomSplit,
    ScaffoldSplit,
)
from qsprpred.utils.interfaces.randomized import Randomized
import pandas as pd


class PCMSplit(DataSplit, Randomized):
    """
    Splits a dataset into train and test set such that the subsets are balanced with
    respect to each of the protein targets.

    This is done with https://github.com/sohviluukkonen/gbmt-splits, linear programming
    of initial clusters (random-, scaffold- or cluster-based) to get a balanced split.

    Attributes:
        dataset (PCMDataSet): The dataset to split.
        splitter (GBMTDataSplit): The splitter to use on the initial clusters.
        targetProp (pd.Series): The protein targets to balance the split on.
        seed (int): The random seed to use for the splitter if it is a RandomSplit or
            ClusterSplit (Can also be set on the splitter itself).
    """
    def __init__(self, splitter: GBMTDataSplit, target_prop: pd.Series, seed = None) -> None:
        super().__init__()
        self.splitter = splitter
        self.targetProp = target_prop

        # Check that splitter is either RandomSplit, ScaffoldSplit or ClusterSplit
        assert isinstance(
            self.splitter, (GBMTRandomSplit, ScaffoldSplit, ClusterSplit)
        ), "Splitter must be either RandomSplit, ScaffoldSplit or ClusterSplit!"

        if isinstance(self.splitter, (GBMTRandomSplit, ClusterSplit)):
            if seed is None:
                self.randomState = self.splitter.randomState
            else:
                self.randomState = seed

    @property
    def randomState(self) -> int:
        return self._seed

    @randomState.setter
    def randomState(self, seed: int | None):
        self._seed = seed
        if isinstance(self.splitter, (GBMTRandomSplit, ClusterSplit)):
            self.splitter.randomState = seed

    def split(self, X, y) -> Iterable[tuple[list[int], list[int]]]:
        """
        Split the PCM dataset into train and test set such that the subsets are balanced
        with respect to the protein targets and there is not data leakage between the
        train and test set.

        Converts the PCM dataset into a multi-task dataset with protein targets as
        columns and uses the given splitter to split the multi-task dataset.

        Args:
            X (np.ndarray | pd.DataFrame): the input data matrix
            y (np.ndarray | pd.DataFrame | pd.Series): the target variable(s)

        Returns:
            an generator over the generated subsets represented as a tuple of
            (train_indices, test_indices) where the indices are the row indices of the
            input data matrix X (note that these are integer indices, rather than a
            pandas index!)
        """
        assert (
            y.shape[1] == 1
        ), "PCMSplit only works for single-task datasets!"
        # TODO: Add support for multi-target (create a multi-task PCM dataset)
        # with all target-task combinations as different columns and split that
        # dataset with the given splitter
        # Pivot dataframe to get a matrix with protein targets as columns
        y_copy = y.copy()
        y_copy["SMILES"] = self.splitter.smilesProp
        y_copy["target_prop"] = self.targetProp
        y_copy.reset_index(drop=True, inplace=True)
        # create dataframe with SMILES as index and protein targets as columns
        # so each SMILES only appears once
        df_mt = y_copy.pivot(
            index="SMILES",
            columns="target_prop",
            values=y.columns[0],
        )
        # Fill NaN values with median of the column
        df_mt = df_mt.fillna(df_mt.median())
        
        # temporarily set the underlying splitter's smilesProp to the SMILES
        # to align them with the multi-task dataset
        self.splitter.smilesProp = pd.Series(df_mt.index, index=df_mt.index)
        
        for train_mt_index, test_mt_index in self.splitter.split(X=df_mt, y=df_mt):
            # Get the SMILES for the train and test set
            train_smiles = df_mt.iloc[train_mt_index].index
            test_smiles = df_mt.iloc[test_mt_index].index
            # Get the numeric indices of the SMILES in the original dataset
            train_indices = y_copy[y_copy["SMILES"].isin(train_smiles)].index.to_list()
            test_indices = y_copy[y_copy["SMILES"].isin(test_smiles)].index.to_list()
            yield train_indices, test_indices
        self.splitter.smilesProp = y_copy["SMILES"]


class LeaveTargetsOut(DataSplit):
    def __init__(self, targets: list[str], target_prop: pd.Series) -> None:
        """Creates a leave target out splitter.

        Args:
            targets (list): the identifiers of the targets to leave out as test set
            targetProp (pd.Series): the protein targets to balance the split on
        """
        self.targets = list(set(targets))
        self.targetProp = target_prop

    def split(self, X, y):
        mask = self.targetProp.isin(self.targets)
        mask = mask.loc[X.index].reset_index(drop=True)
        
        indices = np.array(list(range(len(X))))
        train = indices[mask]
        test = indices[~mask]
        return iter([(train, test)])


class TemporalPerTarget(DataSplit):
    def __init__(
        self,
        smiles_prop: pd.Series,
        target_prop: pd.Series,
        time_prop: pd.Series,
        split_time: dict[str, int],
        first_time_per_compound: bool = True,
    ):
        """Creates a temporal split that is consistent across targets.

        Args:
            smiles_prop (pd.Series):
                a series containing the smiles information
            target_prop (pd.Series):
                a series containing the target information
            time_prop (pd.Series):
                a series that contains the time information for the dataset (e.g. year)
            split_time (dict[str,int]):
                a dictionary with target keys as keys
                and split times as values
            first_time_per_compound (bool):
                if True, the first time a compound appears in the dataset is used
                for all targets
        """
        self.smilesProp = smiles_prop
        self.targetProp = target_prop
        self.splitTime = split_time
        self.timeProp = time_prop
        self.firstTimePerCompound = first_time_per_compound

    def split(self, X, y) -> Iterable[tuple[list[int], list[int]]]:
        # Add the smiles,  target and time properties to target values
        y_copy = y.copy()
        y_copy["smiles"] = self.smilesProp
        y_copy["target_prop"] = self.targetProp
        y_copy["time_prop"] = self.timeProp
        y_copy.reset_index(drop=True, inplace=True)

        # Set the first time a compound appears in the dataset as the time
        # of the compound for all targets
        if self.firstTimePerCompound:
            first_appearances = y_copy.groupby("smiles")["time_prop"].min()
            y_copy["time_prop"] = y_copy["smiles"].map(first_appearances)

        train_indices = []
        test_indices = []

        for target, split_year in self.splitTime.items():
            y_target = y_copy[y_copy["target_prop"] == target]
            # Get indices of the train and test set
            train = y_target[y_target["time_prop"] <= split_year].index.tolist()
            test = y_target[y_target["time_prop"] > split_year].index.tolist()
            # Check if there is data for the target before/after the split year
            if len(train) == 0:
                raise ValueError(
                    f"No training data for target {target} before {split_year}!"
                )
            elif len(test) == 0:
                raise ValueError(
                    f"No test data for target {target} after {split_year}!"
                )
            # Convert to numeric indices
            train_indices.extend(train)
            test_indices.extend(test)

        assert len(set(train_indices)) + len(
            set(test_indices)
        ) == len(y_copy), "Train and test set do not cover the whole dataset!"

        return iter([(train_indices, test_indices)])
