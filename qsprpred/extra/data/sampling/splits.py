"""Different splitters to create train and tests for evalutating QSPR model performance.

To add a new data splitter:
* Add a datasplit subclass for your new splitter
"""

from typing import Iterable

import numpy as np
from sklearn.impute import SimpleImputer

from qsprpred.data.sampling.splits import (
    DataSplit,
    ClusterSplit,
    GBMTDataSplit,
    GBMTRandomSplit,
    ScaffoldSplit,
)
from qsprpred.data.tables.qspr import QSPRTable
from qsprpred.extra.data.tables.pcm import PCMDataSet
from qsprpred.tasks import TargetProperty
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
        seed (int): The random seed to use for the splitter if it is a RandomSplit or
            ClusterSplit (Can also be set on the splitter itself).
    """
    def __init__(self, splitter: GBMTDataSplit, proteins: pd.Series, seed = None) -> None:
        super().__init__()
        self.splitter = splitter
        self.proteins = proteins

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
        y_copy["protein"] = self.proteins
        y_copy.reset_index(drop=True, inplace=True)
        # create dataframe with SMILES as index and protein targets as columns
        # so each SMILES only appears once
        df_mt = y_copy.pivot(
            index="SMILES",
            columns="protein",
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
            
            
        
        # df_mt = df.pivot(
        #     index=ds.smilesProp,
        #     columns=ds.proteinIDProp,
        #     values=ds.targetProperties[0].name,
        # ).reset_index()
        # # Create target properties for multi-task dataset
        # mt_targetProperties = [
        #     TargetProperty(
        #         name=target, task=task, th=th, imputer=SimpleImputer(strategy="median")
        #     ) for target in proteins
        # ]
        # # temporarily create multi-task dataset and split it with the given splitter
        # ds_mt = QSPRTable.fromDF(
        #     name=f"PCM_{self.splitter.__class__.__name__}_{hash(self)}",
        #     df=df_mt,
        #     smiles_col=ds.smilesProp,
        #     target_props=mt_targetProperties,
        # )
        # ds_mt.randomState = ds.randomState
        # ds_mt.split(self.splitter)
        # # Convert MT indices to indices of original PCM dataset
        # test_indices = []
        # for i in ds_mt.X_ind.index:
        #     # Get SMILES and non-NaN targets for index i
        #     smiles = df_mt.loc[i, ds_mt.smilesProp]
        #     cols = df_mt.loc[i, :].dropna().index
        #     targets = [col for col in cols if col in proteins]
        #     for target in targets:
        #         # Get index in the original PCM dataset the SMILES-target pair
        #         a = df[ds.smilesProp] == smiles
        #         b = df[ds.proteinIDProp] == target
        #         if any(a & b):
        #             ds_idx = df[a & b].index.astype(str)[0]
        #             # Convert to numeric index
        #             test_indices.append(indices.index(ds_idx))
        # train_indices = [i for i in range(len(df)) if i not in test_indices]
        # return iter([(train_indices, test_indices)])


class LeaveTargetsOut(DataSplit):
    def __init__(self, targets: list[str], dataset: PCMDataSet | None = None):
        """Creates a leave target out splitter.

        Args:
            targets (list): the identifiers of the targets to leave out as test set
            dataset (PCMDataset): a `PCMDataset` instance to split
        """

        super().__init__(dataset)
        self.targets = list(set(targets))

    def split(self, X, y):
        ds = self.getDataSet()
        ds_targets = ds.getProteinKeys()
        for target in self.targets:
            assert target in ds_targets, f"Target key '{target}' not in dataset!"
            ds_targets.remove(target)
        mask = ds.getProperty(ds.proteinIDProp).isin(ds_targets).values
        indices = np.array(list(range(len(ds))))
        train = indices[mask]
        test = indices[~mask]
        return iter([(train, test)])


class TemporalPerTarget(DataSplit):
    def __init__(
        self,
        year_col: str,
        split_years: dict[str, int],
        firts_year_per_compound: bool = True,
        dataset: PCMDataSet | None = None,
    ):
        """Creates a temporal split that is consistent across targets.

        Args:
            year_col (str):
                the name of the column in the dataframe that
                contains the year information
            split_years (dict[str,int]):
                a dictionary with target keys as keys
                and split years as values
            firts_year_per_compound (bool):
                if True, the first year a compound appears in the dataset is used
                for all targets
            dataset (PCMDataset):
                a `PCMDataset` instance to split
        """
        super().__init__(dataset)
        self.splitYears = split_years
        self.yearCol = year_col
        self.firstYearPerCompound = firts_year_per_compound

    def split(self, X, y) -> Iterable[tuple[list[int], list[int]]]:
        ds = self.getDataSet()
        df = ds.getDF()
        indices = df.index.tolist()

        # Set the first year a compound appears in the dataset as the year
        # of the compound for all targets
        if self.firstYearPerCompound:
            first_years = df.groupby(ds.smilesProp)[self.yearCol].min()
            df[self.yearCol + "_first"] = df[ds.smilesProp].map(first_years)
            self.yearCol += "_first"

        train_indices = []
        test_indices = []

        for target, split_year in self.splitYears.items():
            df_target = df[df[ds.proteinIDProp] == target]
            # Get indices of the train and test set
            train = df_target[df_target[self.yearCol] <= split_year].index.tolist()
            test = df_target[df_target[self.yearCol] > split_year].index.tolist()
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
            train_indices.extend([indices.index(i) for i in train])
            test_indices.extend([indices.index(i) for i in test])

        assert len(set(train_indices)) + len(
            set(test_indices)
        ) == len(ds), "Train and test set do not cover the whole dataset!"

        return iter([(train_indices, test_indices)])
