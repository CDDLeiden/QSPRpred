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
from qsprpred.data.tables.interfaces.data_set_dependent import DataSetDependent
import pandas as pd
from qsprpred.extra.data.tables.pcm import PCMDataSet

from qsprpred.data.tables.qspr import QSPRTable
from qsprpred.tasks import TargetSpec
from sklearn.impute import SimpleImputer


class PCMSplit(DataSplit, Randomized, DataSetDependent):
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
        data_set (PCMDataSet): The data set attached to this object.
    """
    def __init__(
        self,
        splitter: GBMTDataSplit,
        seed = None,
        data_set: PCMDataSet | None = None
    ) -> None:
        super().__init__(data_set)
        self.splitter = splitter

        # Check that splitter is either GBMTRandomSplit, ScaffoldSplit or ClusterSplit
        assert isinstance(
            self.splitter, (GBMTRandomSplit, ScaffoldSplit, ClusterSplit)
        ), "Splitter must be either GBMTRandomSplit, ScaffoldSplit or ClusterSplit!"

        if hasattr(self.splitter, "randomState"):
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
        if hasattr(self.splitter, "randomState"):
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
        assert self.hasDataSet, (
            "No dataset attached to this splitter, set dataset with setDataSet()"
        )
        assert isinstance(self.getDataSet(), PCMDataSet), (
            "PCMSplit only works for PCM datasets, set a PCMDataSet with setDataSet()"
        )

        ds = self.getDataSet()
        df = ds.getDF()
        indices = df.index.tolist()
        proteins = df[ds.proteinIDProp].unique()
        task = ds.targetProperties[0].task
        th = ds.targetProperties[0].th if task.isClassification() else None
        # TODO: Add support for multi-target (create a multi-task PCM dataset)
        # with all target-task combinations as different columns and split that
        # dataset with the given splitter
        assert (
            len(ds.targetProperties) == 1
        ), "PCMSplit only works for single-task datasets!"

        df_mt = df.pivot(
            index=ds.smilesProp,
            columns=ds.proteinIDProp,
            values=ds.targetProperties[0].name,
        ).reset_index()
        # Create target properties for multi-task dataset
        mt_targetProperties = [
            TargetSpec(
                name=target, task=task, th=th
            ) for target in proteins
        ]
        # temporarily create multi-task dataset and split it with the given splitter
        ds_mt = QSPRTable.fromDF(
            name=f"PCM_{self.splitter.__class__.__name__}_{hash(self)}",
            df=df_mt,
            smiles_col=ds.smilesProp,
            target_props=mt_targetProperties,
            random_state=ds.randomState,
            drop_empty_target_props=False,
        )
        # impute missing values in the multi-task dataset
        # FIXME: this is not very intuitive, applying a transformation or step
        # directly on the dataset values should be simplified
        values = pd.DataFrame(
            SimpleImputer(strategy="median").fit_transform(ds_mt.getTargets()),
            columns=ds_mt.getTargetPropertiesNames()
        )
        for target_prop in ds_mt.targetProperties:
            ds_mt.addProperty(
                target_prop.name,
                values[target_prop.name]
            )
        _, mt_test_indices = next(ds_mt.split(self.splitter))

        # Convert MT indices to indices of original PCM dataset
        test_indices = []
        for i in mt_test_indices:
            # Get SMILES and non-NaN targets for index i
            smiles = df_mt.loc[i, ds_mt.smilesProp]
            cols = df_mt.loc[i, :].dropna().index
            targets = [col for col in cols if col in proteins]
            for target in targets:
                # Get index in the original PCM dataset the SMILES-target pair
                a = df[ds.smilesProp] == smiles
                b = df[ds.proteinIDProp] == target
                if any(a & b):
                    ds_idx = df[a & b].index.astype(str)[0]
                    # Convert to numeric index
                    test_indices.append(indices.index(ds_idx))
        train_indices = [i for i in range(len(df)) if i not in test_indices]
        return iter([(train_indices, test_indices)])


class LeaveTargetsOut(DataSplit, DataSetDependent):
    def __init__(self, targets: list[str], data_set: PCMDataSet | None = None):
        """Creates a leave target out splitter.

        Args:
            targets (list): the identifiers of the targets to leave out as test set
            data_set (PCMDataset): the dataset to split
        """
        super().__init__(data_set)
        self.targets = list(set(targets))

    def split(self, X, y):
        assert self.hasDataSet, (
            "No dataset attached to this splitter, set dataset with setDataSet()"
        )
        assert isinstance(self.getDataSet(), PCMDataSet), (
            "LeaveTargetsOut only works for PCM datasets, set a PCMDataSet with setDataSet()"
        )

        protein_prop = self.getDataSet().getDF()[self.getDataSet().proteinIDProp]
        mask = protein_prop.isin(self.targets)
        mask = mask.loc[X.index].reset_index(drop=True)

        indices = np.array(list(range(len(X))))
        train = indices[mask]
        test = indices[~mask]
        return iter([(train, test)])


class TemporalPerTarget(DataSplit, DataSetDependent):
    def __init__(
        self,
        time_prop: str,
        split_time: dict[str, int],
        first_time_per_compound: bool = True,
        data_set: PCMDataSet | None = None,
    ):
        """Creates a temporal split that is consistent across targets.

        Args:
            time_prop (str):
                the name of the property in the dataset that contains the time information
            split_time (dict[str,int]):
                a dictionary with target keys as keys
                and split times as values
            first_time_per_compound (bool):
                if True, the first time a compound appears in the dataset is used
                for all targets
            data_set (PCMDataset):
                the dataset to split containing the time information
        """
        super().__init__(data_set)
        self.timeProp = time_prop
        self.splitTime = split_time
        self.firstTimePerCompound = first_time_per_compound

    def split(self, X, y) -> Iterable[tuple[list[int], list[int]]]:
        assert self.hasDataSet, (
            "No dataset attached to this splitter, set dataset with setDataSet()"
        )
        assert isinstance(self.getDataSet(), PCMDataSet), (
            "TemporalPerTarget only works for PCM datasets, set a PCMDataSet with setDataSet()"
        )
        # Add the smiles, target and time properties to target values
        ds = self.getDataSet()
        df = ds.getDF().copy()
        indices = df.index.tolist()

        # Set the first time a compound appears in the dataset as the time
        # of the compound for all targets
        if self.firstTimePerCompound:
            first_appearances = df.groupby(ds.smilesProp)[self.timeProp].min()
            df["time_prop"] = df[ds.smilesProp].map(first_appearances)

        train_indices = []
        test_indices = []

        for target, split_year in self.splitTime.items():
            df_target = df[df[ds.proteinIDProp] == target]
            # Get indices of the train and test set
            train = df_target[df_target["time_prop"] <= split_year].index.tolist()
            test = df_target[df_target["time_prop"] > split_year].index.tolist()
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
