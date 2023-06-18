"""Different splitters to create train and tests for evalutating QSPR model performance.

To add a new data splitter:
* Add a datasplit subclass for your new splitter
"""
from typing import Iterable

import numpy as np

from qsprpred.data.data import QSPRDataset
from qsprpred.data.interfaces import DataSetDependant, DataSplit
from qsprpred.data.utils.datasplitters import TemporalSplit
from qsprpred.extra.data.data import PCMDataSet


class LeaveTargetsOut(DataSplit, DataSetDependant):
    def __init__(self, targets: list[str], dataset: PCMDataSet = None):
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
        mask = ds.getProperty(ds.proteinCol).isin(ds_targets).values
        indices = np.array(list(range(len(ds))))
        train = indices[mask]
        test = indices[~mask]
        return iter([(train, test)])


class StratifiedPerTarget(DataSplit, DataSetDependant):
    """Splits dataset in train and test subsets based on the specified splitter."""
    def __init__(
        self,
        splitter: DataSplit = None,
        splitters: dict[str, DataSplit] = None,
        dataset: PCMDataSet = None
    ):
        """Creates a split that is consistent across targets.

        Args:
            splitter: a `datasplit` instance to split the target subsets of the dataset
            splitters (dict[str, datasplit]): a dictionary with target keys as keys and
                splitters to use on each protein target as values
            dataset (PCMDataset): a `PCMDataset` instance to split
        """
        super().__init__(dataset)
        self.splitter = splitter
        self.splitters = splitters
        assert self.splitter is not None or self.splitters is not None, \
            "Either a splitter or multiple splitters must be specified!"
        assert (splitter is None) != (splitters is None), \
            "Either one splitter or multiple splitters must be specified, but not both!"

    def split(self, X, y) -> Iterable[tuple[list[int], list[int]]]:
        ds = self.getDataSet()
        df = ds.getDF()
        train = []
        test = []
        indices = np.array(list(range(len(ds))))
        for target in ds.getProteinKeys():
            splitter = self.splitter if self.splitter is not None else self.splitters[
                target]
            df_target = df[df[ds.proteinCol] == target]
            ds_target = QSPRDataset(
                name=f"{target}_scaff_split_{hash(self)}",
                df=df_target,
                smiles_col=ds.smilesCol,
                target_props=ds.targetProperties,
                index_cols=ds.indexCols,
            )
            ds_target.split(splitter)
            train.extend(indices[df.index.isin(ds_target.X.index)])
            test.extend(indices[df.index.isin(ds_target.X_ind.index)])

        assert len(set(train)) + len(set(test)) == len(ds), \
            "Train and test set do not cover the whole dataset!"
        return iter([(train, test)])


class TemporalPerTarget(DataSplit, DataSetDependant):
    def __init__(
        self, year_col: str, split_years: dict[str, int], dataset: PCMDataSet = None
    ):
        """Creates a temporal split that is consistent across targets.

        Args:
            year_col (str):
                the name of the column in the dataframe that
                contains the year information
            split_years (dict[str,int]):
                a dictionary with target keys as keys
                and split years as values
            dataset (PCMDataset):
                a `PCMDataset` instance to split
        """
        super().__init__(dataset)
        self.splitYears = split_years
        self.yearCol = year_col

    def split(self, X, y) -> Iterable[tuple[list[int], list[int]]]:
        splitters = {
            target:
                TemporalSplit(timeprop=self.yearCol, timesplit=self.splitYears[target])
            for target, year in self.splitYears.items()
        }
        return StratifiedPerTarget(dataset=self.getDataSet(),
                                   splitters=splitters).split(X, y)
