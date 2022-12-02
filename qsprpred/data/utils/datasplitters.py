"""Different splitters to create train and tests for evalutating QSPR model performance.

To add a new data splitter:
* Add a datasplit subclass for your new splitter
"""
from collections import defaultdict

import numpy as np
from qsprpred.data.interfaces import datasplit
from qsprpred.logs import logger
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.model_selection import train_test_split


class randomsplit(datasplit):
    """Splits dataset in random train and test subsets.

    Attributes:
        test_fraction (float): fraction of total dataset to testset
    """

    def __init__(self, test_fraction=0.1) -> None:
        self.test_fraction = test_fraction

    def __call__(self, df, Xcol, ycol):
        return train_test_split(df[Xcol], df[ycol], test_size=self.test_fraction)


class temporalsplit(datasplit):
    """Splits dataset train and test subsets based on a threshold in time.

    Attributes:
        timesplit(float): time point after which sample to test set
        timecol (str): column name of column in df that contains the timepoints
    """

    def __init__(self, timesplit: float, timecol: str) -> None:
        self.timesplit = timesplit
        self.timecol = timecol

    def __call__(self, df, Xcol, ycol):
        test_idx = df[df[self.timecol] > self.timesplit].index
        test = df.loc[list(test_idx)]
        train = df.drop(test.index)
        return train[Xcol], test[Xcol], train[ycol], test[ycol]


class scaffoldsplit(datasplit):
    """Splits dataset in train and test subsets based on their Murcko scaffold.

    Attributes:
        test_fraction (float): fraction of total dataset to testset
    """
    
    def __init__(self, test_fraction=0.1) -> None:
        self.test_fraction = test_fraction

    def __call__(self, df, Xcol, ycol, shuffle=True):
        # make sure dataframe is shuffled
        if shuffle:
            df = df.sample(frac=1)

        # Find the scaffold of each smiles
        scaffolds = defaultdict(list)
        invalid_idx = []
        for row in df[[Xcol]].itertuples():
            idx, smiles = row
            try:
                scaffolds[MurckoScaffoldSmiles(smiles)].append(idx)
            except:
                logger.warning(f"Invalid smiles skipped: {smiles}")
                invalid_idx.append(idx)

        # Fill test set with groups of smiles with the same scaffold
        max_in_test = np.ceil(len(df[Xcol]) * self.test_fraction)
        test_idx = []
        for _, scaffold_idx in scaffolds.items():
            if len(test_idx) < max_in_test:
                test_idx.extend(scaffold_idx)
            else:
                break

        # Create train and test set
        test = df.loc[test_idx]
        train = df.drop(test.index).drop(invalid_idx)
        return train[Xcol], test[Xcol], train[ycol], test[ycol]
