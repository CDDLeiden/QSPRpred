"""Different splitters to create train and tests for evalutating QSPR model performance.

To add a new data splitter:
* Add a datasplit subclass for your new splitter
"""
from collections import defaultdict

import numpy as np

from qsprpred.data.interfaces import datasplit
from qsprpred.data.utils.scaffolds import Scaffold
from qsprpred.logs import logger
from sklearn.model_selection import ShuffleSplit


class randomsplit(datasplit):
    """Splits dataset in random train and test subsets.

    Attributes:
        test_fraction (float): fraction of total dataset to testset
    """

    def __init__(self, test_fraction=0.1) -> None:
        self.test_fraction = test_fraction

    def split(self, X, y):
        return ShuffleSplit(1).split(X, y)


class temporalsplit(datasplit):
    """Splits dataset train and test subsets based on a threshold in time.

    Attributes:
        dataset (QSPRDataset): dataset that this splitter will be acting on
        timesplit(float): time point after which sample to test set
        timecol (str): column name of column in df that contains the timepoints
    """

    def __init__(self, dataset, timesplit, timeprop) -> None:
        self.dataset = dataset
        self.timesplit = timesplit
        self.timecol = timeprop

    def split(self, X, y):
        df = self.dataset.getDF()
        assert len(X) == len(df), "X and the current data in the dataset must have same length"
        indices = np.array([x for x in range(len(df))])

        mask = df[self.timecol] > self.timesplit
        mask = mask.values

        test = indices[mask]
        assert len(test) > 0, "No test samples found"
        train = indices[~mask]
        return iter([(train, test)])


class scaffoldsplit(datasplit):
    """Splits dataset in train and test subsets based on their Murcko scaffold.

    Attributes:
        test_fraction (float): fraction of total dataset to testset
    """
    
    def __init__(self, dataset, scaffold : Scaffold, test_fraction=0.1, shuffle=True) -> None:
        self.dataset = dataset
        self.scaffold = scaffold
        self.test_fraction = test_fraction
        self.shuffle = shuffle

    def split(self, X, y):
        self.dataset.addScaffolds([self.scaffold])

        # make sure dataframe is shuffled
        if self.shuffle:
            self.dataset.shuffle()

        # Find the scaffold of each smiles
        df = self.dataset.getDF()
        assert len(X) == len(df), "X and the current data in the dataset must have same length"
        scaffold_list = df[f"Scaffold_{self.scaffold}"]
        scaffolds = defaultdict(list)
        invalid_idx = []
        for idx, scaffold in enumerate(scaffold_list):
            if scaffold:
                scaffolds[scaffold].append(idx)
            else:
                logger.warning(f"Invalid scaffold skipped for: {df.iloc[idx][self.dataset.smilescol]}")
                invalid_idx.append(idx)

        # Fill test set with groups of smiles with the same scaffold
        max_in_test = np.ceil(len(df) * self.test_fraction)
        test_idx = []
        for _, scaffold_idx in scaffolds.items():
            if len(test_idx) < max_in_test:
                test_idx.extend(scaffold_idx)
            else:
                break

        # Get train and test set indices
        return iter([([x for x in range(len(df)) if x not in test_idx], test_idx)])
