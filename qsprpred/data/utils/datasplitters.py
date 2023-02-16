"""Different splitters to create train and tests for evalutating QSPR model performance.

To add a new data splitter:
* Add a datasplit subclass for your new splitter
"""
from collections import defaultdict

import numpy as np

from qsprpred.data.data import QSPRDataset
from qsprpred.data.interfaces import datasplit, DataSetDependant
from qsprpred.data.utils.scaffolds import Scaffold, Murcko
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
        return ShuffleSplit(1, test_size=self.test_fraction).split(X, y)


class temporalsplit(datasplit, DataSetDependant):
    """Splits dataset train and test subsets based on a threshold in time.

    Attributes:
        dataset (QSPRDataset): dataset that this splitter will be acting on
        timesplit(float): time point after which sample to test set
        timecol (str): column name of column in df that contains the timepoints
    """

    def __init__(self, timesplit, timeprop, dataset=None) -> None:
        super().__init__(dataset=dataset)
        self.timesplit = timesplit
        self.timecol = timeprop

    def split(self, X, y):
        df = self.getDataSet().getDF()
        assert len(X) == len(df), "X and the current data in the dataset must have same length"
        indices = np.array([x for x in range(len(df))])

        mask = df[self.timecol] > self.timesplit
        mask = mask.values

        test = indices[mask]
        assert len(test) > 0, "No test samples found"
        train = indices[~mask]
        return iter([(train, test)])


class scaffoldsplit(datasplit, DataSetDependant):
    """Splits dataset in train and test subsets based on their Murcko scaffold.

    Attributes:
        dataset: QSPRDataset object.
        scaffold (qsprpred.data.utils.scaffolds.Scaffold()): `Murcko()` and
            `BemisMurcko()` are currently available, other types can be added through
            the abstract class `Scaffold`. Defaults to Murcko().
        test_fraction (float): fraction of the test set. Defaults to 0.1.
        shuffle (bool): whether to shuffle the data or not. Defaults to True.
    """        
    def __init__(self, scaffold : Scaffold = Murcko(), test_fraction=0.1, shuffle=True, dataset=None) -> None:
        super().__init__(dataset)
        self.scaffold = scaffold
        self.test_fraction = test_fraction
        self.shuffle = shuffle

    def split(self, X, y):
        dataset = self.getDataSet()
        if not dataset:
            raise AttributeError("Dataset not set for splitter. Use 'setDataSet(dataset)' to attach it to this instance.")

        dataset.addScaffolds([self.scaffold])

        # make sure dataframe is shuffled
        if self.shuffle:
            dataset.shuffle()

        # Find the scaffold of each smiles
        df = dataset.getDF()
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
