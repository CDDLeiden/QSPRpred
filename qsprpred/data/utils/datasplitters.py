"""Different splitters to create train and tests for evalutating QSPR model performance.

To add a new data splitter:
* Add a DataSplit subclass for your new splitter
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

from ....qsprpred.logs import logger
from ...data.interfaces import DataSetDependant, DataSplit
from ..utils.scaffolds import Murcko, Scaffold


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
        train = self.splitCol[self.splitCol == self.trainVal].index.values
        test = self.splitCol[self.splitCol == self.testVal].index.values
        return iter([(train, test)])


class RandomSplit(DataSplit):
    """Splits dataset in random train and test subsets.

    Attributes:
        testFraction (float): fraction of total dataset to testset
    """
    def __init__(self, test_fraction=0.1) -> None:
        self.testFraction = test_fraction

    def split(self, X, y):
        return ShuffleSplit(1, test_size=self.testFraction).split(X, y)


class TemporalSplit(DataSplit, DataSetDependant):
    """Splits dataset train and test subsets based on a threshold in time.

    Attributes:
        dataset (QSPRDataset): dataset that this splitter will be acting on
        timesplit(float): time point after which sample to test set
        timecol (str): name of the column within the dataframe with timepoints
    """
    def __init__(self, timesplit, timeprop, dataset=None) -> None:
        """Initialize a TemporalSplit object.

        Attributes:
            dataset (QSPRDataset): dataset that this splitter will be acting on
            timesplit(float): time point after which sample to test set
            timecol (str): name of the column within the dataframe with timepoints
        """
        super().__init__(dataset=dataset)
        self.timeSplit = timesplit
        self.timeCol = timeprop

    def split(self, X, y):
        df = self.getDataSet().getDF()
        assert len(X) == len(
            df
        ), "X and the current data in the dataset must have same length"
        indices = np.array(list(range(len(df))))

        mask = df[self.timeCol] > self.timeSplit
        mask = mask.values

        test = indices[mask]
        assert len(test) > 0, "No test samples found"
        train = indices[~mask]
        return iter([(train, test)])


class ScaffoldSplit(DataSplit, DataSetDependant):
    """Splits dataset in train and test subsets based on their Murcko scaffold.

    Attributes:
        dataset: QSPRDataset object.
        scaffold (qsprpred.data.utils.scaffolds.Scaffold()): `Murcko()` and
            `BemisMurcko()` are currently available, other types can be added through
            the abstract class `Scaffold`. Defaults to Murcko().
        test_fraction (float): fraction of the test set. Defaults to 0.1.
        shuffle (bool): whether to shuffle the data or not. Defaults to True.
        custom_test_list (list): list of molecule indexes to force in test set. If
            forced test contains the totality of the molecules in the dataset, the
            custom_test_list reverts to default None.
    """
    def __init__(
        self,
        scaffold: Scaffold = Murcko(),
        test_fraction=0.1,
        shuffle=True,
        custom_test_list=None,
        dataset=None,
    ) -> None:
        super().__init__(dataset)
        self.scaffold = scaffold
        self.testFraction = test_fraction
        self.shuffle = shuffle
        self.customTestList = custom_test_list

    def split(self, X, y):
        dataset = self.getDataSet()
        if not dataset:
            raise AttributeError(
                "Dataset not set for splitter. Use 'setDataSet(dataset)' to attach it "
                "to this instance."
            )

        dataset.addScaffolds([self.scaffold])

        # make sure dataframe is shuffled
        if self.shuffle:
            dataset.shuffle()

        # Find the scaffold of each smiles
        df = dataset.getDF()
        assert len(X) == len(
            df
        ), "X and the current data in the dataset must have same length"
        scaffold_list = df[[f"Scaffold_{self.scaffold}"]]
        scaffolds = defaultdict(list)
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
