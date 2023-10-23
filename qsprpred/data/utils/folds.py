"""A module that provides a class that creates folds from a given data set."""
from abc import ABC, abstractmethod
from typing import Generator

import pandas as pd
from .feature_standardization import apply_feature_standardizer


class FoldGenerator(ABC):

    @abstractmethod
    def iterFolds(
            self,
            dataset: "QSPRDataset",
            concat=False
    ) -> Generator[tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame | pd.Series,
        pd.DataFrame | pd.Series,
        list[int],
        list[int]
    ], None, None]:
        pass

    def getFolds(self, dataset: "QSPRDataset"):
        """Directly converts the output of `iterFolds` to a `list`."""
        return list(self.iterFolds(dataset))


class FoldsFromDataSplit(FoldGenerator):
    """A class that creates folds from a given data set. It can be used for
    cross-validation."""

    def _standardize_folds(self, folds):
        """A generator that fits and applies feature standardizers to each fold
        returned. They are properly fitted on the training set and applied to the
        test set."""
        for X_train, X_test, y_train, y_test, train_index, test_index in folds:
            X_train, standardizer = apply_feature_standardizer(
                self.featureStandardizer, X_train, fit=True
            )
            X_test, _ = apply_feature_standardizer(standardizer, X_test, fit=False)
            yield X_train, X_test, y_train, y_test, train_index, test_index

    def __init__(self, split: "DataSplit", feature_standardizer=None):
        """Create a new instance of Folds."""
        self.split = split
        self.featureStandardizer = feature_standardizer

    def _make_folds(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series
    ) -> Generator[tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame | pd.Series,
        pd.DataFrame | pd.Series,
        list[int],
        list[int]
    ], None, None]:
        """A generator that converts folds as returned by the splitter to a tuple of
        (X_train, X_test, y_train, y_test, train_index, test_index).

        Arguments:
            X (pd.DataFrame): feature matrix as a DataFrame
            y (pd.Series): target values
        Returns:
            generator: a generator that yields tuples of
                (X_train, X_test, y_train, y_test, train_index, test_index)
        """
        folds = self.split.split(X, y)
        for train_index, test_index in folds:
            yield X.iloc[train_index, :], X.iloc[test_index, :], y.iloc[
                train_index], y.iloc[test_index], train_index, test_index

    def iterFolds(
            self,
            dataset: "QSPRDataset",
            concat=False
    ) -> Generator[tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame | pd.Series,
        pd.DataFrame | pd.Series,
        list[int],
        list[int]
    ], None, None]:
        """Create folds from X and y. Can be used either for cross-validation,
        bootstrapping or train-test split.

        Each split in the resulting generator is represented by a tuple:
        (
            X_train, # feature matrix of the training set
            X_test, # feature matrix of the test set
            y_train, # target values of the training set
            y_test, # target values of the test set
            train_index, # indices of the training set in the original data set
            test_index # indices of the test set in the original data set
        )

        Arguments:
            dataset (QSPRDataset):
                the data set to generate the splits for
        Returns:
            generator:
                a generator that yields a tuple of
                (X_train, X_test, y_train, y_test, train_index, test_index)

        """
        if hasattr(self.split, "setDataSet"):
            self.split.setDataSet(dataset)
        if hasattr(self.split, "setSeed") and hasattr(self.split, "getSeed"):
            if self.split.getSeed() is None:
                self.split.setSeed(dataset.randomState)
        features = dataset.getFeatures(raw=True, concat=concat)
        targets = dataset.getTargetPropertiesValues(concat=concat)
        if not concat:
            features = features[0]
            targets = targets[0]
        if self.featureStandardizer:
            return self._standardize_folds(self._make_folds(features, targets))
        else:
            return self._make_folds(features, targets)
