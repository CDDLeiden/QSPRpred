"""A module that provides a class that creates folds from a given data set."""
from abc import ABC, abstractmethod
from typing import Generator

import pandas as pd
from ...data.processing.feature_standardizers import apply_feature_standardizer


class FoldGenerator(ABC):
    """A generator that creates folds from a given data set."""

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
        """
        Returns the generator of folds to iterate over.

        Args:
            dataset (QSPRDataset):
                the data set to generate the splits for
            concat (bool, optional):
                whether to concatenate the features in the test
                and training set of the data set (default: False)

        Returns:
            generator:
                a generator that yields a tuple of
                (X_train, X_test, y_train, y_test, train_index, test_index)
        """
        pass

    def getFolds(self, dataset: "QSPRDataset"):
        """Directly converts the output of `iterFolds` to a `list`."""
        return list(self.iterFolds(dataset))


class FoldsFromDataSplit(FoldGenerator):
    """This generator takes a scikit-learn or scikit-learn-like splitter
    and creates folds from it. It is possible to pass a standardizer to
    make sure features in the splits are properly standardized.

    Attributes:
        split (DataSplit):
            the splitter to use to create the folds (this can also just be
            a raw scikit-learn splitter)
        featureStandardizer:
            the standardizer to use to standardize the features (this can also
            just be a raw scikit-learn standardizer)
    """

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
        """Initialize the generator with a splitter and a standardizer.

        Args:
            split (DataSplit):
                the splitter to use to create the folds (this can also just be
                a raw scikit-learn splitter)
            feature_standardizer:
                the standardizer to use to standardize the features (this can also
                just be a raw scikit-learn standardizer)
        """
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
        features = dataset.getFeatures(raw=True, concat=concat, ordered=True)
        targets = dataset.getTargetPropertiesValues(concat=concat, ordered=True)
        if not concat:
            features = features[0]
            targets = targets[0]
        if self.featureStandardizer:
            return self._standardize_folds(self._make_folds(features, targets))
        else:
            return self._make_folds(features, targets)
