"""A module that provides a class that creates folds from a given data set."""

from abc import ABC, abstractmethod
from typing import Generator

import pandas as pd
from copy import deepcopy
from ..pipelines.pipeline import Pipeline


class FoldGenerator(ABC):
    """A generator that creates folds from a given data set."""
    @abstractmethod
    def iterFolds(
        self,
        dataset: "QSPRDataSet",
        concat=False
    ) -> Generator[
        tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame | pd.Series,
            pd.DataFrame | pd.Series,
            list[int],
            list[int],
        ],
        None,
        None,
    ]:
        """
        Returns the generator of folds to iterate over.

        Args:
            dataset (QSPRDataSet):
                the data set to generate the splits for
            concat (bool, optional):
                whether to concatenate the features in the test
                and training set of the data set (default: False)

        Returns:
            generator:
                a generator that yields a tuple of
                (X_train, X_test, y_train, y_test, train_index, test_index)
        """

    def getFolds(self, dataset: "QSPRDataSet"):  # noqa: F821
        """Directly converts the output of `iterFolds` to a `list`."""
        return list(self.iterFolds(dataset))


class FoldsFromDataSplit(FoldGenerator):
    """This generator takes a scikit-learn or scikit-learn-like splitter
    and creates folds from it. It is possible to pass a pipeline to
    make sure features in the splits are properly pre-processed.

    Attributes:
        split (DataSplit):
            the splitter to use to create the folds (this can also just be
            a raw scikit-learn splitter)
        pipeline (Pipeline):
            the pipeline to use to pre-process the features, e.g. standardize them
    """
    def _preprocess_folds(self, folds):
        """A generator that fits and applies the pipeline to each fold
        returned. They are properly fitted on the training set and applied to the
        test set."""
        for X_train, X_test, y_train, y_test, train_index, test_index in folds:
            pipeline_copy = deepcopy(self.pipeline)
            X_train, y_train = pipeline_copy.fitTransform(X_train, y_train)
            X_test, y_test = pipeline_copy.transform(X_test, y_test)
            
            yield X_train, X_test, y_train, y_test, train_index, test_index

    def __init__(self, split: "DataSplit", pipeline: Pipeline = None):  # noqa: F821
        """Initialize the generator with a splitter and a pipeline.

        Args:
            split (DataSplit):
                the splitter to use to create the folds (this can also just be
                a raw scikit-learn splitter)
            pipeline:
                the pipeline to use to pre-process the features
        """
        self.split = split
        self.pipeline = pipeline

    def _make_folds(
        self, X: pd.DataFrame, y: pd.DataFrame | pd.Series
    ) -> Generator[
        tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame | pd.Series,
            pd.DataFrame | pd.Series,
            list[int],
            list[int],
        ],
        None,
        None,
    ]:
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
        dataset: "QSPRDataSet",
        concat=False
    ) -> Generator[
        tuple[
            pd.DataFrame,
            pd.DataFrame,
            pd.DataFrame | pd.Series,
            pd.DataFrame | pd.Series,
            list[int],
            list[int],
        ],
        None,
        None,
    ]:
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
            dataset (QSPRDataSet):
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
        if concat:
            X, y = features
        else:
            X, X_ind, y, y_ind = features
        if self.pipeline:
            return self._preprocess_folds(self._make_folds(X, y))
        else:
            return self._make_folds(X, y)
