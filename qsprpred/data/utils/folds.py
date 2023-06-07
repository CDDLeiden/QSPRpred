"""A module that provides a class that creates folds from a given data set."""
from ..interfaces import DataSplit
from .feature_standardization import apply_feature_standardizer


class Folds:
    """A class that creates folds from a given data set. It can be used for
    cross-validation."""
    @staticmethod
    def toArrays(X, y):
        """Convert data frames X and y to numpy arrays.

        Arguments:
            X (pd.DataFrame): feature matrix as a DataFrame
            y (pd.Series): target values

        Returns:
            tuple: (X, y) as numpy arrays
        """
        return X.values, y.values

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

    def _make_folds(self, X, y):
        """A generator that converts folds as returned by the splitter to a tuple of
        (X_train, X_test, y_train, y_test, train_index, test_index).

        Arguments:
            X (pd.DataFrame): feature matrix as a DataFrame
            y (pd.Series): target values
        Returns:
            generator: a generator that yields a tuple of
                (X_train, X_test, y_train, y_test, train_index, test_index)
        """
        X_arr, y_arr = self.toArrays(X, y)
        folds = self.split.split(X_arr, y_arr)

        for train_index, test_index in folds:
            yield X_arr[train_index, :], X_arr[test_index, :], y_arr[
                train_index], y_arr[test_index], train_index, test_index

    def __init__(self, split: DataSplit, feature_standardizer=None):
        """Create a new instance of Folds."""
        self.split = split
        self.featureStandardizer = feature_standardizer

    def iterFolds(self, X, y):
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
            X (pd.DataFrame): feature matrix as a DataFrame
            y (pd.Series): target values
        Returns:
            generator: a generator that yields a tuple of
                (X_train, X_test, y_train, y_test, train_index, test_index)

        """
        if self.featureStandardizer:
            return self._standardize_folds(self._make_folds(X, y))
        else:
            return self._make_folds(X, y)

    def getFolds(self, X, y):
        """Directly converts the output of `iterFolds` to a `list`."""
        return list(self.iterFolds(X, y))
