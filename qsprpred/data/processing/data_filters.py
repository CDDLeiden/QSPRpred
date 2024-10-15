"""Filters for QSPR Datasets.

To add a new filter:
* Add a DataFilter subclass for your new filter
"""

from abc import abstractmethod
from itertools import chain

import numpy as np
import pandas as pd

from ...logs import logger
from ..pipelines.pipeline import Step


class DataFilter(Step):
    """Filter out some rows from a dataframe."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Fit the filter to the data.
        
        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame, optional): training targets
        """

    @abstractmethod
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Remove rows from a dataframe.
        
        Args:
            X (pd.DataFrame): dataframe to be standardized
            y (pd.DataFrame, optional): output dataframe if the standardization method
                requires it
        """


class CategoryFilter(Step):
    """To filter out values from column

    Attributes:
        prop (pd.Series): column based on which to filter.
        values (list[str]): filter values.
        keep (bool): whether to keep or discard values.
    """
    def __init__(self, prop: pd.Series, values: list[str], keep: bool = False) -> None:
        """Initialize the CategoryFilter with the name, values and keep attributes.

        Args:
            prop (pd.Series): column based on which to filter.
            values (list): list of values to filter from props.
            keep (bool, optional): whether to keep or discard the values. Defaults to
                False.
        """
        self.prop = prop
        self.values = values
        self.keep = keep
        
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Fit the filter to the data.

        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame, optional): training targets
        """
        pass
    
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Filter rows from dataframe.

        Args:
            X (pd.DataFrame): dataframe to filter.
            y (pd.DataFrame, optional): output dataframe if the filtering method
                requires it

        Returns:
            pd.DataFrame: filtered dataframe.
        """
        old_len = X.shape[0]
        if self.keep:
            idx_to_keep = self.prop.isin(self.values)
        else:
            idx_to_keep = ~self.prop.isin(self.values)
        X = X[idx_to_keep]
        logger.info(f"{old_len - X.shape[0]} rows filtered out.")

        return X, y

class RepeatsFilter(DataFilter):
    """To filter out duplicate molecules based on descriptor values.

    Attributes:
        keep (str): For duplicate entries determines how properties are treated,
            if False remove both (/all) duplicate entries, if True keep them,
            if first, keep row of first entry (based on time), if last keep row of
            last entry based on time.
            options: 'first', 'last', True, False
        timeCol (str, optional): name of column containing time of publication
            used if keep is 'first' or 'last'
        additionalCols (list[str], optional): additional columns to use for
            determining duplicates (e.g. proteinid, in case of PCM modelling),
            so that compounds with same X but different proteinid
            are not removed.
    """
    def __init__(
        self,
        keep: str | bool = False,
        timecol: pd.Series | None = None,
        additional_cols: dict[str, pd.Series] = None
    ) -> None:
        """Initialize the RepeatsFilter with the keep, timecol and additional_cols
        attributes.

        Args:
            keep (str|bool, optional): For duplicate entries determines how properties
                are treated, if False remove both (/all) duplicate entries, if True
                keep them, if first, keep row of first entry (based on time), if last
                keep row of last entry based on time. Defaults to False.
            timecol (pd.Series, optional): name of column containing time of publication
                used if keep is 'first' or 'last'. Defaults to None.
            additional_cols (dict[str, pd.Series], optional): additional columns to use for
                determining duplicates (e.g. proteinid, in case of PCM modelling),
                so that compounds with same X but different proteinid
                are not removed.
        """
        self.keep = keep
        self.timeCol = timecol
        self.additionalCols = additional_cols
        
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Fit the filter to the data.

        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame, optional): training targets
        """
        pass

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Filter rows from dataframe.

        Arguments:
            X (pandas dataframe): dataframe to filter
            y (pandas dataframe, optional): output dataframe if the filtering method
                requires it
        """
        def group_duplicate_index(df) -> list[list[int]]:
            """Group indices of duplicate rows

            From https://stackoverflow.com/a/46629623

            Args:
                a (numpy array): array of fingerprints

            Returns:
                list[list[int]]: list of lists of indices of duplicate rows
            """
            # Sort by rows
            a = df.values
            sidx = np.lexsort(a.T)
            b = a[sidx]

            # Get unique row mask
            m = np.concatenate(([False], (b[1:] == b[:-1]).all(1), [False]))

            # Get start and stop indices for each group of duplicates
            idx = np.flatnonzero(m[1:] != m[:-1])

            # Get sorted indices
            sort_idxs = df.index[sidx].tolist()

            # Return list of lists of indices of duplicate rows
            return [sort_idxs[i:j] for i, j in zip(idx[::2], idx[1::2] + 1)]

        assert X.shape[1] > 0, "X must have at least one column"

        X_copy = X.copy()

        # Adding additional columns to X
        if self.additionalCols is not None:
            for col in self.additionalCols:
                X_copy[col] = self.additionalCols[col]

        allrepeats = group_duplicate_index(X_copy)

        if self.keep is True:
            if len(allrepeats) > 0:
                logger.warning(
                    "Dataframe contains compounds with duplicate features."
                    f"\nThe following rows contain duplicates: {allrepeats}"
                )
        elif self.keep is False:
            to_drop = list(chain(*allrepeats))
            logger.info(f"{len(to_drop)} duplicate rows filtered out.")
            X = X.drop(to_drop)
        elif self.keep in ["first", "last"]:
            assert (
                self.timeCol is not None
            ), "timecol must be specified if keep is 'first' or 'last'"
            self.timeCol = pd.to_numeric(self.timeCol, errors="coerce")
            for repeat in allrepeats:
                repeat_time = self.timeCol.loc[repeat]
                if self.keep == "first":
                    tokeep = repeat_time.idxmin()  # Use the first occurance
                else:
                    tokeep = repeat_time.idxmax()
                repeat.remove(tokeep)  # Remove the one to keep from the allrepeats list
            X = X.drop(list(chain(*allrepeats)))

        return X, y
