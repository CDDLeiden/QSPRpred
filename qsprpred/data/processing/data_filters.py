"""Filters for QSPR Datasets.

To add a new filter:
* Add a DataFilter subclass for your new filter
"""
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain

import numpy as np
import pandas as pd

from ...logs import logger


class DataFilter(ABC):
    """Filter out some rows from a dataframe."""

    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out some rows from a dataframe.

        Args:
            df (pd.DataFrame): dataframe to be filtered

        Returns:
            The filtered pd.DataFrame
        """


class CategoryFilter(DataFilter):
    """To filter out values from column

    Attributes:
        name (str): column name.
        values (list[str]): filter values.
        keep (bool): whether to keep or discard values.
    """

    def __init__(self, name: str, values: list[str], keep=False) -> None:
        """Initialize the CategoryFilter with the name, values and keep attributes.

        Args:
            name (str): name of the column.
            values (list): list of values to filter.
            keep (bool, optional): whether to keep or discard the values. Defaults to
                False.
        """
        self.name = name
        self.values = values
        self.keep = keep

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter rows from dataframe.

        Args:
            df (pd.DataFrame): dataframe to filter.

        Returns:
            pd.DataFrame: filtered dataframe.
        """
        old_len = df.shape[0]
        try:
            if self.keep:
                df = df[df[self.name].isin(self.values)]
            else:
                df = df[~df[self.name].isin(self.values)]
            logger.info(f"{old_len - df.shape[0]} rows filtered out.")
        except KeyError:
            logger.warning(
                f"Filter column not in dataframe ({self.name}), all data included."
            )

        return df


papyrusLowQualityFilter = partial(CategoryFilter, name="Quality", values=["Low"])


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
            so that compounds with same descriptors but different proteinid
            are not removed.
    """

    def __init__(
        self,
        keep: str | bool = False,
        timecol: str | None = None,
        additional_cols: list[str] = [],
    ) -> None:
        """Initialize the RepeatsFilter with the keep, timecol and additional_cols
        attributes.

        Args:
            keep (str|bool, optional): For duplicate entries determines how properties
                are treated, if False remove both (/all) duplicate entries, if True
                keep them, if first, keep row of first entry (based on time), if last
                keep row of last entry based on time. Defaults to False.
            timecol (str, optional): name of column containing time of publication
                used if keep is 'first' or 'last'. Defaults to None.
            additional_cols (list[str], optional): additional columns to use for
                determining duplicates (e.g. proteinid, in case of PCM modelling),
                so that compounds with same descriptors but different proteinid
                are not removed. Defaults to [].
        """
        self.keep = keep
        self.timeCol = timecol
        self.additionalCols = additional_cols

    def __call__(self, df: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
        """Filter rows from dataframe.

        Arguments:
            df (pandas dataframe): dataframe to filter
            descriptors (pandas dataframe): dataframe containing descriptors
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

        # Adding additional columns to descriptors
        for col in self.additionalCols:
            if col in df.columns:
                descriptors[col] = df[col]

        allrepeats = group_duplicate_index(descriptors)

        if self.keep is True:
            if len(allrepeats) > 0:
                # print QSPRID of duplicate compounds
                logger.warning(
                    "Dataframe contains duplicate compounds and/or compounds with "
                    "identical descriptors.\nThe following rows contain duplicates: "
                    f"{[df.loc[repeats].index.values for repeats in allrepeats]}"
                )
        elif self.keep is False:
            df = df.drop(list(chain(*allrepeats)))
        elif self.keep in ["first", "last"]:
            assert (
                self.timeCol is not None
            ), "timecol must be specified if keep is 'first' or 'last'"
            for repeat in allrepeats:
                years = df.loc[repeat, self.timeCol]
                years = pd.to_numeric(years, errors="coerce")
                if self.keep == "first":
                    tokeep = years.idxmin()  # Use the first occurance
                else:
                    tokeep = years.idxmax()
                repeat.remove(tokeep)  # Remove the one to keep from the allrepeats list
            df = df.drop(list(chain(*allrepeats)))

        return df
