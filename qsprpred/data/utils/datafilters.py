"""Filters for QSPR Datasets.

To add a new filter:
* Add a DataFilter subclass for your new filter
"""
from functools import partial
from itertools import combinations, compress
from typing import Optional

import numpy as np
import pandas as pd

from ...data.interfaces import DataFilter
from ...logs import logger


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


class DuplicateFilter(DataFilter):
    """To filter out duplicate molecules based on descriptor values

    Attributes:
        keep (str): For duplicate entries determines how properties are treated,
            if False remove both (/all) duplicate entries, if True keep them,
            if first, keep row of first entry (based on year), if last keep row of
            last entry based on year.
            options: 'first', 'last', True, False
        year_name (str, optional): name of column containing year of publication
            used if keep is 'first' or 'last'
    """
    def __init__(self, keep: str = "first", year_name: Optional[str] = "Year"):
        self.keep = keep
        self.year_name = year_name

    def __call__(self, df: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
        """Filter rows from dataframe.

        Arguments:
            df (pandas dataframe): dataframe to filter
            descriptors (pandas dataframe): dataframe containing descriptors
        """
        df.shape[0]

        fps = descriptors.values.tolist()

        idxs = list(range(len(df)))
        fps_tpl = list(combinations(fps, 2))
        idxs_tpl = list(combinations(idxs, 2))
        results = [i[0] == i[1] for i in fps_tpl]  # True if fingerprints identical
        duplicate_idxs = list(compress(idxs_tpl, results))

        allrepeats = []  # This part could be refactored but for now it works
        for number in np.unique(np.array(duplicate_idxs)):
            repeat = []
            for x in np.where(np.array(duplicate_idxs) == number)[0]:
                repeat.append(duplicate_idxs[x])
            this_set = list(np.unique(np.array(repeat).flatten()))
            if this_set not in allrepeats:
                allrepeats.append(list(this_set))

        if self.keep is True:
            logger.warning(
                "Dataframe contains duplicate compounds and/or compounds with "
                "identical descriptors.\nThe following rows contain duplicates: "
                f"{[df.index[repeats].to_list() for repeats in allrepeats]}"
            )
        elif self.keep is False:
            remove_idx = [item for repeat in allrepeats for item in repeat]
            df = df.drop(df.index[remove_idx])
        elif self.keep in ["first", "last"]:
            logger.warning(
                "For dataframe with multiple target properties it is "
                "recommended to give target_props to prevent loss of "
                "values for properties only occuring for some datapoints"
            )
            remove_idxs = set()
            for repeat in allrepeats:
                indexes = df.index[repeat]
                years = df.loc[indexes, self.year_name]
                if self.keep == "first":
                    tokeep = years.idxmin()  # Use the first occurance
                else:
                    tokeep = years.idxmax()
                remove_idx = list(indexes)
                remove_idx.remove(tokeep)
                remove_idxs.update(remove_idx)
            df = df.drop(remove_idxs)

        return df
