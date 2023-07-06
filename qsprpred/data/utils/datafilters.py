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
            e.g. keep property of first entry (by year), average of all entries,
            or remove duplicate entries
            options: 'first', 'last', 'average', 'median', False
        year_name (str, optional): name of column containing year of publication
            used if keep is 'first' or 'last'
        target_props (list, optional): name of column(s) containing target properties
            to combine for duplicates
            used if keep is 'average' or 'median'
    """
    def __init__(
        self,
        keep: str = "first",
        year_name: Optional[str] = "Year",
        target_props: Optional[list] = None
    ):
        self.keep = keep
        self.year_name = year_name
        self.target_props = target_props

    def __call__(self, df: pd.DataFrame, descriptors: pd.DataFrame) -> pd.DataFrame:
        """Filter rows from dataframe.

        Arguments:
            df (pandas dataframe): dataframe to filter
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

        if self.keep in ["first", "last"]:
            if self.target_props is None or len(self.target_props) == 1:
                logger.warning(
                    "For dataframe with multiple target properties it is /"
                    "recommended to give target_props to prevent loss of /"
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
            else:
                remove_indexes = []
                for repeat in allrepeats:
                    indexes = df.index[repeat]
                    df_values = df.loc[indexes]
                    for target_prop in self.target_props:
                        if self.keep == "first":
                            index = df_values[self.year_name].idxmin()
                        elif self.keep == "last":
                            index = df_values[self.year_name].idxmax()
                        replace = df_values.loc[index, target_prop]
                        df.at[indexes[0], target_prop] = replace
                        #TODO now all other properties are just those from first row
                        remove_indexes.extend(indexes[1:])
                df = df.drop(remove_indexes)
        elif self.keep in ["mean", "median"]:
            remove_indexes = []
            for repeat in allrepeats:
                indexes = df.index[repeat]
                for target_prop in self.target_props:
                    values = df.loc[indexes, target_prop]
                    if self.keep == "mean":
                        replace = sum(values) / len(values)
                    elif self.keep == "median":
                        replace = values.median()
                    df.at[indexes[0], target_prop] = replace
                remove_indexes.extend(indexes[1:])
            df = df.drop(remove_indexes)
        elif self.keep is False:
            remove_idx = [item for repeat in allrepeats for item in repeat]
            df = df.drop(df.index[remove_idx])

        return df
