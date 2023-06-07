"""Filters for QSPR Datasets.

To add a new filter:
* Add a DataFilter subclass for your new filter
"""
from functools import partial

import pandas as pd

from qsprpred.data.interfaces import DataFilter
from qsprpred.logs import logger


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
