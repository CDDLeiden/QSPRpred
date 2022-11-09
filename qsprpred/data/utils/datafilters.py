from functools import partial

import pandas as pd
from qsprpred.data.interfaces import datafilter
from qsprpred.logs import logger


class CategoryFilter(datafilter):
    """
        To filter out values from categorical column
        Attributes:
            name (str): column name
            values (list of str): filter values
            keep (bool): whether to keep or discard values
    """
    def __init__(self, name: str, values: list, keep=False) -> None:
        self.name = name
        self.values = values
        self.keep = keep

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            filter
            Arguments:
                df (pandas dataframe): dataframe to filter from

        """
        old_len = df.shape[0]
        try:
            if self.keep:
                df = df[df[self.name].isin(self.values)]
            else:
                df = df[~df[self.name].isin(self.values)]
            logger.info(f"{old_len - df.shape[0]} rows filtered out.")
        except KeyError:
            logger.warning(f"Filter column not in dataframe ('{self.name}'), all data included in set.")
        
        return df

papyrusLowQualityFilter = partial(CategoryFilter, name="Quality", values=["Low"])