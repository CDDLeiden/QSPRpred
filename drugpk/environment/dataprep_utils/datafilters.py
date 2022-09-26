from functools import partial
from drugpk.environment.interfaces import datafilter
from drugpk.logs import logger
import pandas as pd

class CategoryFilter(datafilter):
    """
        To filter out values from categorical column
    """
    def __call__(df: pd.DataFrame, name: str, values: list, keep=False) -> pd.DataFrame:
        """
            filter
            Arguments:
                df (pandas dataframe): dataframe to filter from
                name (str): column name
                values (list of str): filter values
                keep (bool): wheter to keep or discard values
        """
        try:
            if keep:
                df = df[df[name] in [values]]
            else:
                df = df[df[name] not in [values]]
        except KeyError:
            logger.warning(f"Filter column not in dataframe ('{name}'), all data included in set.")
        
        return df

lowQualityFilter = partial(CategoryFilter, values=["Low"])