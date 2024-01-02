from abc import abstractmethod

import pandas as pd


class Summarizable:
    @abstractmethod
    def getSummary(self) -> pd.DataFrame:
        """Make a summary with some statistics about this object or action.

        Returns:
            (pd.DataFrame):
                A dataframe with the summary statistics.
        """
