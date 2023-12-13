from abc import abstractmethod
from typing import Literal

import pandas as pd

from qsprpred.data.tables.base import MoleculeDataSet


class Searchable(MoleculeDataSet):

    @abstractmethod
    def searchOnProperty(
            self,
            prop_name: str,
            values: list[str],
            name: str | None = None,
            exact=False
    ) -> MoleculeDataSet:
        """ Search the molecules within this `MoleculeDataSet` on a property value.

        Args:
            prop_name:
                Name of the column to search on.
            values:
                Values to search for.
            name:
                Name of the new table.
            exact:
                Whether to search for exact matches or not.

        Returns:
            (MoleculeDataSet):
                A data set with the molecules that match the search.
        """

    @abstractmethod
    def searchWithSMARTS(
            self,
            patterns: list[str],
            operator: Literal["or", "and"] = "or",
            use_chirality: bool = False,
            name: str | None = None
    ) -> MoleculeDataSet:
        """
        Search the molecules within this `MoleculeDataSet` with SMARTS patterns.

        Args:
            patterns:
                List of SMARTS patterns to search with.
            operator (object):
                Whether to use an "or" or "and" operator on patterns. Defaults to "or".
            use_chirality:
                Whether to use chirality in the search.
            name:
                Name of the new table.

        Returns:
            (MoleculeDataSet): A dataframe with the molecules that match the pattern.
        """


class Summarizable:

    @abstractmethod
    def getSummary(self) -> pd.DataFrame:
        """Make a summary with some statistics about this object or action.

        Returns:
            (pd.DataFrame):
                A dataframe with the summary statistics.
        """

