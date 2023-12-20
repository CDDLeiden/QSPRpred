from abc import abstractmethod
from typing import Literal

from qsprpred.data.tables.base import MoleculeDataTable


class SearchableMolTable(MoleculeDataTable):
    @abstractmethod
    def searchOnProperty(
        self, prop_name: str, values: list[str], name: str | None = None, exact=False
    ) -> MoleculeDataTable:
        """Search the molecules within this `MoleculeDataSet` on a property value.

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
            (MoleculeDataTable):
                A data set with the molecules that match the search.
        """

    @abstractmethod
    def searchWithSMARTS(
        self,
        patterns: list[str],
        operator: Literal["or", "and"] = "or",
        use_chirality: bool = False,
        name: str | None = None,
    ) -> MoleculeDataTable:
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
            (MoleculeDataTable): A dataframe with the molecules that match the pattern.
        """
