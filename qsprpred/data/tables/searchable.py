from abc import abstractmethod, ABC
from typing import Literal


class SMARTSSearchable(ABC):

    @abstractmethod
    def searchWithSMARTS(
            self,
            patterns: list[str],
            operator: Literal["or", "and"] = "or",
            use_chirality: bool = False,
            name: str | None = None,
    ) -> "SMARTSSearchable":
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
            (MoleculeStorage): A dataframe with the molecules that match the pattern.
        """


class PropSearchable(ABC):
    @abstractmethod
    def searchOnProperty(
            self, prop_name: str, values: list[str], name: str | None = None,
            exact=False
    ) -> "PropSearchable":
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
            (MoleculeStorage):
                A data set with the molecules that match the search.
        """
