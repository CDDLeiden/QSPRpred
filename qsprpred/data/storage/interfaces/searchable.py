from abc import ABC, abstractmethod


class SMARTSSearchable(ABC):
    """Instances of this class can be searched with SMARTS patterns."""
    @abstractmethod
    def searchWithSMARTS(
        self,
        patterns: list[str],
    ) -> "SMARTSSearchable":
        """Search the molecules within this instance with SMARTS patterns.

        Args:
            patterns:
                List of SMARTS patterns to search with.

        Returns:
            (SMARTSSearchable): Another instance that can be filtered further.
        """


class PropSearchable(ABC):
    """Interface for searching on properties."""
    @abstractmethod
    def searchOnProperty(
        self,
        prop_name: str,
        values: list[float | int | str],
        exact=False
    ) -> "PropSearchable":
        """Search the molecules within this `MoleculeDataSet` on a property value.

        Args:
            prop_name:
                Name of the column to search on.
            values:
                Values to search for.
            exact:
                Whether to search for exact matches or not.

        Returns:
            (PropSearchable): Another instance that can be filtered further.
        """
