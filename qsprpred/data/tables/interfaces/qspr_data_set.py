from abc import ABC, abstractmethod


from qsprpred import TargetSpec
from qsprpred.data.tables.interfaces.molecule_data_set import MoleculeDataSet


class QSPRDataSet(MoleculeDataSet, ABC):
    """Interface for storing and managing QSPR-specific data sets."""
    @abstractmethod
    def setTargetProperties(
        self,
        target_props: list[TargetSpec | dict],
        drop_empty: bool = True,
    ):
        """Set the target properties for the dataset.

        Args:
            target_props (list[TargetSpec | dict]): The target properties to add.
            drop_empty (bool): If True, drop rows with missing target properties.
        """

    @abstractmethod
    def makeRegression(self, target_property: str):
        """Make this a regression dataset for the given target property.
           This is only possible if the target property was previously converted
           to classification.

        Args:
            target_property (str): The name of the target property.
        """

    @abstractmethod
    def makeClassification(self, target_property: str, threshold: float | list[float]):
        """Make this a classification dataset for the given target property.

        Args:
            target_property (str): The name of the target property.
            threshold (float | list[float]): The threshold for the classification.
        """

    @abstractmethod
    def restoreTargetProperty(self, prop: TargetSpec | str):
        """Restore a target property to the original state.

        Args:
            prop (TargetSpec | str): The target property to restore.
        """

    @abstractmethod
    def addTargetProperty(self, prop: TargetSpec | dict, drop_empty: bool = True):
        """Add a target property to the dataset.

        Args:
            prop (TargetSpec):
                name of the target property to add
            drop_empty (bool):
                whether to drop rows with empty target property values. Defaults to
                `True`.
        """

    @property
    @abstractmethod
    def isMultiTask(self) -> bool:
        """Indicates if the dataset is a multi-task dataset."""

    @abstractmethod
    def unsetTargetProperty(self, name: str | TargetSpec):
        """Unset the target property with the given name.

        Args:
            (str | TargetSpec): name of the target property to unset
        """

    @property
    @abstractmethod
    def targetProperties(self) -> list[TargetSpec]:
        """Get the target properties of the dataset.

        Returns:
            (list): list of target properties
        """

    def getTargetPropertiesNames(self) -> list[str]:
        """Get the names of the target properties.
        Returns:
            (list[str]): list of target property names
        """
        return TargetSpec.getNames(self.targetProperties)
