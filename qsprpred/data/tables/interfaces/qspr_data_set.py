from abc import ABC, abstractmethod


from qsprpred import TargetProperty
from qsprpred.data.tables.interfaces.molecule_data_set import MoleculeDataSet


class QSPRDataSet(MoleculeDataSet, ABC):
    """Interface for storing and managing QSPR-specific data sets."""
    @abstractmethod
    def setTargetProperties(
        self,
        target_props: list[TargetProperty | dict],
        drop_empty: bool = True,
    ):
        """Set the target properties for the dataset.

        Args:
            target_props (list[TargetProperty | dict]): The target properties to add.
            drop_empty (bool): If True, drop rows with missing target properties.
        """

    @abstractmethod
    def makeRegression(self, target_property: str):
        """Make this a regression dataset for the given target property.

        Args:
            target_property (str): The name of the target property.
        """

    @abstractmethod
    def makeClassification(self, target_property: str, threshold: float):
        """Make this a classification dataset for the given target property.

        Args:
            target_property (str): The name of the target property.
            threshold (float): The threshold for the classification.
        """

    @abstractmethod
    def restoreTargetProperty(self, prop: TargetProperty | str):
        """Restore a target property to the original state.

        Args:
            prop (TargetProperty | str): The target property to restore.
        """

    @abstractmethod
    def addTargetProperty(self, prop: TargetProperty | dict, drop_empty: bool = True):
        """Add a target property to the dataset.

        Args:
            prop (TargetProperty):
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
    def unsetTargetProperty(self, name: str | TargetProperty):
        """Unset the target property with the given name.

        Args:
            (str | TargetProperty): name of the target property to unset
        """

    @property
    @abstractmethod
    def targetProperties(self) -> list[TargetProperty]:
        """Get the target properties of the dataset.

        Returns:
            (list): list of target properties
        """
