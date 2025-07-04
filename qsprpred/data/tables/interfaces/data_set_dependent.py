from qsprpred.utils.serialization import JSONSerializable
from qsprpred.data.tables.interfaces.qspr_data_set import QSPRDataSet
from typing import ClassVar


class DataSetDependent(JSONSerializable):
    """Classes that need an attached `QSPRDataSet` should inherit from this class,
    and it will be supplied to them via this API.

    Attributes:
        dataSet (QSPRDataSet): The data set attached to this object.
    """
    _notJSON: ClassVar = ["dataSet"]
    
    def __init__(self, dataset: QSPRDataSet | None = None):
        """Initialize the object with a data set.

        Args:
            dataset (QSPRDataSet, optional):
                The data set to attach to this object. Defaults to None.
        """
        self.dataSet = dataset

    def setDataSet(self, dataset: QSPRDataSet | None) -> None:
        """Set the data set for this object."""
        self.dataSet = dataset

    @property
    def hasDataSet(self) -> bool:
        """Indicates if this object has a data set attached to it."""
        return self.dataSet is not None

    def getDataSet(self) -> QSPRDataSet:
        """Get the data set attached to this object.

        Returns:
            QSPRDataSet: The data set attached to this object

        Raises:
            ValueError: If no data set is attached to this object.
        """
        if self.hasDataSet:
            return self.dataSet
        else:
            raise ValueError("Data set not set.")
        
    def __setstate__(self, state):
        super().__setstate__(state)
        self.dataSet = None