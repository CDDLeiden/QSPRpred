from qsprpred.data.tables.interfaces.qspr_data_set import QSPRDataSet


class DataSetDependent:
    """Classes that need a molecule storage attached can derive from this."""

    def __init__(self, dataset: QSPRDataSet | None = None) -> None:
        self.dataSet = dataset

    def setDataSet(self, dataset: QSPRDataSet):
        self.dataSet = dataset

    @property
    def hasDataSet(self) -> bool:
        """Indicates if this object has a storage attached to it."""
        return self.dataSet is not None

    def getDataSet(self):
        """Get the storage attached to this object.

        Raises:
            ValueError: If no storage is attached to this object.
        """
        if self.hasDataSet:
            return self.dataSet
        else:
            raise ValueError("Data set not set.")
