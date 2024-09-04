from dataclasses import dataclass

from ...data.processing.data_filters import RepeatsFilter
from ...data.processing.feature_standardizers import SKLearnStandardizer
from ...data.sampling.splits import DataSplit


@dataclass
class DataPrepSettings:
    """Class that determines settings for data preparation. These are arguments
    passed to `QSPRDataSet.prepareDataset`.

    Attributes:
        data_filters (list):
            Data filters to use.
        split (DataSplit):
            Data split to use.
        feature_filters (list):
            Feature filters to use.
        feature_standardizer (SKLearnStandardizer):
            Standardizer to use for features.
        feature_fill_value (float):
            Fill value to use for features.
        shuffle (bool):
            Whether to shuffle the data.
    """
    data_filters: list | None = (RepeatsFilter(keep=True),)
    split: DataSplit = None
    feature_filters: list = None
    feature_standardizer: SKLearnStandardizer = None
    feature_fill_value: float = 0.0
    shuffle: bool = True
