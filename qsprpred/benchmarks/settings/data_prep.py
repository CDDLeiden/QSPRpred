from dataclasses import dataclass

from ...data.processing.data_filters import RepeatsFilter
from ...data.sampling.splits import DataSplit
from ...data.pipelines.pipeline import Pipeline


@dataclass
class DataPrepSettings:
    """Class that determines settings for data preparation. These are arguments
    passed to `QSPRDataSet.prepareDataset`.

    Attributes:
        split (DataSplit):
            Data split to use.
        pipeline (Pipeline):
            Feature filters to use.
        feature_standardizer (SKLearnStandardizer):
            Standardizer to use for features.
        feature_fill_value (float):
            Fill value to use for features.
        shuffle (bool):
            Whether to shuffle the data.
    """
    split: DataSplit = None
    pipeline: Pipeline = None
    feature_fill_value: float = 0.0
    shuffle: bool = True
