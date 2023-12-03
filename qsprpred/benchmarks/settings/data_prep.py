from dataclasses import dataclass
from typing import Callable

from ...data.processing.feature_standardizers import SKLearnStandardizer
from ...data.sampling.splits import DataSplit


@dataclass
class DataPrepSettings:
    """Class that determines settings for data preparation. These are arguments
    passed to `QSPRDataset.prepareDataset`.

    Attributes:
        split (DataSplit):
            Data split to use.
        smiles_standardizer (str or callable):
            Standardizer to use for SMILES strings.
        feature_filters (list):
            Feature filters to use.
        feature_standardizer (SKLearnStandardizer):
            Standardizer to use for features.
        feature_fill_value (float):
            Fill value to use for features.
    """

    split: DataSplit = None
    smiles_standardizer: str | Callable = "chembl"
    feature_filters: list = None
    feature_standardizer: SKLearnStandardizer = None
    feature_fill_value: float = 0.0
