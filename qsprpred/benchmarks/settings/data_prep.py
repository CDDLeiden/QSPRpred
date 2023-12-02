from dataclasses import dataclass
from typing import Callable

from ...data.processing.feature_standardizers import SKLearnStandardizer
from ...data.sampling.splits import DataSplit


@dataclass
class DataPrepSettings:
    split: DataSplit = None
    smiles_standardizer: str | Callable = "chembl"
    feature_filters: list = None
    feature_standardizer: SKLearnStandardizer = None
    feature_fill_value: float = 0.0
