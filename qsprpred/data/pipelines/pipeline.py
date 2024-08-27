from abc import ABC, abstractmethod
import pandas as pd

class Step(ABC):
    """"A data preprocessing step that can be applied to a dataset"""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

class Pipeline(ABC):
    """Pipeline class for data preprocessing steps
    
    Pipeline is a sequence of data preprocessing steps that can be applied to a dataset.
    
    Args:
        steps (dict[str, Step]): Dictionary of named steps in the pipeline
    """
    
    def __init__(self, steps: dict[str, Step]):
        self.steps = steps
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        pass
    
    @abstractmethod
    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        pass