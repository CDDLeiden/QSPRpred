from abc import ABC, abstractmethod
import pandas as pd

class Step(ABC):
    """A data preprocessing step that can be applied to a dataset"""
    
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Fit the step to the dataset
        
        If the step requires fitting to the data, this method should be implemented.
        
        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame): training targets
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply the step to the dataset
        
        Args:
            X (pd.DataFrame): data to be transformed
            y (pd.DataFrame): target data to be transformed
        
        Returns:
            pd.DataFrame: transformed data
            pd.DataFrame: (transformed) target data
        """
        pass
    
    def fitTransform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit the step to the dataset and apply it
        
        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame): training targets
            
        Returns:
            pd.DataFrame: transformed data
            pd.DataFrame: (transformed) target data
        """
        self.fit(X, y)
        return self.transform(X, y)

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
    

class QSPRPipeline(Pipeline):
    """Pipeline class for QSPR prediction
    
    QSPRPipeline is a sequence of data preprocessing steps that can be applied to a dataset.
    
    Args:
        steps (dict[str, Step]): Dictionary of named steps in the pipeline
    """
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        for step in self.steps.values():
            X, y = step.fitTransform(X, y)
    
    def apply(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> pd.DataFrame:
        for step in self.steps.values():
            X, y = step.transform(X, y)
        return X, y