"""This module is used for feature standardization and transformation in a pipeline."""

import pandas as pd

from .pipeline import Step
from sklearn.base import BaseEstimator

class FeatureTransformer(Step):
    """Base class for feature transformers
    
    This class is used to standardize or transform feature sets in a pipeline.
    It should be subclassed to implement specific transformations.
    
    Currently, only the SklearnStep class is implemented,
    which wraps a scikit-learn transformer.
    """


class SklearnStep(FeatureTransformer):
    """Step that wraps a scikit-learn transformer
    
    For example, this can be used to wrap a scikit-learn StandardScaler

    Attributes:
        transformer (BaseEstimator): scikit-learn transformer to wrap, should
            have implementations of the `fit` and `transform` methods.
    """
    
    def __init__(self, transformer: BaseEstimator):
        """Initialize the SklearnStep
        
        Args:
            transformer (BaseEstimator): scikit-learn transformer to wrap, should 
                have implementations of the `fit` and `transform` methods.
        """
        self.transformer = transformer
    
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Fit the transformer to the data
        
        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame | None): training targets
        """
        self.transformer.fit(X, y)
    
    def transform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Transform the data using the transformer
        
        Args:
            X (pd.DataFrame): data to be transformed
            y (pd.DataFrame | None): target data to be transformed

        Returns:
            pd.DataFrame: transformed data
            pd.DataFrame | None: (transformed) target data
        """
        return pd.DataFrame(self.transformer.transform(X), columns=X.columns, index=X.index), y        

