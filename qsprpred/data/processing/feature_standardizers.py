"""This module is used for standardizing feature sets."""

import ml2json
import numpy as np
import pandas as pd

from ...logs import logger
from ..pipelines.pipeline import Step
from abc import abstractmethod

class Standardizer(Step):
    """Standardizer for molecular features."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Fit the standardizer to the data.
        
        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame, optional): training targets
        """
    
    @abstractmethod
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Standardize features.	
        
        Args:
            X (pd.DataFrame): dataframe to be standardized
            y (pd.DataFrame, optional): output dataframe if the standardization method
                requires it
        """
        
    @property
    @abstractmethod
    def fitted(self) -> bool:
        """Return True if the standardizer has been fitted."""

class SKLearnStandardizer(Standardizer):
    """Standardizer for molecular features."""
    def __init__(self, scaler):
        """
        Initialize the standardizer.

        Args:
            scaler: sklearn object
        """

        self.scaler = scaler
        self._fitted = False

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["scaler"] = ml2json.to_dict(self.scaler)
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.scaler = ml2json.from_dict(state["scaler"])

    def __str__(self):
        """Return string representation."""
        return f"SKLearnStandardizer_{self.scaler.__class__.__name__}"
    
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Fit the standardizer to the data.
        
        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame, optional): training targets
        """
        self.scaler.fit(X)
        logger.debug("Standardizer fitted")
        self._fitted = True
        
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Standardize features.	
        
        Args:
            X (pd.DataFrame): dataframe to be standardized
            y (pd.DataFrame, optional): output dataframe if the standardization method
                requires it
        """
        X_std = self.scaler.transform(X)
        return pd.DataFrame(X_std, index=X.index, columns=X.columns), y
    
    @property
    def fitted(self) -> bool:
        """Return True if the standardizer has been fitted."""
        return self._fitted
