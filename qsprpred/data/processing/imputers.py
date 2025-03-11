import pandas as pd
from .pipeline import Step
from abc import abstractmethod
from sklearn.impute._base import _BaseImputer
from qsprpred.logs import logger

class Imputer(Step):
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Fit the imputer to the dataset
        
        Args:
            X (pd.DataFrame): training data features
            y (pd.DataFrame): training targets
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Impute values in the dataset.
        
        Args:
            X (pd.DataFrame): features (to be imputed)
            y (pd.DataFrame): target data (to be imputed)
        
        Returns:
            pd.DataFrame: (imputed) data
            pd.DataFrame: (imputed) target data
        """
        pass
    

class TargetImputer(Imputer):
    def __init__(self, imputer: _BaseImputer, target_properties: list[str] | None = None):
        """
        Initialize the target imputer.
        
        Args:
            imputer (callable): imputer function, e.g. from sklearn.impute, should
                have fit and transform methods
            target_properties (list[str], optional): target properties to impute
        """
        self.imputer = imputer
        self.target_properties = target_properties
        self._fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit the imputer to the dataset
        
        Args:
            X (pd.DataFrame): training data features
            y (pd.DataFrame): training targets
        """
        if self.target_properties is None:
            self.target_properties = y.columns
        self.imputer.fit(y[self.target_properties])
        self._fitted = True
    
    def transform(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Impute values in the dataset.
        
        Args:
            X (pd.DataFrame): features (to be imputed)
            y (pd.DataFrame): target data (to be imputed)
        
        Returns:
            pd.DataFrame: (imputed) data
            pd.DataFrame: (imputed) target data
        """
        if not self._fitted:
            raise ValueError("Imputer not fitted.")
        y_imputed = y.copy()
        y_imputed[self.target_properties] = self.imputer.transform(y[self.target_properties])
        return X, y_imputed
    
class FeatureImputer(Imputer):
    def __init__(self, imputer: _BaseImputer, feature_properties: list[str] | None = None):
        """
        Initialize the feature imputer.
        
        Args:
            imputer (callable): imputer function, e.g. from sklearn.impute, should
                have fit and transform methods
            feature_properties (list[str], optional): feature properties to impute,
                if None, all features will be imputed. Note that you can set either
                a DescriptorSet name or a list of feature names prefixed by the
                DescriptorSet name, e.g. ['RDKitDesc', 'MorganFP_0', 'MorganFP_1']
        """
        self.imputer = imputer
        self.feature_properties = feature_properties
        self._fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit the imputer to the dataset
        
        Args:
            X (pd.DataFrame): training data features
            y (pd.DataFrame): training targets
        """
        if self.feature_properties is None:
            self.feature_properties = X.columns
        
        to_be_imputed = self.get_features_to_be_imputed(X)
        self.imputer.fit(X[to_be_imputed])
        self._fitted = True
    
    def transform(self, X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Impute values in the dataset.
        
        Args:
            X (pd.DataFrame): features (to be imputed)
            y (pd.DataFrame): target data (to be imputed)
        
        Returns:
            pd.DataFrame: (imputed) data
            pd.DataFrame: (imputed) target data
        """
        if not self._fitted:
            raise ValueError("Imputer not fitted.")
        X_imputed = X.copy()
        
        to_be_imputed = self.get_features_to_be_imputed(X)
        X_imputed[to_be_imputed] = self.imputer.transform(X[to_be_imputed])
        return X_imputed, y
    
    def get_features_to_be_imputed(self, X: pd.DataFrame) -> list[str]:
        """Get the features that will be imputed.
        
        Args:
            X (pd.DataFrame): features
        
        Returns:
            list[str]: features to be imputed
        """
        return [
            col for col in X.columns
            if any(col.startswith(prop) for prop in self.feature_properties)
        ]