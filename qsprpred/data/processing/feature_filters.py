"""Different filters to select features from trainingset.

To add a new feature filters:
* Add a FeatureFilter subclass for your new filter
"""
from abc import abstractmethod

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from ...logs import logger
from ...utils.interfaces.randomized import Randomized
from ..pipelines.pipeline import Step


class FeatureFilter(Step):
    """Filter out uninformative featureNames from a dataframe."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Fit the filter to the data.
        
        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame, optional): training targets
        """

    @abstractmethod
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Filter out uninformative features from a dataframe.

        Args:
            X (pd.DataFrame): dataframe to be filtered
            y (pd.DataFrame, optional): output dataframe if the filtering method
                requires it

        Returns:
            The filtered pd.DataFrame
        """
        
    def fitTransform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> pd.DataFrame:
        """Fit the filter to the data and transform the data.
        
        Args:
            X (pd.DataFrame): dataframe to be filtered
            y (pd.DataFrame, optional): output dataframe if the filtering method
                requires it

        Returns:
            The filtered pd.DataFrame
        """
        self.fit(X, y)
        return self.transform(X, y)


class LowVarianceFilter(FeatureFilter):
    """Remove features with variance lower than a given threshold after MinMax scaling.

    Attributes:
        th (float): threshold for removing features
    """

    def __init__(self, th: float) -> None:
        self.th = th
        
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Find features with variance lower than a given threshold after MinMax scaling.
        
        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame, optional): training targets
        """
        colnames = X.columns
        data_scaled = MinMaxScaler().fit_transform(X=X.values)
        variance = data_scaled.var(axis=0, ddof=1)
        
        low_var_cols_idx = np.where(variance <= self.th)[0]
        self.low_var_cols = colnames[low_var_cols_idx]
        logger.info(
            f"Number of columns dropped low variance filter: {len(self.low_var_cols)}"
        )
        logger.info(f"Number of columns left: {X.shape[1] - len(self.low_var_cols)}")
        
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Filter out low variance features from a dataframe.

        Args:
            X (pd.DataFrame): dataframe to be filtered
            y (pd.DataFrame, optional): output dataframe if the filtering method
                requires it

        Returns:
            The filtered pd.DataFrame
        """
        assert hasattr(self, "low_var_cols"), "Filter has not been fitted yet."
        assert self.low_var_cols.isin(X.columns).all(), "Columns do not match fitted columns."
        
        X = X.drop(columns=self.low_var_cols)

        return X


class HighCorrelationFilter(FeatureFilter):
    """Remove features with correlation higher than a given threshold.

    Attributes:
        th (float): threshold for correlation
    """

    def __init__(self, th: float) -> None:
        self.th = th
        
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Find features with correlation higher than a given threshold.
        
        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame, optional): training targets
        """
        correlation = np.triu(np.abs(np.corrcoef(X.values.astype(float).T)), k=1)
        high_corr = np.where(np.any(correlation > self.th, axis=0))

        self.high_corr_cols = X.columns[high_corr[0]]
        logger.info(
            f"Number of columns dropped high correlation filter: {len(self.high_corr_cols)}"
        )
        logger.info(f"Number of columns left: {X.shape[1] - len(self.high_corr_cols)}")
        
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Filter out high correlation features from a dataframe.

        Args:
            X (pd.DataFrame): dataframe to be filtered
            y (pd.DataFrame, optional): output dataframe if the filtering method
                requires it

        Returns:
            The filtered pd.DataFrame
        """
        assert hasattr(self, "high_corr_cols"), "Filter has not been fitted yet."
        assert self.high_corr_cols.isin(X.columns).all(), "Columns do not match fitted columns."
        
        X = X.drop(columns=self.high_corr_cols)

        return X


class BorutaFilter(FeatureFilter, Randomized):
    """Boruta filter from BorutaPy: Boruta all-relevant feature selection.

    Uses BorutaPy implementation from https://github.com/scikit-learn-contrib/boruta_py.
    Note that the `boruta` package is not compatible with numpy 1.24.0 and above.
    Therefore, make sure to downgrade numpy to 1.23.0 or older before using this filter.

    Attributes:
        featSelector (BorutaPy): BorutaPy feature selector
        seed (int):
            Random state to use for shuffling and other random operations.
    """

    @property
    def randomState(self) -> int:
        """Get the random state for the object."""
        return self.seed

    @randomState.setter
    def randomState(self, seed: int | None):
        """Set the random state for the object.

        Args:
            seed (int | None):
                The seed to use to randomize the action. If `None`,
                a random seed is used instead of a fixed one.
        """
        self.seed = seed

    def __init__(self, boruta_feat_selector: BorutaPy = None, seed: int | None = None):
        """Initialize the BorutaFilter class.

        Args:
            boruta_feat_selector (BorutaPy, optional): The BorutaPy feature selector.
                If not provided, a default BorutaPy instance will be created.
            seed (int | None, optional): Random state to use for shuffling and other
                random operations. If None, the random state set in the BorutaPy
                instance is used. Defaults to None.
        """
        self.seed = seed
        self.featSelector = boruta_feat_selector
        if self.featSelector is None:
            self.featSelector = BorutaPy(estimator=RandomForestRegressor())
        if seed is not None:
            self.featSelector.random_state = seed

        # set seed from BorutaPy instance to class attribute
        self.randomState = self.featSelector.random_state
        
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit the Boruta filter to the data.

        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame): training targets
        """
        assert y.shape[1] == 1, "Boruta filter only works with one target column."
        
        self.featSelector.fit(X.values, y.values.ravel())
        self.dropped_features = X.columns[~self.featSelector.support_]

        logger.info(
            "Number of columns dropped Boruta filter: "
            f"{len(self.dropped_features)}"
        )
        logger.info(f"Number of columns left: {X.shape[1] - len(self.dropped_features)}")
        
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Filter out uninformative features from a dataframe using BorutaPy.

        Args:
            X (pd.DataFrame): dataframe to be filtered
            y (pd.DataFrame, optional): output dataframe if the filtering method
                requires it

        Returns:
            The filtered pd.DataFrame
        """
        assert hasattr(self, "dropped_features"), "Filter has not been fitted yet."
        assert self.dropped_features.isin(X.columns).all(), "Columns do not match fitted columns."
        
        X = X.drop(columns=self.dropped_features)

        return X