"""Different filters to select features from trainingset.

To add a new feature filters:
* Add a FeatureFilter subclass for your new filter
"""

from abc import abstractmethod

from typing import ClassVar
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from ...logs import logger
from ...utils.interfaces.randomized import Randomized
from .pipeline import Step
import os
from pickle import dump, load
import json


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


class LowVarianceFilter(FeatureFilter):
    """Remove features with variance lower than a given threshold after MinMax scaling.

    Attributes:
        th (float): threshold for removing features
        low_var_cols (pd.Index): columns with low variance (if fitted)
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
        
        if len(self.low_var_cols) == len(colnames):
            logger.warning(
                "All columns have low variance, no columns will be dropped, this filter"
                " will be skipped."
            )
            self.low_var_cols = None
        else:
            logger.info(
                f"Number of columns dropped low variance filter: {len(self.low_var_cols)}"
            )
            logger.info(f"Number of columns left: {X.shape[1] - len(self.low_var_cols)}")
        
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Filter out low variance features from a dataframe.

        Args:
            X (pd.DataFrame): dataframe to be filtered
            y (pd.DataFrame, optional): output dataframe if the filtering method
                requires it

        Returns:
            pd.DataFrame: The filtered dataframe
            pd.DataFrame: The target dataframe
        """
        assert hasattr(self, "low_var_cols"), "Filter has not been fitted yet."
        # assert self.low_var_cols.isin(X.columns).all(), "Columns do not match fitted columns."
        if self.low_var_cols is not None:
            columns_to_drop = self.low_var_cols.intersection(X.columns)
            
            X = X.drop(columns=columns_to_drop)

        return X, y


class HighCorrelationFilter(FeatureFilter):
    """Remove features with correlation higher than a given threshold.

    Attributes:
        th (float): threshold for correlation
        high_corr_cols (pd.Index): columns with high correlation (if fitted)
    """
    def __init__(self, th: float) -> None:
        self.th = th
        
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Find features with correlation higher than a given threshold.
        
        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame, optional): training targets
        """
        # stop if only 1 column
        if X.shape[1] == 1:
            logger.info("Only one column in the dataframe. No correlation check.")
            self.high_corr_cols = None
        else:
            correlation = np.triu(np.abs(np.corrcoef(X.values.astype(float).T)), k=1)
            high_corr = np.where(np.any(correlation > self.th, axis=0))

            self.high_corr_cols = X.columns[high_corr[0]]
            
            if len(self.high_corr_cols) == len(X.columns):
                logger.warning(
                    "All columns have high correlation, no columns will be dropped, "
                    "this filter will be skipped."
                )
                self.high_corr_cols = None
            else:
                logger.info(
                    f"Number of columns dropped high correlation filter: {len(self.high_corr_cols)}"
                )
                logger.info(f"Number of columns left: {X.shape[1] - len(self.high_corr_cols)}")
        
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Filter out high correlation features from a dataframe.

        Args:
            X (pd.DataFrame): dataframe to be filtered
            y (pd.DataFrame, optional): output dataframe if the filtering method
                requires it

        Returns:
            pd.DataFrame: The filtered dataframe
            pd.DataFrame: The target dataframe
        """
        assert hasattr(self, "high_corr_cols"), "Filter has not been fitted yet."
        if self.high_corr_cols is not None:
            #assert self.high_corr_cols.isin(X.columns).all(), "Columns do not match fitted columns."
            
            columns_to_drop = self.high_corr_cols.intersection(X.columns)
            
            X = X.drop(columns=columns_to_drop)

        return X, y


class BorutaFilter(FeatureFilter, Randomized):
    """Boruta filter from BorutaPy: Boruta all-relevant feature selection.

    Attributes:
        featSelector (BorutaPy): BorutaPy feature selector
        droppedFeatures (pd.Index): columns dropped by Boruta filter
        seed (int):
            Random state to use for shuffling and other random operations.
    """
    _notJSON: ClassVar = ["featSelector"]
    
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
    
    def toFile(self, filename: str) -> str:
        """Serialize object to a JSON file. This JSON file should
        contain all  data necessary to reconstruct the object.

        Args:
            filename (str): filename to save object to

        Returns:
            filename (str): absolute path to the saved JSON file of the object
        """
        with open(f"{filename.removesuffix('.json')}_featSelector.pkl", "wb") as f:
            dump(self.featSelector, f)
            
        o_dict = json.loads(self.toJSON())
        o_dict["py/state"]["featSelector"] = os.path.basename(
            f"{filename.removesuffix('.json')}_featSelector.pkl"
        )
        with open(filename, "w") as fh:
            json.dump(o_dict, fh, indent=4)
        return os.path.abspath(filename)
    
    @classmethod
    def fromFile(cls, filename: str) -> "BorutaFilter":
        ret = super().fromFile(filename)
        with open(f"{filename.removesuffix('.json')}_featSelector.pkl", "rb") as f:
            ret.featSelector = load(f)
        return ret

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
        self.droppedFeatures = X.columns[~self.featSelector.support_]

        logger.info(
            "Number of columns dropped Boruta filter: "
            f"{len(self.droppedFeatures)}"
        )
        logger.info(f"Number of columns left: {X.shape[1] - len(self.droppedFeatures)}")
        
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Filter out uninformative features from a dataframe using BorutaPy.

        Args:
            X (pd.DataFrame): dataframe to be filtered
            y (pd.DataFrame, optional): output dataframe if the filtering method
                requires it

        Returns:
            pd.DataFrame: The filtered dataframe
            pd.DataFrame: The target dataframe
        """
        assert hasattr(self, "droppedFeatures"), "Filter has not been fitted yet."
        assert self.droppedFeatures.isin(X.columns).all(), "Columns do not match fitted columns."
        
        X = X.drop(columns=self.droppedFeatures)

        return X, y