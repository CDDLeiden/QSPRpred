"""Different filters to select features from trainingset.

To add a new feature filters:
* Add a FeatureFilter subclass for your new filter
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from ...logs import logger
from ...utils.interfaces.randomized import Randomized


class FeatureFilter(ABC):
    """Filter out uninformative featureNames from a dataframe."""

    @abstractmethod
    def __call__(self, df: pd.DataFrame, y_col: pd.DataFrame = None):
        """Filter out uninformative features from a dataframe.

        Args:
            df (pd.DataFrame): dataframe to be filtered
            y_col (pd.DataFrame, optional): output dataframe if the filtering method
                requires it

        Returns:
            The filtered pd.DataFrame
        """


class LowVarianceFilter(FeatureFilter):
    """Remove features with variance lower than a given threshold after MinMax scaling.

    Attributes:
        th (float): threshold for removing features
    """

    def __init__(self, th: float) -> None:
        self.th = th

    def __call__(self, df: pd.DataFrame, y_col: pd.DataFrame = None) -> pd.DataFrame:
        # scale values between 0 and 1
        colnames = df.columns
        data_scaled = MinMaxScaler().fit_transform(X=df.values)
        variance = data_scaled.var(axis=0, ddof=1)

        low_var_cols = np.where(variance <= self.th)[0]

        # drop from both the minmax scaled descriptors df and the original dataframe
        df = df.drop(list(colnames[low_var_cols]), axis=1)
        logger.info(
            f"number of columns dropped low variance filter: {len(low_var_cols)}"
        )
        logger.info(f"number of columns left: {df.shape[1]}")

        return df


class HighCorrelationFilter(FeatureFilter):
    """Remove features with correlation higher than a given threshold.

    Attributes:
        th (float): threshold for correlation
    """

    def __init__(self, th: float) -> None:
        self.th = th

    def __call__(self, df: pd.DataFrame, y_col: pd.DataFrame = None) -> pd.DataFrame:
        if df.shape[1] == 1:
            return df
        # make absolute, because we also want to filter out large negative correlation
        correlation = np.triu(np.abs(np.corrcoef(df.values.astype(float).T)), k=1)
        high_corr = np.where(np.any(correlation > self.th, axis=0))

        logger.info(
            f"number of columns dropped high correlation filter: {len(high_corr[0])}"
        )
        df = df.drop(df.columns[high_corr[0]], axis=1)
        logger.info(f"number of columns left: {df.shape[1]}")

        return df


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

    def __init__(self, boruta_feat_selector: BorutaPy = None, seed: int | None = None):
        """Initialize the BorutaFilter class.

        Args:
            boruta_feat_selector (BorutaPy, optional): The BorutaPy feature selector.
                If not provided, a default BorutaPy instance will be created.
            seed (int | None, optional): Random state to use for shuffling and other
                random operations. If None, the random state set in the BorutaPy
                instance is used. Defaults to None.
        """
        Randomized.__init__(self, seed)
        self.featSelector = boruta_feat_selector
        if self.featSelector is None:
            self.featSelector = BorutaPy(estimator=RandomForestRegressor())
        if seed is not None:
            self.featSelector.random_state = seed

        # set seed from BorutaPy instance to class attribute
        self.setSeed(self.featSelector.random_state)

    def __call__(
        self, features: pd.DataFrame, y_col: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Filter out uninformative features from a dataframe using BorutaPy.

        Args:
            features (pd.DataFrame): dataframe to be filtered
            y_col (pd.DataFrame): target column(s)

        Returns:
            pd.DataFrame: filtered dataframe
        """
        if y_col.shape[1] > 1:
            raise NotImplementedError(
                "Boruta filter only works with one target column."
            )
        self.featSelector.fit(features.values, y_col.values.ravel())

        selected_features = features.loc[:, self.featSelector.support_]
        logger.info(
            "Number of columns dropped Boruta filter: "
            f"{features.shape[1] - selected_features.shape[1]}"
        )
        logger.info(f"Number of columns left: {selected_features.shape[1]}")

        return selected_features
