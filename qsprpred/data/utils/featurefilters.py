"""Different filters to select features from trainingset.

To add a new feature filters:
* Add a FeatureFilter subclass for your new filter
"""
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from ...logs import logger
from ..interfaces import FeatureFilter


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


class BorutaFilter(FeatureFilter):
    """Boruta filter from BorutaPy: feature selection based on Random Forest predictors.

    Attributes:
        estimator (object): A supervised learning estimator, with a 'fit' method that
            returns the feature_importances attribute. Important features must
            correspond to high absolute values in the feature_importances.
        n_estimators (int or string): If int sets the number of estimators in the chosen
            ensemble method. If 'auto' this is determined automatically based on the
            size of the dataset. The other parameters of the used estimators need to be
            set with initialisation.
        perc (int): Instead of the max we use the percentile defined by the user,
            to pick our threshold for comparison between shadow and real features.
            The max tends to be too stringent. This provides a finer control over this.
            The lower perc is the more false positives will be picked as relevant
            but also the less relevant features will be left out. The usual trade-off.
            The default is essentially the vanilla Boruta corresponding to the max.
        alpha (float): Level at which the corrected p-values will get rejected in
            both correction steps.
        max_iter (int): The number of maximum iterations to perform.
        verbose (int): Controls verbosity of output.

    """
    def __init__(
        self,
        estimator=RandomForestRegressor(),
        n_estimators="auto",
        perc=80,
        alpha=0.05,
        max_iter=200,
        verbose=2,
    ):
        self.estimator = estimator
        self.nEstimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.maxIter = max_iter
        self.verbose = verbose

    def __call__(
        self, features: pd.DataFrame, y_col: pd.DataFrame = None
    ) -> pd.DataFrame:
        feat_selector = BorutaPy(
            estimator=self.estimator,
            n_estimators=self.nEstimators,
            perc=self.perc,
            alpha=self.alpha,
            max_iter=self.maxIter,
            verbose=self.verbose,
        )
        feat_selector.fit(features.values, y_col.values)

        selected_features = features.loc[:, feat_selector.support_]
        logger.info(
            "Number of columns dropped Boruta filter: "
            f"{features.shape[1] - selected_features.shape[1]}"
        )
        logger.info(f"Number of columns left: {selected_features.shape[1]}")

        return selected_features
