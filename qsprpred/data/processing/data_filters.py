"""Filters for QSPR Datasets.

To add a new filter:
* Add a DataFilter subclass for your new filter
"""

from abc import abstractmethod
from itertools import chain

import numpy as np
import pandas as pd

from ...logs import logger
from .pipeline import Step
from .applicability_domain import ApplicabilityDomain, MLChemAD
from mlchemad.base import ApplicabilityDomain as MLChemADApplicabilityDomain
from ..tables.interfaces.data_set_dependent import DataSetDependent
from ..tables.interfaces.qspr_data_set import QSPRDataSet


class DataFilter(Step, DataSetDependent):
    """Filter out some rows from a dataframe."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Fit the filter to the data.

        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame, optional): training targets
        """

    @abstractmethod
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Remove rows from a dataframe.

        Args:
            X (pd.DataFrame): dataframe to be standardized
            y (pd.DataFrame, optional): output dataframe if the standardization method
                requires it
        """

class CategoryFilter(DataFilter):
    """To filter out values from column

    Attributes:
        prop (str): column based on which to filter.
        values (list[str]): filter values.
        keep (bool): whether to keep or discard values.
    """
    def __init__(
        self,
        prop: str,
        values: list[str],
        data_set: QSPRDataSet | None = None,
        keep: bool = False
    ) -> None:
        """Initialize the CategoryFilter with the name, values and keep attributes.

        Args:
            prop (str): column based on which to filter.
            values (list): list of values to filter from props.
            data_set (QSPRDataSet): dataset to filter.
            keep (bool, optional): whether to keep or discard the values. Defaults to
                False.
        """
        super().__init__(data_set)
        self.prop = prop
        self.values = values
        self.keep = keep
        

    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Fit the filter to the data.

        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame, optional): training targets
        """

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """Filter rows from dataframe.

        Args:
            X (pd.DataFrame): dataframe to filter.
            y (pd.DataFrame, optional): output dataframe if the filtering method
                requires it

        Returns:
            pd.DataFrame: filtered dataframe.
        """
        assert self.hasDataSet, (
            "No dataset attached to this filter, set dataset with setDataSet()"
        )
        prop_col = self.dataSet.getDF()[self.prop].copy()
        old_len = X.shape[0]
        if self.keep:
            idx_to_keep = prop_col.isin(self.values)
        else:
            idx_to_keep = ~prop_col.isin(self.values)
        X = X.loc[idx_to_keep]
        if y is not None:
            y = y.loc[idx_to_keep]
        logger.info(f"{old_len - X.shape[0]} rows filtered out.")

        return X, y

class RepeatsFilter(DataFilter):
    """To filter out duplicate molecules based on descriptor values.

    Attributes:
        keep (str): For duplicate entries determines how properties are treated,
            if False remove both (/all) duplicate entries, if True keep them,
            if first, keep row of first entry (based on time), if last keep row of
            last entry based on time.
            options: 'first', 'last', True, False
        timeCol (str, optional): name of column containing time of publication
            used if keep is 'first' or 'last'
        additionalCols (list[str], optional): additional columns to use for
            determining duplicates (e.g. proteinid, in case of PCM modelling),
            so that compounds with same X but different proteinid
            are not removed.
    """
    
    def __init__(
        self,
        keep: str | bool = False,
        timecol: str | None = None,
        additional_cols: list[str] | None = None,
        data_set: QSPRDataSet | None = None
    ) -> None:
        """Initialize the RepeatsFilter with the keep, timecol and additional_cols
        attributes.

        Args:
            keep (str|bool, optional): For duplicate entries determines how properties
                are treated, if False remove both (/all) duplicate entries, if True
                keep them, if first, keep row of first entry (based on time), if last
                keep row of last entry based on time. Defaults to False.
            timecol (str, optional): name of column containing time of publication
                used if keep is 'first' or 'last'. Defaults to None.
            additional_cols (list[str], optional): additional columns to use for
                determining duplicates (e.g. proteinid, in case of PCM modelling),
                so that compounds with same X but different proteinid
                are not removed. Defaults to None.
            data_set (QSPRDataSet, optional): dataset to filter. Defaults to None.
        """
        super().__init__(data_set)
        self.keep = keep
        self.timeCol = timecol
        self.additionalCols = additional_cols

    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        """Fit the filter to the data.

        Args:
            X (pd.DataFrame): training data
            y (pd.DataFrame, optional): training targets
        """

    def transform(self, X: pd.DataFrame, y: pd.DataFrame | None = None) -> pd.DataFrame:
        """Filter rows from dataframe.

        Arguments:
            X (pandas dataframe): dataframe to filter
            y (pandas dataframe, optional): output dataframe if the filtering method
                requires it
        """
        def group_duplicate_index(df) -> list[list[int]]:
            """Group indices of duplicate rows

            From https://stackoverflow.com/a/46629623

            Args:
                a (numpy array): array of fingerprints

            Returns:
                list[list[int]]: list of lists of indices of duplicate rows
            """
            # Sort by rows
            a = df.values
            sidx = np.lexsort(a.T)
            b = a[sidx]

            # Get unique row mask
            m = np.concatenate(([False], (b[1:] == b[:-1]).all(1), [False]))

            # Get start and stop indices for each group of duplicates
            idx = np.flatnonzero(m[1:] != m[:-1])

            # Get sorted indices
            sort_idxs = df.index[sidx].tolist()

            # Return list of lists of indices of duplicate rows
            return [sort_idxs[i:j] for i, j in zip(idx[::2], idx[1::2] + 1)]
        assert self.hasDataSet, (
            "No dataset attached to this filter, set dataset with setDataSet()"
        )
        if self.timeCol is not None:
            assert (
                self.timeCol in self.dataSet.getDF().columns
            ), f"Column {self.timeCol} not found in dataset."
            timecol = self.dataSet.getDF()[self.timeCol].copy()
        if self.additionalCols is not None:
            assert isinstance(self.additionalCols, list), (
                "additionalCols must be a list of column names."
            )
            assert all(
                col in self.dataSet.getDF().columns for col in self.additionalCols
            ), f"Columns {self.additionalCols} not found in dataset."
            additional_cols = {
                col: self.dataSet.getDF()[col].copy() for col in self.additionalCols
            }

        if X.shape[1] == 0:
            logger.warning("Dataframe is empty, nothing to filter.")
            return X, y

        X_copy = X.copy()

        # Adding additional columns to X
        if self.additionalCols is not None:
            for col in additional_cols:
                X_copy[col] = additional_cols[col]

        allrepeats = group_duplicate_index(X_copy)

        if self.keep is True:
            if len(allrepeats) > 0:
                logger.warning(
                    "Dataframe contains compounds with duplicate features."
                    f"\nThe following rows contain duplicates: {allrepeats}"
                )
        else:
            if self.keep in ["first", "last"]:
                assert (
                    self.timeCol is not None
                ), "timecol must be specified if keep is 'first' or 'last'"
                timecol = pd.to_numeric(timecol, errors="coerce")
                for repeat in allrepeats:
                    repeat_time = timecol.loc[repeat]
                    if self.keep == "first":
                        tokeep = repeat_time.idxmin()  # Use the first occurance
                    else:
                        tokeep = repeat_time.idxmax()
                    # Remove the data point to keep from the allrepeats list
                    repeat.remove(tokeep)

            to_drop = list(chain(*allrepeats))
            logger.info(f"{len(to_drop)} duplicate rows filtered out.")
            X = X.drop(list(chain(*allrepeats)))
            if y is not None:
                y = y.drop(list(chain(*allrepeats)))

        return X, y

class NaNFilter(DataFilter):
    """Step that removes rows containing NaN values in a specified column"""
    
    def __init__(self, features: list[str] | None = None, keep: bool = False):
        """Initialize the step with the columns to check for NaN values
        
        If no columns are specified, all columns are checked for NaN values.
        
        Args:
            features (list[str] | None): columns to check for NaN values
            keep (bool): whether to keep or discard rows with NaN values,
                if True only warn about NaN values, if False remove rows with NaN values
        """
        self.keep = keep
        self.selected_features = features
        
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        pass
    
    def transform(self, X: pd.DataFrame, y: None | pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Remove rows containing NaN values in the specified columns"""
        if self.selected_features is None:
            self.selected_features = X.columns
        # only take selected features that are in the current data
        selected_features = list(set(self.selected_features) & set(X.columns))
        
        if self.keep:
            nan_mask = X[selected_features].isnull()
            nan_rows, nan_features = nan_mask.index[nan_mask.any(axis=1)], nan_mask.columns
            nan_features_per_row = nan_mask.loc[nan_rows].apply(lambda row: nan_features[row].tolist(), axis=1)
            for row, features in zip(nan_rows, nan_features_per_row):
                logger.warning(f"Entry {row} contains NaN values in features {features}.")
        else:
            drop_rows = X.index[X[selected_features].isnull().any(axis=1)]
            if len(drop_rows) > 0:
                logger.info(
                    f"Removing rows {drop_rows} with NaN values in features."
                )
            X = X.dropna(subset=selected_features)
            if y is not None:
                y = y.loc[X.index]
        return X, y

class OutlierFilter(DataFilter):
    def __init__(self, ad: ApplicabilityDomain):
        if isinstance(ad, MLChemADApplicabilityDomain):
            ad = MLChemAD(ad)
        self.ad = ad
        
    def fit(self, X: pd.DataFrame, y: None | pd.DataFrame = None):
        self.ad.fit(X)
        
    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        indomain = self.ad.contains(X)
        logger.info(f"Removing {len(X) - indomain.sum()} samples outside the applicability domain.")
        logger.debug(f"Removing samples {X.index[~indomain].tolist()} outside the applicability domain.")
        X = X.loc[indomain]
        if y is not None:
            y = y.loc[indomain]
        return X, y