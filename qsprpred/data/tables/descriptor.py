from typing import Optional

import pandas as pd

from qsprpred.data.descriptors.sets import DescriptorSet
from qsprpred.data.tables.pandas import PandasDataTable


class DescriptorTable(PandasDataTable):
    """Pandas table that holds descriptor data for modelling and other analyses.

    Attributes:
        calculator (DescriptorSet):
            `DescriptorSet` used for descriptor calculation.
    """

    def __init__(
            self,
            calculator: DescriptorSet,
            name: str,
            df: Optional[pd.DataFrame] = None,
            store_dir: str = ".",
            overwrite: bool = False,
            key_cols: list | None = None,
            n_jobs: int = 1,
            chunk_size: int = 1000,
            autoindex_name: str = "ID",
            random_state: int | None = None,
            store_format: str = "pkl",
    ):
        """Initialize a `DescriptorTable` object.

        Args:
            calculator (DescriptorSet):
                `DescriptorSet` used for descriptor calculation.
            name (str):
                Name of the  new  descriptor table.
            df (pd.DataFrame):
                data frame containing the descriptors. If you provide a
                dataframe for a dataset that already exists on disk,
                the dataframe from disk will override the supplied data
                frame. Set 'overwrite' to `True` to override
                the data frame on disk.
            store_dir (str):
                Directory to store the dataset files. Defaults to the
                current directory. If it already contains files with the same name,
                the existing data will be loaded.
            overwrite (bool):
                Overwrite existing dataset.
            key_cols (list):
                list of columns to use as index. If None, the index
                will be a custom generated ID.
            n_jobs (int):
                Number of jobs to use for parallel processing. If <= 0,
                all available cores will be used.
            chunk_size (int):
                Size of chunks to use per job in parallel processing.
            autoindex_name (str):
                Column name to use for automatically generated IDs.
            random_state (int):
                Random state to use for shuffling and other random ops.
            store_format (str):
                Format to use for storing the data ('pkl' or 'csv').
        """
        super().__init__(
            name,
            df,
            store_dir,
            overwrite,
            key_cols,
            n_jobs,
            chunk_size,
            autoindex_name,
            random_state,
            store_format,
        )
        self.calculator = calculator

    def getDescriptors(self, active_only=True):
        """Get the descriptors stored in this table."""
        return self.df[self.getDescriptorNames(active_only=active_only)]

    def getDescriptorNames(self, active_only=True):
        """Get the names of the descriptors in this represented by this table.
        By default, only active descriptors are returned. You can use active_only=False
        to get all descriptors saved in the table.

        Args:
            active_only (bool): Whether to return only descriptors that are active in
                the current descriptor set. Defaults to `True`.

        """
        if active_only:
            return self.calculator.transformToFeatureNames()
        else:
            return self.df.columns[~self.df.columns.isin(self.indexCols)].tolist()

    def fillMissing(self, fill_value, names):
        """Fill missing values in the descriptor table.

        Args:
            fill_value (float): Value to fill missing values with.
            names (list): List of descriptor names to fill. If `None`, all descriptors
                are filled.
        """
        columns = names if names else self.getDescriptorNames()
        self.df[columns] = self.df[columns].fillna(fill_value)

    def keepDescriptors(self, descriptors: list[str]) -> list[str]:
        """Mark only the given descriptors as active in this set.

        Args:
            descriptors (list): list of descriptor names to keep

        Returns:
            list[str]: list of descriptor names that were kept

        Raises:
            ValueError: If any of the descriptors are not present in the table.
        """
        all_descs = self.getDescriptorNames(active_only=False)
        to_keep = set(all_descs) & set(descriptors)
        prefix = str(self.calculator) + "_"
        self.calculator.descriptors = [
            x.replace(prefix, "", 1)  # remove prefix
            for x in self.calculator.transformToFeatureNames()
            if x in to_keep
        ]
        return self.getDescriptorNames()

    def restoreDescriptors(self) -> list[str]:
        """Restore all descriptors to active in this set."""
        all_descs = self.getDescriptorNames(active_only=False)
        prefix = str(self.calculator) + "_"
        self.calculator.descriptors = [
            x.replace(prefix, "", 1) for x in all_descs  # remove prefix
        ]
        return self.getDescriptorNames()
