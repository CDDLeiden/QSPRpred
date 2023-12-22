import concurrent
import json
import os
import shutil
import warnings
from typing import ClassVar, Callable, Optional

import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm

from .base import DataTable
from ...logs import logger
from ...utils.serialization import JSONSerializable
from ...utils.stringops import generate_padded_index


class PandasDataTable(DataTable, JSONSerializable):
    """A Pandas DataFrame wrapper class to enable data processing functions on
    QSPRpred data.

    Attributes:
        name (str): Name of the data set. You can use this name to load the dataset
            from disk anytime and create a new instance.
        df (pd.DataFrame): Pandas dataframe containing the data. If you provide a
            dataframe for a dataset that already exists on disk, the dataframe from
            disk will override the supplied data frame. Set 'overwrite' to `True` to
            override the data frame on disk.
        storeDir (str): Directory to store the dataset files. Defaults to the
            current directory. If it already contains files with the same name,
            the existing data will be loaded.
        indexCols (List): List of columns to use as index. If None, the index
            will be a custom generated ID.
        nJobs (int): Number of jobs to use for parallel processing. If <= 0,
            all available cores will be used.
        chunkSize (int): Size of chunks to use per job in parallel processing.
        randomState (int): Random state to use for all random operations.

    """

    _notJSON: ClassVar = [*JSONSerializable._notJSON, "df"]

    class ParallelApplyWrapper:
        """A wrapper class to parallelize pandas apply functions."""

        def __init__(
            self,
            func: Callable,
            func_args: list | None = None,
            func_kwargs: dict | None = None,
            axis: int = 0,
            raw: bool = False,
            result_type: str = "expand",
        ):
            """Initialize the instance with pandas parameters to apply to data chunks.

            See `pandas.DataFrame.apply` for more information.

            Args:
                func (Callable): Function to apply to the data frame.
                func_args (list): Positional arguments to pass to the function.
                func_kwargs (dict): Keyword arguments to pass to the function.
                axis (int): Axis to apply func along (0 for columns, 1 for rows).
                raw (bool): Whether to pass Series object to func or raw array.
                result_type (str): whether to expand/ignore the results.

            """
            self.args = func_args
            self.kwargs = func_kwargs
            self.func = func
            self.axis = axis
            self.raw = raw
            self.result_type = result_type

        def __call__(self, data: pd.DataFrame):
            """Apply the function to the current chunk of data.

            Args:
                data (pd.DataFrame): chunk of data to apply function to

            Returns:
                result of applying function to chunk of data
            """
            return data.apply(
                self.func,
                raw=self.raw,
                axis=self.axis,
                result_type=self.result_type,
                args=self.args,
                **self.kwargs if self.kwargs else {},
            )

    def __init__(
        self,
        name: str,
        df: Optional[pd.DataFrame] = None,
        store_dir: str = ".",
        overwrite: bool = False,
        index_cols: Optional[list[str]] = None,
        n_jobs: int = 1,
        chunk_size: int = 1000,
        autoindex_name: str = "QSPRID",
        random_state: int | None = None,
    ):
        """Initialize a `PandasDataTable` object.
        Args
            name (str): Name of the data set. You can use this name to load the dataset
                from disk anytime and create a new instance.
            df (pd.DataFrame): Pandas dataframe containing the data. If you provide a
                dataframe for a dataset that already exists on disk, the dataframe from
                disk will override the supplied data frame. Set 'overwrite' to `True` to
                override the data frame on disk.
            store_dir (str): Directory to store the dataset files. Defaults to the
                current directory. If it already contains files with the same name,
                the existing data will be loaded.
            overwrite (bool): Overwrite existing dataset.
            index_cols (List): List of columns to use as index. If None, the index
                will be a custom generated ID.
            n_jobs (int): Number of jobs to use for parallel processing. If <= 0,
                all available cores will be used.
            chunk_size (int): Size of chunks to use per job in parallel processing.
            autoindex_name (str): Column name to use for automatically generated IDs.
            random_state (int): Random state to use for all random operations
                for reproducibility. If not specified, the state is generated randomly.
                The state is saved upon `save` so if you want to change the state later,
                call the `setRandomState` method after loading.
        """
        self.randomState = None
        self.setRandomState(
            random_state or int(np.random.randint(0, 2**32 - 1, dtype=np.int64))
        )
        self.name = name
        self.indexCols = index_cols
        # parallel settings
        self.nJobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.chunkSize = chunk_size
        # paths
        self._storeDir = store_dir.rstrip("/")
        # data frame initialization
        self.df = None
        if df is not None:
            if self._isInStore("df") and not overwrite:
                warnings.warn(
                    "Existing data set found, but also found a data frame in store. "
                    "Refusing to overwrite data. If you want to overwrite data in "
                    "store, set overwrite=True.",
                    stacklevel=2,
                )
                self.reload()
            else:
                self.clearFiles()
                self.df = df
                if index_cols is not None:
                    self.setIndex(index_cols)
                else:
                    self.generateIndex(name=autoindex_name)
        else:
            if not self._isInStore("df"):
                raise ValueError(
                    f"No data frame found in store for '{self.name}'. Are you sure "
                    "this is the correct dataset? If you are creating a new data set, "
                    "make sure to supply a data frame."
                )
            self.reload()
        assert self.df is not None, "Unknown error in data set creation."

    def __len__(self):
        """
        Get the number of rows in the data frame.

        Returns: number of rows in the data frame.
        """
        return len(self.df)

    def __getstate__(self):
        o_dict = super().__getstate__()
        os.makedirs(self.storeDir, exist_ok=True)
        self.df.to_pickle(self.storePath)
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.reload()

    @property
    def baseDir(self):
        return self._storeDir

    @property
    def storeDir(self):
        return f"{self.baseDir}/{self.name}"

    @property
    def storePath(self):
        return f"{self.storePrefix}_df.pkl"

    @property
    def storePrefix(self):
        return f"{self.storeDir}/{self.name}"

    @property
    def metaFile(self):
        return f"{self.storePrefix}_meta.json"

    def setIndex(self, cols: list[str]):
        """
        Set the index of the data frame.

        Args:
            cols (list[str]): list of columns to use as index.
        """
        self.df.set_index(cols, inplace=True, verify_integrity=True, drop=False)
        self.df.drop(
            inplace=True,
            columns=[c for c in self.df.columns if c.startswith("Unnamed")],
        )
        self.df.index.name = "~".join(cols)
        self.indexCols = cols

    def generateIndex(self, name: str = "QSPRID", prefix: str | None = None):
        """Generate a custom index for the data frame.

        Args:
            name (str): name of the index column.
            prefix (str): prefix to use for the index column values.
        """
        prefix = prefix if prefix is not None else self.name
        self.df[name] = generate_padded_index(self.df.index, prefix=prefix)
        self.setIndex([name])

    def _isInStore(self, name):
        """Check if a pickled file with the given suffix exists.

        Args:
            name (str): Suffix of the file to check.

        Returns:
            bool: `True` if the file exists, `False` otherwise.
        """
        return os.path.exists(self.storePath) and self.storePath.endswith(
            f"_{name}.pkl"
        )

    def getProperty(self, name: str) -> pd.Series:
        """Get a property of the data set.

        Args:
            name (str): Name of the property to get.

        Returns:
            pd.Series: List of values for the property.
        """
        return self.df[name]

    def getProperties(self):
        """Get the properties of the data set.

        Returns: list of properties of the data set.
        """
        return self.df.columns

    def addProperty(self, name: str, data: list):
        """Add a property to the data set.

        Args:
            name (str): Name of the property.
            data (list): List of values for the property.
        """
        if isinstance(data, pd.Series):
            if not self.df.index.equals(data.index):
                logger.info(
                    f"Adding property '{name}' to data set might be introducing 'nan' "
                    "values due to index with pandas series. Make sure the index of "
                    "the data frame and the series match or convert series to list."
                )
        self.df[name] = data

    def removeProperty(self, name: str):
        """Remove a property from the data set.

        Args:
            name (str): Name of the property to remove.
        """
        self.df.drop(columns=[name], inplace=True)

    def getSubset(self, prefix: str):
        """Get a subset of the data set by providing a prefix for the column names or a
        column name directly.

        Args:
            prefix (str): Prefix of the column names to select.
        """
        if self.df.columns.str.startswith(prefix).any():
            return self.df[self.df.columns[self.df.columns.str.startswith(prefix)]]

    def apply(
        self,
        func: Callable,
        func_args: list | None = None,
        func_kwargs: dict | None = None,
        axis: int = 0,
        raw: bool = False,
        result_type: str = "expand",
        subset: list | None = None,
    ):
        """Apply a function to the data frame.

        In addition to the arguments of `pandas.DataFrame.apply`, this method also
        supports parallelization using `multiprocessing.Pool`.

        Args:
            func (Callable): Function to apply to the data frame.
            func_args (list): Positional arguments to pass to the function.
            func_kwargs (dict): Keyword arguments to pass to the function.
            axis (int): Axis along which the function is applied
                (0 for column, 1 for rows).
            raw (bool): Whether to pass the data frame as-is to the function or to pass
                each row/column as a Series to the function.
            result_type (str): Whether to expand the result of the function to columns
                or to leave it as a Series.
            subset (list): list of column names if only a subset of the data should be
                used (reduces memory consumption).
        """
        n_cpus = self.nJobs
        chunk_size = self.chunkSize
        if (
            n_cpus
            and n_cpus > 1
            and not (
                hasattr(func, "noParallelization") and func.noParallelization is True
            )
        ):
            return self.papply(
                func,
                func_args,
                func_kwargs,
                axis,
                raw,
                result_type,
                subset,
                n_cpus,
                chunk_size,
            )
        else:
            df_sub = self.df[subset if subset else self.df.columns]
            return df_sub.apply(
                func,
                raw=raw,
                axis=axis,
                result_type=result_type,
                args=func_args,
                **func_kwargs if func_kwargs else {},
            )

    def papply(
        self,
        func: Callable,
        func_args: list | None = None,
        func_kwargs: dict | None = None,
        axis: int = 0,
        raw: bool = False,
        result_type: str = "expand",
        subset: list | None = None,
        n_cpus: int = 1,
        chunk_size: int = 1000,
    ):
        """Parallelized version of `MoleculeTable.apply`.

        Args:
            func (Callable): Function to apply to the data frame.
            func_args (list): Positional arguments to pass to the function.
            func_kwargs (dict): Keyword arguments to pass to the function.
            axis (int): Axis along which the function is applied
                (0 for column, 1 for rows).
            raw (bool): Whether to pass the data frame as-is to the function or to pass
                each row/column as a Series to the function.
            result_type (str): Whether to expand the result of the function to columns
                or to leave it as a Series.
            subset (list): list of column names if only a subset of the data should be
                used (reduces memory consumption).
            n_cpus (int): Number of CPUs to use for parallelization.
            chunk_size (int): Number of rows to process in each chunk.

        Returns:
            result of applying function to data in chunks
        """
        n_cpus = n_cpus if n_cpus else os.cpu_count()
        df_sub = self.df[subset if subset else self.df.columns]
        data = [df_sub[i : i + chunk_size] for i in range(0, len(df_sub), chunk_size)]
        # size of batches use in the process - more is faster, but uses more memory
        batch_size = n_cpus
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as executor:
            batches = [
                data[i : i + batch_size] for i in range(0, len(data), batch_size)
            ]
            for batch in tqdm(
                batches, desc=f"Parallel apply in progress for {self.name}."
            ):
                wrapped = self.ParallelApplyWrapper(
                    func,
                    func_args=func_args,
                    func_kwargs=func_kwargs,
                    result_type=result_type,
                    axis=axis,
                    raw=raw,
                )
                for result in executor.map(wrapped, batch):
                    results.append(result)

        return pd.concat(results, axis=0)

    def transform(
        self, targets: list[str], transformer: Callable, add_as: list[str] | None = None
    ):
        """Transform the data frame (or its part) using a list of transformers.

        Each transformer is a function that takes the data frame (or a subset of it as
        defined by the `targets` argument) and returns a transformed data frame. The
        transformed data frame can then be added to the original data frame if `add_as`
        is set to a `list` of new column names. If `add_as` is not `None`, the result of
        the application of transformers must have the same number of rows as the
        original data frame.

        Args:
            targets (list[str]): list of column names to transform.
            transformer (Callable): Function that transforms the data in target columns
                to a new representation.
            add_as (list): If `True`, the transformed data is added to the original data
                frame and the
            names in this list are used as column names for the new data.
        """
        ret = self.df[targets]
        ret = transformer(ret)
        if add_as:
            self.df[add_as] = ret
        return ret

    def filter(self, table_filters: list[Callable]):
        """Filter the data frame using a list of filters.

        Each filter is a function that takes the data frame and returns a
        a new data frame with the filtered rows. The new data frame is then used as the
        input for the next filter. The final data frame is saved as the new data frame
        of the `MoleculeTable`."""
        df_filtered = None
        for table_filter in table_filters:
            if len(self.df) == 0:
                logger.warning("Dataframe is empty")
            if table_filter.__class__.__name__ == "CategoryFilter":
                df_filtered = table_filter(self.df)
            elif table_filter.__class__.__name__ == "DuplicateFilter":
                descriptors = self.getDescriptors()
                if len(descriptors.columns) == 0:
                    logger.warning(
                        "Removing duplicates based on descriptors does not \
                                    work if there are no descriptors"
                    )
                else:
                    df_filtered = table_filter(self.df, descriptors)
            if df_filtered is not None:
                self.df = df_filtered.copy()

    def toFile(self, filename: str):
        """Save the metafile and all associated files to a custom location.

        Args:
            filename (str): absolute path to the saved metafile.
        """
        os.makedirs(self.storeDir, exist_ok=True)
        o_dict = json.loads(self.toJSON())
        o_dict["py/state"]["storeDir"] = f"./{self.name}"
        with open(filename, "w") as fh:
            json.dump(o_dict, fh, indent=4)
        return os.path.abspath(filename)

    def save(self):
        """Save the data frame to disk and all associated files.

        Returns:
            str: Path to the saved data frame.
        """
        return self.toFile(f"{self.storePrefix}_meta.json")

    def clearFiles(self):
        """Remove all files associated with this data set from disk."""
        if os.path.exists(self.storeDir):
            shutil.rmtree(self.storeDir)

    def reload(self):
        """Reload the data table from disk."""
        self.df = pd.read_pickle(self.storePath)
        self.indexCols = self.df.index.name.split("~")
        assert all(col in self.df.columns for col in self.indexCols)

    @classmethod
    def fromFile(cls, filename: str) -> "PandasDataTable":
        with open(filename, "r") as f:
            json_f = f.read()
        o_dict = json.loads(json_f)
        o_dict["py/state"]["storeDir"] = os.path.dirname(filename)
        return cls.fromJSON(json.dumps(o_dict))

    def getDF(self):
        """Get the data frame this instance manages.

        Returns:
            pd.DataFrame: The data frame this instance manages.
        """
        return self.df

    def shuffle(self, random_state: Optional[int] = None):
        """Shuffle the internal data frame."""
        self.df = self.df.sample(
            frac=1, random_state=random_state if random_state else self.randomState
        )

    def setRandomState(self, random_state: int):
        """Set the random state for this instance.

        Args:
            random_state (int):
                Random state to use for shuffling and other random operations.
        """
        self.randomState = random_state
