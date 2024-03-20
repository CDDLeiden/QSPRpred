import json
import os
import shutil
from typing import ClassVar, Callable, Optional, Generator, Any

import numpy as np
import pandas as pd

from .base import DataTable
from ...logs import logger
from ...utils.parallel import batched_generator, parallel_jit_generator
from ...utils.serialization import JSONSerializable
from ...utils.stringops import generate_padded_index


class PandasDataTable(DataTable, JSONSerializable):
    """A Pandas DataFrame wrapper class to enable data processing functions on
    QSPRpred data.

    Attributes:
        name (str):
            Name of the data set. You can use this name to load the dataset
            from disk anytime and create a new instance.
        df (pd.DataFrame):
            Pandas dataframe containing the data. You can modify this one directly,
            but note that removing rows, adding rows, or changing the index or other
            automatic properties of the data frame might break the data set. In that
            case, it is recommended to recreate the data set from scratch.
        indexCols (List):
            List of columns to use as index. If `None`, the index
            will be a custom generated ID. Note that if you specify multiple columns
            their values will be joined with a '~' character rather than using the
            default pandas multi-index.
        nJobs (int):
            Number of jobs to use for parallel processing. If set to `None` or `0`,
            all available cores will be set.
        chunkSize (int):
            Size of chunks to use per job in parallel processing. This is automatically
            set to the number of rows in the data frame divided by `nJobs`. However,
            you can also set it manually if you want to use a different chunk size.
            Set to `None` to again use the default value determined by `nJobs`.
        randomState (int):
            Random state to use for all random operations.
        idProp (str):
            Column name to use for automatically generated IDs. Defaults to 'QSPRID'.
            If `indexCols` is set, this will be the names of the columns joined by '~'.
        storeFormat (str):
            Format to use for storing the data frame. Currently only
            'pkl' and 'csv' are supported. Defaults to 'pkl' because it is faster.
            However, 'csv' is more portable and can be opened in other programs.
        parallelGenerator (Callable):
            A parallel generator function to use for parallel processing.
            Defaults to `qsprpred.utils.parallel.parallel_jit_generator`.
            You can replace this with your own parallel generator function if you
            want to use a different parallelization strategy (i.e. utilize
            remote servers instead of local processes).
    """

    _notJSON: ClassVar = [*JSONSerializable._notJSON, "df"]

    def __init__(
        self,
        name: str,
        df: Optional[pd.DataFrame] = None,
        store_dir: str = ".",
        overwrite: bool = False,
        index_cols: Optional[list[str]] = None,
        n_jobs: int = 1,
        chunk_size: int | None = None,
        autoindex_name: str = "QSPRID",
        random_state: int | None = None,
        store_format: str = "pkl",
    ):
        """Initialize a `PandasDataTable` object.
        Args
            name (str):
                Name of the data set. You can use this name to load the dataset
                from disk anytime and create a new instance.
            df (pd.DataFrame):
                Pandas dataframe containing the data. If you provide a
                dataframe for a dataset that already exists on disk, the dataframe from
                disk will override the supplied data frame. Set 'overwrite' to `True` to
                override the data frame on disk.
            store_dir (str):
                Directory to store the dataset files. Defaults to the
                current directory. If it already contains files with the same name,
                the existing data will be loaded.
            overwrite (bool):
                Overwrite existing dataset.
            index_cols (List):
                List of columns to use as index. If None, the index
                will be a custom generated ID.
            n_jobs (int):
                Number of jobs to use for parallel processing. If <= 0,
                all available cores will be used.
            chunk_size (int):
                Size of chunks to use per job in parallel processing. If `None`, the
                chunk size will be set to the number of rows in the data frame divided
                by `nJobs`.
            autoindex_name (str):
                Column name to use for automatically generated IDs.
            random_state (int):
                Random state to use for all random operations
                for reproducibility. If not specified, the state is generated randomly.
                The state is saved upon `save` so if you want to change the state later,
                call the `setRandomState` method after loading.
            store_format (str):
                Format to use for storing the data frame.
                Currently only 'pkl' and 'csv' are supported.
        """
        self.idProp = autoindex_name
        self.storeFormat = store_format
        self.randomState = None
        self.setRandomState(
            random_state or int(np.random.randint(0, 2**31-1, dtype=np.int64))
        )
        self.name = name
        self.indexCols = index_cols
        # paths
        self._storeDir = store_dir.rstrip("/")
        # data frame initialization
        self.df = None
        if df is not None:
            if self._isInStore("df") and not overwrite:
                logger.warning(
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
                    self.generateIndex()
        else:
            if not self._isInStore("df"):
                raise ValueError(
                    f"No data frame found in store for '{self.name}'. Are you sure "
                    "this is the correct dataset? If you are creating a new data set, "
                    "make sure to supply a data frame."
                )
            self.reload()
        assert self.df is not None, "Unknown error in data set creation."
        # parallel settings
        self.nJobs = n_jobs
        self.chunkSize = chunk_size
        self.parallelGenerator = parallel_jit_generator

    def __len__(self) -> int:
        """Get the number of rows in the data frame."""
        return len(self.df)

    def __getstate__(self):
        o_dict = super().__getstate__()
        os.makedirs(self.storeDir, exist_ok=True)
        if self.storeFormat == "csv":
            self.df.to_csv(self.storePath, index=False)
        else:
            self.df.to_pickle(self.storePath)
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.reload()

    @property
    def chunkSize(self) -> int:
        return self._chunkSize

    @chunkSize.setter
    def chunkSize(self, value: int | None):
        self._chunkSize = value if value is not None else int(len(self) / self.nJobs)
        if self._chunkSize < 1:
            self._chunkSize = len(self)
        if self._chunkSize > len(self):
            self._chunkSize = len(self)

    @property
    def nJobs(self):
        return self._nJobs

    @nJobs.setter
    def nJobs(self, value: int | None):
        self._nJobs = value if value is not None and value > 0 else os.cpu_count()
        self.chunkSize = None

    @property
    def baseDir(self) -> str:
        """The base directory of the data set folder."""
        return self._storeDir

    @property
    def storeDir(self):
        """The data set folder containing the data set files after saving."""
        return f"{self.baseDir}/{self.name}"

    @property
    def storePath(self):
        """The path to the main data set file."""
        if self.storeFormat == "csv":
            return f"{self.storePrefix}_df.csv"
        else:
            return f"{self.storePrefix}_df.pkl"

    @property
    def storePrefix(self):
        """The prefix of the data set files."""
        return f"{self.storeDir}/{self.name}"

    @property
    def metaFile(self):
        """The path to the meta file of this data set."""
        return f"{self.storePrefix}_meta.json"

    def setIndex(self, cols: list[str]):
        """Create and index column from several columns of the data set.
        This also resets the `idProp` attribute to be the name of the index columns
        joined by a '~' character. The values of the columns are also joined in the
        same way to create the index. Thus, make sure the values of the columns are
        unique together and can be joined to a string.

        Args:
            cols (list[str]): list of columns to use as index.
        """
        self.indexCols = cols
        self.idProp = "~".join(self.indexCols)
        self.df[self.idProp] = self.df[self.indexCols].apply(
            lambda x: "~".join(map(str, x.tolist())), axis=1
        )
        self.df.set_index(self.idProp, inplace=True, verify_integrity=True, drop=False)
        self.df.drop(
            inplace=True,
            columns=[c for c in self.df.columns if c.startswith("Unnamed")],
        )
        self.df.index.name = self.idProp

    def generateIndex(self, name: str | None = None, prefix: str | None = None):
        """Generate a custom index for the data frame automatically.

        Args:
            name (str | None): name of the resulting index column.
            prefix (str | None): prefix to use for the index column values.
        """
        name = name if name is not None else self.idProp
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
            f"_{name}.{self.storeFormat}"
        )

    def hasProperty(self, name):
        """Check whether a property is present in the data frame.

        Args:
            name (str): Name of the property.

        Returns:
            bool: Whether the property is present.
        """
        return name in self.df.columns

    def getProperty(self, name: str) -> pd.Series:
        """Get property values from the data set.

        Args:
            name (str): Name of the property to get.

        Returns:
            pd.Series: List of values for the property.
        """
        return self.df[name]

    def getProperties(self) -> list[str]:
        """Get names of all properties/variables saved in the data frame (all columns).

        Returns:
            list: list of property names.
        """
        return self.df.columns.tolist()

    def addProperty(self, name: str, data: list):
        """Add a property to the data frame.

        Args:
            name (str): Name of the property.
            data (list): list of property values.
        """
        if isinstance(data, pd.Series):
            if not self.df.index.equals(data.index):
                logger.info(
                    f"Adding property '{name}' to data set might be introducing 'nan' "
                    "values due to index with pandas series. Make sure the index of "
                    "the data frame and the series match or convert series to list."
                )
        self.df[name] = data

    def removeProperty(self, name):
        """Remove a property from the data frame.

        Args:
            name (str): Name of the property to delete.
        """
        del self.df[name]

    def dropEmptyProperties(self, names: list[str]):
        """Drop rows with empty target property value from the data set.

        Args:
            names (list[str]): list of property names to check for empty values.
        """
        self.df.dropna(subset=names, how="all", inplace=True)

    def getSubset(self, prefix: str):
        """Get a subset of the data set by providing a prefix for the column names or a
        column name directly.

        Args:
            prefix (str): Prefix of the column names to select.
        """
        if self.df.columns.str.startswith(prefix).any():
            return self.df[self.df.columns[self.df.columns.str.startswith(prefix)]]

    def iterChunks(
        self,
        include_props: list[str] | None = None,
        as_dict: bool = False,
        chunk_size: int | None = None,
    ) -> Generator[pd.DataFrame | dict, None, None]:
        """Batch a data frame into chunks of the given size.

        Args:
            include_props (list[str]):
                list of properties to include, if `None`, all
                properties are included.
            as_dict (bool):
                If `True`, the generator yields dictionaries instead of data frames.
            chunk_size (int):
                Size of chunks to use per job in parallel processing.
                If `None`, `self.chunkSize` is used.

        Returns:
            Generator[pd.DataFrame, None, None]:
                Generator that yields batches of the data frame as smaller data frames.
        """
        chunk_size = chunk_size if chunk_size is not None else self.chunkSize
        include_props = include_props or self.df.columns
        df_batches = batched_generator(
            self.df[self.idProp] if include_props is not None else self.df.iterrows(),
            chunk_size,
        )
        for ids in df_batches:
            df_batch = self.df.loc[ids]
            ret = {}
            if self.idProp not in include_props:
                include_props.append(self.idProp)
            for prop in include_props:
                ret[prop] = df_batch[prop].tolist()
            yield ret if as_dict else df_batch

    def apply(
        self,
        func: Callable[[dict[str, list[Any]] | pd.DataFrame, ...], Any],
        func_args: list | None = None,
        func_kwargs: dict | None = None,
        on_props: list[str] | None = None,
        as_df: bool = False,
        chunk_size: int | None = None,
        n_jobs: int | None = None,
    ) -> Generator:
        """Apply a function to the data frame. The properties of the data set
        are passed as the first positional argument to the function. This
        will be a dictionary of the form `{'prop1': [...], 'prop2': [...], ...}`.
        If `as_df` is `True`, the properties will be passed as a data frame instead.

        Any additional arguments specified in `func_args` and `func_kwargs` will
        be passed to the function after the properties as positional and keyword
        arguments, respectively.

        If `on_props` is specified, only the properties
        in this list will be passed to the function. If `on_props` is `None`,
        all properties will be passed to the function.

        Args:
            func (Callable):
                Function to apply to the data frame.
            func_args (list):
                Positional arguments to pass to the function.
            func_kwargs (dict):
                Keyword arguments to pass to the function.
            on_props (list[str]):
                list of properties to send to the function as arguments
            as_df (bool):
                If `True`, the function is applied to chunks represented as data frames.
            chunk_size (int):
                Size of chunks to use per job in parallel processing. If `None`,
                the chunk size will be set to `self.chunkSize`. The chunk size will
                always be set to the number of rows in the data frame if `n_jobs`
                or `self.nJobs is 1.
            n_jobs (int):
                Number of jobs to use for parallel processing. If `None`,
                `self.nJobs` is used.

        Returns:
            Generator:
                Generator that yields the results of the function applied to each chunk
                of the data frame as determined by `chunk_size` and `n_jobs`. Each
                item in the generator will be the result of the function
                applied to one chunk of the data set.
        """
        n_jobs = self.nJobs if n_jobs is None else n_jobs
        chunk_size = chunk_size if chunk_size is not None else self.chunkSize
        if n_jobs > 1:
            logger.debug(
                f"Applying function '{func!r}' in parallel on {n_jobs} CPUs, "
                f"using chunk size: {chunk_size}."
            )
            for result in self.parallelGenerator(
                self.iterChunks(
                    include_props=on_props, as_dict=not as_df, chunk_size=chunk_size
                ),
                func,
                n_jobs,
                args=func_args,
                kwargs=func_kwargs,
            ):
                logger.debug(f"Result for chunk returned: {result!r}")
                if not isinstance(result, Exception):
                    yield result
                else:
                    raise result
        else:
            logger.debug(f"Applying function '{func!r}' in serial.")
            for props in self.iterChunks(
                include_props=on_props, as_dict=not as_df, chunk_size=len(self)
            ):
                result = func(props, *func_args, **func_kwargs)
                logger.debug(f"Result for chunk returned: {result!r}")
                yield result

    def transformProperties(self, names: list[str], transformer: Callable):
        """Transform property values using a transformer function.

        Args:
            targets (list[str]): list of column names to transform.
            transformer (Callable): Function that transforms the data in target columns
                to a new representation.
        """
        assert all(
            name in self.df.columns for name in names
        ), "Not all properties in dataframe columns for transformation."
        names_old = [f"{name}_before_transform" for name in names]
        self.df[names_old] = self.df[names]
        self.df[names] = transformer(self.df[names])
        logger.debug(f"Transformed properties in: {names}")
        logger.debug(f"Old values saved in: {names_old}")

    def imputeProperties(self, names: list[str], imputer: Callable):
        """Impute missing property values.

        Args:
            names (list):
                List of property names to impute.
            imputer (Callable):
                imputer object implementing the `fit_transform`
                 method from scikit-learn API.
        """
        assert hasattr(imputer, "fit_transform"), (
            "Imputer object must implement the `fit_transform` "
            "method from scikit-learn API."
        )
        assert all(
            name in self.df.columns for name in names
        ), "Not all properties in dataframe columns for imputation."
        names_old = [f"{name}_before_impute" for name in names]
        self.df[names_old] = self.df[names]
        self.df[names] = imputer.fit_transform(self.df[names])
        logger.debug(f"Imputed missing values for properties: {names}")
        logger.debug(f"Old values saved in: {names_old}")

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
        if self.storeFormat == "csv":
            self.df = pd.read_csv(self.storePath)
            self.df.set_index(self.indexCols, inplace=True, drop=False)
        else:
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
