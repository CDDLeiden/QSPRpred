"""This module contains the QSPRDataset that holds and prepares data for modelling."""

import concurrent
import copy
import json
import multiprocessing
import os
import pickle
import warnings
from multiprocessing import Pool
from typing import Callable, List, Literal, Union

import numpy as np
import pandas as pd
from qsprpred.data.interfaces import MoleculeDataSet, datasplit, DataSet
from qsprpred.data.utils.descriptorcalculator import DescriptorsCalculator, MoleculeDescriptorsCalculator, \
    ProteinDescriptorCalculator, CustomDescriptorsCalculator
from qsprpred.data.utils.feature_standardization import (
    SKLearnStandardizer,
    apply_feature_standardizer,
)
from qsprpred.data.utils.folds import Folds
from qsprpred.data.utils.scaffolds import Scaffold
from qsprpred.data.utils.smiles_standardization import (
    chembl_smi_standardizer,
    old_standardize_sanitize,
    check_smiles_valid,
)
from qsprpred.models.tasks import TargetTasks
from qsprpred.logs import logger
from rdkit.Chem import PandasTools
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from qsprpred.utils.inspect import import_class


class PandasDataSet(DataSet):

    class ParallelApplyWrapper:
        """A wrapper class to parallelize pandas apply functions."""

        def __init__(self, func, func_args=None, func_kwargs=None, axis=0, raw=False, result_type='expand'):
            """Initialize the instance with pandas parameters to apply to chunks of data.

            See `pandas.DataFrame.apply` for more information.

            Args:
                func: function to apply
                func_args: arguments to pass to func
                func_kwargs: keyword arguments to pass to func
                axis: axis to apply func on (0 for columns, 1 for rows)
                raw: whether to pass Series object to func or raw array
                result_type: whether to expand/ignore the results. See pandas.DataFrame.apply for more info.

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
                data: chunk of data to apply function to

            Returns:
                result of applying function to chunk of data
            """
            return data.apply(self.func, raw=self.raw, axis=self.axis, result_type=self.result_type,
                              args=self.args, **self.kwargs if self.kwargs else {})

    def __init__(self, name, df: pd.DataFrame = None, store_dir=".", overwrite=False, index_cols=None, n_jobs=1, chunk_size=1000, id_prefix='QSPRID'):
        self.name = name
        self.indexCols = index_cols

        # parallel settings
        self.nJobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.chunkSize = chunk_size

        # paths
        self.storeDir = store_dir.rstrip("/")
        self.storePrefix = f"{self.storeDir}/{self.name}"
        if not os.path.exists(self.storeDir):
            raise FileNotFoundError(f"Directory '{self.storeDir}' does not exist.")
        self.storePath = f'{self.storePrefix}_df.pkl'

        # data frame initialization
        self.df = None
        if df is not None:
            if self._isInStore('df') and not overwrite:
                warnings.warn(
                    'Existing data set found, but also found a data frame in store. Refusing to overwrite data. If you want to overwrite data in store, set overwrite=True.')
                self.reload()
            else:
                self.clearFiles()
                self.df = df
                if index_cols is not None:
                    self.setIndex(index_cols)
                else:
                    df[id_prefix] = [f"{self.name}_{i}" for i in range(len(self.df))]
                    self.setIndex([id_prefix])
        else:
            if not self._isInStore('df'):
                raise ValueError(
                    f"No data frame found in store for '{self.name}'. Are you sure this is the correct dataset? If you are creating a new data set, make sure to supply a data frame.")
            self.reload()
        assert self.df is not None, "Unknown error in data set creation."

    def __len__(self):
        return len(self.df)

    def setIndex(self, cols: List[str]):
        """
        Set the index of the data frame.

        Args:
            cols (List[str]): List of columns to use as index.
        """
        self.df.set_index(cols, inplace=True, verify_integrity=True, drop=False)
        self.df.drop(inplace=True, columns=[c for c in self.df.columns if c.startswith('Unnamed')])
        self.df.index.name = "~".join(cols)
        self.indexCols = cols

    def _isInStore(self, name):
        """Check if a pickled file with the given suffix exists.

        Args:
            name (str): Suffix of the file to check.

        Returns:
            bool: `True` if the file exists, `False` otherwise.
        """
        return os.path.exists(self.storePath) and self.storePath.endswith(f'_{name}.pkl')

    def getProperties(self):
        return self.df.columns

    def addProperty(self, name, data):
        self.df[name] = data

    def removeProperty(self, name):
        self.df.drop(name, axis=1, inplace=True)

    def getSubset(self, prefix: str):
        """Get a subset of the data set by providing a prefix for the column names or a column name directly.

        Args:
            prefix (str): Prefix of the column names to select.
        """
        if self.df.columns.str.startswith(prefix).any():
            return self.df[self.df.columns[self.df.columns.str.startswith(prefix)]]

    def apply(self, func, func_args=None, func_kwargs=None, axis=0, raw=False,
              result_type='expand', subset=None):
        """Apply a function to the data frame.

        In addition to the arguments of `pandas.DataFrame.apply`, this method also supports parallelization using `multiprocessing.Pool`.

        Args:
            func (callable): Function to apply to the data frame.
            func_args (list): Positional arguments to pass to the function.
            func_kwargs (dict): Keyword arguments to pass to the function.
            axis (int): Axis along which the function is applied (0 for column, 1 for rows).
            raw (bool): Whether to pass the data frame as-is to the function or to pass each row/column as a Series to the function.
            result_type (str): Whether to expand the result of the function to columns or to leave it as a Series.
            subset (list): List of column names if only a subset of the data should be used (reduces memory consumption).
        """
        n_cpus = self.nJobs
        chunk_size = self.chunkSize
        if n_cpus and n_cpus > 1:
            return self.papply(func, func_args, func_kwargs, axis, raw, result_type, subset, n_cpus, chunk_size)
        else:
            df_sub = self.df[subset if subset else self.df.columns]
            return df_sub.apply(func, raw=raw, axis=axis, result_type=result_type,
                                args=func_args, **func_kwargs if func_kwargs else {})

    def papply(self, func, func_args=None, func_kwargs=None, axis=0, raw=False,
               result_type='expand', subset=None, n_cpus=1, chunk_size=1000):
        """Parallelized version of `MoleculeTable.apply`.

        Args:
            func (callable): Function to apply to the data frame.
            func_args (list): Positional arguments to pass to the function.
            func_kwargs (dict): Keyword arguments to pass to the function.
            axis (int): Axis along which the function is applied (0 for column, 1 for rows).
            raw (bool): Whether to pass the data frame as-is to the function or to pass each row/column as a Series to the function.
            result_type (str): Whether to expand the result of the function to columns or to leave it as a Series.
            subset (list): List of column names if only a subset of the data should be used (reduces memory consumption).
            n_cpus (int): Number of CPUs to use for parallelization.
            chunk_size (int): Number of rows to process in each chunk.
            n_cpus (int): Number of CPUs to use for parallelization.
            chunk_size (int): Number of rows to process in each chunk.
        """
        n_cpus = n_cpus if n_cpus else os.cpu_count()
        df_sub = self.df[subset if subset else self.df.columns]
        data = [df_sub[i: i + chunk_size] for i in range(0, len(df_sub), chunk_size)]
        batch_size = n_cpus  # how many batches to prefetch into the process (more is faster, but uses more memory)
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as executor:
            batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
            for batch in tqdm(batches,
                              desc=f"Parallel apply in progress for {self.name}."):
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

    def transform(self, targets, transformer, addAs=None):
        """Transform the data frame (or its part) using a list of transformers.

        Each transformer is a function that takes the data frame (or a subset of it as defined by the `targets` argument) and returns a transformed data frame. The transformed
        data frame can then be added to the original data frame if `addAs` is set to a `list` of new column names. If
        `addAs` is not `None`, the result of the application of transformers must have the same number of rows as the
        original data frame.

        Args:
            targets (list): List of column names to transform.
            transformer (callable): Function that transforms the data in target columns to a new representation.
            addAs (list): If `True`, the transformed data is added to the original data frame and the
            names in this list are used as column names for the new data.
        """
        ret = self.df[targets]
        ret = transformer(ret)

        if addAs:
            self.df[addAs] = ret
        return ret

    def filter(self, table_filters: List[Callable]):
        """Filter the data frame using a list of filters.

        Each filter is a function that takes the data frame and returns a
        a new data frame with the filtered rows. The new data frame is then used as the input for the next filter. The
        final data frame is saved as the new data frame of the `MoleculeTable`.
        """
        df_filtered = None
        for table_filter in table_filters:
            df_filtered = table_filter(self.df)

        if df_filtered is not None:
            self.df = df_filtered.copy()

    def save(self):
        """Save the data frame to disk and all associated files.

        Returns:
            str: Path to the saved data frame.
        """
        # save data frame
        self.df.to_pickle(self.storePath)

        return self.storePath

    def clearFiles(self):
        """Remove all files associated with this data set from disk."""
        for file in [f for f in os.listdir(self.storeDir) if f.endswith('.pkl') or f.endswith('.json')]:
            if file.startswith(self.name):
                os.remove(f'{self.storeDir}/{file}')

    def reload(self):
        """Reload the data table from disk."""
        self.df = pd.read_pickle(self.storePath)
        self.indexCols = self.df.index.name.split('~')
        assert all([col in self.df.columns for col in self.indexCols])

    @staticmethod
    def fromFile(filename, *args, **kwargs) -> 'PandasDataSet':
        """Create a `MoleculeTable` instance from by providing a direct path to the pickled data frame in storage."""
        store_dir = os.path.dirname(filename)
        name = os.path.basename(filename).split('.')[0]
        return PandasDataSet(name=name, store_dir=store_dir, *args, **kwargs)

    def getDF(self):
        """Get the data frame this instance manages.

        Returns:
            pd.DataFrame: The data frame this instance manages.
        """
        return self.df

    def shuffle(self, random_state=None):
        """Shuffle the internal data frame."""
        self.df = self.df.sample(frac=1, random_state=random_state)


class DescriptorTable(PandasDataSet):

    def __init__(self, calculator, name_prefix : str,  df: pd.DataFrame = None, store_dir=".", overwrite=False, key_cols=None, n_jobs=1, chunk_size=1000):
        """Initialize a `DescriptorTable` object."""
        super().__init__(f"{name_prefix}_{calculator.getPrefix()}", df, store_dir, overwrite, key_cols, n_jobs, chunk_size)
        self.calculator = calculator

    @property
    def prefix(self):
        return self.name

    @property
    def keyCols(self):
        return self.indexCols

    def getDescriptors(self):
        return self.df[self.getDescriptorNames()]

    def getDescriptorNames(self):
        return [x for x in self.df.columns if x.startswith(f"{self.prefix}_")]

    def fillMissing(self, fill_value, names):
        columns = names if names else self.getDescriptorNames()
        self.df[columns] = self.df[columns].fillna(fill_value)

class MoleculeTable(PandasDataSet, MoleculeDataSet):
    """Class that holds and prepares molecule data for modelling and other analyses."""

    def __init__(
            self,
            name: str,
            df: pd.DataFrame = None,
            smilescol: str = "SMILES",
            proteincol: str = None,
            proteinseqprovider: Callable = None,
            add_rdkit: bool = False,
            store_dir: str = '.',
            overwrite: bool = False,
            n_jobs: int = 1,
            chunk_size: int = 50,
            drop_invalids: bool = True,
            index_cols: List[str] = None,
    ):
        """Initialize a `MoleculeTable` object.

        This object wraps a pandas dataframe and provides short-hand methods to prepare molecule data for modelling and analysis.

        Args:
            name (str): Name of the dataset. You can use this name to load the dataset from disk anytime and create a new instance.
            df (pd.DataFrame): Pandas dataframe containing the data. If you provide a dataframe for a dataset that already exists on disk,
            the dataframe from disk will override the supplied data frame. Set 'overwrite' to `True` to override the data frame on disk.
            smilescol (str): Name of the column containing the SMILES sequences of molecules.
            proteincol (str): name of column in df containing the protein target identifier (usually a UniProt ID). Setting this value enables some PCM features (i.e. allows for calculation of protein descriptors).
            proteinseqprovider (Callable): Function that takes a protein ID and returns the protein sequence. If None, no sequences are passed to protein descriptor sets.
            add_rdkit (bool): Add RDKit molecule instances to the dataframe. WARNING: This can take a lot of memory.
            store_dir (str): Directory to store the dataset files. Defaults to the current directory. If it already contains files with the same name, the existing data will be loaded.
            overwrite (bool): Overwrite existing dataset.
            n_jobs (int): Number of jobs to use for parallel processing. If <= 0, all available cores will be used.
            chunk_size (int): Size of chunks to use per job in parallel processing.
            drop_invalids (bool): Drop invalid molecules from the data frame.
            index_cols (List[str]): List of columns to use as index. If None, the index will be a custom generated ID.
        """
        self.descriptorCalculators = [] # holds all descriptor calculators
        self.descriptors = [] # holds descriptor tables for each calculator
        self.descriptorCalculatorsPathPrefix = None
        super().__init__(name, df, store_dir, overwrite, index_cols, n_jobs, chunk_size)
        if not self.descriptorCalculatorsPathPrefix:
            self.descriptorCalculatorsPathPrefix = f"{self.storePrefix}_descriptor_calculator"

        # settings
        self.smilescol = smilescol
        self.proteincol = proteincol
        self.proteinseqprovider = proteinseqprovider
        self.includesRdkit = add_rdkit

        # add rdkit molecules if requested
        if self.includesRdkit and 'RDMol' not in self.df.columns:
            PandasTools.AddMoleculeColumnToFrame(
                self.df, smilesCol=self.smilescol, molCol='RDMol', includeFingerprints=False)

        # drop invalid columns
        if drop_invalids:
            self.dropInvalids()

    def save(self):
        super().save()

        # save descriptor calculator
        for idx, calc in enumerate(self.descriptorCalculators):
            calc.toFile(f"{self.descriptorCalculatorsPathPrefix}_{calc}.json")
            self.descriptors[idx].save()

        return self.storePath

    def reload(self):
        super().reload()

        # load descriptor calculators and tables
        self.descriptorCalculatorsPathPrefix = f"{self.storePrefix}_descriptor_calculator"
        files = [f for f in os.listdir(self.storeDir) if
                 f.endswith('.json') and f.startswith(os.path.basename(self.descriptorCalculatorsPathPrefix))]
        for file in files:
            path = f"{self.storeDir}/{file}"
            if os.path.exists(path):
                data = json.load(open(path, "r", encoding="utf-8"))
                if not "calculator" in data:
                    calc_cls = "qsprpred.data.utils.descriptorcalculator.MoleculeDescriptorsCalculator"
                else:
                    calc_cls = data["calculator"]
                calc_cls = import_class(calc_cls)
                calc = calc_cls.fromFile(path)
                self.descriptorCalculators.append(calc)
                self.descriptors.append(self.loadDescriptorsTable(calc))

    def loadDescriptorsTable(self, calc):
        """Load a descriptor table from disk.

        Args:
            calc (DescriptorsCalculator): Descriptor

        Returns:
            DescriptorTable: Descriptor table.
        """

        return DescriptorTable(calc, name_prefix=self.name, store_dir=self.storeDir, n_jobs=self.nJobs, chunk_size=self.chunkSize)

    @staticmethod
    def fromFile(filename, *args, **kwargs) -> 'MoleculeTable':
        """Create a `MoleculeTable` instance from by providing a direct path to the pickled data frame in storage."""
        store_dir = os.path.dirname(filename)
        name = os.path.basename(filename).split('.')[0]
        return MoleculeTable(name=name, store_dir=store_dir, *args, **kwargs)

    @staticmethod
    def fromSMILES(name, smiles, *args, **kwargs):
        """Create a `MoleculeTable` instance from a list of SMILES sequences.

        Args:
            name (str): Name of the data set.
            smiles (list): List of SMILES sequences.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: Additional keyword arguments to pass to the `MoleculeTable` constructor.
        """
        smilescol = "SMILES"
        df = pd.DataFrame({smilescol: smiles})
        return MoleculeTable(name, df, *args, smilescol=smilescol, **kwargs)

    @staticmethod
    def fromTableFile(name, filename, sep="\t", *args, **kwargs):
        """Create a `MoleculeTable` instance from a file containing a table of molecules (i.e. a CSV file).

        Args:
            name (str): Name of the data set.
            filename (str): Path to the file containing the table.
            sep (str): Separator used in the file for different columns.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: Additional keyword arguments to pass to the `MoleculeTable` constructor.
        """
        return MoleculeTable(name, pd.read_table(filename, sep=sep), *args, **kwargs)

    @staticmethod
    def fromSDF(name, filename, smiles_prop, *args, **kwargs):
        """Create a `MoleculeTable` instance from an SDF file.

        Args:
            name (str): Name of the data set.
            filename (str): Path to the SDF file.
            smiles_prop (str): Name of the property in the SDF file containing the SMILES sequence.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: Additional keyword arguments to pass to the `MoleculeTable` constructor.
        """
        return MoleculeTable(name, PandasTools.LoadSDF(filename, molColName="RDMol"), smilescol=smiles_prop, *args, **
                             kwargs)  # FIXME: in this case the RDKit molecule is always added, which can in most cases is an unnecessary overhead

    def checkMols(self, throw=True):
        """
        Returns a boolean array indicating whether each molecule is valid or not. If `throw` is `True`, an exception is thrown if any molecule is invalid.

        Args:
            throw (bool): Whether to throw an exception if any molecule is invalid.

        Returns:
            mask (pd.Series): Boolean series indicating whether each molecule is valid or not.
        """

        if self.nJobs > 1:
            with multiprocessing.Pool(self.nJobs) as pool:
                mask = pool.starmap(check_smiles_valid, zip(self.df[self.smilescol], [throw] * len(self.df)))
            return pd.Series(mask, index=self.df.index)
        else:
            return self.df[self.smilescol].apply(check_smiles_valid, throw=throw)

    def dropDescriptors(self, calculator: DescriptorsCalculator):
        to_remove = []
        for idx, calc in enumerate(self.descriptorCalculators):
            if calc.getPrefix() == calculator.getPrefix():
                logger.debug(f"Removing existing descriptors with prefix '{calculator.getPrefix()}'")
                to_remove.append(idx)

        for idx in reversed(to_remove):
            self.descriptors[idx].clearFiles()
            self.descriptors.pop(idx)
            self.descriptorCalculators.pop(idx)

    def addCustomDescriptors(self, calculator: CustomDescriptorsCalculator, recalculate=False):
        """
        Add custom descriptors to the data frame using a `CustomDescriptorsCalculator` object.

        Args:
            calculator (CustomDescriptorsCalculator): CustomDescriptorsCalculator object to use for descriptor calculation.
            recalculate (bool): Whether to recalculate descriptors even if they are already present in the data frame.
                If `False`, existing descriptors are kept and no calculation takes place.
        """
        if recalculate:
            self.dropDescriptors(calculator)
        elif self.getDescriptorNames(prefix=calculator.getPrefix()):
            logger.warning(f"Custom molecular descriptors already exist in {self.name}. Use `recalculate=True` to overwrite them.")
            return

        descriptors = calculator(self.df.index)
        descriptors[self.indexCols] = self.df[self.indexCols]

        self.attachDescriptors(calculator, descriptors, self.indexCols)

    def attachDescriptors(self, calculator: DescriptorsCalculator, descriptors: pd.DataFrame, index_cols):
        if not self.descriptorCalculators:
            self.descriptorCalculators = []
        self.descriptorCalculators.append(calculator)
        self.descriptors.append(DescriptorTable(calculator, self.name, descriptors, store_dir=self.storeDir, n_jobs=self.nJobs, overwrite=True, key_cols=index_cols, chunk_size=self.chunkSize))

    def addDescriptors(self, calculator: MoleculeDescriptorsCalculator, recalculate=False, fail_on_invalid=True):
        """
        Add descriptors to the data frame using a `DescriptorsCalculator` object.

        Args:
            calculator (MoleculeDescriptorsCalculator): DescriptorsCalculator object to use for descriptor calculation.
            recalculate (bool): Whether to recalculate descriptors even if they are already present in the data frame.
                If `False`, existing descriptors are kept and no calculation takes place.
            fail_on_invalid (bool): Whether to throw an exception if any molecule is invalid.
        """
        if recalculate:
            self.dropDescriptors(calculator)
        elif self.getDescriptorNames(prefix=calculator.getPrefix()):
            logger.warning(f"Molecular descriptors already exist in {self.name}. Use `recalculate=True` to overwrite them.")
            return

        if fail_on_invalid:
            try:
                self.checkMols(throw=True)
            except Exception as exp:
                logger.error(f"Cannot add descriptors to {self.name} because it contains one or more invalid molecules. Remove the invalid molecules from your data or try to standardize the data set first with 'standardizeSmiles()'. You can also pass 'fail_on_invalid=False' to remove this exception, but the calculation might not be successful or correct. See the following list of invalid molecule SMILES for more information:")
                logger.error(self.df[~self.checkMols(throw=False)][self.smilescol].to_numpy())
                raise exp

        # get the data frame with the descriptors
        descriptors = self.apply(
            calculator,
            axis=0,
            subset=[
                self.smilescol],
            result_type='reduce'
        )
        descriptors = descriptors.to_list()
        descriptors = pd.concat(descriptors, axis=0)
        descriptors.index = self.df.index
        descriptors[self.indexCols] = self.df[self.indexCols]

        # add the descriptors to the descriptor list
        self.attachDescriptors(calculator, descriptors, self.indexCols)

    def getDescriptors(self):
        """Get the subset of the data frame that contains only descriptors.

        Returns:
            pd.DataFrame: Data frame containing only descriptors.
        """

        join_cols = set()
        for descriptors in self.descriptors:
            join_cols.update(set(descriptors.keyCols))
        join_cols = list(join_cols)
        ret = self.df[join_cols].copy()
        ret.reset_index(drop=True, inplace=True)
        for descriptors in self.descriptors:
            df_descriptors = descriptors.getDF()
            ret = ret.merge(df_descriptors, left_on=descriptors.keyCols, how='left', right_index=True, suffixes=('_left', '_right'))
            for x in descriptors.keyCols:
                ret.drop(columns=[f"{x}_right"], inplace=True)
                ret.rename(columns={f"{x}_left": x}, inplace=True)
        ret.set_index(self.df.index, inplace=True)
        ret.drop(columns=join_cols, inplace=True)
        return ret

    def getDescriptorNames(self, prefix=None):
        """Get the names of the descriptors in the data frame.

        Returns:
            list: List of descriptor names.
        """
        if not prefix:
            prefixes = [x.getPrefix() for x in self.descriptorCalculators] if self.descriptorCalculators else []
        else:
            prefixes = [prefix]

        names = []
        for x in self.descriptors:
            if x.prefix in prefixes:
                names.extend(x.getDescriptorNames())
        return names


    @property
    def hasDescriptors(self):
        """Check whether the data frame contains descriptors."""
        return len(self.getDescriptorNames()) > 0

    def getProperties(self):
        """Get names of all properties/variables saved in the data frame (all columns).

        Returns:
            list: List of property names.
        """
        return self.df.columns

    def hasProperty(self, name):
        """Check whether a property is present in the data frame.

        Args:
            name (str): Name of the property.

        Returns:
            bool: Whether the property is present.
        """
        return name in self.df.columns

    def addProperty(self, name, data):
        """Add a property to the data frame.

        Args:
            name (str): Name of the property.
            data (list): List of property values.
        """
        self.df[name] = data

    def removeProperty(self, name):
        """Remove a property from the data frame.

        Args:
            name (str): Name of the property to delete.
        """
        del self.df[name]

    @staticmethod
    def _scaffold_calculator(mol, scaffold: Scaffold):
        """Just a helper function to calculate the scaffold of a molecule more easily."""
        return scaffold(mol[0])

    def addScaffolds(self, scaffolds: List[Scaffold], add_rdkit_scaffold=False, recalculate=False):
        """Add scaffolds to the data frame.

        A new column is created that contains the SMILES of the corresponding scaffold.
        If `add_rdkit_scaffold` is set to `True`, a new column is created that contains the RDKit scaffold of the
        corresponding molecule.

        Args:
            scaffolds (list): List of `Scaffold` calculators.
            add_rdkit_scaffold (bool): Whether to add the RDKit scaffold of the molecule as a new column.
            recalculate (bool): Whether to recalculate scaffolds even if they are already present in the data frame.
        """
        for scaffold in scaffolds:
            if not recalculate and f"Scaffold_{scaffold}" in self.df.columns:
                continue

            self.df[f"Scaffold_{scaffold}"] = self.apply(
                self._scaffold_calculator, func_args=(
                    scaffold,), subset=[
                    self.smilescol], axis=1, raw=True)
            if add_rdkit_scaffold:
                PandasTools.AddMoleculeColumnToFrame(self.df, smilesCol=f"Scaffold_{scaffold}",
                                                     molCol=f"Scaffold_{scaffold}_RDMol")

    def getScaffoldNames(self, include_mols=False):
        """Get the names of the scaffolds in the data frame.

        Args:
            include_mols (bool): Whether to include the RDKit scaffold columns as well.
        """
        return [col for col in self.df.columns if
                col.startswith("Scaffold_") and (include_mols or not col.endswith("_RDMol"))]

    def getScaffolds(self, includeMols=False):
        """Get the subset of the data frame that contains only scaffolds.

        Args:
            includeMols (bool): Whether to include the RDKit scaffold columns as well.
        """
        if includeMols:
            return self.df[[col for col in self.df.columns if col.startswith("Scaffold_")]]
        else:
            return self.df[self.getScaffoldNames()]

    @property
    def hasScaffolds(self):
        """Check whether the data frame contains scaffolds.

        Returns:
            bool: Whether the data frame contains scaffolds.
        """
        return len(self.getScaffoldNames()) > 0

    def createScaffoldGroups(self, mols_per_group=10):
        """Create scaffold groups.

        A scaffold group is a list of molecules that share the same scaffold. New columns are
        created that contain the scaffold group ID and the scaffold group size.

        Args:
            mols_per_group (int): Number of molecules per scaffold group.
        """
        scaffolds = self.getScaffolds(includeMols=False)
        for scaffold in scaffolds.columns:
            counts = pd.value_counts(self.df[scaffold])
            mask = counts.lt(mols_per_group)
            name = f'ScaffoldGroup_{scaffold}_{mols_per_group}'
            if name not in self.df.columns:
                self.df[name] = np.where(self.df[scaffold].isin(counts[mask].index), 'Other',
                                         self.df[scaffold])

    def getScaffoldGroups(self, scaffold_name: str, mol_per_group: int = 10):
        """Get the scaffold groups for a given combination of scaffold and number of molecules per scaffold group.

        Args:
            scaffold_name (str): Name of the scaffold.
            mol_per_group (int): Number of molecules per scaffold group.

        Returns:
            list: List of scaffold groups.
        """
        return self.df[
            self.df.columns[self.df.columns.str.startswith(f"ScaffoldGroup_{scaffold_name}_{mol_per_group}")][0]]

    @property
    def hasScaffoldGroups(self):
        """Check whether the data frame contains scaffold groups.

        Returns:
            bool: Whether the data frame contains scaffold groups.
        """
        return len([col for col in self.df.columns if col.startswith("ScaffoldGroup_")]) > 0

    def addProteinDescriptors(self, calculator: ProteinDescriptorCalculator, recalculate=False):
        """
        Add protein descriptors to the data frame.

        Args:
            calculator (ProteinDescriptorCalculator): DescriptorsCalculator to use.
            recalculate (bool): Whether to recalculate descriptors even if they are already present in the data frame.
        """

        if recalculate:
            self.dropDescriptors(calculator)
        elif self.getDescriptorNames(prefix=calculator.getPrefix()):
            logger.warning(f"Protein descriptors already exist in {self.name}. Use `recalculate=True` to overwrite them.")
            return

        if not self.proteincol:
            raise ValueError("Protein column not set. Cannot calculate protein descriptors.")

        sequences, info = self.proteinseqprovider(self.df[self.proteincol].unique().tolist()) if self.proteinseqprovider else (None, dict())
        descriptors = calculator(self.df[self.proteincol].unique(), sequences, **info)
        descriptors[self.proteincol] = descriptors.index.values

        # add the descriptors to the descriptor list
        self.attachDescriptors(calculator, descriptors, [self.proteincol])

    def standardizeSmiles(self, smiles_standardizer, drop_invalid=True):
        """Apply smiles_standardizer to the compounds in parallel

        Args:
            drop_invalid (bool): whether to drop invalid SMILES from the data set. Defaults to `True`. If `False`, invalid SMILES will be retained in their original form.
            smiles_standardizer (Union[str, callable]): either `None` to skip the standardization,
                `chembl`, `old`, or a partial function that reads and standardizes smiles.

        Raises:
            ValueError: when smiles_standardizer is not a callable or one of the predefined strings.
        """        
        std_jobs = self.nJobs
        if smiles_standardizer is None:
            return
        if callable(smiles_standardizer):
            try: # Prevents weird error if the user inputs a lambda function
                pickle.dumps(smiles_standardizer)
            except pickle.PicklingError:
                logger.warning("Standardizer is not pickleable. Will set n_jobs to 1")
                std_jobs = 1
            std_func = smiles_standardizer
        elif smiles_standardizer.lower() == 'chembl':
            std_func = chembl_smi_standardizer
        elif smiles_standardizer.lower() == 'old':
            std_func = old_standardize_sanitize
        else:
            raise ValueError("Standardizer must be either 'chembl', or a callable")
        
        if std_jobs == 1:
            std_smi = [std_func(smi) for smi in self.df[self.smilescol].values]
        else:
            with Pool(std_jobs) as pool:
                std_smi = pool.map(std_func, self.df[self.smilescol].values)
        self.df[self.smilescol] = std_smi
        if drop_invalid:
            self.dropInvalids()

    def dropInvalids(self):
        """
        Drops invalid molecules from the data set.

        Returns:
            mask (pd.Series): Boolean mask of invalid molecules in the original data set.
        """
        invalid_mask = self.checkMols(throw=False)
        self.df.drop(self.df.index[~invalid_mask], inplace=True)
        invalids = (~invalid_mask).sum()
        if invalids > 0:
            logger.warning(f"Dropped {invalids} invalid molecules from the data set.")

        return ~invalid_mask


class TargetProperty:
    """Target property for QSPRmodelling class.

    Attributes:
        name (str): name of the target property
        task (Literal[TargetTasks.REGRESSION, TargetTasks.SINGLECLASS, TargetTasks.MULTICLASS]): task type for the target property
        th (int): threshold for the target property, only used for classification tasks
        nClasses (int): number of classes for the target property, only used for classification tasks
        transformer (Callable): function to transform the target property
    """

    def __init__(
            self, name: str,
            task: Literal[TargetTasks.REGRESSION, TargetTasks.SINGLECLASS, TargetTasks.MULTICLASS],
            originalName: str = None,
            th: Union[List[float], str] = None,
            nClasses: int = None,
            transformer: Callable = None):
        """Initialize a TargetProperty object.

        Args:
            name (str): name of the target property
            task (Literal[TargetTasks.REGRESSION, TargetTasks.SINGLECLASS, TargetTasks.MULTICLASS]): task type for the target property
            originalName (str): original name of the target property, if not specified, the name is used
            th (Union[List[float], str]): threshold for the target property, only used for classification tasks
            nClasses (int): number of classes for the target property (only used if th is precomputed, otherwise it is inferred)
            transformer (Callable): function to transform the target property
        """
        self.name = name
        self.originalName = originalName if originalName is not None else name
        self.task = task
        if task.isClassification():
            assert th is not None, f"Threshold not specified for classification task {name}"
            self.th = th
            if isinstance(th, str) and th == "precomputed":
                self.nClasses = nClasses
        self.transformer = transformer

    @property
    def th(self):
        """Set the threshold for the target property.

        Args:
            th (Union[List[int], str]): threshold for the target property
        """
        return self._th

    @th.setter
    def th(self, th: Union[List[float], str]):
        """Set the threshold for the target property and the number of classes if th is not precomputed."""
        assert self.task.isClassification(), f"Threshold can only be set for classification tasks"
        self._th = th
        if isinstance(th, str):
            assert th == "precomputed", f"Invalid threshold {th}"
        else:
            self._nClasses = len(self.th) - 1 if len(self.th) > 1 else 2

    @th.deleter
    def th(self):
        """Delete the threshold for the target property and the number of classes."""
        del self._th
        del self._nClasses

    @property
    def nClasses(self):
        """Get the number of classes for the target property."""
        return self._nClasses

    @nClasses.setter
    def nClasses(self, nClasses: int):
        """Set the number of classes for the target property if th is precomputed.

        Args:
            nClasses (int): number of classes
        """
        assert self.th == "precomputed", f"Number of classes can only be set if threshold is precomputed"
        self._nClasses = nClasses

    def __repr__(self):
        """Representation of the TargetProperty object."""
        if self.task.isClassification() and self.th is not None:
            return f"TargetProperty(name={self.name}, task={self.task}, th={self.th})"
        else:
            return f"TargetProperty(name={self.name}, task={self.task})"

    def __str__(self):
        """Return string identifier of the TargetProperty object."""
        return self.name

    @classmethod
    def fromDict(cls, d: dict):
        """Create a TargetProperty object from a dictionary.

        Args:
            d (dict): dictionary containing the target property information
            task_from_str (bool): whether to convert the task from a string

        Returns:
            TargetProperty: TargetProperty object
        """
        if isinstance(d["task"], str):
            return TargetProperty(**{k: TargetTasks[v] if k == "task" else v for k, v in d.items()})
        else:
            return TargetProperty(**d)

    @classmethod
    def fromList(cls, l: List[dict], task_from_str: bool = False):
        """Create a list of TargetProperty objects from a list of dictionaries.

        Args:
            l (list): list of dictionaries containing the target property information\
            task_from_str (bool): whether to convert the task from a string

        Returns:
            List[TargetProperty]: list of TargetProperty objects
        """
        if task_from_str:
            return [TargetProperty(**{k: TargetTasks[v] if k == "task" else v for k, v in d.items()}) for d in l]
        else:
            return [TargetProperty(**d) for d in l]

    @staticmethod
    def toList(l: list, task_as_str: bool = False, drop_transformer: bool = True):
        """Convert a list of TargetProperty objects to a list of dictionaries.

        Args:
            l (list): list of TargetProperty objects
            task_as_str (bool): whether to convert the task to a string

        Returns:
            List[dict]: list of dictionaries containing the target property information
        """
        target_props = []
        for target_prop in l:
            target_props.append({
                "name": target_prop.name,
                "task": target_prop.task.name if task_as_str else target_prop.task,
                "originalName": target_prop.originalName,
            })
            if target_prop.task.isClassification():
                target_props[-1].update({"th": target_prop.th, "nClasses": target_prop.nClasses})
            if not drop_transformer:
                target_props[-1].update({"transformer": target_prop.transformer})

        return target_props

    @staticmethod
    def selectFromList(l: list, names: list, original_names: bool = False):
        """Select a subset of TargetProperty objects from a list of TargetProperty objects.

        Args:
            l (list): list of TargetProperty objects
            names (list): list of names of the target properties to be selected
            original_names (bool): whether to use the original names of the target properties

        Returns:
            List[TargetProperty]: list of TargetProperty objects
        """
        if original_names:
            return [t for t in l if t.originalName in names]
        return [t for t in l if t.name in names]

    @staticmethod
    def getNames(l: list):
        """Get the names of the target properties from a list of TargetProperty objects.

        Args:
            l (list): list of TargetProperty objects

        Returns:
            List[str]: list of names of the target properties
        """
        return [t.name for t in l]

    @staticmethod
    def getOriginalNames(l: list):
        """Get the original names of the target properties from a list of TargetProperty objects.

        Args:
            l (list): list of TargetProperty objects

        Returns:
            List[str]: list of original names of the target properties
        """
        return [t.originalName for t in l]


class QSPRDataset(MoleculeTable):
    """Prepare dataset for QSPR model training.

    It splits the data in train and test set, as well as creating cross-validation folds.
    Optionally low quality data is filtered out.
    For classification the dataset samples are labelled as active/inactive.

    Attributes:
        targetProperties (str) : property to be predicted with QSPRmodel
        df (pd.dataframe) : dataset
        X (np.ndarray/pd.DataFrame) : m x n feature matrix for cross validation, where m is
            the number of samplesand n is the number of features.
        y (np.ndarray/pd.DataFrame) : m-d label array for cross validation, where m is the
            number of samples and equals to row of X.
        X_ind (np.ndarray/pd.DataFrame) : m x n Feature matrix for independent set, where m
            is the number of samples and n is the number of features.
        y_ind (np.ndarray/pd.DataFrame) : m-l label array for independent set, where m is
            the number of samples and equals to row of X_ind, and l is the number of types.
        featureNames (list of str) : feature names
    """

    def __init__(
        self,
        name: str,
        target_props: List[Union[TargetProperty, dict]],
        df: pd.DataFrame = None,
        smilescol: str = "SMILES",
        proteincol: str = None,
        proteinseqprovider: Callable = None,
        add_rdkit: bool = False,
        store_dir: str = '.',
        overwrite: bool = False,
        n_jobs: int = 1,
        chunk_size: int = 50,
        drop_invalids: bool = True,
        drop_empty: bool = True,
        target_imputer: Callable = None,
        index_cols: List[str] = None,
    ):
        """Construct QSPRdata, also apply transformations of output property if specified.

        Args:
            name (str): data name, used in saving the data
            target_props (List[Union[TargetProperty, dict]]): target properties, names should correspond with target columnname in df
            df (pd.DataFrame, optional): input dataframe containing smiles and target property. Defaults to None.
            smilescol (str, optional): name of column in df containing SMILES. Defaults to "SMILES".
            proteincol (str, optional): name of column in df containing the protein target identifier (usually a UniProt ID) to use for protein descriptors for PCM modelling and other protein related tasks. Defaults to None.
            proteinseqprovider: Callable = None, optional): function that takes a 'proteincol' value and returns the appropriate protein sequence. Defaults to None.
            add_rdkit (bool, optional): if true, column with rdkit molecules will be added to df. Defaults to False.
            store_dir (str, optional): directory for saving the output data. Defaults to '.'.
            overwrite (bool, optional): if already saved data at output dir if should be overwritten. Defaults to False.
            n_jobs (int, optional): number of parallel jobs. If <= 0, all available cores will be used. Defaults to 1.
            chunk_size (int, optional): chunk size for parallel processing. Defaults to 50.
            drop_invalids (bool, optional): if true, invalid SMILES will be dropped. Defaults to True.
            drop_empty (bool, optional): if true, rows with empty target property will be removed.
            target_imputer (Callable, optional): imputer for missing target property values. Defaults to None.
            index_cols (List[str], optional): columns to be used as index in the dataframe. Defaults to `None` in which case a custom ID will be generated.

        Raises:
            `ValueError`: Raised if threshold given with non-classification task.
        """
        super().__init__(name, df, smilescol, proteincol, proteinseqprovider, add_rdkit, store_dir, overwrite, n_jobs, chunk_size, drop_invalids, index_cols)
        self.metaInfo = None
        try:
            self.metaInfo = QSPRDataset.loadMetadata(name, store_dir)
        except FileNotFoundError:
            pass

        # load names of descriptors to use as training features
        self.featureNames = self.getFeatureNames()

        # load target properties
        self.setTargetProperties(target_props, drop_empty, target_imputer)

        # load standardizers for features
        self.feature_standardizer = self.loadFeatureStandardizer()
        if not self.feature_standardizer:
            self.feature_standardizer = None
        self.fold_generator = self.getDefaultFoldGenerator()

        # populate feature matrix and target property array
        self.X = None
        self.y = None
        self.X_ind = None
        self.y_ind = None
        self.restoreTrainingData()

        logger.info(f"Dataset '{self.name}' created for target targetProperties: '{self.targetProperties}'.")

    @staticmethod
    def fromTableFile(name, filename, sep="\t", *args, **kwargs):
        r"""Create QSPRDataset from table file (i.e. CSV or TSV).

        Args:
            name (str): name of the data set
            filename (str): path to the table file
            sep (str, optional): separator in the table file. Defaults to "\t".
            *args: additional arguments for QSPRDataset constructor
            **kwargs: additional keyword arguments for QSPRDataset constructor
        Returns:
            QSPRDataset: `QSPRDataset` object
        """
        return QSPRDataset(name, df=pd.read_table(filename, sep=sep), *args, **kwargs)

    @staticmethod
    def fromSDF(name, filename, smiles_prop, *args, **kwargs):
        """Create QSPRDataset from SDF file.

        It is currently not implemented for QSPRDataset, but you can convert from 'MoleculeTable' with the 'fromMolTable' method.

        Args:
            name (str): name of the data set
            filename (str): path to the SDF file
            smiles_prop (str): name of the property in the SDF file containing SMILES
            *args: additional arguments for QSPRDataset constructor
            **kwargs: additional keyword arguments for QSPRDataset constructor
        """
        raise NotImplementedError(
            f"SDF loading not implemented for {QSPRDataset.__name__}, yet. You can convert from 'MoleculeTable' with 'fromMolTable'.")

    def setTargetProperties(
            self, target_props: List[TargetProperty],
            drop_empty: bool = True, target_imputer: Callable = None):
        """Set list of target properties and apply transformations if specified.

        Args:
            target_props (List[TargetProperty]): list of target properties
        """
        # check target properties validity
        assert isinstance(
            target_props, list), "target_props should be a list of TargetProperty objects or dictionaries to initialize TargetProperties from."
        if isinstance(target_props[0], dict):
            assert all([isinstance(d, dict) for d in target_props]
                       ), "target_props should be a list of TargetProperty objects or dictionaries to initialize TargetProperties from, not a mix."
            self.targetProperties = TargetProperty.fromList(target_props)
        else:
            assert all([isinstance(d, TargetProperty) for d in target_props]
                       ), "target_props should be a list of TargetProperty objects or dictionaries to initialize TargetProperties from, not a mix."
            self.targetProperties = target_props
        assert all([prop in self.df.columns for prop in self.targetPropertyNames]
                   ), "Not all target properties in dataframe columns."

        # transform target properties
        for target_prop in self.targetProperties:
            if target_prop.transformer is not None:
                transformed_prop = f'{target_prop.name}_transformed'
                self.transform([target_prop.name], target_prop.transformer, addAs=[transformed_prop])
                target_prop.name = transformed_prop

        # drop rows with missing smiles or no target property value for any of the target properties
        if drop_empty:
            self.dropEmpty()

        # impute missing target property values
        if target_imputer is not None:
            self.imputeTargetProperties(target_imputer)

        # convert classification targets to integers
        for target_prop in self.targetProperties:
            if target_prop.task.isClassification():
                self.makeClassification(target_prop)

    def dropEmpty(self):
        """Drop rows with empty target property value from the data set."""
        self.df.dropna(subset=([self.smilescol]), inplace=True)
        self.df.dropna(subset=(self.targetPropertyNames), how='all', inplace=True)

    def imputeTargetProperties(self, imputer):
        """Impute missing target property values.

        Args:
            imputer: imputer object, should have a fit and transform method.
        """
        names = self.targetPropertyNames
        for idx, target_prop in enumerate(self.targetProperties):
            self.targetProperties[idx].name = f"{target_prop.name}_imputed"
        self.df[self.targetPropertyNames] = imputer.fit_transform(self.df[names])

    @property
    def hasFeatures(self):
        """Check whether the currently selected set of features is not empty."""
        return True if (self.featureNames and len(self.featureNames) > 0) else False

    def getFeatureNames(self) -> List[str]:
        """Get current feature names for this data set.

        Returns:
            List[str]: list of feature names
        """
        features = None if not self.hasDescriptors else self.getDescriptorNames()
        if self.descriptorCalculators:
            features = []
            for calc in self.descriptorCalculators:
                prefix = calc.getPrefix()
                for descset in calc.descsets:
                    features.extend([f"{prefix}_{descset}_{x}" for x in descset.descriptors])

        if self.metaInfo and not features and ('feature_names' in self.metaInfo) and (self.metaInfo['feature_names'] is not None):
            features.extend([x for x in self.metaInfo['feature_names'] if x not in features])

        return features

    def restoreTrainingData(self):
        """Restore training data from the data frame.

        If the data frame contains a column 'Split_IsTrain',
        the data will be split into training and independent sets. Otherwise, the independent set will
        be empty. If descriptors are available, the resulting training matrices will be featurized.
        """
        self.loadDataToSplits()
        self.featurizeSplits()

    def makeRegression(self, target_property: Union[TargetProperty, str]):
        """Switch to regression task using the given target property.

        Args:
            target_property (str): name of the target property to use for regression
        """
        if isinstance(target_property, str):
            target_property = self.getTargetProperties([target_property], original_names=True)[0]
        target_property.name = target_property.originalName
        target_property.task = TargetTasks.REGRESSION
        del target_property.th
        self.restoreTrainingData()

    def makeClassification(self, target_property: Union[TargetProperty, str], th: List[float] = None):
        """Switch to classification task using the given threshold values.

        Args:
            target_property (TargetProperty): Target property to use for classification or name of the target property.
            th (List[float], optional): list of threshold values. If not provided, the values will be inferred from th specified in TargetProperty. Defaults to None.
        """
        if th is not None:
            assert isinstance(
                th, list) or th == "precomputed", "Threshold values should be provided as a list of floats."
            if isinstance(th, list):
                assert len(th) > 0, "Threshold values should be provided as a list of floats."

        if isinstance(target_property, str):
            target_property = self.getTargetProperties([target_property], original_names=True)[0]

        # check if the column only has nan values
        if self.df[target_property.name].isna().all():
            logger.debug(f"Target property {target_property.name} is all nan, assuming predictor.")
            return target_property

        # if no threshold values are provided, use the ones specified in the TargetProperty
        if th is None:
            assert hasattr(
                target_property, 'th'), "Target property does not have a threshold attribute and no threshold specified in function args."
            th = target_property.th

        new_prop = f"{target_property.originalName}_class"

        if th == 'precomputed':
            self.df[new_prop] = self.df[target_property.originalName]
            assert all([value is None or (type(value) in (int, bool)) or (isinstance(value, float) and value.is_integer())
                       for value in self.df[new_prop]]), "Precomputed classification target must be integers or booleans."
            nClasses = len(self.df[new_prop].dropna().unique())
            target_property.task = TargetTasks.MULTICLASS if nClasses > 2 else TargetTasks.SINGLECLASS
            target_property.th = th
            target_property.nClasses = nClasses
            target_property.name = new_prop
        else:
            assert len(th) > 0, "Threshold list must contain at least one value."
            if len(th) > 1:
                assert (
                    len(th) > 3
                ), "For multi-class classification, set more than 3 values as threshold."
                assert max(self.df[target_property.originalName]) <= max(
                    th
                ), "Make sure final threshold value is not smaller than largest value of property"
                assert min(self.df[target_property.originalName]) >= min(
                    th
                ), "Make sure first threshold value is not larger than smallest value of property"
                self.df[f"{new_prop}_intervals"] = pd.cut(
                    self.df[target_property.originalName], bins=th, include_lowest=True
                ).astype(str)
                self.df[new_prop] = LabelEncoder().fit_transform(self.df[f"{new_prop}_intervals"])
            else:
                self.df[new_prop] = self.df[target_property.originalName] > th[0]
            target_property.task = TargetTasks.SINGLECLASS if len(th) == 1 else TargetTasks.MULTICLASS
            target_property.th = th
            target_property.name = new_prop
        self.restoreTrainingData()
        logger.info("Target property converted to classification.")
        return target_property

    @ staticmethod
    def loadMetadata(name, store_dir):
        """Load metadata from a JSON file.

        Args:
            name (str): name of the data set
            store_dir (str): directory where the data set is stored
        """
        with open(os.path.join(store_dir, f"{name}_meta.json")) as f:
            meta = json.load(f)
            meta['init']['target_props'] = TargetProperty.fromList(meta['init']['target_props'], task_from_str=True)
            return meta

    @ staticmethod
    def fromFile(filename, *args, **kwargs) -> 'QSPRDataset':
        """Load QSPRDataset from the saved file directly.

        Args:
            filename (str): path to the saved file
            args: additional arguments to pass to the constructor
            kwargs: additional keyword arguments to pass to the constructor

        Returns:
            QSPRDataset: loaded data set
        """
        store_dir = os.path.dirname(filename)
        name = os.path.basename(filename).rsplit('_', 1)[0]
        meta = QSPRDataset.loadMetadata(name, store_dir)
        return QSPRDataset(*args, name=name, store_dir=store_dir, **meta['init'], **kwargs)

    @staticmethod
    def fromMolTable(
            mol_table: MoleculeTable, target_props: List[Union[TargetProperty, dict]],
            name=None, **kwargs) -> 'QSPRDataset':
        """Create QSPRDataset from a MoleculeTable.

        Args:
            mol_table (MoleculeTable): MoleculeTable to use as the data source
            target_props (list): list of target properties to use
            name (str, optional): name of the data set. Defaults to None.
            kwargs: additional keyword arguments to pass to the constructor

        Returns:
            QSPRDataset: created data set
        """
        kwargs['store_dir'] = mol_table.storeDir if 'store_dir' not in kwargs else kwargs['store_dir']
        name = mol_table.name if name is None else name
        ds = QSPRDataset(name, target_props, mol_table.getDF(), **kwargs)
        ds.descriptorCalculators = mol_table.descriptorCalculators
        ds.descriptors = mol_table.descriptors
        return ds

    def addCustomDescriptors(self, calculator: CustomDescriptorsCalculator, recalculate=False, featurize=True):
        """Add custom descriptors to the data set.

        If descriptors are already present, they will be recalculated if `recalculate` is `True`.

        Args:
            calculator (CustomDescriptorsCalculator): calculator instance to use for descriptor calculation
            recalculate (bool, optional): whether to recalculate descriptors if they are already present. Defaults to `False`.
            featurize (bool, optional): whether to featurize the data set splits after adding descriptors. Defaults to `True`.
        """
        super().addCustomDescriptors(calculator, recalculate)
        self.featurize(update_splits=featurize)

    def addDescriptors(self, calculator: MoleculeDescriptorsCalculator, recalculate=False, featurize=True):
        """Add descriptors to the data set.

        If descriptors are already present, they will be recalculated if `recalculate` is `True`.
        Featurization will be performed after adding descriptors if `featurize` is `True`. Featurazation
        converts current data matrices to pure numeric matrices of selected descriptors (features).

        Args:
            calculator (MoleculeDescriptorsCalculator): calculator instance to use for descriptor calculation
            recalculate (bool, optional): whether to recalculate descriptors if they are already present. Defaults to `False`.
            featurize (bool, optional): whether to featurize the data set splits after adding descriptors. Defaults to `True`.
        """
        super().addDescriptors(calculator, recalculate)
        self.featurize(update_splits=featurize)

    def addProteinDescriptors(self, calculator: ProteinDescriptorCalculator, recalculate=False, featurize=True):
        """Add protein descriptors to the data set.

        If descriptors are already present, they will be recalculated if `recalculate` is `True`.
        Featurization will be performed after adding descriptors if `featurize` is `True`. Featurazation
        converts current data matrices to pure numeric matrices of selected descriptors (features).

        Args:
            calculator (ProteinDescriptorCalculator): calculator instance to use for descriptor calculation
            recalculate (bool, optional): whether to recalculate descriptors if they are already present. Defaults to `False`.
            featurize (bool, optional): whether to featurize the data set splits after adding descriptors. Defaults to `True`.
        """
        super().addProteinDescriptors(calculator, recalculate)
        self.featurize(update_splits=featurize)

    def featurize(self, update_splits=True):
        self.featureNames = self.getFeatureNames()
        if update_splits:
            self.featurizeSplits()

    def saveSplit(self):
        """Save split data to the managed data frame."""
        if self.X is not None:
            self.df["Split_IsTrain"] = self.df.index.isin(self.X.index)
        else:
            logger.debug("No split data available. Skipping split data save.")

    def save(self, save_split=True):
        """Save the data set to file and serialize metadata."""
        if save_split:
            self.saveSplit()
        super().save()

        # save metadata
        self.saveMetadata()

    def split(self, split: datasplit):
        """Split dataset into train and test set.

        Args:
            split (datasplit) : split instance orchestrating the split
        """
        if hasattr(split, "hasDataSet") and hasattr(split, "setDataSet") and not split.hasDataSet:
            split.setDataSet(self)

        folds = Folds(split)
        self.X, self.X_ind, self.y, self.y_ind, train_index, test_index = next(
            folds.iterFolds(self.df, self.df[self.targetPropertyNames]))
        self.X = self.df.iloc[train_index, :]
        self.X_ind = self.df.iloc[test_index, :]
        self.y = self.df.iloc[train_index, :][self.targetPropertyNames]
        self.y_ind = self.df.iloc[test_index, :][self.targetPropertyNames]

        logger.info("Total: train: %s test: %s" % (len(self.y), len(self.y_ind)))
        for prop in self.targetProperties:
            logger.info("Target property: %s" % prop.name)
            if prop.task == TargetTasks.SINGLECLASS:
                logger.info(
                    "    In train: active: %s not active: %s"
                    % (sum(self.y[prop.name]), len(self.y[prop.name]) - sum(self.y[prop.name]))
                )
                logger.info(
                    "    In test:  active: %s not active: %s\n"
                    % (sum(self.y_ind[prop.name]), len(self.y_ind[prop.name]) - sum(self.y_ind[prop.name]))
                )
            if prop.task == TargetTasks.MULTICLASS:
                logger.info("train: %s" % self.y[prop.name].value_counts())
                logger.info("test: %s\n" % self.y_ind[prop.name].value_counts())
                try:
                    assert np.all([x > 0 for x in self.y[prop.name].value_counts()])
                    assert np.all([x > 0 for x in self.y_ind[prop.name].value_counts()])
                except AssertionError as err:
                    logger.exception(
                        "All bins in multi-class classification should contain at least one sample"
                    )
                    raise err

                if self.y[prop.name].dtype.name == "category":
                    self.y[prop.name] = self.y[prop.name].cat.codes
                    self.y_ind[prop.name] = self.y_ind[prop.name].cat.codes

    def loadDataToSplits(self):
        """Load the data frame into the train and test splits.

        Loads only if the information is available. Otherwise, the whole data
        set will be regarded as the training set and the test set will have zero length.
        """
        self.X = self.df
        self.y = self.df[self.targetPropertyNames]

        # split data into training and independent sets if saved previously
        if "Split_IsTrain" in self.df.columns:
            self.X = self.df[self.df["Split_IsTrain"] == True]
            self.X_ind = self.df[self.df["Split_IsTrain"] == False]
            self.y = self.X[self.targetPropertyNames]
            self.y_ind = self.X_ind[self.targetPropertyNames]
        else:
            self.X_ind = self.X.drop(self.X.index)
            self.y_ind = self.y.drop(self.y.index)

    def loadDescriptorsToSplits(self):
        """Load all available descriptors into the train and test splits.

        If no descriptors are available, an exception will be raised.

        Raises:
            ValueError: if no descriptors are available
        """
        if not self.hasDescriptors:
            raise ValueError("No descriptors available. Cannot load descriptors to splits.")

        descriptors = self.getDescriptors()
        self.X = descriptors.loc[self.X.index, :]
        self.y = self.df.loc[self.y.index, self.targetPropertyNames]

        if self.X_ind is not None and self.y_ind is not None:
            self.X_ind = descriptors.loc[self.X_ind.index, :]
            self.y_ind = self.df.loc[self.y_ind.index, self.targetPropertyNames]
        else:
            self.X_ind = pd.DataFrame(columns=self.X.columns)
            self.y_ind = pd.DataFrame(columns=[self.targetPropertyNames])

    def featurizeSplits(self):
        """If the data set has descriptors, load them into the train and test splits.

        If no descriptors are available, remove all features from
        the splits They will become zero length along the feature axis (columns), but will retain their original length
        along the sample axis (rows). This is useful for the case where the data set has no descriptors, but the user
        wants to retain train and test splits.
        """
        if self.featureNames:
            self.loadDescriptorsToSplits()
            self.X = self.X[self.featureNames]
            self.X_ind = self.X_ind[self.featureNames]
        else:
            self.X = self.X.drop(self.X.columns, axis=1)
            self.X_ind = self.X_ind.drop(self.X_ind.columns, axis=1)

    def fillMissing(self, fill_value: float, columns: List[str] = None):
        """Fill missing values in the data set with a given value.

        Args:
            fill_value (float): value to fill missing values with
            columns (List[str], optional): columns to fill missing values in. Defaults to None.
        """
        for desc in self.descriptors:
            desc.fillMissing(fill_value, columns)
        logger.warning('Missing values filled with %s' % fill_value)

    def filterFeatures(self, feature_filters: List[Callable]):
        """Filter features in the data set.

        Args:
            feature_filters (List[Callable]): list of feature filter functions that take X feature matrix and y target vector as arguments
        """
        if not self.hasFeatures:
            raise ValueError("No features to filter")

        if self.X.shape[1] == 1:
            logger.warning("Only one feature present. Skipping feature filtering.")
            return
        else:
            for featurefilter in feature_filters:
                self.X = featurefilter(self.X, self.y)

            self.featureNames = self.X.columns.to_list()
            if self.X_ind is not None:
                self.X_ind = self.X_ind[self.featureNames]
            logger.info(f"Selected features: {self.featureNames}")

            # update descriptor calculator
            if self.descriptorCalculators is not None:
                for calc in self.descriptorCalculators:
                    prefix = calc.getPrefix()
                    calc.keepDescriptors([x for x in self.featureNames if x.startswith(prefix)])

    def setFeatureStandardizer(self, feature_standardizer):
        """Set feature standardizer.

        Args:
            feature_standardizer (Union[SKLearnStandardizer, BaseEstimator]): feature standardizer
        """
        if not hasattr(feature_standardizer, 'toFile'):
            feature_standardizer = SKLearnStandardizer(feature_standardizer)
        self.feature_standardizer = feature_standardizer

    def prepareDataset(
        self,
        smiles_standardizer : Union[str, Callable, None] = 'chembl',
        datafilters=None,
        split=None,
        fold=None,
        feature_calculators : List[DescriptorsCalculator] = None,
        feature_filters=None,
        feature_standardizer=None,
        feature_fill_value=np.nan,
        recalculate_features=False
    ):
        """Prepare the dataset for use in QSPR model.

        Arguments:
            smiles_standardizer (Union[str, Callable]): either `chembl`, `old`, or a partial function that reads and standardizes smiles. If `None`, no standardization will be performed. Defaults to `chembl`.
            datafilters (list of datafilter obj): filters number of rows from dataset
            split (datasplitter obj): splits the dataset into train and test set
            fold (datasplitter obj): splits the train set into folds for cross validation
            feature_calculators (List[DescriptorsCalculator]): calculate features using different information from the data set
            feature_filters (list of feature filter objs): filters features
            feature_standardizer (SKLearnStandardizer or sklearn.base.BaseEstimator): standardizes and/or scales features
            recalculate_features (bool): recalculate features even if they are already present in the file
            feature_fill_value (float): value to fill missing values with, defaults to `numpy.nan`
        """
        # apply sanitization and standardization
        if smiles_standardizer is not None:
            self.standardizeSmiles(smiles_standardizer)

        # calculate features
        if feature_calculators is not None:
            for calc in feature_calculators:
                if isinstance(calc, MoleculeDescriptorsCalculator):
                    self.addDescriptors(calc, recalculate=recalculate_features, featurize=False)
                elif isinstance(calc, ProteinDescriptorCalculator):
                    self.addProteinDescriptors(calc, recalculate=recalculate_features, featurize=False)
                else:
                    raise ValueError("Unknown feature calculator type: %s" % type(calc))

        # apply data filters
        if datafilters is not None:
            self.filter(datafilters)

        # Replace any NaN values in featureNames by 0
        # FIXME: this is not very good, we should probably add option to do custom
        # data imputation here or drop rows with NaNs
        if feature_fill_value is not None:
            self.fillMissing(feature_fill_value)

        # split dataset
        if split is not None:
            self.split(split)
        else:
            self.X = self.df
            self.y = self.df[self.targetPropertyNames]

        # featurize splits
        if self.hasDescriptors:
            self.featurizeSplits()
        else:
            logger.warning("Attempting to featurize splits without descriptors. Skipping this step...")

        # apply feature filters on training set
        if feature_filters and self.hasDescriptors:
            self.filterFeatures(feature_filters)
        elif not self.hasDescriptors:
            logger.warning(
                "No descriptors present, feature filters will be skipped."
            )

        # set feature standardizers
        if feature_standardizer:
            self.setFeatureStandardizer(feature_standardizer)
            if self.fold_generator:
                self.fold_generator = Folds(self.fold_generator.split, self.feature_standardizer)
            if not self.hasDescriptors:
                logger.warning(
                    "No descriptors present, feature standardizers were initialized, but might fail or have no effect."
                )

        # create fold generator
        if fold:
            self.fold_generator = Folds(fold, self.feature_standardizer)

    def getDefaultFoldSplit(self):
        """Return the default fold split for the model task.

        Returns:
            datasplit (datasplit): default fold split implementation
        """
        if len(self.targetProperties) > 1 or self.targetProperties[0].task == TargetTasks.REGRESSION:
            return KFold(5)
        else:
            return StratifiedKFold(5)

    def getDefaultFoldGenerator(self):
        """Return the default fold generator. The fold generator is used to create folds for cross validation.

        Returns:
            Folds (Folds): default fold generator implementation
        """
        return Folds(self.getDefaultFoldSplit(), self.feature_standardizer)

    def checkFeatures(self):
        """Check consistency of features and descriptors."""
        if not self.hasDescriptors:
            raise ValueError("No descriptors exist in the data set. Cannot create folds.")
        elif not self.hasFeatures:
            raise ValueError("No features exist in the data set. Cannot create folds.")
        elif self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f"X and y have different number of rows: {self.X.shape[0]} != {self.y.shape[0]}")
        elif self.X.shape[0] == 0:
            raise ValueError("X has no rows.")

    def createFolds(self, split: datasplit = None):
        """Create folds for cross validation.

        Args:
            split (datasplit, optional): split to use for creating folds. Defaults to None.
        """
        self.checkFeatures()

        if split is None and not self.fold_generator:
            self.fold_generator = self.getDefaultFoldGenerator()
        elif split is not None:
            self.fold_generator = Folds(split, self.feature_standardizer)

        return self.fold_generator.iterFolds(self.X, self.y)

    def fitFeatureStandardizer(self):
        """Fit the feature standardizers on the training set.

        Returns:
            X (pd.DataFrame): standardized training set
        """
        if self.hasDescriptors:
            X = self.getDescriptors()
            if self.featureNames is not None:
                X = X[self.featureNames]
            return apply_feature_standardizer(self.feature_standardizer, X, fit=True)[0]

    def getFeatures(self, inplace=False, concat=False, raw=False):
        """Get the current feature sets (training and test) from the dataset.

        This method also applies any feature standardizers that have been set on the dataset during preparation.

        Arguments:
            inplace (bool): If `True`, the created feature matrices will be saved to the dataset object itself as 'X' and 'X_ind' attributes.
                            Note that this will overwrite any existing feature matrices and if the data preparation workflow changes, these are not kept up to date.
                            Therefore, it is recommended to generate new feature sets after any data set changes.
            concat (bool): If `True`, the training and test feature matrices will be concatenated into a single matrix. This is useful for
                training models that do not require separate training and test sets (i.e. the final optimized models).
            raw (bool): If `True`, the raw feature matrices will be returned without any standardization applied.
        """
        self.checkFeatures()

        if concat:
            df_X = pd.concat([self.X[self.featureNames], self.X_ind[self.featureNames]], axis=0)
            df_X_ind = None
        else:
            df_X = self.X[self.featureNames]
            df_X_ind = self.X_ind[self.featureNames]

        X = df_X.values
        X_ind = df_X_ind.values if df_X_ind is not None else None
        if not raw and self.feature_standardizer:
            X, self.feature_standardizer = apply_feature_standardizer(
                self.feature_standardizer,
                df_X,
                fit=True
            )
            if X_ind is not None and X_ind.shape[0] > 0:
                X_ind, _ = apply_feature_standardizer(
                    self.feature_standardizer,
                    df_X_ind,
                    fit=False
                )

        X = pd.DataFrame(X, index=df_X.index, columns=df_X.columns)
        if X_ind is not None:
            X_ind = pd.DataFrame(X_ind, index=df_X_ind.index, columns=df_X_ind.columns)

        if inplace:
            self.X = X
            self.X_ind = X_ind

        return (X, X_ind) if not concat else X

    def getTargetPropertiesValues(self, concat=False):
        """Get the response values (training and test) for the set target property.

        Args:
            concat (bool): if `True`, return concatenated training and validation set target properties

        Returns:
            `tuple` of (train_responses, test_responses) or `pandas.DataFrame` of all target property values
        """
        if concat:
            return pd.concat([self.y, self.y_ind] if self.y_ind is not None else [self.y])
        else:
            return self.y, self.y_ind if self.y_ind is not None else self.y

    def getTargetProperties(self, names, original_names=False):
        """Get the target properties with the given names.

        Args:
            names (List[str]): name of the target properties
            original_names (bool): if `True`, use the original names of the target properties

        Returns:
            `TargetProperty`: target property with the given name
        """
        return TargetProperty.selectFromList(self.targetProperties, names, original_names=original_names)

    @property
    def targetPropertyNames(self):
        """Get the names of the target properties."""
        return TargetProperty.getNames(self.targetProperties)

    @property
    def targetPropertyOriginalNames(self):
        """Get the original names of the target properties."""
        return TargetProperty.getOriginalNames(self.targetProperties)

    def loadFeatureStandardizer(self):
        """Load feature standardizer from the metadata.

        Returns:
            `SKLearnStandardizer`
        """
        if self.metaInfo is not None and 'standardizer_path' in self.metaInfo and self.metaInfo['standardizer_path']:
            return SKLearnStandardizer.fromFile(f"{self.storePrefix}{self.metaInfo['standardizer_path']}")
        else:
            return None

    def saveFeatureStandardizer(self):
        """Save feature standardizers to the metadata.

        Returns:
            `str`: paths to the saved standardizers
        """
        path = f'{self.storePrefix}_feature_standardizer.json'

        if self.feature_standardizer and self.featureNames is not None and len(self.featureNames) > 0:
            # make sure feature standardizers are fitted before serialization
            self.fitFeatureStandardizer()
            self.feature_standardizer.toFile(path)
            return path
        elif self.feature_standardizer:
            self.feature_standardizer.toFile(path)
            return path

    def saveMetadata(self):
        """Save metadata to file.

        Returns:
            `str`: path to the saved metadata file
        """
        path = self.saveFeatureStandardizer()

        meta_init = {
            'target_props': TargetProperty.toList(copy.deepcopy(self.targetProperties), task_as_str=True),
            'smilescol': self.smilescol,
        }
        ret = {
            'init': meta_init,
            'standardizer_path': path.replace(
                self.storePrefix, '') if path else None,
            'descriptorcalculator_path': self.descriptorCalculatorsPathPrefix.replace(
                self.storePrefix, '') if self.descriptorCalculatorsPathPrefix else None,
            'feature_names': list(
                self.featureNames) if self.featureNames is not None else None,
        }
        path = f"{self.storePrefix}_meta.json"
        with open(path, 'w') as f:
            json.dump(ret, f)

        return path

    @property
    def isMultiTask(self):
        """Check if the dataset contains multiple target properties.

        Returns:
            `bool`: `True` if the dataset contains multiple target properties
        """
        return len(self.targetProperties) > 1

    @property
    def nTasks(self):
        """Get the number of target properties in the dataset."""
        return len(self.targetProperties)

    def dropTask(self, task):
        """Drop the given task from the dataset.

        Args:
            task (str): name of the task to drop
        """
        assert task in self.targetPropertyNames, f"Task {task} not found in dataset."
        assert len(self.targetProperties) > 1, "Cannot drop task from single-task dataset."
        self.targetProperties = [tp for tp in self.targetProperties if tp.name != task]
        self.restoreTrainingData()

    def addTask(self, task: Union[TargetProperty, dict]):
        """Add a task to the dataset.

        Args:
            task (TargetProperty): name of the task to add
        """
        if isinstance(task, dict):
            task = TargetProperty.fromDict(task)

        assert task.name not in self.targetPropertyNames, f"Task {task} already exists in dataset."
        assert task.name in self.df.columns, f"Task {task} not found in dataset."

        self.targetProperties.append(task)
        self.restoreTrainingData()
