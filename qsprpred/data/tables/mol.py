import os
import pickle
from multiprocessing import Pool
from typing import Optional, ClassVar, Generator, Literal, Callable, Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

from qsprpred.data.tables.searchable import SearchableMolTable
from .pandas import PandasDataTable
from ..chem.matching import match_mol_to_smarts
from ..descriptors.sets import DescriptorSet
from ..processing.mol_processor import MolProcessor
from ...data.chem.scaffolds import Scaffold
from ...data.chem.standardization import (
    CheckSmilesValid,
    chembl_smi_standardizer,
    old_standardize_sanitize,
)
from ...logs import logger
from ...utils.interfaces.summarizable import Summarizable


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
        autoindex_name: str = "QSPRID",
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
        all_descs = self.df.columns[~self.df.columns.isin(self.indexCols)].tolist()
        if active_only:
            return [x for x in all_descs if x in self.calculator.descriptors]
        else:
            return all_descs

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
        self.calculator.descriptors = [
            x for x in self.calculator.descriptors if x in to_keep
        ]
        return self.getDescriptorNames()


class MoleculeTable(PandasDataTable, SearchableMolTable, Summarizable):
    """Class that holds and prepares molecule data for modelling and other analyses.

    Attributes:
        smilesCol (str):
            Name of the column containing the SMILES sequences
            of molecules.
        includesRdkit (bool):
            Whether the data frame contains RDKit molecules as one of
            the properties.
        descriptors (list[DescriptorTable]):
            List of `DescriptorTable` objects containing the descriptors
            calculated for this table.
    """

    _notJSON: ClassVar = PandasDataTable._notJSON + ["descriptors"]

    def __init__(
        self,
        name: str,
        df: Optional[pd.DataFrame] = None,
        smiles_col: str = "SMILES",
        add_rdkit: bool = False,
        store_dir: str = ".",
        overwrite: bool = False,
        n_jobs: int | None = 1,
        chunk_size: int | None = None,
        drop_invalids: bool = True,
        index_cols: Optional[list[str]] = None,
        autoindex_name: str = "QSPRID",
        random_state: int | None = None,
        store_format: str = "pkl",
    ):
        """Initialize a `MoleculeTable` object.

        This object wraps a pandas dataframe and provides short-hand methods to prepare
        molecule data for modelling and analysis.

        Args:
            name (str): Name of the dataset. You can use this name to load the dataset
                from disk anytime and create a new instance.
            df (pd.DataFrame): Pandas dataframe containing the data. If you provide a
                dataframe for a dataset that already exists on disk,
            the dataframe from disk will override the supplied data frame. Set
                'overwrite' to `True` to override the data frame on disk.
            smiles_col (str): Name of the column containing the SMILES sequences
                of molecules.
            add_rdkit (bool): Add RDKit molecule instances to the dataframe.
                WARNING: This can take a lot of memory.
            store_dir (str): Directory to store the dataset files. Defaults to the
                current directory. If it already contains files with the same name,
                the existing data will be loaded.
            overwrite (bool): Overwrite existing dataset.
            n_jobs (int): Number of jobs to use for parallel processing. If <= 0, all
                available cores will be used.
            chunk_size (int): Size of chunks to use per job in parallel processing.
            drop_invalids (bool): Drop invalid molecules from the data frame.
            index_cols (list[str]): list of columns to use as index. If None, the index
                will be a custom generated ID.
            autoindex_name (str): Column name to use for automatically generated IDs.
            random_state (int): Random state to use for shuffling and other random ops.
            store_format (str): Format to use for storing the data ('pkl' or 'csv').
        """
        super().__init__(
            name,
            df,
            store_dir,
            overwrite,
            index_cols,
            n_jobs,
            chunk_size,
            autoindex_name,
            random_state,
            store_format,
        )
        # the descriptors
        self.descriptors = []
        # settings
        self.smilesCol = smiles_col
        self.includesRdkit = add_rdkit
        # drop invalid columns
        if drop_invalids:
            self.dropInvalids()
            # update chunk size if count changed
            self.chunkSize = chunk_size
        # add rdkit molecules if requested
        if self.includesRdkit and "RDMol" not in self.df.columns:
            PandasTools.AddMoleculeColumnToFrame(
                self.df,
                smilesCol=self.smilesCol,
                molCol="RDMol",
                includeFingerprints=False,
            )
            self.includesRdkit = True

    def searchWithIndex(
        self, index: pd.Index, name: str | None = None
    ) -> "MoleculeTable":
        """Search in this table using a pandas index. The return values
        is a new table with the molecules from the old table with the given indices.

        Args:
            index(pd.Index):
                Indices to search for in this table.
            name(str):
                Name of the new table. Defaults to the name of the old table,
                plus the `_searched` suffix.

        Returns:
            MoleculeTable:
                A new table with the molecules from the
                old table with the given indices.
        """
        name = f"{self.name}_searched" if name is None else name
        ret = MoleculeTable(
            name=name,
            df=self.df.loc[index, :],
            smiles_col=self.smilesCol,
            add_rdkit=False,
            store_dir=self.storeDir,
            overwrite=True,
            n_jobs=self.nJobs,
            chunk_size=self.chunkSize,
            drop_invalids=False,
            index_cols=self.indexCols,
            random_state=self.randomState,
            store_format=self.storeFormat,
        )
        for table in self.descriptors:
            ret.descriptors.append(
                DescriptorTable(
                    table.calculator,
                    name=ret.generateDescriptorDataSetName(table.calculator),
                    df=table.getDF().loc[index, :],
                    store_dir=table.storeDir,
                    overwrite=True,
                    key_cols=table.indexCols,
                    n_jobs=table.nJobs,
                    chunk_size=table.chunkSize,
                    store_format=table.storeFormat,
                    random_state=table.randomState,
                )
            )
        return ret

    def searchOnProperty(
        self, prop_name: str, values: list[str], name: str | None = None, exact=False
    ) -> "MoleculeTable":
        """Search in this table using a property name and a list of values. It is
        assumed that the property is searchable with string matching. Either an
        exact match or a partial match can be used. If 'exact' is `False`, the
        search will be performed with partial matching, i.e. all molecules that
        contain any of the given values in the property will be returned. If
        'exact' is `True`, only molecules that have the exact property value for
        any of the given values will be returned.

        Args:
            prop_name (str):
                Name of the property to search on.
            values (list[str]):
                List of values to search for. If any of the values is found in the
                property, the molecule will be considered a match.
            name (str | None, optional):
                Name of the new table. Defaults to the name of
                the old table, plus the `_searched` suffix.
            exact (bool, optional):
                Whether to use exact matching, i.e. whether to
                search for exact strings or just substrings. Defaults to False.

        Returns:
            MoleculeTable:
                A new table with the molecules from the
                old table with the given property values.
        """
        mask = [False] * len(self.df)
        for value in values:
            mask = (
                mask | (self.df[prop_name].str.contains(value))
                if not exact
                else mask | (self.df[prop_name] == value)
            )
        matches = self.df.index[mask]
        return self.searchWithIndex(matches, name)

    def searchWithSMARTS(
        self,
        patterns: list[str],
        operator: Literal["or", "and"] = "or",
        use_chirality: bool = False,
        name: str | None = None,
        match_function: Callable = match_mol_to_smarts,
    ) -> "MoleculeTable":
        """Search the molecules in the table with a SMARTS pattern.

        Args:
            patterns:
                List of SMARTS patterns to search with.
            operator (object):
                Whether to use an "or" or "and" operator on patterns. Defaults to "or".
            use_chirality:
                Whether to use chirality in the search.
            name:
                Name of the new table. Defaults to the name of the old table,
                plus the `smarts_searched` suffix.
            match_function:
                Function to use for matching the molecules to the SMARTS patterns.
                Defaults to `match_mol_to_smarts`.

        Returns:
            (MolTable): A dataframe with the molecules that match the pattern.
        """
        matches = self.df.index[
            self.df[self.smilesCol].apply(
                lambda x: match_function(
                    x, patterns, operator=operator, use_chirality=use_chirality
                )
            )
        ]
        return self.searchWithIndex(
            matches, name=f"{self.name}_smarts_searched" if name is None else name
        )

    def getSummary(self):
        """
        Make a summary with some statistics about the molecules in this table.
        The summary contains the number of molecules per target and the number of
        unique molecules per target.

        Requires this data set to be imported from Papyrus for now.

        Returns:
            (pd.DataFrame): A dataframe with the summary statistics.

        """
        summary = {
            "mols_per_target": self.df.groupby("accession")
            .count()["InChIKey"]
            .to_dict(),
            "mols_per_target_unique": self.df.groupby("accession")
            .aggregate(lambda x: len(set(x)))["InChIKey"]
            .to_dict(),
        }
        return pd.DataFrame(summary)

    def sample(
        self, n: int, name: str | None = None, random_state: int | None = None
    ) -> "MoleculeTable":
        """
        Sample n molecules from the table.

        Args:
            n (int):
                Number of molecules to sample.
            name (str):
                Name of the new table. Defaults to the name of the old
                table, plus the `_sampled` suffix.
            random_state (int):
                Random state to use for shuffling and other random ops.

        Returns:
            (MoleculeTable): A dataframe with the sampled molecules.
        """
        random_state = random_state or self.randomState
        name = f"{self.name}_sampled" if name is None else name
        index = self.df.sample(n=n, random_state=random_state).index
        return self.searchWithIndex(index, name=name)

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["descriptors"] = []
        for desc in self.descriptors:
            o_dict["descriptors"].append(os.path.basename(desc.storeDir))
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.descriptors = []
        for desc in state["descriptors"]:
            desc = os.path.join(self.storeDir, desc, f"{desc}_meta.json")
            self.descriptors.append(DescriptorTable.fromFile(desc))

    def toFile(self, filename: str):
        ret = super().toFile(filename)
        for desc in self.descriptors:
            desc.save()
        return ret

    @property
    def descriptorSets(self):
        """Get the descriptor calculators for this table."""
        return [x.calculator for x in self.descriptors]

    @staticmethod
    def fromSMILES(name: str, smiles: list, *args, **kwargs):
        """Create a `MoleculeTable` instance from a list of SMILES sequences.

        Args:
            name (str): Name of the data set.
            smiles (list): list of SMILES sequences.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: Additional keyword arguments to pass to the `MoleculeTable`
                constructor.
        """
        smilescol = "SMILES"
        df = pd.DataFrame({smilescol: smiles})
        return MoleculeTable(name, df, *args, smiles_col=smilescol, **kwargs)

    @staticmethod
    def fromTableFile(name: str, filename: str, sep="\t", *args, **kwargs):
        """Create a `MoleculeTable` instance from a file containing a table of molecules
        (i.e. a CSV file).

        Args:
            name (str): Name of the data set.
            filename (str): Path to the file containing the table.
            sep (str): Separator used in the file for different columns.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: Additional keyword arguments to pass to the `MoleculeTable`
                constructor.
        """
        return MoleculeTable(name, pd.read_table(filename, sep=sep), *args, **kwargs)

    @staticmethod
    def fromSDF(name, filename, smiles_prop, *args, **kwargs):
        """Create a `MoleculeTable` instance from an SDF file.

        Args:
            name (str): Name of the data set.
            filename (str): Path to the SDF file.
            smiles_prop (str): Name of the property in the SDF file containing the
                SMILES sequence.
            *args: Additional arguments to pass to the `MoleculeTable` constructor.
            **kwargs: Additional keyword arguments to pass to the `MoleculeTable`
                constructor.
        """
        # FIXME: the RDKit mols are always added here, which might be unnecessary
        return MoleculeTable(
            name,
            PandasTools.LoadSDF(filename, molColName="RDMol"),
            smiles_col=smiles_prop,
            *args,  # noqa: B026 # FIXME: this is a bug in flake8...
            **kwargs,
        )

    @classmethod
    def runMolProcess(
        cls,
        props: dict[str, list] | pd.DataFrame,
        func: MolProcessor,
        add_rdkit: bool,
        smiles_col: str,
        *args,
        **kwargs,
    ):
        """A helper method to run a `MolProcessor` on a list of molecules via `apply`.
        It converts the SMILES to RDKit molecules if required and then applies the
        function to the `MolProcessor` object.

        Args:
            props (dict):
                Dictionary of properties that will be passed in addition to the
                molecule structure.
            func (MolProcessor):
                `MolProcessor` object to use for processing.
            add_rdkit (bool):
                Whether to convert the SMILES to RDKit molecules before
                applying the function.
            smiles_col (str):
                Name of the property containing the SMILES sequences.
            *args:
                Additional positional arguments to pass to the function.
            **kwargs:
                Additional keyword arguments to pass to the function.
        """
        if add_rdkit:
            mols = (
                props["RDMol"]
                if "RDMol" in props
                else [Chem.MolFromSmiles(x) for x in props[smiles_col]]
            )
        else:
            mols = props[smiles_col]
        return func(mols, props, *args, **kwargs)

    def processMols(
        self,
        processor: MolProcessor,
        proc_args: tuple[Any] | None = None,
        proc_kwargs: dict[str, Any] | None = None,
        add_props: list[str] | None = None,
        as_rdkit: bool = False,
        chunk_size: int | None = None,
        n_jobs: int | None = None,
    ) -> Generator:
        """Apply a function to the molecules in the data frame.
        The SMILES  or an RDKit molecule will be supplied as the first
        positional argument to the function. Additional properties
        to provide from the data set can be specified with 'add_props', which will be
        a dictionary supplied as an additional positional argument to the function.

        IMPORTANT: For successful parallel processing, the processor must be picklable.
        Also note that
        the returned generator will produce results as soon as they are ready,
        which means that the chunks of data will
        not be in the same order as the original data frame. However, you can pass the
        value of `idProp` in `add_props` to identify the processed molecules.
        See `CheckSmilesValid` for an example.

        Args:
            processor (MolProcessor):
                `MolProcessor` object to use for processing.
            proc_args (list, optional):
                Any additional positional arguments to pass to the processor.
            proc_kwargs (dict, optional):
                Any additional keyword arguments to pass to the processor.
            add_props (list, optional):
                List of data set properties to send to the processor. If `None`, all
                properties will be sent.
            as_rdkit (bool, optional):
                Whether to convert the molecules to RDKit
                molecules before applying the processor.
            chunk_size (int, optional):
                Size of chunks to use per job in parallel.
                If not specified, `self.chunkSize` is used.
            n_jobs (int, optional):
                Number of jobs to use for parallel processing.
                If not specified, `self.nJobs` is used.

        Returns:
            Generator:
                A generator that yields the results of the supplied processor on
                the chunked molecules from the data set.
        """
        chunk_size = chunk_size or self.chunkSize
        n_jobs = n_jobs or self.nJobs
        proc_args = proc_args or ()
        proc_kwargs = proc_kwargs or {}
        add_props = add_props or self.df.columns.tolist()
        if add_props is not None and self.smilesCol not in add_props:
            add_props.append(self.smilesCol)
            add_props.append(self.idProp)
        for prop in processor.requiredProps:
            if prop not in self.df.columns:
                raise ValueError(
                    f"Cannot apply function '{processor}' to {self.name} because "
                    f"it requires the property '{prop}', which is not present in the "
                    "data set."
                )
            if prop not in add_props:
                add_props.append(prop)
        if self.nJobs > 1 and processor.supportsParallel:
            logger.debug(
                f"Applying processor '{processor}' to '{self.name}' in parallel."
            )
            for result in self.apply(
                self.runMolProcess,
                func_args=[processor, as_rdkit, self.smilesCol, *proc_args],
                func_kwargs=proc_kwargs,
                on_props=add_props,
                as_df=False,
                chunk_size=chunk_size,
                n_jobs=n_jobs,
            ):
                yield result
        else:
            logger.debug(
                f"Applying processor '{processor}' to '{self.name}' in serial."
            )
            for result in self.iterChunks(
                include_props=add_props, as_dict=True, chunk_size=len(self)
            ):
                yield self.runMolProcess(
                    result,
                    processor,
                    as_rdkit,
                    self.smilesCol,
                    *proc_args,
                    **proc_kwargs,
                )

    def checkMols(self, throw: bool = True):
        """
        Returns a boolean array indicating whether each molecule is valid or not.
        If `throw` is `True`, an exception is thrown if any molecule is invalid.

        Args:
            throw (bool): Whether to throw an exception if any molecule is invalid.

        Returns:
            mask (pd.Series): Boolean series indicating whether each molecule is valid.
        """
        mask = pd.Series([False] * len(self), index=self.df.index, dtype=bool)
        for result in self.processMols(
            CheckSmilesValid(id_prop=self.idProp), proc_kwargs={"throw": throw}
        ):
            mask.loc[result.index] = result.values
        return mask

    def generateDescriptorDataSetName(self, ds_set: str | DescriptorSet):
        """Generate a descriptor set name from a descriptor set."""
        return f"Descriptors_{self.name}_{ds_set}"

    def dropDescriptors(
        self,
        descriptors: list[DescriptorSet] | list[str],
    ):
        """
        Drop descriptors from the data frame
        that were calculated with a specific calculator.

        Args:
            descriptors (list): list of `DescriptorSet` objects or prefixes of
                descriptors to drop.
        """
        # sanity check
        assert (
            len(self.descriptors) != 0
        ), "Cannot drop descriptors because the data set does not contain any."
        if len(descriptors) == 0:
            logger.warning(
                "No descriptors specified to drop. All descriptors will be retained."
            )
            return
        # convert descriptors to descriptor set names
        descriptors = [self.generateDescriptorDataSetName(x) for x in descriptors]
        # drop the descriptors
        to_remove = []
        for idx, ds in enumerate(self.descriptors):
            if ds.name in descriptors:
                logger.info(f"Removing descriptor set: {ds.name}")
                to_remove.append(idx)
        for idx in reversed(to_remove):
            self.descriptors[idx].clearFiles()
            self.descriptors.pop(idx)

    def dropEmptySmiles(self):
        """Drop rows with empty SMILES from the data set."""
        self.df.dropna(subset=[self.smilesCol], inplace=True)

    def attachDescriptors(
        self,
        calculator: DescriptorSet,
        descriptors: pd.DataFrame,
        index_cols: list,
    ):
        """Attach descriptors to the data frame.

        Args:
            calculator (DescriptorsCalculator): DescriptorsCalculator object to use for
                descriptor calculation.
            descriptors (pd.DataFrame): DataFrame containing the descriptors to attach.
            index_cols (list): List of column names to use as index.
        """
        self.descriptors.append(
            DescriptorTable(
                calculator,
                self.generateDescriptorDataSetName(calculator),
                descriptors,
                store_dir=self.storeDir,
                n_jobs=self.nJobs,
                overwrite=True,
                key_cols=index_cols,
                chunk_size=self.chunkSize,
                random_state=self.randomState,
                store_format=self.storeFormat,
            )
        )

    def addDescriptors(
        self,
        descriptors: list[DescriptorSet],
        recalculate: bool = False,
        fail_on_invalid: bool = True,
        *args,
        **kwargs,
    ):
        """Add descriptors to the data frame with the given descriptor calculators.

        Args:
            descriptors (list[DescriptorSet]):
                List of `DescriptorSet` objects to use for descriptor
                calculation.
            recalculate (bool):
                Whether to recalculate descriptors even if they are
                already present in the data frame. If `False`, existing descriptors are
                kept and no calculation takes place.
            fail_on_invalid (bool):
                Whether to throw an exception if any molecule
                is invalid.
            *args:
                Additional positional arguments to pass to each descriptor set.
            **kwargs:
                Additional keyword arguments to pass to each descriptor set.
        """
        if recalculate and self.hasDescriptors():
            self.dropDescriptors(descriptors)
        to_calculate = []
        for desc_set, exists in zip(descriptors, self.hasDescriptors(descriptors)):
            if exists:
                logger.warning(
                    f"Molecular descriptors already exist in {self.name}. "
                    "Calculation will be skipped. "
                    "Use `recalculate=True` to overwrite them."
                )
            else:
                to_calculate.append(desc_set)
        # check for invalid molecules if required
        if fail_on_invalid:
            try:
                self.checkMols(throw=True)
            except Exception as exp:
                logger.error(
                    f"Cannot add descriptors to {self.name} because it contains one or "
                    "more invalid molecules. Remove the invalid molecules from your "
                    "data or try to standardize the data set first with "
                    "'standardizeSmiles()'. You can also pass 'fail_on_invalid=False' "
                    "to remove this exception, but the calculation might not be "
                    "successful or correct. See the following list of invalid molecule "
                    "SMILES for more information:"
                )
                logger.error(
                    self.df[~self.checkMols(throw=False)][self.smilesCol].to_numpy()
                )
                raise exp
        # get the data frame with the descriptors
        # and attach it to this table as descriptors
        for calculator in to_calculate:
            df_descriptors = []
            for result in self.processMols(
                calculator, proc_args=args, proc_kwargs=kwargs
            ):
                df_descriptors.append(result)
            df_descriptors = pd.concat(df_descriptors, axis=0)
            df_descriptors[self.indexCols] = None
            df_descriptors.loc[self.df.index, self.indexCols] = self.df[self.indexCols]
            self.attachDescriptors(calculator, df_descriptors, [self.idProp])

    def getDescriptors(self):
        """Get the calculated descriptors as a pandas data frame.

        Returns:
            pd.DataFrame: Data frame containing only descriptors.
        """
        # join_cols = set()
        # for descriptors in self.descriptors:
        #     join_cols.update(set(descriptors.indexCols))
        # join_cols = list(join_cols)
        # ret = self.df[join_cols].copy()
        # ret.reset_index(drop=True, inplace=True)
        ret = pd.DataFrame(index=pd.Index(self.df.index.values, name=self.idProp))
        for descriptors in self.descriptors:
            df_descriptors = descriptors.getDescriptors()
            ret = ret.join(df_descriptors, how="left")
        # ret.set_index(self.df.index, inplace=True)
        # ret.drop(columns=join_cols, inplace=True)
        return ret

    def getDescriptorNames(self):
        """Get the names of the descriptors present for  molecules  in  this data  set.

        Returns:
            list: list of descriptor names.
        """
        names = []
        for ds in self.descriptors:
            names.extend(ds.getDescriptorNames())
        return names

    def hasDescriptors(
        self, descriptors: list[DescriptorSet | str] | None = None
    ) -> bool | list[bool]:
        """Check whether the data frame contains given descriptors.

        Args:
            descriptors (list): list of `DescriptorSet` objects or prefixes of
                descriptors to check for. If `None`,
                all descriptors are checked for and
                a single boolean is returned if any descriptors are found.

        Returns:
            list: list of booleans indicating whether each descriptor is present or not.
        """
        if not descriptors:
            return len(self.getDescriptorNames()) > 0
        else:
            descriptors = [self.generateDescriptorDataSetName(x) for x in descriptors]
            descriptors_in = [x.name for x in self.descriptors]
            ret = []
            for name in descriptors:
                if name in descriptors_in:
                    ret.append(True)
                else:
                    ret.append(False)
            return ret

    @property
    def smiles(self) -> Generator[str, None, None]:
        """Get the SMILES strings of the molecules in the data frame.

        Returns:
            Generator[str, None, None]: Generator of SMILES strings.
        """
        return iter(self.df[self.smilesCol].values)

    def addScaffolds(
        self,
        scaffolds: list[Scaffold],
        add_rdkit_scaffold: bool = False,
        recalculate: bool = False,
    ):
        """Add scaffolds to the data frame.

        A new column is created that contains the SMILES of the corresponding scaffold.
        If `add_rdkit_scaffold` is set to `True`, a new column is created that contains
        the RDKit scaffold of the corresponding molecule.

        Args:
            scaffolds (list): list of `Scaffold` calculators.
            add_rdkit_scaffold (bool): Whether to add the RDKit scaffold of the molecule
                as a new column.
            recalculate (bool): Whether to recalculate scaffolds even if they are
                already present in the data frame.
        """
        for scaffold in scaffolds:
            if not recalculate and f"Scaffold_{scaffold}" in self.df.columns:
                continue
            for scaffolds in self.processMols(scaffold):
                self.df.loc[scaffolds.index, f"Scaffold_{scaffold}"] = scaffolds.values
            if add_rdkit_scaffold:
                PandasTools.AddMoleculeColumnToFrame(
                    self.df,
                    smilesCol=f"Scaffold_{scaffold}",
                    molCol=f"Scaffold_{scaffold}_RDMol",
                )

    def getScaffoldNames(
        self, scaffolds: list[Scaffold] | None = None, include_mols: bool = False
    ):
        """Get the names of the scaffolds in the data frame.

        Args:
            include_mols (bool): Whether to include the RDKit scaffold columns as well.


        Returns:
            list: List of scaffold names.
        """
        all_names = [
            col
            for col in self.df.columns
            if col.startswith("Scaffold_")
            and (include_mols or not col.endswith("_RDMol"))
        ]
        if scaffolds:
            wanted = [str(x) for x in scaffolds]
            return [x for x in all_names if x.split("_", 1)[1] in wanted]
        return all_names

    def getScaffolds(
        self, scaffolds: list[Scaffold] | None = None, include_mols: bool = False
    ):
        """Get the subset of the data frame that contains only scaffolds.

        Args:
            include_mols (bool): Whether to include the RDKit scaffold columns as well.

        Returns:
            pd.DataFrame: Data frame containing only scaffolds.
        """
        names = self.getScaffoldNames(scaffolds, include_mols=include_mols)
        return self.df[names]

    @property
    def hasScaffolds(self):
        """Check whether the data frame contains scaffolds.

        Returns:
            bool: Whether the data frame contains scaffolds.
        """
        return len(self.getScaffoldNames()) > 0

    def createScaffoldGroups(self, mols_per_group: int = 10):
        """Create scaffold groups.

        A scaffold group is a list of molecules that share the same scaffold. New
        columns are created that contain the scaffold group ID and the scaffold group
        size.

        Args:
            mols_per_group (int): Number of molecules per scaffold group.
        """
        scaffolds = self.getScaffolds(include_mols=False)
        for scaffold in scaffolds.columns:
            counts = pd.value_counts(self.df[scaffold])
            mask = counts.lt(mols_per_group)
            name = f"ScaffoldGroup_{scaffold}_{mols_per_group}"
            if name not in self.df.columns:
                self.df[name] = np.where(
                    self.df[scaffold].isin(counts[mask].index),
                    "Other",
                    self.df[scaffold],
                )

    def getScaffoldGroups(self, scaffold_name: str, mol_per_group: int = 10):
        """Get the scaffold groups for a given combination of scaffold and number of
        molecules per scaffold group.

        Args:
            scaffold_name (str): Name of the scaffold.
            mol_per_group (int): Number of molecules per scaffold group.

        Returns:
            list: list of scaffold groups.
        """
        return self.df[
            self.df.columns[
                self.df.columns.str.startswith(
                    f"ScaffoldGroup_{scaffold_name}_{mol_per_group}"
                )
            ][0]
        ]

    @property
    def hasScaffoldGroups(self):
        """Check whether the data frame contains scaffold groups.

        Returns:
            bool: Whether the data frame contains scaffold groups.
        """
        return (
            len([col for col in self.df.columns if col.startswith("ScaffoldGroup_")])
            > 0
        )

    def standardizeSmiles(self, smiles_standardizer, drop_invalid=True):
        """Apply smiles_standardizer to the compounds in parallel

        Args:
            smiles_standardizer (): either `None` to skip the
                standardization, `chembl`, `old`, or a partial function that reads
                and standardizes smiles.
            drop_invalid (bool): whether to drop invalid SMILES from the data set.
                Defaults to `True`. If `False`, invalid SMILES will be retained in
                their original form.

        Raises:
            ValueError: when smiles_standardizer is not a callable or one of the
                predefined strings.
        """
        std_jobs = self.nJobs
        if smiles_standardizer is None:
            return
        if callable(smiles_standardizer):
            try:  # Prevents weird error if the user inputs a lambda function
                pickle.dumps(smiles_standardizer)
            except pickle.PicklingError:
                logger.warning("Standardizer is not pickleable. Will set n_jobs to 1")
                std_jobs = 1
            std_func = smiles_standardizer
        elif smiles_standardizer.lower() == "chembl":
            std_func = chembl_smi_standardizer
        elif smiles_standardizer.lower() == "old":
            std_func = old_standardize_sanitize
        else:
            raise ValueError("Standardizer must be either 'chembl', or a callable")
        if std_jobs == 1:
            std_smi = [std_func(smi) for smi in self.df[self.smilesCol].values]
        else:
            with Pool(std_jobs) as pool:
                std_smi = pool.map(std_func, self.df[self.smilesCol].values)
        self.df[self.smilesCol] = std_smi
        if drop_invalid:
            self.dropInvalids()

    def dropInvalids(self):
        """
        Drops invalid molecules from the data set.

        Returns:
            mask (pd.Series): Boolean mask of invalid molecules in the original
                data set.
        """
        invalid_mask = self.checkMols(throw=False)
        self.df.drop(self.df.index[~invalid_mask], inplace=True)
        invalids = (~invalid_mask).sum()
        if invalids > 0:
            logger.warning(f"Dropped {invalids} invalid molecules from the data set.")
        return ~invalid_mask
