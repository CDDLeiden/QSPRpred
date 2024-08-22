import os
import shutil
from typing import ClassVar, Iterable, Any, Generator, Sized, Callable, Literal

import pandas as pd
from rdkit import Chem

from qsprpred.data.chem.identifiers import ChemIdentifier
from qsprpred.data.chem.matching import SMARTSMatchProcessor
from qsprpred.data.chem.standardizers import ChemStandardizer
from qsprpred.data.processing.mol_processor import MolProcessor
from qsprpred.data.storage.interfaces.chem_store import ChemStore
from qsprpred.data.storage.interfaces.searchable import SMARTSSearchable, PropSearchable
from qsprpred.data.storage.interfaces.stored_mol import StoredMol
from qsprpred.data.storage.tabular.stored_mol import TabularMol
from qsprpred.data.tables.pandas import PandasDataTable
from qsprpred.logs import logger
from qsprpred.utils.interfaces.summarizable import Summarizable
from qsprpred.utils.parallel import ParallelGenerator, MultiprocessingJITGenerator


class TabularStorageBasic(ChemStore, SMARTSSearchable, PropSearchable, Summarizable):
    _notJSON: ClassVar = ChemStore._notJSON + ["_libraries"]

    def __init__(
            self,
            name: str,
            path: str,
            df: pd.DataFrame | None = None,
            smiles_col: str = "SMILES",
            add_rdkit: bool = False,
            overwrite: bool = False,
            save: bool = False,
            standardizer=None,
            identifier=None,
            id_col: str = "ID",
            store_format: str = "pkl",
            chunk_processor: ParallelGenerator = None,
            chunk_size: int | None = None,
            n_jobs: int = 1,
    ) -> None:
        super().__init__()
        if df is not None and smiles_col not in df.columns:
            raise ValueError(
                f"Column '{smiles_col}' not found in the data frame. "
                "Please provide a valid column name for the SMILES representations."
            )
        self.name = name
        self.path = os.path.abspath(os.path.join(path, self.name))
        self.storeFormat = store_format
        self._libraries = dict()
        self.chunkSize = chunk_size
        self.nJobs = n_jobs
        self.chunkProcessor = MultiprocessingJITGenerator(
            n_workers=self.nJobs) if chunk_processor is None else chunk_processor
        self._standardizer = standardizer
        self._identifier = identifier
        if overwrite and os.path.exists(self.metaFile):
            self.clear()
        if not os.path.exists(self.metaFile):
            columns = [
                self.idProp,
                self.smilesProp,
            ]
            if df is None:
                df = pd.DataFrame(columns=columns)
            self.add_library(
                f"{self.name}_library",
                df,
                smiles_col,
                id_col,
                add_rdkit,
                store_format,
                save=save,
            )
        else:
            self.reload()

    @property
    def libsPath(self):
        return os.path.join(self.path, "libs")

    @property
    def smilesProp(self) -> str:
        return "SMILES"

    @property
    def originalSmilesProp(self) -> str:
        return "original_smiles"

    @property
    def idProp(self) -> str:
        return "ID"

    @property
    def chunkSize(self) -> int:
        return self._chunkSize

    @chunkSize.setter
    def chunkSize(self, value: int | None):
        self._chunkSize = value
        for lib in self._libraries.values():
            lib.chunkSize = value
        if self._chunkSize is None:
            self._chunkSize = len(self)

    @property
    def nJobs(self):
        return self._nJobs

    @nJobs.setter
    def nJobs(self, value: int | None):
        self._nJobs = value if value is not None and value > 0 else os.cpu_count()
        for lib in self._libraries.values():
            lib.nJobs = value

    def add_library(
            self,
            name: str,
            df,
            smiles_col: str = "SMILES",
            id_col: str = "ID",
            add_rdkit=False,
            store_format="pkl",
            save=False,
    ):
        """
        Reads molecules from a file and adds standardized SMILES to the store.

        :param path: path to the library file
        :param smiles_col: name of the column containing the SMILES

        :return: `StoredMol` instance of the added molecule
        """
        if len(df) == 0 and len(self._libraries) > 0:
            logger.warning(
                "No valid or unique molecules found in the data frame."
                "Nothing will be added."
            )
            return
        if add_rdkit:
            raise NotImplementedError("Adding RDKit molecules is not yet implemented.")
        if name in self._libraries:
            raise ValueError(
                f"Library with name {name} already exists in the store. "
                f"Remove it first or use a different name for the new library."
            )
        df[self.smilesProp] = df[smiles_col]
        df[self.originalSmilesProp] = df[smiles_col]
        if smiles_col != self.smilesProp:
            del df[smiles_col]
        pd_table = PandasDataTable(
            name=name,
            df=df,
            store_dir=self.libsPath,
            overwrite=False,
            n_jobs=self.nJobs,
            chunk_size=self.chunkSize,
            autoindex_name=self.idProp,
            index_cols=[
                id_col] if self._identifier is None and id_col in df.columns else None,
            store_format=store_format,
            parallel_generator=self.chunkProcessor,
        )
        # apply standardizer
        if self._standardizer and len(pd_table) > 0:
            self._apply_standardizer_to_library(pd_table)
        self._drop_invalids_from_table(pd_table)
        # create IDs for compounds
        if self._identifier:
            # replace the default ID with own identifier if requested
            ids = self._apply_identifier_to_library(pd_table)
        else:
            ids = pd_table.getProperty(self.idProp)
        # resolve duplicates within the table by taking only the first occurrence
        self._remove_duplicates_from_table(pd_table, ids)
        # FIXME: add RDKit molecules here if requested
        self._remove_duplicates_from_libs(pd_table, pd_table.getProperty(self.idProp))
        self._libraries[pd_table.name] = pd_table
        # make sure all properties of the new library are present in all libraries
        props = set()
        for lib in self._libraries.values():
            props.update(lib.getProperties())
        for lib in self._libraries.values():
            for prop in props:
                if prop not in lib.getProperties():
                    lib.addProperty(prop, [None] * len(lib))
        if save:
            self.save()

    def applyIdentifier(self, identifier: ChemIdentifier):
        """
        Apply an identifier to the SMILES in the store.

        Args:
            identifier (ChemIdentifier): Identifier to apply to the SMILES.
        """
        self._identifier = identifier
        for lib in self._libraries.values():
            ids = self._apply_identifier_to_library(lib)
            self._remove_duplicates_from_libs(lib, ids)

    def applyStandardizer(self, standardizer: ChemStandardizer):
        """
        Apply a standardizer to the SMILES in the store.

        Args:
            standardizer (ChemStandardizer): Standardizer to apply to the SMILES.
        """
        self._standardizer = standardizer
        for lib in self._libraries.values():
            self._apply_standardizer_to_library(lib)
            self._drop_invalids_from_table(lib)
        if self._identifier:
            self.applyIdentifier(self.identifier)

    def _drop_invalids_from_table(self, pd_table: PandasDataTable):
        pd_table.dropEmptyProperties([self.smilesProp])

    @classmethod
    def fromDF(cls, df: pd.DataFrame, *args, name: str | None = None,
               **kwargs) -> "TabularStorageBasic":
        """
        Create a new instance from a pandas DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to create the instance from.
            name (str): Name of the new instance. Defaults to the name of the DataFrame.
            *args: Additional arguments to pass to the constructor.
            **kwargs: Additional keyword arguments to pass to the constructor.

        Returns:
            PropertyStorage: New instance created from the DataFrame.
        """
        name = name or repr(df)
        return cls(df=df, *args, name=name, **kwargs)

    @staticmethod
    def apply_standardizer_to_data_frame(
            df: pd.DataFrame,
            smiles_prop: str,
            standardizer: ChemStandardizer
    ):
        smiles = df[smiles_prop].values
        output = []
        for i, smi in enumerate(smiles):
            try:
                standardized = standardizer(smi)[0]
                if standardized is None:
                    raise ValueError(f"Standardizer {standardizer} returned None.")
            except Exception as e:
                logger.error(
                    f"Error ({e}) standardizing SMILES: {smi}. "
                    f"Molecule will not be added."
                )
                standardized = None
            output.append((df.index[i], standardized, smi))
        return output

    def _remove_duplicates_from_table(self, pd_table: PandasDataTable, ids: pd.Series):
        """
        Remove duplicates from the table using a list of identifiers.
        If duplicates are found, the first occurrence is kept. The index
        property of the table is also updated with the ids provided.

        Args:
            pd_table:
                The table to remove duplicates from.
            ids:
                A list of values to use as identifiers. Duplicates are
                identified based on these values.
        """
        duplicated = ids.duplicated(keep="first")
        if sum(duplicated) > 0:
            logger.warning(
                f"Duplicated identifiers found in {pd_table}."
                f"Dropping duplicates, keeping only the first occurrence."
                f"Molecules dropped (ID, original SMILES): {
                ids[duplicated].index.tolist(),
                pd_table.getProperty(
                    self.originalSmilesProp,
                    duplicated[duplicated].index
                )
                }"
            )
        pd_table.dropEntries(duplicated[duplicated].index, ignore_missing=True)
        ids = ids[~duplicated]
        pd_table.addProperty(self.idProp, ids, ids[~duplicated].index)

    def _remove_duplicates_from_libs(self, pd_table: PandasDataTable, ids: pd.Series):
        for lib in self._libraries.values():
            overlap = tuple(set(lib.getProperty(self.idProp)) & set(ids))
            if len(overlap) > 0:
                logger.warning(
                    f"Duplicated identifiers found in library: {lib}."
                    f"Dropping duplicates from: {pd_table}."
                    f"Molecules dropped (ID, original SMILES): {
                    pd_table.getProperty(self.idProp, overlap).tolist(),
                    pd_table.getProperty(
                        self.originalSmilesProp,
                        overlap
                    )
                    }"
                )
                pd_table.dropEntries(overlap, ignore_missing=True)

    @staticmethod
    def _apply_identifier_to_data_frame(
            df: pd.DataFrame,
            smiles_col: str,
            id_prop: str,
            identifier: Callable[[str], str]
    ) -> pd.Series:
        identifiers = df[smiles_col].apply(identifier)
        ids = df[id_prop]
        return pd.Series(identifiers, index=ids)

    def addEntries(self, ids: list[str], props: dict[str, list],
                   raise_on_existing: bool = True, library: str | None = None):
        lib = self._libraries[library] if library else self._libraries[self.name]
        lib.addEntries(ids, props, raise_on_existing)

    def add_mols(
            self,
            smiles: Iterable[str],
            props: dict[str, list] | None = None,
            library: str | None = None,
            raise_on_existing: bool = True,
            add_rdkit: bool = False,
            store_format: str = "pkl",
            save: bool = False,
            chunk_size: int | None = None,
            chunk_processor: ParallelGenerator | None = None,
    ) -> list[TabularMol]:
        """
        Add a molecule to the store using its raw SMILES. The SMILES will be standardized and an identifier will be
        calculated.

        :param smiles: SMILES of the molecule to add
        :param mol_id: identifier of the molecule to add
        :param metadata: additional metadata to store with the molecule
        :ligprep_metadata: metadata from the ligprep process
        :param sdf_path: path to the SDF file containing the molecule processed via ligprep
        :param library: name of the library the molecule belongs to
        :param raise_on_existing: whether to raise an error if the molecule already exists in the store
        :param update_existing: whether to update the existing molecule if it already exists in the store

        :return: `StoredMol` instance of the added molecule

        :raises ValueError: if the molecule already exists in the store
        """
        data = {self.smilesProp: smiles}
        if props:
            data.update(props)
        df = pd.DataFrame(data)
        library = library or f"{self.name}_library"
        if library not in self._libraries:
            self.add_library(
                name=library,
                df=df,
                smiles_col=self.smilesProp,
                add_rdkit=add_rdkit,
                store_format=store_format,
                save=save,
            )
        else:
            random_temp_name = f"{library}_temp"
            self.add_library(
                name=random_temp_name,
                df=df,
                smiles_col=self.smilesProp,
                add_rdkit=False,
                store_format="pkl",
                save=False,
            )
            if library in self._libraries:
                self._libraries[library].addEntries(
                    self._libraries[random_temp_name].getProperty(self.idProp),
                    self._libraries[random_temp_name].getDF().to_dict(orient="list"),
                    raise_on_existing,
                )
                self._libraries.pop(random_temp_name)
        if len(df) == 0:
            logger.warning(
                "No new or valid molecules detected in the list of SMILES."
                "Nothing was be added to the store."
            )
            return []
        return [self.get_mol(x) for x in
                self._libraries[library].getProperty(self.idProp) if x in df.index]

    def hasProperty(self, name: str) -> bool:
        return name in self.getProperties()

    def __getstate__(self):
        o_dict = super().__getstate__()
        os.makedirs(self.libsPath, exist_ok=True)
        o_dict["_libraries"] = {}
        for lib in self._libraries.values():
            o_dict["_libraries"][lib.name] = os.path.relpath(lib.save(), self.libsPath)
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        if hasattr(self, "_json_main"):
            self.path = os.path.abspath(os.path.dirname(self._json_main))
        self._libraries = {}
        for name, lib in state["_libraries"].items():
            lib = os.path.join(self.libsPath, lib)
            self._libraries[name] = PandasDataTable.fromFile(lib)

    def save(self):
        """Save the whole storage to disk."""
        os.makedirs(self.path, exist_ok=True)
        return self.toFile(self.metaFile)

    def processMols(
            self,
            processor: MolProcessor,
            proc_args: Iterable[Any] | None = None,
            proc_kwargs: dict[str, Any] | None = None,
            mol_type: Literal["smiles", "mol", "rdkit"] = "mol",
            add_props: Iterable[str] | None = None,
            chunk_processor: ParallelGenerator | None = None,
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
            mol_type (str, optional):
                Type of molecule to send to the processor. Can be 'smiles', 'mol', or
                'rdkit'. Defaults to 'mol', which implies `TabularMol` objects.
            chunk_processor (ParallelGenerator, optional):
                The parallel generator to use for processing. If not specified,
                `self.chunkProcessor` is used.

        Returns:
            Generator:
                A generator that yields the results of the supplied processor on
                the chunked molecules from the data set.
        """
        proc_args = proc_args or ()
        proc_kwargs = proc_kwargs or {}
        if add_props is None:
            add_props = self.getProperties()
        else:
            add_props = list(add_props)
        add_props = add_props + list(processor.requiredProps)
        chunk_processor = chunk_processor or self.chunkProcessor
        for prop in add_props:
            if prop not in self.getProperties():
                raise ValueError(
                    f"Cannot apply function '{processor}' to {self.name} because "
                    f"it requires the property '{prop}', which is not present in the "
                    "data set."
                )
        for result in self.apply(
                processor,
                func_args=proc_args,
                func_kwargs=proc_kwargs,
                on_props=add_props,
                chunk_type=mol_type,
                chunk_processor=chunk_processor,
                no_parallel=not processor.supportsParallel,
        ):
            yield result

    def getProperty(self, name: str, ids: list[str] | None = None) -> pd.Series:
        # find the libraries that contain the specified ids if any
        subsets = []
        for lib in self._libraries.values():
            subset = lib.getProperty(name, ids, ignore_missing=True)
            if len(subset) > 0:
                subsets.append(subset)
        return pd.concat(subsets) if len(subsets) > 0 else pd.Series(
            index=pd.Index([], name=self.idProp), name=name)

    def getProperties(self) -> list[str]:
        ret = set()
        for lib in self._libraries.values():
            ret.update(lib.getProperties())
        return list(ret)

    def addProperty(self, name: str, data: Sized, ids: list[str] | None = None):
        for lib in self._libraries.values():
            lib.addProperty(name, data, ids, ignore_missing=True)

    def removeProperty(self, name: str):
        for lib in self._libraries.values():
            lib.removeProperty(name)

    def getSubset(
            self, subset: list[str],
            ids: list[str] | None = None,
            name: str | None = None
    ) -> "TabularStorageBasic":
        name = name or f"{self.name}_subset"
        if self.smilesProp not in subset:
            subset = [self.smilesProp, *subset]
        subsets = []
        for lib in self._libraries.values():
            subsets.append(lib.getSubset(subset, ids, ignore_missing=True).getDF())
        ret = pd.concat(subsets)
        return self.fromDF(
            ret,
            name=name,
            path=self.path,
            overwrite=True,
            save=False,
            store_format=self.storeFormat,
            chunk_processor=self.chunkProcessor,
            standardizer=None,
            identifier=None,
            id_col=self.idProp,
            smiles_col=self.smilesProp,
            chunk_size=self.chunkSize,
            n_jobs=self.nJobs,
        )

    def getDF(self) -> pd.DataFrame:
        if len(self) > 0:
            return pd.concat([lib.getDF() for lib in self._libraries.values()])
        else:
            return pd.DataFrame(index=pd.Index([], name=self.idProp),
                                columns=self.getProperties())

    def reload(self):
        self.__dict__.update(self.fromFile(self.metaFile).__dict__)

    def clear(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    @property
    def metaFile(self) -> str:
        return os.path.join(self.path, "meta.json")

    def apply(
            self,
            func: callable,
            func_args: list | None = None,
            func_kwargs: dict | None = None,
            on_props: tuple[str, ...] | None = None,
            chunk_type: Literal["mol", "smiles", "rdkit", "df"] = "mol",
            chunk_processor: ParallelGenerator | None = None,
            no_parallel: bool = False,
    ) -> Generator[Iterable[Any], None, None]:
        chunk_processor = chunk_processor or self.chunkProcessor
        if self.nJobs > 1 and not no_parallel:
            for result in chunk_processor(
                    self.iterChunks(self.chunkSize, chunk_type=chunk_type,
                                    on_props=on_props),
                    func,
                    *func_args,
                    **func_kwargs,
            ):
                yield result
        else:
            # do not use the parallel generator if n_jobs is 1
            for chunk in self.iterChunks(self.chunkSize, chunk_type=chunk_type,
                                         on_props=on_props):
                yield func(chunk, *func_args, **func_kwargs)

    def searchOnProperty(
            self,
            prop_name: str,
            values: list[float | int | str],
            exact=False,
            name: str | None = None,
    ) -> "TabularStorageBasic":
        """Search in this table using a property name and a list of values.
        It is assumed that the property is searchable with string matching
        or direct comparison if a number is supplied.
        Note that the types of the query list need to be consistent.
        Otherwise, a `ValueError` will be raised.

        In the case of string comparison,
        if 'exact' is `False`, the
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
        Raises:
            ValueError: If the types of the query list are not consistent.
        """
        assert len(values) > 0, "No values provided for the search."
        # check type consistency
        value_type = type(values[0])
        if not all(isinstance(x, value_type) for x in values):
            raise ValueError(
                "Inconsistent types in the query list. "
                "All values must be of the same type."
            )
        name = name or f"{self.name}_{prop_name}_searched"
        if value_type is str:
            prop = self.getProperty(prop_name)
            mask = [False] * len(prop)
            for value in values:
                mask = (
                    mask | (prop.str.contains(value))
                    if not exact
                    else mask | (prop == value)
                )
            matches = self.getSubset(
                self.getProperties(),
                ids=self.getProperty(self.idProp)[mask],
                name=name,
            )
            return matches
        elif value_type in (int, float):
            prop = self.getProperty(prop_name)
            mask = [False] * len(prop)
            for value in values:
                mask = mask | (prop == value)
            matches = self.getSubset(
                self.getProperties(),
                ids=self.getProperty(self.idProp)[mask],
                name=name,
            )
            return matches

    @staticmethod
    def _apply_match_function(
            iterable: Iterable[StoredMol],
            match_function: Callable[[Chem.Mol, list[str], ...], bool],
            *args: list[str],
            **kwargs: dict[str, Any],
    ):
        res = []
        for mol in iterable:
            rd_mol = mol.as_rd_mol()
            res.append(
                match_function(rd_mol, *args, **kwargs)
            )
        return res

    def searchWithSMARTS(
            self,
            patterns: list[str],
            operator: Literal["or", "and"] = "or",
            use_chirality: bool = False,
            name: str | None = None,
            match_function: MolProcessor | None = None,
    ) -> "TabularStorageBasic":
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
        match_function = match_function or SMARTSMatchProcessor()
        results = []
        for result in self.processMols(
                match_function,
                proc_args=(patterns, operator, use_chirality),

        ):
            results.append(result)
        results = pd.concat(results)
        results = results[results["match"]]
        return self.getSubset(
            self.getProperties(),
            ids=results.index.values,
            name=name,
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
            "n_mols": [sum(len(lib) for lib in self._libraries.values())],
            "n_libs": [len(self._libraries)],
            "standardizer": [self._standardizer.__class__.__name__],
            "identifier": [self._identifier.__class__.__name__],
            "libraries": [",".join(self._libraries.keys())],
        }
        return pd.DataFrame(summary)

    @property
    def standardizer(self) -> ChemStandardizer:
        return self._standardizer

    @property
    def identifier(self):
        return self._identifier

    @property
    def nLibs(self):
        return len(self._libraries)

    def get_mol(self, mol_id) -> TabularMol:
        smiles = self.getProperty(self.smilesProp, [mol_id])
        if len(smiles) == 0:
            raise ValueError(f"Molecule with ID {mol_id} not found.")
        props = None
        for lib in self._libraries.values():
            if mol_id in lib:
                props = {prop: lib.getProperty(prop, [mol_id])[0] for prop in
                         lib.getProperties()}
                break
        return TabularMol(mol_id, smiles[0], props=props)

    def remove_mol(self, mol_id):
        """
        Remove a molecule from the store.
        """
        for lib in self._libraries.values():
            lib.dropEntries([mol_id], ignore_missing=True)

    def get_mol_ids(self) -> tuple[str, ...]:
        """
        Returns a set of all molecule IDs in the store.
        Good for checking possible overlaps between stores.
        """
        ids = []
        for lib in self._libraries.values():
            ids.extend(list(lib.getProperty(self.idProp)))
        return tuple(ids)

    def get_mol_count(self) -> int:
        return sum(len(lib) for lib in self._libraries.values())

    def iterChunks(
            self,
            size=1000,
            on_props: Iterable[str] | None = None,
            chunk_type: Literal["mol", "smiles", "rdkit", "df"] = "mol",
    ) -> Generator[list[StoredMol | str | Chem.Mol | pd.DataFrame], None, None]:
        on_props = on_props or self.getProperties()
        for lib in self._libraries.values():
            for chunk in lib.iterChunks(size, on_props=on_props):
                chunk_converters = {
                    "df": self._convert_chunk_df,
                    "mol": self._convert_chunk_mol,
                    "smiles": self._convert_chunk_smiles,
                    "rdkit": self._convert_chunk_rdkit,
                }
                yield chunk_converters[chunk_type](chunk, on_props)

    def _convert_chunk_df(self, chunk, on_props):
        return chunk[list({self.idProp, self.smilesProp, *on_props})]

    def _convert_chunk_mol(self, chunk, on_props):
        ids = chunk[self.idProp]
        smiles = chunk[self.smilesProp]
        props = {prop: chunk[prop] for prop in on_props}
        mols = []
        for idx, _id in enumerate(ids):
            mol_props = {prop: props[prop][idx] for prop in on_props} if props else None
            mols.append(TabularMol(_id, smiles[idx], props=mol_props))
        return mols

    def _convert_chunk_smiles(self, chunk, on_props):
        return chunk[self.smilesProp]

    def _convert_chunk_rdkit(self, chunk, on_props):
        mols = []
        for idx, mol in enumerate(chunk[self.smilesProp]):
            mol = Chem.MolFromSmiles(mol)
            for prop in on_props:
                mol.SetProp(prop, str(chunk[prop].iloc[idx]))
            mols.append(mol)
        return mols

    def iter_mols(self) -> Generator[TabularMol, None, None]:
        for chunk in self.iterChunks():
            for mol in chunk:
                yield mol

    def dropEntries(self, ids: tuple[str, ...]):
        for lib in self._libraries.values():
            lib.dropEntries(ids, ignore_missing=True)

    def _apply_identifier_to_library(self, pd_table):
        ids = []
        for chunk in pd_table.apply(
                self._apply_identifier_to_data_frame,
                func_args=(self.smilesProp, self.idProp, self._identifier),
                on_props=(self.smilesProp, self.idProp),
                as_df=True,
        ):
            ids.append(chunk)
        ids = pd.concat(ids) if len(ids) > 0 else pd.Series(
            index=pd_table.getProperty(self.idProp))
        return ids

    def _apply_standardizer_to_library(self, pd_table):
        output = []
        for chunk in pd_table.apply(
                self.apply_standardizer_to_data_frame,
                func_args=(self.smilesProp, self._standardizer),
                on_props=(self.smilesProp, self.idProp),
                as_df=True,
        ):
            output.extend(chunk)
        pd_table.addProperty(
            self.smilesProp,
            [x[1] for x in output],  # standardized SMILES
            [x[0] for x in output]  # IDs
        )
