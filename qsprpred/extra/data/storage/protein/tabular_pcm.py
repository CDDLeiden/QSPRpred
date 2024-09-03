from typing import Any, Iterable, Optional, Callable

import pandas as pd

from qsprpred.data.tables.pandas import PandasDataTable
from qsprpred.extra.data.storage.protein.interfaces.protein_storage import \
    ProteinStorage
from qsprpred.extra.data.storage.protein.interfaces.storedprotein import StoredProtein
from qsprpred.logs import logger
from qsprpred.utils.parallel import ParallelGenerator
from qsprpred.utils.serialization import function_as_string, function_from_string


class TabularProtein(StoredProtein):
    """A protein object that is stored in a tabular format.
    
    Attributes:
        id (str): id of the protein
        sequence (str): sequence of the protein
        props (dict[str, Any]): properties of the protein
        representations (Iterable[TabularProtein]): representations of the protein
    """

    def __init__(
            self,
            protein_id: str,
            sequence: str | None = None,
            parent: Optional["TabularProtein"] = None,
            props: dict[str, Any] | None = None,
            representations: Iterable["TabularProtein"] | None = None,
    ) -> None:
        """Create a new protein instance.
        
        Args:
            protein_id (str): identifier of the protein
            sequence (str): sequence of the protein
            parent (TabularProtein): parent protein
            props (dict[str, Any]): properties of the protein
            representations (Iterable[TabularProtein]): representations of the protein
        """
        self._parent = parent
        self._id = protein_id
        self._sequence = sequence
        self._props = props
        self._representations = representations

    @property
    def id(self) -> str:
        """Get the id of the protein."""
        return self._id

    @property
    def sequence(self) -> str | None:
        """Get the sequence of the protein."""
        return self._sequence

    @property
    def props(self) -> dict[str, Any] | None:
        """Get the properties of the protein."""
        return self._props

    def as_pdb(self) -> str | None:
        """Return the protein as a PDB file."""
        return self._props["pdb"] if "pdb" in self._props else None

    def as_fasta(self) -> str | None:
        """Return the protein as a FASTA file."""
        return self._props["fasta"] if "fasta" in self._props else None

    @property
    def representations(self) -> Iterable["TabularProtein"]:
        """Get all representations of the protein."""
        return self._representations


class TabularProteinStorage(ProteinStorage, PandasDataTable):
    """A storage class for proteins stored in a tabular format.
    
    Attributes:
        sequenceCol (str): name of the column that contains all protein sequences
        proteinSeqProvider (Callable): function that provides protein
        sequenceProp (str): name of the property that contains all protein sequences
        proteins (Iterable[TabularProtein]): all proteins in the store
    """

    def __init__(
            self,
            name: str,
            df: pd.DataFrame | None = None,
            sequence_col: str = "Sequence",
            sequence_provider: Optional[Callable] = None,
            store_dir: str = ".",
            overwrite: bool = False,
            index_cols: list[str] | None = None,
            n_jobs: int = 1,
            chunk_size: int | None = None,
            protein_col: str = "accession",
            random_state: int | None = None,
            store_format: str = "pkl",
            parallel_generator: ParallelGenerator | None = None,
    ):
        """Create a new protein storage instance.	
        
        Args:
            name (str): name of the storage
            df (pd.DataFrame): data frame containing the proteins
            sequence_col (str): name of the column that contains all protein sequences
            sequence_provider (Callable): function that provides protein
            store_dir (str): directory to store the data
            overwrite (bool): overwrite the existing data
            index_cols (list[str]): columns to use as index
            n_jobs (int): number of parallel jobs
            chunk_size (int): size of the chunks
            protein_col (str): name of the column that contains the protein ids
            random_state (int): random state
            store_format (str): format to store the data
            parallel_generator (ParallelGenerator): parallel generator
        """
        super().__init__(
            name,
            df if df is not None else pd.DataFrame(columns=[sequence_col,
                                                            *index_cols] if index_cols else [
                sequence_col,
                protein_col]),
            store_dir,
            overwrite,
            index_cols or [protein_col],
            n_jobs,
            chunk_size,
            protein_col,
            random_state,
            store_format,
            parallel_generator,
        )
        self._sequenceCol = sequence_col
        self.proteinSeqProvider = sequence_provider
        if self.proteinSeqProvider is not None:
            self.getPCMInfo()
        else:
            assert self.sequenceProp in self.getProperties()

    def __getstate__(self):
        o_dict = super().__getstate__()
        if self.proteinSeqProvider:
            o_dict["proteinSeqProvider"] = function_as_string(self.proteinSeqProvider)
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        if self.proteinSeqProvider and type(self.proteinSeqProvider) is str:
            try:
                self.proteinSeqProvider = function_from_string(self.proteinSeqProvider)
            except Exception as e:
                logger.warning(
                    "Failed to load protein sequence provider from metadata. "
                    f"The function object could not be recreated from the code. "
                    f"\nError: {e}"
                    f"\nDeserialized Code: {self.proteinSeqProvider}"
                    f"\nSetting protein sequence provider to `None` for now."
                )
                self.proteinSeqProvider = None

    def getPCMInfo(self) -> tuple[dict[str, str], dict]:
        """Return a dictionary of protein sequences for the proteins
        in the data frame and the additional metadata separately.

        Returns:
            sequences (dict): Dictionary of protein sequences.
        """
        if self.proteinSeqProvider is not None:
            mapping, props = self.proteinSeqProvider(set(self.getProperty(self.idProp)))
            assert set(mapping.keys()) == set(self.getProperty(self.idProp)), (
                "Protein sequence provider did not return sequences "
                "for all proteins. Could"
                " not get sequences for the following proteins: "
                f"{set(self.getProperty(self.idProp)) - set(mapping.keys())}"
            )
            for protein_id in mapping:
                self.addProperty(
                    self.sequenceProp,
                    [mapping[protein_id]],
                    [protein_id]
                )
                if props:
                    for prop in props[protein_id]:
                        self.addProperty(prop, [props[protein_id][prop]], [protein_id])
            return mapping, props
        else:
            return {
                key: seq for key, seq in zip(
                    self.getProperty(self.idProp),
                    self.getProperty(self.sequenceProp)
                )
            }, {  # return all remaining props as metadata
                prop: self.getProperty(prop)
                for prop in self.getProperties()
                if prop not in [self.idProp, self.sequenceProp]
            }

    @property
    def sequenceProp(self) -> str:
        """Get the name of the property that contains all protein sequences."""	
        return self._sequenceCol

    def add_protein(self, protein: TabularProtein, raise_on_existing=True):
        """Add a protein to the store.
        
        Args:
            protein (TabularProtein): protein sequence
            raise_on_existing (bool): 
                raise an exception if the protein already exists in the store
        """
        self.addEntries(
            [protein.id],
            {prop: [val] for prop, val in protein.props},
            raise_on_existing
        )

    def _make_proteins_from_chunk(self, df: pd.DataFrame) -> list[TabularProtein]:
        """Create a list of proteins from a chunk of the data frame.
        
        Args:
            df (pd.DataFrame): chunk of the data frame
            
        Returns:
            list[TabularProtein]: list of proteins
        """
        ids = df[self.idProp].values
        sequences = df[self.sequenceProp].values
        props = df.columns.difference([self.idProp, self.sequenceProp])
        return [
            TabularProtein(
                protein_id=ids[i],
                sequence=sequences[i],
                props={prop: df[prop].values[i] for prop in props},
            )
            for i in range(len(df))
        ]

    @property
    def proteins(self) -> list[TabularProtein]:
        """Get all proteins in the store.
        
        Returns:
            list[TabularProtein]: list of proteins
        """
        ret = []
        for chunk in self.iterChunks(len(self)):
            ret.extend(self._make_proteins_from_chunk(chunk))
        return ret

    def getProtein(self, protein_id: str) -> TabularProtein:
        """Get a protein from the store using its name.	
        
        Args:
            protein_id (str): name of the protein to search
        
        Returns:
            TabularProtein: instance of `Protein`
            
        Raises:
            ValueError: if the protein is not found
        """
        df = self.getDF()
        protein = df[df[self.idProp] == protein_id]
        if protein.empty:
            raise ValueError(f"Protein {protein_id} not found.")
        return self._make_proteins_from_chunk(protein)[0]
