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

    def __init__(
            self,
            protein_id: str,
            sequence: str | None = None,
            parent: Optional["TabularProtein"] = None,
            props: dict[str, Any] | None = None,
            representations: Iterable["TabularProtein"] | None = None,
    ) -> None:
        """
        Create a new protein instance.

        :param parent: parent protein
        :param protein_id: identifier of the protein
        :param sequence: sequence of the protein
        """
        self._parent = parent
        self._id = protein_id
        self._sequence = sequence
        self._props = props
        self._representations = representations

    @property
    def id(self) -> str:
        return self._id

    @property
    def sequence(self) -> str | None:
        return self._sequence

    @property
    def props(self) -> dict[str, Any] | None:
        return self._props

    def as_pdb(self) -> str | None:
        return self._props["pdb"] if "pdb" in self._props else None

    def as_fasta(self) -> str | None:
        return self._props["fasta"] if "fasta" in self._props else None

    @property
    def representations(self) -> Iterable["TabularProtein"]:
        return self._representations


class TabularProteinStorage(ProteinStorage, PandasDataTable):

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
            autoindex_name: str = "ID",
            random_state: int | None = None,
            store_format: str = "pkl",
            parallel_generator: ParallelGenerator | None = None,
    ):
        super().__init__(
            name,
            df if df is not None else pd.DataFrame(columns=[sequence_col,
                                                            *index_cols] if index_cols else [
                sequence_col,
                autoindex_name]),
            store_dir,
            overwrite,
            index_cols,
            n_jobs,
            chunk_size,
            autoindex_name,
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
        return self._sequenceCol

    def add_protein(self, protein: TabularProtein, raise_on_existing=True):
        self.addEntries(
            [protein.id],
            {prop: [val] for prop, val in protein.props},
            raise_on_existing
        )

    def _make_proteins_from_chunk(self, df: pd.DataFrame) -> list[TabularProtein]:
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
        ret = []
        for chunk in self.iterChunks(len(self)):
            ret.extend(self._make_proteins_from_chunk(chunk))
        return ret

    def getProtein(self, protein_id: str) -> TabularProtein:
        df = self.getDF()
        protein = df[df[self.idProp] == protein_id]
        if protein.empty:
            raise ValueError(f"Protein {protein_id} not found.")
        return self._make_proteins_from_chunk(protein)[0]
