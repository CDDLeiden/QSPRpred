"""
Various implementations of multiple sequence alignment (MSA).

The MSA providers are used to align sequences for protein descriptor calculation. This
is required for the calculation of descriptors that are based on sequence alignments,
such as `ProDec`.
"""

import shutil
from abc import ABC, abstractmethod

import Bio
import Bio.SeqIO as Bio_SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline, MafftCommandline

from qsprpred.utils.serialization import FileSerializable, JSONSerializable


class MSAProvider(FileSerializable, ABC):
    """Interface for multiple sequence alignment providers.

    This interface defines how calculation and storage of
    multiple sequence alignments (MSAs) is handled.
    """

    @abstractmethod
    def __call__(
        self, sequences: dict[str:str] | None = None, **kwargs
    ) -> dict[str:str] | None:
        """
        Aligns the sequences and returns the alignment.

        Args:
            sequences:
                dictionary of sequences to align, keys are sequence IDs
                (i.e. accession keys) and values are sequences themselves.
                If no sequences are passed, the current alignment is returned.
            **kwargs:
                additional arguments to be passed to the alignment algorithm,
                can also just be metadata to be stored with the alignment

        Returns:
            alignment (dict[str:str] | None):
                the alignment, `None` if no sequences are and
                current alignment is `None` (see `current` property)
        """

    @property
    @abstractmethod
    def current(self) -> dict[str:str] | None:
        """The current alignment.

        Returns the current alignment as a dictionary where keys are sequence IDs
        as `str` and values are aligned sequences as `str`. The values are of the
        same length and contain gaps ("-") where necessary. If the alignment
        is not yet calculated, `None` is returned.

        Returns:
            alignment (dict[str:str] | None): the current alignment
        """


class BioPythonMSA(MSAProvider, JSONSerializable, ABC):
    """
    Common functionality for MSA providers using BioPython command line wrappers.

    Attributes:
        outDir: directory to save the alignment to
        fname: file name of the alignment file
        cache: cache of alignments performed so far by the provider
    """

    def __init__(self, out_dir: str = ".", fname: str = "alignment.aln-fasta.fasta"):
        """Initializes the MSA provider.

        Args:
            out_dir (str): directory to save the alignment to
            fname (str): file name of the alignment file
        """
        if not self.checkTool():
            raise RuntimeError(
                f"The tool commnad '{self.cmd}' " f"was not found. Please install it."
            )
        self.outDir = out_dir
        self.fName = fname
        self.cache = {}
        self._current = None

    @property
    @abstractmethod
    def cmd(self) -> str:
        """The command that runs the alignment algorithm.

        Returns:
            cmd (str): the command to run the alignment algorithm
        """

    def getFromCache(self, target_ids: list[str]) -> dict[str:str] | None:
        """
        Gets the alignment from the cache if it exists for a `list` of sequence IDs.
        Args:
            target_ids (list[str]):
                list of sequence IDs to get the alignment for,

        Returns:
            alignment (dict[str:str] | None):
                the alignment if it exists in the cache, `None` otherwise

        """
        key = "~".join(target_ids)
        if key in self.cache:
            return self.cache[key]

    def saveToCache(self, target_ids: list[str], alignment: dict[str:str]):
        """Saves the alignment to the cache for a `list` of sequence IDs.

        Args:
            target_ids (list[str]):
                list of sequence IDs to save the alignment for
            alignment (dict[str:str]):
                the alignment to save

        """
        key = "~".join(target_ids)
        self.cache[key] = alignment

    @property
    def current(self):
        return self._current

    def parseSequences(self, sequences: dict[str, str], **kwargs) -> tuple[str, int]:
        """Create object with sequences and the passed metadata.

        Saves the sequences to a file that will serve
        as input to the command line tools.

        Args:
            sequences (dict[str,str]): sequences to align
            **kwargs: metadata to be stored with the alignment

        Returns:
            sequences_path (str): path to the file with the sequences
            n_sequences (int): number of sequences in the file
        """
        records = []
        target_ids = []
        for target_id in sequences:
            records.append(
                Bio_SeqIO.SeqRecord(
                    seq=Bio.Seq.Seq(sequences[target_id]),
                    id=target_id,
                    name=target_id,
                    description=" ".join([f"{k}={v}" for k, v in kwargs.items()]),
                )
            )
            target_ids.append(target_id)
        sequences_path = f"{self.outDir}/sequences.fasta"
        # Write sequences as .fasta file
        return sequences_path, Bio_SeqIO.write(records, sequences_path, "fasta")

    def parseAlignment(self, sequences: dict[str:str]) -> dict[str, str]:
        """
        Parse the alignment from the output file of the alignment algorithm.

        Args:
            sequences: the original dictionary of sequences that were aligned

        Returns:
            the aligned sequences mapped to their IDs
        """

        alignment = dict(
            zip(
                sequences.keys(),
                [
                    str(seq.seq)
                    for seq in Bio.SeqIO.parse(f"{self.outDir}/{self.fName}", "fasta")
                ],
            )
        )
        self.saveToCache(sorted(sequences.keys()), alignment)
        self._current = alignment
        return alignment

    def checkTool(self) -> bool:
        """Check if the MAFFT tool is installed"""
        return shutil.which(self.cmd) is not None


class MAFFT(BioPythonMSA):
    """
    Multiple sequence alignment provider using the MAFFT cross-platform program
    - https://mafft.cbrc.jp/alignment/software/

    Uses the BioPython wrapper for MAFFT:
    - https://biopython.org/docs/1.76/api/Bio.Align.Applications.html#Bio.Align.Applications.MafftCommandline
    """

    def __call__(
        self, sequences: dict[str:str] | None = None, **kwargs
    ) -> dict[str, str] | None:
        """
        MSA with MAFFT and BioPython.

        Args:
            sequences (dict[str,str]):
                dictionary of sequences to align, keys are sequence IDs
                (i.e. accession keys) and values are sequences themselves
            **kwargs:
                additional arguments to be passed to the alignment algorithm,
                can also just be metadata to be stored with the alignment

        Returns:
            alignment (dict[str,str]):
                dictionary of aligned sequences, keys are sequence IDs

        """
        # if no sequences are provided, return current alignment
        if not sequences:
            return self.current
        # check if we have the alignment cached
        alignment = self.getFromCache(sorted(sequences.keys()))
        if alignment:
            self._current = alignment
            return alignment
        # Parse sequences
        sequences_path, _ = self.parseSequences(sequences, **kwargs)
        # Run mafft
        cmd = MafftCommandline(
            cmd="mafft",
            auto=True,
            input=sequences_path,
            clustalout=False,
        )
        stdout, stderr = cmd()
        with open(f"{self.outDir}/{self.fName}", "w") as handle:
            handle.write(stdout)
        alignment = self.parseAlignment(sequences)
        self.saveToCache(sorted(sequences.keys()), alignment)
        return alignment

    @property
    def cmd(self) -> str:
        return "mafft"


class ClustalMSA(BioPythonMSA):
    """
    Multiple sequence alignment provider using the Clustal Omega Linux program
    - http://www.clustal.org/omega/

    Uses the BioPython wrapper for Clustal Omega
    - https://biopython.org/docs/1.76/api/Bio.Align.Applications.html#Bio.Align.Applications.ClustalOmegaCommandline
    """

    def __call__(
        self, sequences: dict[str:str] = None, **kwargs
    ) -> dict[str, str] | None:
        """
        MSA with Clustal Omega and BioPython.

        Args:
            sequences (dict[str,str]):
                dictionary of sequences to align, keys are sequence IDs
            **kwargs:
                additional arguments to be passed to the alignment algorithm,

        Returns:
            alignment (dict[str,str]):
                the aligned sequences mapped to their IDs
        """

        # if no sequences are provided, return current alignment
        if not sequences:
            return self.current
        # check if we have the alignment cached
        alignment = self.getFromCache(sorted(sequences.keys()))
        if alignment:
            self._current = alignment
            return alignment
        # Parse sequences
        sequences_path, _ = self.parseSequences(sequences, **kwargs)
        # Run clustal omega
        clustal_omega_cline = ClustalOmegaCommandline(
            infile=sequences_path,
            outfile=f"{self.outDir}/alignment.aln-fasta.fasta",
            verbose=True,
            auto=True,
            force=True,
            outfmt="fasta",
        )
        clustal_omega_cline()
        alignment = self.parseAlignment(sequences)
        self.saveToCache(sorted(sequences.keys()), alignment)
        return alignment

    @property
    def cmd(self) -> str:
        return "clustalo"
