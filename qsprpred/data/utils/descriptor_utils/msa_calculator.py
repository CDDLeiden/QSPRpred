"""
msa_calculator

Created by: Martin Sicho
On: 05.04.23, 12:34
"""
import json
from abc import ABC, abstractmethod

import Bio
import Bio.SeqIO as Bio_SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Align.Applications import MafftCommandline

from qsprpred.logs import logger

class MSAProvider(ABC):

    @abstractmethod
    def __call__(self, sequences: dict[str : str], **kwargs):
        """
        Aligns the sequences and returns the alignment.

        Args:
            sequences: dictionary of sequences to align, keys are sequence IDs (i.e. accession keys) and values are sequences themselves
            **kwargs: additional arguments to be passed to the alignment algorithm, can also just be metadata to be stored with the alignment
        """
        pass

    @property
    @abstractmethod
    def current(self):
        """
        Returns the current alignment.
        """
        pass

    @classmethod
    @abstractmethod
    def fromFile(cls, fname : str) -> 'MSAProvider':
        """
        Creates an MSA provider object from a JSON file.

        Args:
            fname: file name of the JSON file to load the provider from

        Returns:
            the loaded alignment as a dictionary
        """
        pass

    @abstractmethod
    def toFile(self, fname : str):
        """
        Saves the MSA provider to a JSON file.

        Args:
            fname: file name of the JSON file to save the provider to
        """
        pass

class BioPythonMSA(MSAProvider, ABC):
    """
    Common functionality for MSA providers using BioPython command line wrappers.
    """

    def __init__(self, out_dir : str = ".", fname : str = "alignment.aln-fasta.fasta"):
        self.outDir = out_dir
        self.fname = fname
        self.cache = dict()
        self._current = None

    def getFromCache(self, target_ids : list[str]):
        key = "~".join(target_ids)
        if key in self.cache:
            return self.cache[key]

    def saveToCache(self, target_ids : list[str], alignment : dict[str : str]):
        key = "~".join(target_ids)
        self.cache[key] = alignment

    def currentToFile(self, fname : str):
        """
        Saves the current alignment to a JSON file.
        """
        if self.current:
            with open(fname, "w") as f:
                json.dump(self.current, f)
        else:
            logger.warning("No current alignment to save. File not created.")

    def currentFromFile(self, fname : str):
        """
        Loads the alignment from a JSON file.
        """

        with open(fname, "r") as f:
            self._current = json.load(f)
            self.saveToCache(sorted(self.current.keys()), self.current)
            return self.current

    @property
    def current(self):
        return self._current

    @classmethod
    def fromFile(cls, fname: str) -> 'MSAProvider':
        with open(fname, 'r') as f:
            data = json.load(f)

        ret = cls(data["out_dir"], data["fname"])
        ret.currentFromFile(data["current"])
        return ret

    def toFile(self, fname: str):
        with open(fname, 'w') as f:
            json.dump({
                "out_dir": self.outDir,
                "fname": self.fname,
                "current": f"{fname}.msa",
                "class": f"{self.__class__.__module__}.{self.__class__.__name__}"
            }, f)
        self.currentToFile(f"{fname}.msa")

    def parseSequences(self, sequences, **kwargs):
        """
        Create object with sequences and the passed metadata. Save the sequences to a file that will serve as input to the command line tools.

        Args:
            sequences: sequences to align
            **kwargs: metadata to be stored with the alignment

        Returns:
            the object with sequences and metadata
        """

        records = []
        target_ids = []
        for target_id in sequences:
            records.append(
                Bio_SeqIO.SeqRecord(
                    seq=Bio.Seq.Seq(sequences[target_id]),
                    id=target_id,
                    name=target_id,
                    description=" ".join([f"{k}={v}" for k, v in kwargs.items()])
                )
            )
            target_ids.append(target_id)
        sequences_path = f"{self.outDir}/sequences.fasta"
        # Write sequences as .fasta file
        return sequences_path, Bio_SeqIO.write(records, sequences_path, "fasta")

    def parseAlignment(self, sequences: dict[str: str]):
        """
        Parse the alignment from the output file of the alignment algorithm.

        Args:
            sequences: the original dictionary of sequences that were aligned

        Returns:
            the aligned sequences mapped to their IDs
        """

        alignment = {tid: seq for tid, seq in zip(sequences.keys(), [str(seq.seq) for seq in
                                                                     Bio.SeqIO.parse(f"{self.outDir}/{self.fname}",
                                                                                     "fasta")])}
        self.saveToCache(sorted(sequences.keys()), alignment)
        self._current = alignment
        return alignment

class MAFFT(BioPythonMSA):
    """
    Multiple sequence alignment provider using the MAFFT cross-platform program (https://mafft.cbrc.jp/alignment/software/).
    Uses the BioPython wrapper (https://biopython.org/docs/1.76/api/Bio.Align.Applications.html#Bio.Align.Applications.MafftCommandline).

    """

    def __call__(self, sequences: dict[str: str], **kwargs):
        """
        MSA with MAFFT and BioPython.

        Args:
            sequences: dictionary of sequences to align, keys are sequence IDs (i.e. accession keys) and values are sequences themselves
            **kwargs: additional arguments to be passed to the alignment algorithm, can also just be metadata to be stored with the alignment

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
        cmd = MafftCommandline(
            cmd="mafft",
            auto=True,
            input=sequences_path,
            clustalout=False,
        )
        stdout, stderr = cmd()
        with open(f"{self.outDir}/{self.fname}", "w") as handle:
            handle.write(stdout)

        return self.parseAlignment(sequences)


class ClustalMSA(BioPythonMSA):

    def __call__(self, sequences: dict[str : str], **kwargs):
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

        return self.parseAlignment(sequences)



