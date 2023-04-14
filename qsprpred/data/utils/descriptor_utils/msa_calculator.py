"""
msa_calculator

Created by: Martin Sicho
On: 05.04.23, 12:34
"""
import json

import Bio
import Bio.SeqIO as Bio_SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline

from qsprpred.logs import logger


class ClustalMSA:

    def __init__(self, out_dir : str = ".", fname : str = "alignment.aln-fasta.fasta"):
        self.outDir = out_dir
        self.fname = fname
        self.cache = dict()
        self.current = None

    def getFromCache(self, target_ids : list[str]):
        key = "~".join(target_ids)
        if key in self.cache:
            return self.cache[key]

    def saveToCache(self, target_ids : list[str], alignment : dict[str : str]):
        key = "~".join(target_ids)
        self.cache[key] = alignment

    def __call__(self, sequences: dict[str : str], **kwargs):
        # if no sequences are provided, return current alignment
        if not sequences:
            return self.current

        # check if we have the alignment cached
        alignment = self.getFromCache(sorted(sequences.keys()))
        if alignment:
            self.current = alignment
            return alignment

        # Create object with sequences and descriptions
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
        _ = Bio_SeqIO.write(records, sequences_path, "fasta")

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

        # Read alignment and return the aligned sequences mapped to IDs
        alignment = {tid : seq for tid, seq in zip(target_ids, [str(seq.seq) for seq in Bio.SeqIO.parse(f"{self.outDir}/alignment.aln-fasta.fasta", "fasta")])}
        self.saveToCache(sorted(sequences.keys()), alignment)
        self.current = alignment
        return alignment

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
            self.current = json.load(f)
            self.saveToCache(sorted(self.current.keys()), self.current)
            return self.current



