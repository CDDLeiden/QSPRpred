"""
msa_calculator

Created by: Martin Sicho
On: 05.04.23, 12:34
"""
import Bio
import Bio.SeqIO as Bio_SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline

class ClustalMSA:

    def __init__(self, email : str, out_dir : str = "."):
        self.email = email
        self.outDir = out_dir

    def __call__(self, sequences: dict[str : str], **kwargs):
        # Create object with sequences and descriptions
        records = []
        target_ids = []
        for target_id in sequences:
            records.append(
                Bio_SeqIO.SeqRecord(
                    seq=Bio.Seq.Seq(sequences[target_id]),
                    id=target_id,
                    name=target_id,
                    description=" ".join([kwargs[target_id]["UniProtID"], kwargs[target_id]["Organism"], kwargs[target_id]["Classification"]]),
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
        return {tid : seq for tid, seq in zip(target_ids, [str(seq.seq) for seq in Bio.SeqIO.parse(f"{self.outDir}/alignment.aln-fasta.fasta", "fasta")])}


