"""  Filter Papyrus data.  """

import os.path

import pandas as pd
from papyrus_scripts.preprocess import (
    consume_chunks,
    keep_accession,
    keep_quality,
    keep_type,
)
from papyrus_scripts.reader import read_papyrus

from qsprpred.logs import logger


def papyrus_filter(
    acc_key: list[str],
    quality: str,
    outdir: str,
    activity_types: list[str] | str = "all",
    prefix: str | None = None,
    drop_duplicates: bool = True,
    chunk_size: int = 1e5,
    use_existing: bool = True,
    stereo: bool = False,
    plusplus: bool = False,
    papyrus_dir: str | None = None,
):
    """Filters the downloaded Papyrus dataset for quality and accession key (UniProt)
    and outputs a .tsv file of all compounds fulfilling these requirements.

    Args:
        acc_key (list): list of UniProt accession keys
        quality (str): str with minimum quality of dataset to keep
        outdir (str): path to the location of Papyrus data
        activity_types (list, str): list of activity types to keep
        prefix (str): prefix for the output file
        drop_duplicates (bool): boolean to drop duplicates from the final dataset
        chunk_size (int): integer of chunks to process one at the time
        use_existing (bool): if `True`, use existing data if available
        stereo (bool): if `True`, read stereochemistry data (if available)
        plusplus (bool): if `True`, read high quality Papyrus++ data (if available)
        papyrus_dir: path to the location of Papyrus database

    Returns:
        dataset (pd.DataFrame): filtered dataset
        outfile (str): path to the output file
    """
    prefix = prefix or f"{'_'.join(acc_key)}_{quality}"
    outfile = os.path.join(outdir, f"{prefix}.tsv")

    papyrus_dir = outdir if not papyrus_dir else papyrus_dir

    if use_existing and os.path.exists(outfile):
        logger.info(f"Using existing data from {outfile}...")
        return pd.read_table(outfile, sep="\t", header=0), outfile

    # read data
    logger.info(f"Reading data from {papyrus_dir}...")
    sample_data = read_papyrus(
        is3d=stereo, chunksize=chunk_size, source_path=papyrus_dir, plusplus=plusplus
    )
    logger.info("Read all data.")

    # data filters
    filter1 = keep_quality(data=sample_data, min_quality=quality)
    filter2 = keep_accession(data=filter1, accession=acc_key)
    filter3 = keep_type(data=filter2, activity_types=activity_types)
    logger.info("Initialized filters.")

    # filter data per chunk
    filtered_data = consume_chunks(generator=filter3)
    logger.info(f"Number of compounds:{filtered_data.shape[0]}")

    # filter out duplicate InChiKeys
    if drop_duplicates:
        logger.info("Filtering out duplicate molecules")
        amnt_mols_i = len(filtered_data["InChIKey"])
        filtered_data.drop_duplicates(
            subset=["InChIKey"], inplace=True, ignore_index=True
        )
        amnt_mols_f = len(filtered_data["InChIKey"])
        logger.info(f"Filtered out {amnt_mols_i - amnt_mols_f} duplicate molecules")

    # write filtered data to .tsv file
    filtered_data.to_csv(outfile, sep="\t", index=False)
    logger.info(f"Wrote data to file '{outfile}'.")

    return filtered_data, outfile
