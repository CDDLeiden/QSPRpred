"""  Creating dataset from Papyrus database.  """

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import papyrus_scripts
from papyrus_scripts.download import download_papyrus

from qsprpred.logs import logger
from .papyrus_filter import papyrus_filter
from ..data_source import DataSource
from ...tables.mol import MoleculeTable


class Papyrus(DataSource):
    """Create new instance of Papyrus dataset.
    See `papyrus_filter` and `Papyrus.download` and `Papyrus.getData` for more details.

    Attributes:
        DEFAULT_DIR (str): default directory for Papyrus database and the extracted data
        dataDir (str): storage directory for Papyrus database and the extracted data
        _papyrusDir (str): directory where the Papyrus database is located, os.path.join(dataDir, "papyrus")
        version (list): Papyrus database version
        descriptors (list, str, None): descriptors to download if not already present
        stereo (bool): use version with stereochemistry
        nostereo (bool): use version without stereochemistry
        plusplus (bool): use plusplus version
        diskMargin (float): the disk space margin to leave free
    """

    DEFAULT_DIR = os.path.join(Path.home(), ".Papyrus")

    def __init__(
        self,
        data_dir: str = DEFAULT_DIR,
        version: str = "latest",
        descriptors: str | list[str] | None = None,
        stereo: bool = False,
        disk_margin: float = 0.01,
        plus_only: bool = True,
    ):
        """Create new instance of Papyrus dataset. See `papyrus_filter` and
        `Papyrus.download` and `Papyrus.getData` for more details.

        Args:
            data_dir (str):
                storage directory for Papyrus database and the extracted data
            version (str):
                Papyrus database version
            descriptors (str, list, None):
                descriptors to download if not already present (set to 'all' for
                all descriptors, otherwise a list of descriptor names, see
                https://github.com/OlivierBeq/Papyrus-scripts)
            stereo (str):
                include stereochemistry in the database
            disk_margin (float):
                the disk space margin to leave free
            plus_only (bool):
                use only plusplus version, only high quality data
        """
        self.dataDir = data_dir
        self._papyrusDir = os.path.join(self.dataDir, "papyrus")
        self.version = version
        self.descriptors = descriptors
        self.stereo = stereo
        self.nostereo = not self.stereo
        self.plusplus = plus_only
        self.diskMargin = disk_margin

    def download(self):
        """Download Papyrus database with the required information.

        Only newly requested data is downloaded. Remove the files if you want to
        reload the data completely.
        """
        if not os.path.exists(self._papyrusDir):
            os.makedirs(self.dataDir, exist_ok=True)
            logger.info("Downloading Papyrus database...")
            download_papyrus(
                outdir=self.dataDir,
                version=self.version,
                descriptors=self.descriptors,
                stereo=self.stereo,
                nostereo=self.nostereo,
                disk_margin=self.diskMargin,
                only_pp=self.plusplus,
            )
        else:
            logger.info(
                "Papyrus database already downloaded. Using existing data. "
                f"Delete the following folder to reload the data: {self._papyrusDir}"
            )

    def getData(
        self,
        name: str | None = None,
        acc_keys: list[str] | None = None,
        quality: str = "high",
        activity_types: list[str] | str = "all",
        output_dir: Optional[str] = None,
        drop_duplicates: bool = False,
        chunk_size: int = 1e5,
        use_existing: bool = True,
        **kwargs,
    ) -> MoleculeTable:
        """Get the data from the Papyrus database as a `DataSetTSV` instance.

        Args:
            acc_keys (list): protein accession keys
            quality (str): desired minimum quality of the dataset
            activity_types (list, str): list of activity types to include in the dataset
            output_dir (str): path to the directory where the data set will be stored
            name (str): name of the dataset (the prefix of the generated .tsv file)
            drop_duplicates (bool): remove duplicates after filtering
            chunk_size (int): data is read in chunks of this size (see `papyrus_filter`)
            use_existing (bool): use existing if available
            kwargs: additional keyword arguments passed to `MoleculeTable.fromTableFile`

        Returns:
            MolculeTable: the filtered data set
        """
        logger.debug("Getting data from Papyrus data source...")
        assert acc_keys is not None, "Please provide a list of accession keys."
        name = name or "papyrus"
        self.download()
        logger.debug("Papyrus data set finished downloading.")
        output_dir = output_dir or self.dataDir
        if not os.path.exists(output_dir):
            raise ValueError(f"Output directory '{output_dir}' does not exist.")
        logger.debug(f"Filtering Papyrus for accession keys: {acc_keys}")
        data, path = papyrus_filter(
            acc_key=acc_keys,
            quality=quality,
            outdir=output_dir,
            activity_types=activity_types,
            prefix=name,
            drop_duplicates=drop_duplicates,
            chunk_size=chunk_size,
            use_existing=use_existing,
            stereo=self.stereo,
            plusplus=self.plusplus,
            papyrus_dir=self.dataDir,
        )
        logger.debug("Finished filtering Papyrus data set.")
        logger.debug(f"Creating MoleculeTable from '{path}'.")
        ret = MoleculeTable.fromTableFile(name, path, store_dir=output_dir, **kwargs)
        logger.debug(f"Finished creating MoleculeTable from '{path}'.")
        return ret

    def getProteinData(
        self,
        acc_keys: list[str],
        output_dir: Optional[str] = None,
        name: Optional[str] = None,
        use_existing: bool = True,
    ) -> pd.DataFrame:
        """Get the protein data from the Papyrus database.

        Args:
            acc_keys (list): protein accession keys
            output_dir (str): path to the directory where the data set will be stored
            name (str): name of the dataset (the prefix of the generated .tsv file)
            use_existing (bool): use existing if available

        Returns:
            pd.DataFrame: the protein data
        """
        self.download()
        output_dir = output_dir or self.dataDir
        if not os.path.exists(output_dir):
            raise ValueError(f"Output directory '{output_dir}' does not exist.")
        path = os.path.join(output_dir, f"{name or os.path.basename(output_dir)}.tsv")
        if os.path.exists(path) and use_existing:
            return pd.read_table(path)
        else:
            protein_data = papyrus_scripts.read_protein_set(
                source_path=self.dataDir, version=self.version
            )
            protein_data["accession"] = protein_data["target_id"].apply(
                lambda x: x.split("_")[0]
            )
            targets = protein_data[protein_data.accession.isin(acc_keys)]
            targets.to_csv(path, sep="\t", header=True, index=False)
            return targets
