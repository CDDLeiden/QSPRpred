"""  Creating dataset from Papyrus database.  """

import os
from pathlib import Path

import pandas as pd
import papyrus_scripts
from papyrus_scripts.download import download_papyrus

from ...data import MoleculeTable
from .papyrus_filter import papyrus_filter


class Papyrus:
    """ Create new instance of Papyrus dataset.
    See `papyrus_filter` and `Papyrus.download` and `Papyrus.getData` for more details.

    Attributes:
        DEFAULT_DIR (str): default directory for Papyrus database and the extracted data
        dataDir (str): storage directory for Papyrus database and the extracted data
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
        descriptors: str | list[str] | None = "all",
        stereo: bool = False,
        disk_margin: float = 0.01,
        plus_only: bool = True,
    ):
        """Create new instance of Papyrus dataset. See `papyrus_filter` and
        `Papyrus.download` and `Papyrus.getData` for more details.

        Args:
            data_dir (str): storage directory for Papyrus database and the extracted data
            version (str): Papyrus database version
            descriptors (str, list, None): descriptors to download if not already present
            stereo (str): include stereochemistry in the database
            disk_margin (float): the disk space margin to leave free
            plus_only (bool): use only plusplus version, only high quality data
        """
        self.dataDir = data_dir
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
        os.makedirs(self.dataDir, exist_ok=True)
        download_papyrus(
            outdir=self.dataDir,
            version=self.version,
            descriptors=self.descriptors,
            stereo=self.stereo,
            nostereo=self.nostereo,
            disk_margin=self.diskMargin,
            only_pp=self.plusplus,
        )

    def getData(
        self,
        acc_keys: list[str],
        quality: str,
        activity_types: list[str] | str = "all",
        output_dir: str = None,
        name: str = None,
        drop_duplicates: bool = False,
        chunk_size: int = 1e5,
        use_existing: bool = True,
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

        Returns:
            MolculeTable: the filtered data set
        """
        self.download()
        output_dir = output_dir or self.dataDir
        if not os.path.exists(output_dir):
            raise ValueError(f"Output directory '{output_dir}' does not exist.")
        data, path = papyrus_filter(
            acc_key=acc_keys,
            quality=quality,
            outdir=output_dir,
            activity_types=activity_types,
            prefix=name or os.path.basename(output_dir),
            drop_duplicates=drop_duplicates,
            chunk_size=chunk_size,
            use_existing=use_existing,
            stereo=self.stereo,
            plusplus=self.plusplus,
            papyrus_dir=self.dataDir,
        )
        return MoleculeTable.fromTableFile(name, path, store_dir=output_dir)

    def getProteinData(
        self,
        acc_keys: list[str],
        output_dir: str = None,
        name: str = None,
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
            protein_data = papyrus_scripts.read_protein_set(version=self.version)
            protein_data["accession"] = protein_data["target_id"].apply(
                lambda x: x.split("_")[0]
            )
            targets = protein_data[protein_data.accession.isin(acc_keys)]
            targets.to_csv(path, sep="\t", header=True, index=False)
            return targets
