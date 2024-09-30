import logging

from qsprpred.data.descriptors.sets import DescriptorSet
from qsprpred.data.storage.interfaces.chem_store import ChemStore
from qsprpred.data.tables.mol import MoleculeTable
from qsprpred.data.tables.qspr import QSPRTable
from qsprpred.extra.data.descriptors.sets import ProteinDescriptorSet
from qsprpred.extra.data.storage.protein.interfaces.protein_storage import (
    ProteinStorage,
)
from qsprpred.logs import logger
from qsprpred.tasks import TargetProperty


class PCMDataSet(QSPRTable):
    """Extension of `QSARDataset` for PCM modelling.

    It allows specification of a column with protein identifiers
    and the calculation of protein descriptors.

    Attributes:
        proteinIDProp (str):
            name of column in df containing the protein target identifier (usually a
            UniProt ID) to use for protein descriptors for PCM modelling
            and other protein related tasks.
        proteinSeqProvider (Callable):
            function that takes a list of protein identifiers and returns a `dict`
            mapping those identifiers to their sequences. Defaults to `None`.
    """
    def __init__(
        self,
        storage: ChemStore | None = None,
        name: str | None = None,
        target_props: list[TargetProperty | dict] | None = None,
        path: str = ".",
        random_state: int | None = None,
        store_format: str = "pkl",
        drop_empty_target_props: bool = True,
        proteins: ProteinStorage | None = None,
    ):
        """Construct QSPRdata, also apply transformations of output property if
                specified.

        Args:
            name (str):
                data name, used in saving the data
            target_props (list[TargetProperty | dict] | None):
                target properties, names
                should correspond with target column name in df. If `None`, target
                properties will be inferred if this data set has been saved
                previously. Defaults to `None`.
            random_state (int, optional):
                random state for splitting the data.
            store_format (str, optional):
                format to use for storing the data ('pkl' or 'csv').
            drop_empty_target_props (bool, optional):
                whether to ignore entries with empty target properties. Defaults to
                `True`.

        Raises:
            `ValueError`: Raised if threshold given with non-classification task.
        """
        super().__init__(
            storage=storage,
            name=name,
            path=path,
            target_props=target_props,
            random_state=random_state,
            store_format=store_format,
            drop_empty_target_props=drop_empty_target_props,
        )
        self.proteins = None
        if proteins is not None:
            self.attachProteins(proteins)
        else:
            logging.warning(
                "No protein storage provided. Protein descriptors will not be "
                "possible to calculate on this data set. Use `attachProteins` to "
                "attach a protein storage to provide protein data."
            )

    @property
    def proteinIDProp(self) -> str:
        """Return the name of the property in the data frame containing
        the protein target identifier.

        Returns:
            proteinIDProp (str): Name of the protein target identifier column.
        """
        return self.proteins.idProp

    def attachProteins(self, proteins: ProteinStorage):
        self.proteins = proteins
        if self.proteins.idProp not in self.storage.getProperties():
            raise ValueError(
                f"Protein ID property '{self.proteins.idProp}' not found in "
                f"storage '{self.storage}' properties: {self.storage.getProperties()}"
            )
        # check if all protein IDs can be found
        if not set(self.getProteinKeys()
                  ).issubset(set(self.storage.getProperty(self.proteins.idProp))):
            missing_ids = set(self.getProteinKeys()
                             ) - set(self.storage.getProperty(self.proteins.idProp))
            logger.warning(
                f"Not all protein IDs found in storage '{self.storage}' properties: "
                f"{self.storage.getProperty(self.proteins.idProp)}. Missing the"
                f" following protein IDs: {missing_ids}"
            )

    def getProteinKeys(self) -> list[str]:
        """Return a list of keys identifying the proteins in the data frame.

        Returns:
            keys (list): List of protein keys.
        """
        return list(set(self.proteins.getProperty(self.proteins.idProp)))

    def getPCMInfo(self) -> tuple[dict[str, str], dict]:
        """Return a dictionary of protein sequences for the proteins in the data frame.

        Returns:
            sequences (dict): Dictionary of protein sequences.
        """
        return self.proteins.getPCMInfo()

    def addDescriptors(
        self,
        descriptors: list[DescriptorSet | ProteinDescriptorSet],
        recalculate: bool = False,
        featurize: bool = True,
        *args,
        **kwargs,
    ):
        # get protein sequences and metadata
        sequences, info = self.proteins.getPCMInfo()
        # append sequences and metadata to kwargs
        kwargs["sequences"] = sequences
        kwargs["protein_ids"] = sequences.keys()
        kwargs["protein_id_prop"] = self.proteins.idProp
        for key in info:
            kwargs[key] = info[key]
        # pass everything to the descriptor calculation
        return super().addDescriptors(
            descriptors, recalculate, featurize, *args, **kwargs
        )

    def getSubset(
        self,
        subset: list[str],
        ids: list[str] | None = None,
        name: str | None = None,
        path: str | None = None,
        **kwargs,
    ) -> "QSPRTable":
        ds = super().getSubset(subset, ids, name, path, **kwargs)
        ds.__class__ = PCMDataSet
        ds.attachProteins(self.proteins)
        return ds

    @classmethod
    def fromMolTable(
        cls,
        mol_table: MoleculeTable,
        target_props: list[TargetProperty | dict],
        *args,
        proteins: ProteinStorage | None = None,
        name: str | None = None,
        path: str = ".",
        **kwargs,
    ) -> "PCMDataSet":
        """Construct a data set to handle PCM data from a `MoleculeTable`.

        Args:
            mol_table (MoleculeTable):
                `MoleculeTable` instance containing the PCM data.
            target_props (list[TargetProperty | dict], optional):
                target properties,
                names should correspond with target column name in `df`
            *args:
                additional arguments to be passed to the `PCMDataset` constructor.
            proteins (ProteinStorage):
                `ProteinStorage` instance containing the protein data.
            name (str, optional):
                data name, used in saving the data. Defaults to `None`.
            path (str, optional):
                path to save the data. Defaults to `'.'`.
            **kwargs:
                keyword arguments to be passed to the `PCMDataset` constructor.

        Returns:
            PCMDataSet:
                `PCMDataset` instance containing the PCM data.
        """
        name = name or f"{mol_table.name}_PCM"
        ret = QSPRTable.fromMolTable(
            mol_table, target_props, *args, name=name, path=path, **kwargs
        )
        ret.__class__ = PCMDataSet
        if proteins is not None:
            ret.attachProteins(proteins)
        return ret
