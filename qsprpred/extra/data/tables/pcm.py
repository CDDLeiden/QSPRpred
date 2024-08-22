from typing import Literal

from qsprpred.data.descriptors.sets import DescriptorSet
from qsprpred.data.storage.interfaces.chem_store import ChemStore
from qsprpred.data.tables.mol import MoleculeTable
from qsprpred.data.tables.qspr import QSPRDataset
from qsprpred.extra.data.descriptors.sets import ProteinDescriptorSet
from qsprpred.extra.data.storage.protein.interfaces.protein_storage import \
    ProteinStorage
from qsprpred.tasks import TargetProperty


class PCMDataSet(QSPRDataset):
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
            storage: ChemStore,
            proteins: ProteinStorage,
            name: str | None = None,
            target_props: list[TargetProperty | dict] | None = None,
            path: str = ".",
            random_state: int | None = None,
            store_format: str = "pkl",
            drop_empty_target_props: bool = True,
    ):
        """Construct QSPRdata, also apply transformations of output property if
                specified.

        Args:
            name (str):
                data name, used in saving the data
            target_props (list[TargetProperty | dict] | None):
                target properties, names
                should correspond with target columnname in df. If `None`, target
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
            storage,
            name=name,
            path=path,
            target_props=target_props,
            random_state=random_state,
            store_format=store_format,
            drop_empty_target_props=drop_empty_target_props,
        )
        self.proteins = proteins
        if self.proteins.idProp not in self.storage.getProperties():
            raise ValueError(
                f"Protein ID property '{self.proteins.idProp}' not found in "
                f"storage '{self.storage}' properties: {self.storage.getProperties()}"
            )
        # check if all protein IDs can be found
        if not set(self.getProteinKeys()).issubset(
                set(self.storage.getProperty(self.proteins.idProp))
        ):
            raise ValueError(
                f"Not all protein IDs found in storage '{self.storage}' properties: "
                f"{self.storage.getProperty(self.proteins.idProp)}. Missing the"
                f" following protein IDs: {
                set(self.getProteinKeys())
                - set(self.storage.getProperty(self.proteins.idProp))
                }"
            )

    @property
    def proteinIDProp(self) -> str:
        """Return the name of the property in the data frame containing
        the protein target identifier.

        Returns:
            proteinIDProp (str): Name of the protein target identifier column.
        """
        return self.proteins.idProp

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

    @classmethod
    def fromMolTable(
            cls,
            mol_table: MoleculeTable,
            proteins: ProteinStorage,
            target_props: list[TargetProperty | dict] | None = None,
            name: str | None = None,
            **kwargs,
    ) -> "PCMDataSet":
        """Construct a data set to handle PCM data from a `MoleculeTable`.

        Args:
            mol_table (MoleculeTable):
                `MoleculeTable` instance containing the PCM data.
            proteins (ProteinStorage):
                `ProteinStorage` instance containing the protein data.
            target_props (list[TargetProperty | dict], optional):
                target properties,
                names should correspond with target column name in `df`
            name (str, optional):
                data name, used in saving the data. Defaults to `None`.
            **kwargs:
                keyword arguments to be passed to the `PCMDataset` constructor.

        Returns:
            PCMDataSet:
                `PCMDataset` instance containing the PCM data.
        """
        ret = QSPRDataset.fromMolTable(mol_table, target_props, name, **kwargs)
        ret.proteins = proteins
        ret.__class__ = PCMDataSet
        return ret

    def searchWithSMARTS(self, patterns: list[str],
                         operator: Literal["or", "and"] = "or",
                         use_chirality: bool = False,
                         name: str | None = None,
                         path: str | None = None
                         ) -> "PCMDataSet":
        ret = super().searchWithSMARTS(patterns, operator, use_chirality, name, path)
        ret.proteins = self.proteins
        ret.__class__ = PCMDataSet
        return ret

    def searchOnProperty(self, prop_name: str, values: list[float | int | str],
                         exact=False, name: str | None = None,
                         path: str | None = None) -> "PCMDataSet":
        ret = super().searchOnProperty(prop_name, values, exact, name, path)
        ret.proteins = self.proteins
        ret.__class__ = PCMDataSet
        return ret
