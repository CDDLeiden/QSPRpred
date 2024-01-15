import itertools
import os
import tempfile
from typing import Callable

import pandas as pd

from qsprpred import TargetProperty, TargetTasks
from qsprpred.data.descriptors.sets import DescriptorSet
from qsprpred.extra.data.descriptors.fingerprints import (
    CDKFP,
    CDKExtendedFP,
    CDKEStateFP,
    CDKGraphOnlyFP,
    CDKMACCSFP,
    CDKPubchemFP,
    CDKSubstructureFP,
    CDKKlekotaRothFP,
    CDKAtomPairs2DFP,
)
from qsprpred.extra.data.descriptors.sets import (
    Mordred,
    Mold2,
    PaDEL,
    ExtendedValenceSignature,
    ProteinDescriptorSet,
    ProDec,
)
from qsprpred.extra.data.tables.pcm import PCMDataSet
from qsprpred.extra.data.utils.msa_calculator import ClustalMSA
from qsprpred.utils.testing.path_mixins import DataSetsPathMixIn


class DataSetsMixInExtras(DataSetsPathMixIn):
    """MixIn class for testing data sets in extras."""

    def setUpPaths(self):
        super().setUpPaths()
        self.dataPathPCM = f"{os.path.dirname(__file__)}/test_files/data"

    @classmethod
    def getAllDescriptors(cls) -> list[DescriptorSet]:
        """Return a list of all available molecule descriptor sets.

        Returns:
            list: list of `MoleculeDescriptorSet` objects
        """
        return [
            Mordred(),
            Mold2(),
            CDKFP(size=2048, search_depth=7),
            CDKExtendedFP(),
            CDKEStateFP(),
            CDKGraphOnlyFP(size=2048, search_depth=7),
            CDKMACCSFP(),
            CDKPubchemFP(),
            CDKSubstructureFP(use_counts=False),
            CDKKlekotaRothFP(use_counts=True),
            CDKAtomPairs2DFP(use_counts=False),
            CDKSubstructureFP(use_counts=True),
            CDKKlekotaRothFP(use_counts=False),
            CDKAtomPairs2DFP(use_counts=True),
            PaDEL(),
            ExtendedValenceSignature(1),
        ]

    @classmethod
    def getAllProteinDescriptors(cls) -> list[ProteinDescriptorSet]:
        """Return a list of all available protein descriptor sets.

        Returns:
            list: list of `ProteinDescriptorSet` objects
        """

        return [
            ProDec(
                sets=["Zscale Hellberg"],
                msa_provider=cls.getMSAProvider(tempfile.mkdtemp()),
            ),
            ProDec(
                sets=["Sneath"], msa_provider=cls.getMSAProvider(tempfile.mkdtemp())
            ),
        ]

    @classmethod
    def getDefaultCalculatorCombo(cls):
        mol_descriptor_calculators = [super().getDefaultCalculatorCombo()[0]]
        feature_sets_pcm = cls.getAllProteinDescriptors()
        protein_descriptor_calculators = list(
            itertools.combinations(feature_sets_pcm, 1)
        ) + list(itertools.combinations(feature_sets_pcm, 2))
        descriptor_calculators = (
            mol_descriptor_calculators + protein_descriptor_calculators
        )
        # make combinations of molecular and PCM descriptor calculators
        descriptor_calculators += [
            mol + prot
            for mol, prot in zip(
                mol_descriptor_calculators, protein_descriptor_calculators
            )
        ]
        return descriptor_calculators

    def getPCMDF(self) -> pd.DataFrame:
        """Return a test dataframe with PCM data.

        Returns:
            pd.DataFrame: dataframe with PCM data
        """
        return pd.read_csv(f"{self.dataPathPCM}/pcm_sample.csv")

    def getPCMTargetsDF(self) -> pd.DataFrame:
        """Return a test dataframe with PCM targets and their sequences.

        Returns:
            pd.DataFrame: dataframe with PCM targets and their sequences
        """
        return pd.read_csv(f"{self.dataPathPCM}/pcm_sample_targets.csv")

    def getPCMSeqProvider(
        self,
    ) -> Callable[[list[str]], tuple[dict[str, str], dict[str, dict]]]:
        """Return a function that provides sequences for given accessions.

        Returns:
            Callable[[list[str]], tuple[dict[str, str], dict[str, dict]]]:
                function that provides sequences for given accessions
        """
        df_seq = self.getPCMTargetsDF()
        mapper = {}
        kwargs_map = {}
        for i, row in df_seq.iterrows():
            mapper[row["accession"]] = row["Sequence"]
            kwargs_map[row["accession"]] = {
                "Classification": row["Classification"],
                "Organism": row["Organism"],
                "UniProtID": row["UniProtID"],
            }

        return lambda acc_keys: (
            {acc: mapper[acc] for acc in acc_keys},
            {acc: kwargs_map[acc] for acc in acc_keys},
        )

    @classmethod
    def getMSAProvider(cls, out_dir: str):
        return ClustalMSA(out_dir=out_dir)

    def createPCMDataSet(
        self,
        name: str = "QSPRDataset_test_pcm",
        target_props: list[TargetProperty]
        | list[dict] = [
            {"name": "pchembl_value_Median", "task": TargetTasks.REGRESSION}
        ],
        preparation_settings: dict | None = None,
        protein_col: str = "accession",
        random_state: int | None = None,
    ):
        """Create a small dataset for testing purposes.

        Args:
            name (str, optional):
                name of the dataset. Defaults to "QSPRDataset_test".
            target_props (list[TargetProperty] | list[dict], optional):
                target properties.
            preparation_settings (dict | None, optional):
                preparation settings. Defaults to None.
            protein_col (str, optional):
                name of the column with protein accessions. Defaults to "accession".
            random_state (int, optional):
                random seed to use in the dataset. Defaults to `None`
        Returns:
            QSPRDataset: a `QSPRDataset` object
        """
        df = self.getPCMDF()
        ret = PCMDataSet(
            name,
            protein_col=protein_col,
            protein_seq_provider=self.getPCMSeqProvider(),
            target_props=target_props,
            df=df,
            store_dir=self.generatedDataPath,
            random_state=random_state,
        )
        if preparation_settings:
            ret.prepareDataset(**preparation_settings)
        return ret
