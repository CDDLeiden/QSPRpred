import platform
from typing import Type
from unittest import TestCase, skipIf

import numpy as np
from parameterized import parameterized
from sklearn.preprocessing import StandardScaler

from qsprpred import TargetTasks, TargetProperty
from qsprpred.data import RandomSplit
from qsprpred.data.descriptors.fingerprints import MorganFP
from qsprpred.data.descriptors.sets import (
    DrugExPhyschem,
    DescriptorSet,
)
from qsprpred.data.processing.feature_filters import (
    LowVarianceFilter,
    HighCorrelationFilter,
)
from qsprpred.extra.data.descriptors.fingerprints import (
    CDKFP,
    CDKExtendedFP,
    CDKGraphOnlyFP,
    CDKMACCSFP,
    CDKPubchemFP,
    CDKEStateFP,
    CDKSubstructureFP,
    CDKKlekotaRothFP,
    CDKAtomPairs2DFP,
)
from qsprpred.extra.data.descriptors.sets import (
    Mold2,
    PaDEL,
    Mordred,
    ExtendedValenceSignature,
    ProDec,
)
from qsprpred.extra.data.tables.pcm import PCMDataSet
from qsprpred.extra.data.utils.msa_calculator import MAFFT, ClustalMSA, BioPythonMSA
from qsprpred.extra.data.utils.testing.path_mixins import DataSetsMixInExtras
from qsprpred.utils.testing.base import QSPRTestCase
from qsprpred.utils.testing.check_mixins import DescriptorInDataCheckMixIn


class TestDescriptorSetsExtra(DataSetsMixInExtras, QSPRTestCase):
    """Test descriptor sets with extra features.

    Attributes:
        dataset (QSPRDataset): dataset for testing, shuffled
    """

    def setUp(self):
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)
        self.dataset.shuffle()

    @skipIf(platform.system() == "Darwin", "Mold2 not supported on Mac OS")
    def testMold2(self):
        """Test the Mold2 descriptor calculator."""
        self.dataset.addDescriptors([Mold2()])
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 777))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    def testPaDELDescriptors(self):
        """Test the PaDEL descriptor calculator."""
        self.dataset.addDescriptors([PaDEL()])
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), 1444))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    @parameterized.expand(
        [
            (CDKFP, 1024),
            (CDKExtendedFP, 1024),
            (CDKGraphOnlyFP, 1024),
            (CDKMACCSFP, 166),
            (CDKPubchemFP, 881),
            (CDKEStateFP, 79),
            (CDKSubstructureFP, 307),
            (CDKKlekotaRothFP, 4860),
            (CDKAtomPairs2DFP, 780),
        ]
    )
    def testPaDELFingerprints(self, fp_type, nbits):
        dataset = self.createSmallTestDataSet(f"{self.__class__.__name__}_{fp_type}")
        dataset.addDescriptors([fp_type()])
        self.assertEqual(dataset.X.shape, (len(dataset), nbits))
        self.assertTrue(dataset.X.any().any())
        self.assertTrue(dataset.X.any().sum() > 1)

    def testMordred(self):
        """Test the Mordred descriptor calculator."""
        import mordred
        from mordred import descriptors

        self.dataset.addDescriptors([Mordred()])
        self.assertEqual(
            self.dataset.X.shape,
            (
                len(self.dataset),
                len(mordred.Calculator(descriptors).descriptors),
            ),
        )
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)

    def testExtendedValenceSignature(self):
        """Test the SMILES based signature descriptor calculator."""
        desc_set = ExtendedValenceSignature(1)
        self.dataset.addDescriptors([desc_set], recalculate=True)
        self.dataset.featurize()
        self.assertTrue(self.dataset.X.shape[1] == len(desc_set))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)


class TestPCMDataSet(DataSetsMixInExtras, TestCase):
    """Test the PCM data set features.

    Attributes:
        dataset (QSPRDataset): dataset for testing
        sampleDescSet (DescriptorSet): descriptor set for testing
        defaultMSA (BioPythonMSA): MSA provider for testing
    """

    def setUp(self):
        """Set up the test Dataframe."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createPCMDataSet(self.__class__.__name__)
        self.defaultMSA = self.getMSAProvider(self.generatedDataPath)
        self.sampleDescSet = ProDec(
            sets=["Zscale Hellberg"], msa_provider=self.defaultMSA
        )

    @parameterized.expand(
        [
            ("MAFFT", MAFFT),
            ("ClustalMSA", ClustalMSA),
        ]
    )
    def testSerialization(self, _, msa_provider_cls: Type[BioPythonMSA]):
        """Test the serialization of dataset with data split.

        Args:
            msa_provider_cls (BioPythonMSA): MSA provider class
        """
        provider = msa_provider_cls(out_dir=self.generatedDataPath)
        self.sampleDescSet.msaProvider = provider
        dataset = self.createPCMDataSet(self.__class__.__name__)
        split = RandomSplit(test_fraction=0.2)
        dataset.prepareDataset(
            split=split,
            feature_calculators=[self.sampleDescSet],
            feature_standardizer=StandardScaler(),
            feature_filters=[LowVarianceFilter(0.05), HighCorrelationFilter(0.9)],
        )
        ndata = dataset.getDF().shape[0]
        self.validate_split(dataset)
        self.assertEqual(dataset.X_ind.shape[0], round(ndata * 0.2))
        test_ids = dataset.X_ind.index.values
        train_ids = dataset.y_ind.index.values
        dataset.save()
        # load dataset and test if all checks out after loading
        dataset_new = PCMDataSet.fromFile(dataset.metaFile)
        self.assertIsInstance(dataset_new, PCMDataSet)
        self.validate_split(dataset_new)
        self.assertEqual(dataset.X_ind.shape[0], round(ndata * 0.2))
        self.assertEqual(len(dataset_new.descriptorSets), len(dataset_new.descriptors))
        self.assertTrue(dataset_new.featureStandardizer)
        self.assertTrue(len(dataset_new.featureNames) == len(self.sampleDescSet))
        self.assertTrue(all(mol_id in dataset_new.X_ind.index for mol_id in test_ids))
        self.assertTrue(all(mol_id in dataset_new.y_ind.index for mol_id in train_ids))
        # clear files and try saving again
        dataset_new.clearFiles()
        dataset_new.save()

    def testSwitching(self):
        """Test if the feature calculator can be switched to a new dataset."""
        dataset = self.createPCMDataSet(self.__class__.__name__)
        split = RandomSplit(test_fraction=0.5)
        lv = LowVarianceFilter(0.05)
        hc = HighCorrelationFilter(0.9)
        dataset.prepareDataset(
            split=split,
            feature_calculators=[self.sampleDescSet],
            feature_filters=[lv, hc],
            recalculate_features=True,
            feature_fill_value=np.nan,
        )
        ndata = dataset.getDF().shape[0]
        self.assertEqual(len(dataset.descriptorSets), len(dataset.descriptors))
        self.assertEqual(
            dataset.X_ind.shape, (round(ndata * 0.5), len(self.sampleDescSet))
        )
        # create new dataset with different feature calculator
        dataset_next = self.createPCMDataSet(f"{self.__class__.__name__}_next")
        dataset_next.prepareDataset(
            split=split,
            feature_calculators=[self.sampleDescSet],
            feature_filters=[lv, hc],
            recalculate_features=True,
            feature_fill_value=np.nan,
        )
        self.assertEqual(
            len(dataset_next.descriptorSets), len(dataset_next.descriptors)
        )
        self.assertEqual(
            dataset_next.X_ind.shape, (round(ndata * 0.5), len(self.sampleDescSet))
        )
        self.assertEqual(
            dataset_next.X.shape, (round(ndata * 0.5), len(self.sampleDescSet))
        )

    def testWithMolDescriptors(self):
        """Test the calculation of protein and molecule descriptors."""
        calcs = [
            ProDec(sets=["Zscale Hellberg"], msa_provider=self.defaultMSA),
            MorganFP(radius=2, nBits=128),
            DrugExPhyschem(),
        ]
        self.dataset.prepareDataset(
            feature_calculators=calcs,
            feature_standardizer=StandardScaler(),
            split=RandomSplit(test_fraction=0.2),
        )
        # test if all descriptors are there
        expected_length = 0
        for calc in calcs:
            expected_length += len(calc)
        self.assertEqual(self.dataset.X.shape[1], expected_length)
        # filter features and test if they are there after saving and loading
        self.dataset.filterFeatures(
            [LowVarianceFilter(0.05), HighCorrelationFilter(0.9)]
        )
        feats_left = self.dataset.X.shape[1]
        self.dataset.save()
        dataset_new = PCMDataSet.fromFile(self.dataset.metaFile)
        self.assertEqual(dataset_new.X.shape[1], feats_left)

    @parameterized.expand(
        [
            ("MAFFT", MAFFT),
            ("ClustalMSA", ClustalMSA),
        ]
    )
    def testProDec(self, _, provider_class):
        provider = provider_class(out_dir=self.generatedDataPath)
        descset = ProDec(sets=["Zscale Hellberg"], msa_provider=provider)
        self.dataset.addDescriptors([descset])
        self.assertEqual(self.dataset.X.shape, (len(self.dataset), len(descset)))
        self.assertTrue(self.dataset.X.any().any())
        self.assertTrue(self.dataset.X.any().sum() > 1)


class TestDescriptorsExtra(
    DataSetsMixInExtras, DescriptorInDataCheckMixIn, QSPRTestCase
):
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    @parameterized.expand(
        [
            (
                f"{desc_set}",
                desc_set,
                [{"name": "CL", "task": TargetTasks.REGRESSION}],
            )
            for desc_set in DataSetsMixInExtras.getAllDescriptors()
        ]
    )
    def testDescriptorsExtraAll(
        self,
        _,
        desc_set: DescriptorSet,
        target_props: list[dict | TargetProperty],
    ):
        """Test the calculation of extra descriptors with data preparation."""
        dataset = self.createLargeTestDataSet(
            name=self.getDatSetName(desc_set, target_props),
            target_props=target_props,
            n_jobs=2,
            chunk_size=self.chunkSize,
        )
        self.checkDataSetContainsDescriptorSet(
            dataset, desc_set, self.getDefaultPrep(), target_props
        )


class TestDescriptorsPCM(DataSetsMixInExtras, DescriptorInDataCheckMixIn, TestCase):
    """Test the calculation of PCM descriptors with data preparation.

    Attributes:
        defaultMSA (MSAProvider): Default MSA provider.
    """

    def setUp(self):
        super().setUp()
        self.setUpPaths()
        self.defaultMSA = self.getMSAProvider(self.generatedDataPath)

    @parameterized.expand(
        [
            (
                f"{desc_set}_{TargetTasks.MULTICLASS}",
                desc_set,
                [
                    {
                        "name": "pchembl_value_Median",
                        "task": TargetTasks.MULTICLASS,
                        "th": [2.0, 5.5, 6.5, 12.0],
                    }
                ],
            )
            for desc_set in DataSetsMixInExtras.getAllProteinDescriptors()
        ]
        + [
            (
                f"{desc_set}_{TargetTasks.REGRESSION}",
                desc_set,
                [{"name": "pchembl_value_Median", "task": TargetTasks.REGRESSION}],
            )
            for desc_set in DataSetsMixInExtras.getAllProteinDescriptors()
        ]
        + [
            (
                f"{desc_set}_Multitask",
                desc_set,
                [
                    {"name": "pchembl_value_Median", "task": TargetTasks.REGRESSION},
                    {
                        "name": "pchembl_value_Mean",
                        "task": TargetTasks.SINGLECLASS,
                        "th": [6.5],
                    },
                ],
            )
            for desc_set in DataSetsMixInExtras.getAllProteinDescriptors()
        ]
    )
    def testDescriptorsPCMAll(self, _, desc_set, target_props):
        """Tests all available descriptor sets with data set preparation.

        Note that they are not checked with all possible settings
        and all possible preparations,
        but only with the default settings provided
        by `DataSetsPathMixIn.getDefaultPrep()`.
        The list itself is defined and configured
        by `DataSetsPathMixIn.getAllDescriptors()`,
        so if you need a specific descriptor tested, add it there.
        """
        dataset = self.createPCMDataSet(
            name=f"{self.getDatSetName(desc_set, target_props)}_pcm",
            target_props=target_props,
        )
        self.checkDataSetContainsDescriptorSet(
            dataset, desc_set, self.getDefaultPrep(), target_props
        )
        self.checkDataSetContainsDescriptorSet(
            dataset, desc_set, self.getDefaultPrep(), target_props
        )
