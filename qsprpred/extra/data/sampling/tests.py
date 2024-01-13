from unittest import TestCase

from parameterized import parameterized

from qsprpred.data import RandomSplit, ScaffoldSplit, ClusterSplit
from qsprpred.data.descriptors.sets import RDKitDescs
from qsprpred.extra.data.descriptors.sets import ProDec
from qsprpred.extra.data.sampling.splits import (
    PCMSplit,
    LeaveTargetsOut,
    TemporalPerTarget,
)
from qsprpred.extra.data.tables.pcm import PCMDataSet
from qsprpred.extra.data.utils.testing.path_mixins import DataSetsMixInExtras


class TestPCMSplitters(DataSetsMixInExtras, TestCase):
    def setUp(self):
        super().setUp()
        self.setUpPaths()
        self.msaProvider = self.getMSAProvider(self.generatedDataPath)
        self.dataset = self.createPCMDataSet(f"{self.__class__.__name__}_test")
        self.dataset.addDescriptors([ProDec(["Zscale Hellberg"], self.msaProvider)])
        self.dataset.addDescriptors([RDKitDescs()])

    @parameterized.expand([(RandomSplit(),), (ScaffoldSplit(),), (ClusterSplit(),)])
    def testPCMSplit(self, splitter):
        splitter = PCMSplit(splitter)
        self.dataset.split(splitter, featurize=True)
        train, test = self.dataset.getFeatures()
        train, test = train.index, test.index
        test_targets = self.dataset.getProperty(self.dataset.proteinCol).loc[test]
        train_targets = self.dataset.getProperty(self.dataset.proteinCol).loc[train]
        test_smiles = self.dataset.getProperty(self.dataset.smilesCol).loc[test]
        train_smiles = self.dataset.getProperty(self.dataset.smilesCol).loc[train]
        self.assertEqual(len(test_targets), len(test))
        self.assertEqual(len(train_targets), len(train))
        self.assertTrue(
            set(test_smiles.unique()).isdisjoint(set(train_smiles.unique()))
        )

    def testPCMSplitRandomShuffle(self):
        seed = self.dataset.randomState
        self.dataset.save()
        splitter = PCMSplit(RandomSplit(), dataset=self.dataset)
        self.dataset.split(splitter, featurize=True)
        train, test = self.dataset.getFeatures()
        train_order = train.index.tolist()
        test_order = test.index.tolist()
        # reload and check if orders are the same if we redo the split
        dataset = PCMDataSet.fromFile(self.dataset.metaFile)
        splitter = PCMSplit(RandomSplit(), dataset=dataset)
        dataset.split(splitter, featurize=True)
        train, test = dataset.getFeatures()
        self.assertEqual(dataset.randomState, seed)
        self.assertListEqual(train.index.tolist(), train_order)
        self.assertListEqual(test.index.tolist(), test_order)

    def testLeaveTargetOut(self):
        target = self.dataset.getProteinKeys()[0:2]
        splitter = LeaveTargetsOut(targets=target)
        self.dataset.split(splitter, featurize=True)
        train, test = self.dataset.getFeatures()
        train, test = train.index, test.index
        test_targets = self.dataset.getProperty(self.dataset.proteinCol).loc[test]
        train_targets = self.dataset.getProperty(self.dataset.proteinCol).loc[train]
        self.assertEqual(len(test_targets), len(test))
        self.assertEqual(len(train_targets), len(train))
        self.assertTrue(
            set(test_targets.unique()).isdisjoint(set(train_targets.unique()))
        )

    def testPerTargetTemporal(self):
        year_col = "Year"
        year = 2015
        splitter = TemporalPerTarget(
            year_col=year_col,
            split_years={key: year for key in self.dataset.getProteinKeys()},
        )
        self.dataset.split(splitter, featurize=True)
        train, test = self.dataset.getFeatures()
        self.assertTrue(self.dataset.getDF()[year_col].loc[train.index].max() <= year)
        self.assertTrue(self.dataset.getDF()[year_col].loc[test.index].min() > year)
