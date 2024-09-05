from parameterized import parameterized

from ... import TargetTasks
from ...data.chem.clustering import (
    FPSimilarityLeaderPickerClusters,
    FPSimilarityMaxMinClusters,
    RandomClusters,
    ScaffoldClusters,
)
from ...data.chem.scaffolds import BemisMurcko, BemisMurckoRDKit
from ...data.tables.qspr import QSPRTable
from ...utils.testing.base import QSPRTestCase
from ...utils.testing.path_mixins import DataSetsPathMixIn
from .standardizers.chembl import ChemblStandardizer


class TestScaffolds(DataSetsPathMixIn, QSPRTestCase):
    """Test calculation of scaffolds."""
    def setUp(self):
        """Create a small dataset."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createSmallTestDataSet(self.__class__.__name__)

    @parameterized.expand(
        [
            ("Murcko", BemisMurckoRDKit()),
            ("BemisMurcko", BemisMurcko()),
            ("BemisMurckoCSK", BemisMurcko(True, True)),
            ("BemisMurckoJustCSK", BemisMurcko(False, True)),
            ("BemisMurckoOff", BemisMurcko(False, False)),
        ]
    )
    def testScaffoldAdd(self, _, scaffold):
        """Test the adding and getting of scaffolds."""
        self.dataset.addScaffolds([scaffold])
        scaffs = self.dataset.getScaffolds()
        self.assertEqual(scaffs.shape, (len(self.dataset), 1))
        self.dataset.addScaffolds(
            [scaffold], add_rdkit_scaffold=False, recalculate=True
        )
        scaffs = self.dataset.getScaffolds(include_mols=True)
        self.assertEqual(scaffs.shape, (len(self.dataset), 1))
        # for mol in scaffs[f"Scaffold_{scaffold}_RDMol"]:
        #     self.assertTrue(isinstance(mol, Chem.rdchem.Mol))


class TestClusters(DataSetsPathMixIn, QSPRTestCase):
    """Test calculation of clusters."""
    def setUp(self):
        """Create a test dataset."""
        super().setUp()
        self.setUpPaths()
        self.dataset = self.createLargeTestDataSet(self.__class__.__name__)

    @parameterized.expand(
        [
            ("Random", RandomClusters()),
            ("FPSimilarityMaxMin", FPSimilarityMaxMinClusters()),
            ("FPSimilarityLeaderPicker", FPSimilarityLeaderPickerClusters()),
            ("Scaffold", ScaffoldClusters(BemisMurckoRDKit())),
        ]
    )
    def testClusterAdd(self, _, cluster):
        """Test the adding and getting of clusters."""
        self.dataset.addClusters([cluster])
        clusters = self.dataset.getClusters()
        self.assertEqual(clusters.shape, (len(self.dataset), 1))
        self.dataset.addClusters([cluster], recalculate=True)
        self.assertEqual(clusters.shape, (len(self.dataset), 1))


class TestStandardizers(DataSetsPathMixIn, QSPRTestCase):
    """Test the standardizers."""
    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def testInvalidFilter(self):
        """Test the invalid filter."""
        df = self.getSmallDF()
        orig_len = len(df)
        mask = [False] * orig_len
        mask[0] = True
        df.loc[mask, "SMILES"] = "C(C)(C)(C)(C)(C)(C)(C)(C)(C)"  # bad valence example
        dataset = QSPRTable.fromDF(
            "standardization_test_invalid_filter",
            df=df,
            target_props=[{
                "name": "VDss",
                "task": TargetTasks.REGRESSION
            }],
            path=self.generatedDataPath,
        )
        self.assertEqual(len(dataset), len(df))
        dataset.applyStandardizer(ChemblStandardizer())
        self.assertEqual(len(dataset), orig_len - 1)
