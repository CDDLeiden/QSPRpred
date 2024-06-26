from parameterized import parameterized

from ... import TargetTasks
from ...data import QSPRDataset
from ...data.chem.scaffolds import BemisMurckoRDKit, BemisMurcko
from ...data.chem.clustering import RandomClusters, FPSimilarityMaxMinClusters, FPSimilarityLeaderPickerClusters, ScaffoldClusters
from ...utils.testing.base import QSPRTestCase
from ...utils.testing.path_mixins import DataSetsPathMixIn


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
            [scaffold],
            add_rdkit_scaffold=False,
            recalculate=True
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
        dataset = QSPRDataset(
            "standardization_test_invalid_filter",
            df=df,
            target_props=[{"name": "CL", "task": TargetTasks.REGRESSION}],
            drop_invalids=False,
            drop_empty=False,
        )
        dataset.standardizeSmiles("chembl", drop_invalid=False)
        self.assertEqual(len(dataset), len(df))
        dataset.standardizeSmiles("chembl", drop_invalid=True)
        self.assertEqual(len(dataset), orig_len - 1)
