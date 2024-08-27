from qsprpred.data import MoleculeTable

from qsprpred.data.sources.papyrus import Papyrus

from qsprpred.utils.testing.base import QSPRTestCase

from qsprpred.utils.testing.path_mixins import DataSetsPathMixIn


class PapyrusSourceTest(DataSetsPathMixIn, QSPRTestCase):

    def setUp(self):
        super().setUp()
        self.setUpPaths()

    def test_papyrus_source(self):
        data_dir = self.generatedDataPath
        acc_keys = [
            "P29274",  # A2A
        ]  # Adenosine receptor A2A (https://www.uniprot.org/uniprotkb/P29274/entry)
        dataset_name = "A2A_LIGANDS"  # name of the file to be generated
        quality = "high"  # choose minimum quality from {"high", "medium", "low"}
        papyrus_version = "05.6"  # Papyrus database version

        papyrus = Papyrus(
            data_dir=data_dir,
            stereo=False,
            version=papyrus_version,
            plus_only=True,
        )

        dataset = papyrus.getData(
            dataset_name, acc_keys, quality, use_existing=True
        )
        self.assertTrue(isinstance(dataset, MoleculeTable))
        self.assertTrue(len(dataset) > 0)
