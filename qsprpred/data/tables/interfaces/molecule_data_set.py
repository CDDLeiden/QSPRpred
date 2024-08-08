from abc import ABC

from qsprpred.data.storage.interfaces.chunk_iterable import ChunkIterable
from qsprpred.data.storage.interfaces.descriptor_provider import DescriptorProvider
from qsprpred.data.storage.interfaces.mol_processable import MolProcessable
from qsprpred.data.storage.interfaces.property_storage import PropertyStorage
from qsprpred.data.storage.interfaces.searchable import SMARTSSearchable
from qsprpred.utils.interfaces.randomized import Randomized
from qsprpred.utils.interfaces.summarizable import Summarizable


class MoleculeDataSet(
    PropertyStorage,
    DescriptorProvider,
    MolProcessable,
    SMARTSSearchable,
    Summarizable,
    ChunkIterable,
    Randomized,
    ABC
):
    pass
