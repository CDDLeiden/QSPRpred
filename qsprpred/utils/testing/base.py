import logging
import os
from unittest import TestCase

from ...logs import logger, setLogger


class QSPRTestCase(TestCase):
    def setUp(self):
        self.nCPU = os.cpu_count()
        self.GPUs = [0]
        self.chunkSize = None
        logger.setLevel(logging.DEBUG)
        setLogger(logger)
