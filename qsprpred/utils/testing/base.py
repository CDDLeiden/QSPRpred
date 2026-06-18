import logging
from unittest import TestCase

from ...logs import logger, setLogger


class QSPRTestCase(TestCase):
    def setUp(self):
        self.nCPU = 2
        self.GPUs = [0]
        self.chunkSize = None
        logger.setLevel(logging.INFO)
        setLogger(logger)
