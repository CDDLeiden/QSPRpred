import logging
import sys

logger = None

if not logger:
    logger = logging.getLogger("qsprpred")
    logger.setLevel(logging.INFO)


def setLogger(log):
    sys.modules[__name__].qsprpred = log
