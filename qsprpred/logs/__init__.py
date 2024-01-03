import logging
import sys

logger = None
if not logger:
    logger = logging.getLogger("qsprpred")
    logger.setLevel(logging.WARNING)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
    )
    sh.setFormatter(formatter)
    logger.addHandler(sh)


def setLogger(log):
    sys.modules[__name__].qsprpred = log
