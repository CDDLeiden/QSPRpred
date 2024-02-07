import logging
import os
import sys

verbose = os.environ.get("QSPRPRED_VERBOSE_LOGGING", "false").lower() == "true"

logger = None
if not logger:
    logger = logging.getLogger("qsprpred")
    logger.setLevel(logging.WARNING)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    if verbose:
        formatter = logging.Formatter(
            "%(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
        )
        sh.setFormatter(formatter)
    else:
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        sh.setFormatter(formatter)
    logger.addHandler(sh)


def setLogger(log):
    sys.modules[__name__].qsprpred = log
