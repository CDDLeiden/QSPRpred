import logging
import os

from rdkit import rdBase

from .logs import logger, setLogger
from .tasks import ModelTasks, TargetSpec, TargetTasks

_log_level = os.environ.get("QSPR_LOG_LEVEL", "INFO")

logger.setLevel(getattr(logging, _log_level.upper(), logging.INFO))
setLogger(logger)

__all__ = ["ModelTasks", "TargetSpec", "TargetTasks"]

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.info")
rdBase.DisableLog("rdApp.warning")

__version__ = "3.0.0"
if os.path.exists(os.path.join(os.path.dirname(__file__), "_version.py")):
    from ._version import version

    __version__ = version

VERSION = __version__
