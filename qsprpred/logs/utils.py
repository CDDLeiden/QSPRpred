"""
utils

Created by: Martin Sicho
On: 17.05.22, 9:55
"""
import datetime
import json
import logging
import os
import re
import shutil

import git

from . import config, setLogger
from .config import LogFileConfig

BACKUP_DIR_FOLDER_PREFIX = "backup"

def enable_file_logger(
    log_folder,
    filename,
    debug=False,
    log_name=None,
    init_data=None,
    disable_existing_loggers=True,
):

    path = os.path.join(log_folder, filename)
    config.config_logger(path, debug, disable_existing_loggers=disable_existing_loggers)

    # get logger and init configuration
    log = logging.getLogger(filename) if not log_name else logging.getLogger(log_name)
    log.setLevel(logging.INFO)
    setLogger(log)
    settings = LogFileConfig(path, log, debug)

    # Begin log file
    config.init_logfile(log, json.dumps(init_data, sort_keys=False, indent=2))

    return settings


def generate_backup_runID(path="."):
    """
    Generates runID for generation backups of files to be overwritten.

    If no previous backfiles (starting with #) exists, runid is set to 0, else
    to previous runid+1
    """

    regex = f"{BACKUP_DIR_FOLDER_PREFIX}_[0-9]+"
    previous = [
        int(re.search("[0-9]+", _file)[0])
        for _file in os.listdir(path) if re.match(regex, _file)
    ]

    runid = 1
    if previous:
        runid = max(previous) + 1

    # backup_files = sorted([ _file for _file in os.listdir(path) if _file.startswith('#')]) # noqa: E501
    # if len(backup_files) == 0 :
    #     runid = 0
    # else :
    #     previous_id = max([int(_file.split('.')[-1][:-1]) for _file in backup_files])
    #     runid = previous_id + 1

    return runid


def generate_backup_dir(root: str, backup_id: int):
    """Generates backup directory for files to be overwritten.

    Args:
        root (str): the root directory
        backup_id (int): the ID of the backup

    Returns:
        new_dir: the path to the new backup directory
    """
    new_dir = os.path.join(
        root, f"{BACKUP_DIR_FOLDER_PREFIX}_{str(backup_id).zfill(5)}"
    )
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir


def backup_files_in_folder(
    _dir: str,
    backup_id: int,
    output_prefixes,
    output_extensions="dummy",
    cp_suffix=None
):
    """Backs up files in a specified directory to a backup directory.

    Args:
        _dir (str): The directory path where the files to be backed up are located.
        backup_id (int): The ID of the backup.
        output_prefixes (str): The prefix of the output files to be backed up.
        output_extensions (str, optional): The extension of the output files to be
            backed up. Defaults to "dummy".
        cp_suffix (list of str, optional): The suffix of the files to be copied instead
            of moved. Defaults to None.

    Returns:
        str: A message indicating which files were backed up and where they
            were moved/copied.
    """
    message = ""
    existing_files = os.listdir(_dir)
    if cp_suffix and all(
        any(_file.split(".")[0].endswith(suff) for suff in cp_suffix)
        for _file in existing_files
    ):
        return message
    for _file in existing_files:
        if _file.startswith(output_prefixes) or _file.endswith(output_extensions):
            backup_dir = generate_backup_dir(_dir, backup_id)
            backup_log = open(os.path.join(backup_dir, "backuplog.log"), "w")
            if cp_suffix is not None and any(
                _file.split(".")[0].endswith(suff) for suff in cp_suffix
            ):
                shutil.copyfile(
                    os.path.join(_dir, _file), os.path.join(backup_dir, _file)
                )
                message += (
                    f"Already existing '{_file}' "
                    "was copied to {os.path.abspath(backup_dir)}\n"
                )
            else:
                os.rename(os.path.join(_dir, _file), os.path.join(backup_dir, _file))
                backup_log.write(
                    f"[{datetime.datetime.now()}] : {_file} "
                    f"was moved from {os.path.abspath(_dir)}"
                )
                message += (
                    f"Already existing '{_file}' "
                    "was moved to {os.path.abspath(backup_dir)}\n"
                )
    return message


def backup_files(base_dir: str, folder: str, output_prefixes: tuple, cp_suffix=None):
    dir = base_dir + "/" + folder
    if os.path.exists(dir):
        backup_id = generate_backup_runID(dir)
        if folder in "qspr/data":
            message = backup_files_in_folder(
                dir,
                backup_id,
                output_prefixes,
                output_extensions=("json", "log"),
                cp_suffix=cp_suffix,
            )
        if folder == "qspr/models":
            message = backup_files_in_folder(
                dir,
                backup_id,
                output_prefixes,
                output_extensions=("json", "log"),
                cp_suffix=cp_suffix,
            )

        if folder == "qspr/predictions":
            message = backup_files_in_folder(
                dir,
                backup_id,
                output_prefixes,
                output_extensions=("json", "log"),
                cp_suffix=cp_suffix,
            )
        return message
    else:
        return ""
