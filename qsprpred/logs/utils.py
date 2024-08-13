import datetime
import json
import logging
import os
import re
import shutil
import subprocess
from typing import Optional

from . import config, setLogger
from .config import LogFileConfig

BACKUP_DIR_FOLDER_PREFIX = "backup"


def export_conda_environment(filepath: str):
    """Export the conda environment to a yaml file.

    Args:
        filepath (str): path to the yaml file

    Raises:
        subprocess.CalledProcessError: if the command fails
        Exception: if an unexpected error occurs
    """
    try:
        cmd = f"conda env export > {filepath}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"Environment exported to {filepath} successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error exporting the environment: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def enable_file_logger(
    log_folder: str,
    filename: str,
    debug: bool = False,
    log_name: Optional[str] = None,
    init_data: Optional[dict] = None,
    disable_existing_loggers: bool = False,
):
    """Enable file logging.

    Args:
        log_folder (str): path to the folder where the log file should be stored
        filename (str): name of the log file
        debug (bool): whether to enable debug logging. Defaults to False.
        log_name (str, optional): name of the logger. Defaults to None.
        init_data (dict, optional): initial data to be logged. Defaults to None.
        disable_existing_loggers (bool): whether to disable existing loggers.
    """
    # create log folder if it does not exist
    path = os.path.join(log_folder, filename)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # configure logging
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

    # backup_files = sorted([ _file for _file in os.listdir(path) if _file.startswith('#')])
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
    cp_suffix=None,
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
    # if only files with cp_suffix are already existing, only return empty message
    if cp_suffix and all(
        any(_file.split(".")[0].endswith(suff) for suff in cp_suffix)
        for _file in existing_files
    ):
        return message
    for _file in existing_files:
        if _file.startswith(output_prefixes) or _file.endswith(output_extensions):
            # create backup directory
            backup_dir = generate_backup_dir(_dir, backup_id)
            backup_log = open(os.path.join(backup_dir, "backuplog.log"), "w")
            # copy file if it has a suffix in cp_suffix
            if cp_suffix is not None and any(
                _file.split(".")[0].endswith(suff) for suff in cp_suffix
            ):
                shutil.copyfile(
                    os.path.join(_dir, _file), os.path.join(backup_dir, _file)
                )
                message += (
                    f"Already existing '{_file}' "
                    f"was copied to {os.path.abspath(backup_dir)}\n"
                )
            # move file otherwise
            else:
                os.rename(os.path.join(_dir, _file), os.path.join(backup_dir, _file))
                backup_log.write(
                    f"[{datetime.datetime.now()}] : {_file} "
                    f"was moved from {os.path.abspath(_dir)}"
                )
                message += (
                    f"Already existing '{_file}' "
                    f"was moved to {os.path.abspath(backup_dir)}\n"
                )
    return message


def backup_files(output_dir: str, output_prefixes: tuple, cp_suffix=None):
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        backup_id = generate_backup_runID(output_dir)
        message = backup_files_in_folder(
            output_dir,
            backup_id,
            output_prefixes,
            output_extensions=("json", "log"),
            cp_suffix=cp_suffix,
        )
        return message
    else:
        return ""
