#!/bin/bash

# prepare tutorial data
set -e

export DATA_DIR="../../tutorials/tutorial_data"

# Create the dataset if it doesn't exist
if [ ! -d "${DATA_DIR}/papyrus" ]; then
    python "${DATA_DIR}/create_tutorial_data.py"
fi